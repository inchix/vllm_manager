package safepath

import (
	"fmt"
	"io/fs"
	"os"
	"strings"
	"time"

	"golang.org/x/sys/unix"
)

// Resolver selects which implementation resolves a path.
type Resolver int

const (
	// ResolverAuto uses openat2 where available and the component walk otherwise.
	ResolverAuto Resolver = iota
	// ResolverOpenat2 requires openat2 and fails if it is unavailable.
	ResolverOpenat2
	// ResolverWalk requires the component walk, even where openat2 exists.
	ResolverWalk
)

func (r Resolver) String() string {
	switch r {
	case ResolverOpenat2:
		return "openat2"
	case ResolverWalk:
		return "walk"
	}
	return "auto"
}

// forced overrides resolver selection process-wide.
//
// This exists so the test suite and CI can drive every case through the fallback on a
// modern kernel. A degradation path that is never exercised decays into fiction and is
// discovered by a user on an old NAS; see docs/14-preflight.md §5.
var forced = resolverFromEnv()

func resolverFromEnv() Resolver {
	switch strings.ToLower(os.Getenv("MODELFSD_SAFEPATH_RESOLVER")) {
	case "walk":
		return ResolverWalk
	case "openat2":
		return ResolverOpenat2
	}
	return ResolverAuto
}

// SetResolver forces a resolver process-wide and returns a function restoring the
// previous setting. Intended for tests and for the preflight self-check.
func SetResolver(r Resolver) (restore func()) {
	prev := forced
	forced = r
	return func() { forced = prev }
}

// ActiveResolver reports which implementation resolution will actually use.
func ActiveResolver() Resolver {
	switch forced {
	case ResolverOpenat2, ResolverWalk:
		return forced
	}
	if HasOpenat2() {
		return ResolverOpenat2
	}
	return ResolverWalk
}

// Available reports whether any safe resolver works in this process. The appliance
// refuses to start when this is false: there is no acceptable degraded mode for path
// containment. See docs/00-overview.md §6.
func Available() bool {
	return HasOpenat2() || HasProcFD() || walkUsable()
}

// walkUsable reports whether the component walk can run without /proc, using the
// inode-verified reopen instead.
func walkUsable() bool {
	fd, err := unix.Open("/", unix.O_PATH|unix.O_DIRECTORY|unix.O_CLOEXEC, 0)
	if err != nil {
		return false
	}
	_ = unix.Close(fd)
	return true
}

// resolve opens rel beneath the root with the requested flags, refusing any path that
// would escape.
func (r *Root) resolve(rel string, mode Mode, flags int) (int, error) {
	if _, err := ValidateRel(rel); err != nil {
		return -1, err
	}
	rootfd, err := r.fd()
	if err != nil {
		return -1, err
	}

	switch ActiveResolver() {
	case ResolverOpenat2:
		if !HasOpenat2() {
			return -1, fmt.Errorf("safepath: openat2 forced but unavailable")
		}
		return resolveOpenat2(rootfd, rel, mode, flags)
	default:
		return resolveWalk(rootfd, rel, mode, flags)
	}
}

// OpenFile opens a regular file beneath the root for reading.
func (r *Root) OpenFile(rel string, mode Mode) (*os.File, error) {
	fd, err := r.resolve(rel, mode, unix.O_RDONLY)
	if err != nil {
		return nil, r.wrap(rel, mode, err)
	}
	return os.NewFile(uintptr(fd), r.display(rel)), nil
}

// OpenDir opens a directory beneath the root.
func (r *Root) OpenDir(rel string, mode Mode) (*os.File, error) {
	fd, err := r.resolve(rel, mode, unix.O_RDONLY|unix.O_DIRECTORY)
	if err != nil {
		return nil, r.wrap(rel, mode, err)
	}
	return os.NewFile(uintptr(fd), r.display(rel)), nil
}

// FileInfo is the subset of stat the appliance uses. It is deliberately not
// os.FileInfo: nothing above this layer should be tempted to reach for a path from it.
type FileInfo struct {
	Name    string // the final component as supplied, not as resolved
	Size    int64
	Mode    fs.FileMode
	ModTime time.Time
	IsDir   bool
	IsLink  bool
	Dev     uint64
	Ino     uint64
	Nlink   uint64
	UID     uint32
	GID     uint32
}

// Stat returns metadata for a path beneath the root.
//
// In ModeStrict a final symlink is refused rather than reported, because a caller in
// strict mode has no legitimate use for one. In ModeBeneath a symlink is followed, and
// refused only if it leaves the root.
func (r *Root) Stat(rel string, mode Mode) (*FileInfo, error) {
	fd, err := r.resolve(rel, mode, unix.O_PATH)
	if err != nil {
		return nil, r.wrap(rel, mode, err)
	}
	defer unix.Close(fd)

	var st unix.Stat_t
	if err := unix.Fstat(fd, &st); err != nil {
		return nil, r.wrap(rel, mode, err)
	}
	return statToInfo(baseName(rel), &st), nil
}

func statToInfo(name string, st *unix.Stat_t) *FileInfo {
	fi := &FileInfo{
		Name:    name,
		Size:    st.Size,
		Mode:    fs.FileMode(st.Mode & 0o7777),
		ModTime: time.Unix(st.Mtim.Sec, st.Mtim.Nsec),
		Dev:     uint64(st.Dev),
		Ino:     st.Ino,
		Nlink:   uint64(st.Nlink),
		UID:     st.Uid,
		GID:     st.Gid,
	}
	switch st.Mode & unix.S_IFMT {
	case unix.S_IFDIR:
		fi.IsDir = true
		fi.Mode |= fs.ModeDir
	case unix.S_IFLNK:
		fi.IsLink = true
		fi.Mode |= fs.ModeSymlink
	}
	return fi
}

// Entry is one directory entry. Name is returned as the raw bytes the filesystem holds:
// Linux permits filenames that are not valid UTF-8, and silently mangling them would make
// files unreachable. Callers rendering a name for a browser must sanitise it there.
type Entry struct {
	Name  string
	IsDir bool
}

// ReadDir lists a directory beneath the root.
func (r *Root) ReadDir(rel string, mode Mode) ([]Entry, error) {
	d, err := r.OpenDir(rel, mode)
	if err != nil {
		return nil, err
	}
	defer d.Close()

	names, err := d.Readdirnames(-1)
	if err != nil {
		return nil, r.wrap(rel, mode, err)
	}

	out := make([]Entry, 0, len(names))
	dirfd := int(d.Fd())
	for _, name := range names {
		e := Entry{Name: name}
		var st unix.Stat_t
		// AT_SYMLINK_NOFOLLOW: report the entry itself. Whether a link is followed is
		// a decision for the caller's next resolve, not for a listing.
		if err := unix.Fstatat(dirfd, name, &st, unix.AT_SYMLINK_NOFOLLOW); err == nil {
			e.IsDir = st.Mode&unix.S_IFMT == unix.S_IFDIR
		}
		out = append(out, e)
	}
	return out, nil
}

// Exists reports whether a path resolves beneath the root. A containment refusal is
// returned as an error, never as a plain false: "it escaped" and "it is not there" are
// different answers and conflating them hides attacks.
func (r *Root) Exists(rel string, mode Mode) (bool, error) {
	_, err := r.Stat(rel, mode)
	if err == nil {
		return true, nil
	}
	if isNotExist(err) {
		return false, nil
	}
	return false, err
}

// display renders a path for error messages and logs. It is never used for resolution.
func (r *Root) display(rel string) string {
	if rel == "" || rel == "." {
		return r.id + ":/"
	}
	return r.id + ":/" + rel
}

// wrap annotates an error with the root, path and mode, without leaking the resolved
// absolute path into anything a caller might log or return to a browser.
func (r *Root) wrap(rel string, mode Mode, err error) error {
	if err == nil {
		return nil
	}
	return &PathError{Root: r.id, Rel: rel, Mode: mode, Err: err}
}

// PathError reports a failed resolution.
type PathError struct {
	Root string
	Rel  string
	Mode Mode
	Err  error
}

func (e *PathError) Error() string {
	return fmt.Sprintf("safepath: %s:/%s (%s): %v", e.Root, e.Rel, e.Mode, e.Err)
}

func (e *PathError) Unwrap() error { return e.Err }

func baseName(rel string) string {
	if i := strings.LastIndexByte(rel, '/'); i >= 0 {
		return rel[i+1:]
	}
	return rel
}
