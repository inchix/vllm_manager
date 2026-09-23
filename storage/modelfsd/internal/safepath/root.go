package safepath

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"golang.org/x/sys/unix"
)

// Root is a deployment-declared directory that paths may be resolved beneath.
//
// Roots come from the unit file or compose definition, never from the API. The API can
// only name a root by id; it cannot add one, rename one, or repoint one. That is a
// deliberate friction: a compromise of the web plane cannot reach /etc, /home or the
// container socket. See docs/04-data-model.md §2.
type Root struct {
	id   string
	path string

	mu  sync.Mutex
	dir *os.File // O_PATH|O_DIRECTORY handle, pinning the root against rename
}

// OpenRoot pins path as a resolution root. The directory handle is held open for the
// lifetime of the Root, so that renaming or replacing the directory underneath us cannot
// silently redirect later resolutions.
func OpenRoot(id, path string) (*Root, error) {
	if id == "" {
		return nil, fmt.Errorf("safepath: root id must not be empty")
	}
	if !filepath.IsAbs(path) {
		return nil, fmt.Errorf("safepath: root %q: path %q is not absolute", id, path)
	}
	clean := filepath.Clean(path)

	fd, err := unix.Open(clean, unix.O_PATH|unix.O_DIRECTORY|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("safepath: root %q: open %s: %w", id, clean, err)
	}
	f := os.NewFile(uintptr(fd), clean)

	var st unix.Stat_t
	if err := unix.Fstat(fd, &st); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("safepath: root %q: fstat: %w", id, err)
	}
	if st.Mode&unix.S_IFMT != unix.S_IFDIR {
		_ = f.Close()
		return nil, fmt.Errorf("safepath: root %q: %s is not a directory", id, clean)
	}
	return &Root{id: id, path: clean, dir: f}, nil
}

// ID returns the root's deployment-declared identifier.
func (r *Root) ID() string { return r.id }

// Path returns the root's absolute path. For display and logging only: never resolve
// against this string, always resolve against the Root.
func (r *Root) Path() string { return r.path }

// Close releases the root's directory handle.
func (r *Root) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.dir == nil {
		return nil
	}
	err := r.dir.Close()
	r.dir = nil
	return err
}

func (r *Root) fd() (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.dir == nil {
		return -1, fmt.Errorf("safepath: root %q is closed", r.id)
	}
	return int(r.dir.Fd()), nil
}

// Registry is the set of roots this process may resolve beneath. It is built once at
// startup from the deployment contract and is immutable thereafter.
type Registry struct {
	roots map[string]*Root
}

// NewRegistry pins every root in spec, a map of id to absolute path. If any root fails to
// open, every root opened so far is closed and the error is returned: a partially
// initialised registry is never handed back.
func NewRegistry(spec map[string]string) (*Registry, error) {
	reg := &Registry{roots: make(map[string]*Root, len(spec))}
	for id, path := range spec {
		root, err := OpenRoot(id, path)
		if err != nil {
			_ = reg.Close()
			return nil, err
		}
		reg.roots[id] = root
	}
	return reg, nil
}

// Root returns the named root, or ErrNoRoot if it was not declared at deploy time.
func (reg *Registry) Root(id string) (*Root, error) {
	r, ok := reg.roots[id]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrNoRoot, id)
	}
	return r, nil
}

// IDs returns the declared root identifiers, unordered.
func (reg *Registry) IDs() []string {
	out := make([]string, 0, len(reg.roots))
	for id := range reg.roots {
		out = append(out, id)
	}
	return out
}

// Close releases every root handle.
func (reg *Registry) Close() error {
	var first error
	for _, r := range reg.roots {
		if err := r.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}
