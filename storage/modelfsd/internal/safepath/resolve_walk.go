package safepath

import (
	"errors"
	"fmt"

	"golang.org/x/sys/unix"
)

// The fallback resolver, for kernels without openat2 (pre-5.6) or where seccomp filters
// it. It walks the path one component at a time, holding an O_PATH descriptor for each
// directory it has entered, so that containment is decided by descriptor identity rather
// than by string comparison. A "safe path" check that compares cleaned strings is not
// safe; this one never forms a path string at all.
//
// It is slower than openat2 (one syscall per component instead of one per path) and it
// carries a small TOCTOU window that the kernel closes for openat2. It exists so the
// appliance runs on older hosts, and doc 14 requires that it be exercised on every build
// rather than left to rot.

// walker holds the chain of directory descriptors from the root to the current position.
// stack[0] is always the root itself and is never popped, which is what makes ".."
// containment a length check rather than a path comparison.
type walker struct {
	stack []int // O_PATH|O_DIRECTORY fds; stack[0] is borrowed from Root and not closed
	links int
}

func (w *walker) top() int { return w.stack[len(w.stack)-1] }

func (w *walker) push(fd int) { w.stack = append(w.stack, fd) }

// pop closes and removes the current directory. It refuses to pop the root, which is how
// ".." is contained.
func (w *walker) pop() error {
	if len(w.stack) == 1 {
		return ErrEscape
	}
	fd := w.stack[len(w.stack)-1]
	w.stack = w.stack[:len(w.stack)-1]
	return unix.Close(fd)
}

// close releases every descriptor the walker opened, leaving the borrowed root alone.
func (w *walker) close() {
	for i := len(w.stack) - 1; i >= 1; i-- {
		_ = unix.Close(w.stack[i])
	}
	w.stack = w.stack[:1]
}

// resolveWalk opens rel beneath rootfd without openat2.
func resolveWalk(rootfd int, rel string, mode Mode, flags int) (int, error) {
	parts, err := ValidateRel(rel)
	if err != nil {
		return -1, err
	}

	w := &walker{stack: []int{rootfd}}
	defer w.close()

	// queue is consumed left to right. Resolving a symlink prepends its target's
	// components, which is how a link is followed without ever building a path string.
	queue := make([]string, len(parts))
	copy(queue, parts)

	for len(queue) > 0 {
		comp := queue[0]
		queue = queue[1:]

		switch comp {
		case ".", "":
			continue
		case "..":
			// Only reachable from inside a symlink target; ValidateRel rejects ".."
			// in caller-supplied input.
			if err := w.pop(); err != nil {
				return -1, err
			}
			continue
		}

		// O_PATH|O_NOFOLLOW on a symlink succeeds and yields a descriptor for the link
		// itself rather than its target, which is exactly what we need to inspect it.
		fd, err := unix.Openat(w.top(), comp, unix.O_PATH|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
		if err != nil {
			return -1, err
		}

		var st unix.Stat_t
		if err := unix.Fstat(fd, &st); err != nil {
			_ = unix.Close(fd)
			return -1, err
		}

		if st.Mode&unix.S_IFMT == unix.S_IFLNK {
			_ = unix.Close(fd)
			if mode == ModeStrict {
				return -1, ErrSymlink
			}
			w.links++
			if w.links > MaxSymlinks {
				return -1, fmt.Errorf("%w: more than %d symlinks", unix.ELOOP, MaxSymlinks)
			}
			target, err := readlinkat(w.top(), comp)
			if err != nil {
				return -1, err
			}
			if target == "" {
				return -1, fmt.Errorf("%w: empty symlink target", ErrInvalid)
			}
			if target[0] == '/' {
				// An absolute target leaves the root by definition. We do not
				// attempt to reinterpret it relative to the root: that would make
				// the meaning of a symlink depend on who opened it.
				return -1, ErrEscape
			}
			queue = append(splitTarget(target), queue...)
			continue
		}

		if len(queue) > 0 {
			// Interior component: must be a directory to continue into.
			if st.Mode&unix.S_IFMT != unix.S_IFDIR {
				_ = unix.Close(fd)
				return -1, unix.ENOTDIR
			}
			w.push(fd)
			continue
		}

		// Final component, and not a symlink. Reopen with the caller's flags.
		out, err := finalOpen(w.top(), comp, fd, flags)
		_ = unix.Close(fd)
		if err != nil {
			return -1, err
		}
		return out, nil
	}

	// Every component was consumed without a final open: the target is the directory we
	// are standing in, which for an empty path is the root itself.
	return reopenPath(w.top(), flags)
}

// finalOpen reopens the resolved leaf with the caller's flags.
//
// Reopening by name would reintroduce the race the walk exists to avoid: the entry could
// be replaced by a symlink between our inspection and our open. So we reopen through the
// descriptor we already hold, and only fall back to a by-name open, verified against the
// inode we inspected, where /proc is unavailable.
func finalOpen(dirfd int, name string, pathfd int, flags int) (int, error) {
	if HasProcFD() {
		return reopenPath(pathfd, flags)
	}

	var want unix.Stat_t
	if err := unix.Fstat(pathfd, &want); err != nil {
		return -1, err
	}
	fd, err := unix.Openat(dirfd, name, flags|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return -1, err
	}
	var got unix.Stat_t
	if err := unix.Fstat(fd, &got); err != nil {
		_ = unix.Close(fd)
		return -1, err
	}
	if got.Dev != want.Dev || got.Ino != want.Ino {
		// Swapped underneath us between inspection and open.
		_ = unix.Close(fd)
		return -1, ErrEscape
	}
	return fd, nil
}

func readlinkat(dirfd int, name string) (string, error) {
	buf := make([]byte, unix.PathMax)
	for {
		n, err := unix.Readlinkat(dirfd, name, buf)
		if err != nil {
			return "", err
		}
		if n < len(buf) {
			return string(buf[:n]), nil
		}
		if len(buf) >= 64*1024 {
			return "", fmt.Errorf("%w: symlink target too long", ErrInvalid)
		}
		buf = make([]byte, len(buf)*2)
	}
}

// splitTarget breaks a relative symlink target into components. Unlike ValidateRel it
// tolerates "." and "..", because those are legitimate inside a link target and are
// contained by the walker's descriptor stack rather than rejected up front.
func splitTarget(t string) []string {
	out := make([]string, 0, 8)
	start := 0
	for i := 0; i <= len(t); i++ {
		if i == len(t) || t[i] == '/' {
			if i > start {
				out = append(out, t[start:i])
			}
			start = i + 1
		}
	}
	return out
}

// isNotExist reports whether err is a plain "not there", as opposed to a containment
// refusal. Callers use it to distinguish a missing file from an attempted escape.
func isNotExist(err error) bool {
	return errors.Is(err, unix.ENOENT) || errors.Is(err, unix.ENOTDIR)
}
