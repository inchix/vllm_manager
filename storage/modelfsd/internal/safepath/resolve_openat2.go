package safepath

import (
	"errors"
	"os"
	"sync"

	"golang.org/x/sys/unix"
)

// openat2 is the preferred resolver: the kernel enforces containment itself, atomically,
// with no window for a concurrent rename to change the answer between our check and our
// open. Available from Linux 5.6.
//
// It can be missing for two reasons, and we treat them identically: an older kernel
// (ENOSYS) or a seccomp profile that filters it (EPERM). Either way we fall back to the
// component walk, which the appliance refuses to start without.

var (
	openat2Once sync.Once
	openat2OK   bool
)

// HasOpenat2 reports whether openat2(2) with RESOLVE_BENEATH is usable in this process.
// The result is probed once, by actually calling it, rather than inferred from a kernel
// version string: a backported kernel reports one version and behaves like another, and a
// seccomp profile can remove a syscall the kernel supports.
func HasOpenat2() bool {
	openat2Once.Do(func() {
		fd, err := unix.Openat2(unix.AT_FDCWD, ".", &unix.OpenHow{
			Flags:   unix.O_PATH | unix.O_DIRECTORY | unix.O_CLOEXEC,
			Resolve: unix.RESOLVE_BENEATH | unix.RESOLVE_NO_MAGICLINKS,
		})
		if err != nil {
			openat2OK = false
			return
		}
		_ = unix.Close(fd)
		openat2OK = true
	})
	return openat2OK
}

// resolveOpenat2 opens rel beneath the root in a single syscall.
func resolveOpenat2(rootfd int, rel string, mode Mode, flags int) (int, error) {
	resolve := uint64(unix.RESOLVE_BENEATH | unix.RESOLVE_NO_MAGICLINKS)
	if mode == ModeStrict {
		resolve |= unix.RESOLVE_NO_SYMLINKS
	}
	if rel == "" {
		// RESOLVE_BENEATH rejects "" outright; "." denotes the root itself and is
		// permitted because it does not ascend.
		rel = "."
	}
	fd, err := unix.Openat2(rootfd, rel, &unix.OpenHow{
		Flags:   uint64(flags) | unix.O_CLOEXEC,
		Resolve: resolve,
	})
	if err != nil {
		return -1, translateOpenat2Err(err, mode)
	}
	return fd, nil
}

// translateOpenat2Err maps the kernel's containment refusals onto our sentinel errors, so
// that callers and tests can distinguish "you tried to escape" from "it isn't there".
func translateOpenat2Err(err error, mode Mode) error {
	switch {
	case errors.Is(err, unix.EXDEV):
		// RESOLVE_BENEATH refusal: the path ascended out of the root, via "..", an
		// absolute symlink, or a symlink pointing outside.
		return ErrEscape
	case errors.Is(err, unix.ELOOP) && mode == ModeStrict:
		// RESOLVE_NO_SYMLINKS refusal.
		return ErrSymlink
	case errors.Is(err, unix.EAGAIN):
		// The kernel could not guarantee containment because of concurrent renames.
		// Refusing is correct; the caller may retry.
		return ErrEscape
	}
	return err
}

// reopenPath turns an O_PATH descriptor into a usable one with the requested flags, by
// reopening it through /proc/self/fd. This is the standard technique and the reason the
// appliance requires /proc; see HasProcFD.
func reopenPath(pathfd int, flags int) (int, error) {
	name := "/proc/self/fd/" + itoa(pathfd)
	return unix.Open(name, flags|unix.O_CLOEXEC, 0)
}

var (
	procOnce sync.Once
	procOK   bool
)

// HasProcFD reports whether /proc/self/fd is usable for reopening O_PATH descriptors.
// Only the fallback resolver needs it; openat2 opens with the final flags directly.
func HasProcFD() bool {
	procOnce.Do(func() {
		st, err := os.Stat("/proc/self/fd")
		procOK = err == nil && st.IsDir()
	})
	return procOK
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
