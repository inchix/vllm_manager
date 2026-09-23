// Package safepath resolves caller-supplied relative paths beneath a fixed set of
// deployment-declared roots, without ever permitting escape.
//
// It is the appliance's most security-critical primitive: the web API can name a path,
// and the agent that opens it runs as root. See docs/01-architecture.md §3 and
// docs/02-threat-model.md boundary B5.
//
// Two resolution modes are offered:
//
//   - [ModeStrict] refuses to traverse any symlink. Use for configuration paths and for
//     anything the agent will write, where a symlink is never legitimate.
//   - [ModeBeneath] permits symlinks that stay within the root and refuses those that
//     escape. Use for serving media, where symlink and hardlink layouts are common.
//
// Both modes refuse "..", absolute paths, and magic links (/proc/*/fd, ...).
package safepath

import (
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"
)

// Limits on a caller-supplied relative path. Deliberately smaller than the kernel's,
// so that a malformed request is rejected by us with a clear error rather than by a
// syscall with an opaque one.
const (
	MaxPathLen  = 4096
	MaxNameLen  = 255
	MaxDepth    = 64
	MaxSymlinks = 40 // matches the kernel's ELOOP budget
)

var (
	// ErrEscape means the path left, or tried to leave, its root.
	ErrEscape = errors.New("safepath: path escapes its root")
	// ErrInvalid means the path was malformed before resolution was attempted.
	ErrInvalid = errors.New("safepath: invalid relative path")
	// ErrSymlink means a symlink was encountered in a mode that forbids them.
	ErrSymlink = errors.New("safepath: symlink not permitted here")
	// ErrNoRoot means the named root was not declared at deploy time.
	ErrNoRoot = errors.New("safepath: unknown root")
)

// Mode selects how symlinks are treated during resolution.
type Mode int

const (
	// ModeStrict refuses to traverse any symlink.
	ModeStrict Mode = iota
	// ModeBeneath permits symlinks that resolve within the root.
	ModeBeneath
)

func (m Mode) String() string {
	if m == ModeBeneath {
		return "beneath"
	}
	return "strict"
}

// ValidateRel checks a caller-supplied relative path and splits it into components.
//
// It deliberately does NOT clean the path. Cleaning is where traversal bugs live: a
// cleaner that turns "a/../../b" into "../b" has already done the attacker's work. Any
// ".." is rejected outright, and a caller that legitimately wants a parent directory can
// name it directly.
//
// The empty string and "." both denote the root itself and yield no components.
func ValidateRel(rel string) ([]string, error) {
	if len(rel) > MaxPathLen {
		return nil, fmt.Errorf("%w: longer than %d bytes", ErrInvalid, MaxPathLen)
	}
	if strings.ContainsRune(rel, 0) {
		return nil, fmt.Errorf("%w: contains NUL", ErrInvalid)
	}
	if !utf8.ValidString(rel) {
		// Linux filenames may be arbitrary bytes, but a path *arriving from the API*
		// came through JSON and must be valid UTF-8. Names read back off the
		// filesystem are returned raw and are not subject to this.
		return nil, fmt.Errorf("%w: not valid UTF-8", ErrInvalid)
	}
	if rel == "" || rel == "." {
		return nil, nil
	}
	if strings.HasPrefix(rel, "/") {
		return nil, fmt.Errorf("%w: absolute", ErrInvalid)
	}
	// Reject a Windows-style drive or UNC prefix outright rather than letting it
	// become a strange relative name.
	if strings.HasPrefix(rel, `\`) {
		return nil, fmt.Errorf("%w: backslash-rooted", ErrInvalid)
	}

	parts := strings.Split(rel, "/")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		switch p {
		case "":
			// Covers a trailing slash, a leading slash (already caught) and "a//b".
			// Rejected rather than silently collapsed, so that what we validate is
			// exactly what we resolve.
			return nil, fmt.Errorf("%w: empty component", ErrInvalid)
		case ".":
			return nil, fmt.Errorf("%w: %q component", ErrInvalid, ".")
		case "..":
			return nil, fmt.Errorf("%w: %q component", ErrEscape, "..")
		}
		if len(p) > MaxNameLen {
			return nil, fmt.Errorf("%w: component longer than %d bytes", ErrInvalid, MaxNameLen)
		}
		out = append(out, p)
	}
	if len(out) > MaxDepth {
		return nil, fmt.Errorf("%w: deeper than %d components", ErrInvalid, MaxDepth)
	}
	return out, nil
}
