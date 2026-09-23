// Package nfsd serves NFSv3 and MOUNTv3, read-only, on a network address.
//
// It was ported from the sibling project wgshare (internal/nfsd), which served a
// household's media index over a WireGuard tunnel. In modelfsd the same wire core serves
// one local directory of immutable model weights to the cluster fabric: metadata is
// answered from the local filesystem and READ reaches local disk, so there is no peer and
// no remote round trip anywhere in the path.
//
// v3 rather than v4.1 because the parts v4.1 adds — sessions, the slot-table replay
// cache, open state, a callback channel — are exactly the parts read-only access does
// not need, and we generate the consumer's mount line ourselves so we choose the
// version. The usual objection to v3 is its retinue of separate network services;
// MOUNT is handled in this process on the same port and NLM, statd and quota are not
// offered at all, so that objection does not apply here.
//
// Everything this package parses arrives from the network, from clients the server does
// not control. Every length is bounded before anything is allocated, every decoder is
// bounds-checked, and a malformed message produces an error or an RPC-level refusal —
// never a panic and never an unbounded allocation.
//
// The package holds no filesystem of its own. It is defined entirely against [Backend];
// modelfsd's local-directory implementation lives in internal/localfs.
package nfsd

import (
	"context"
	"errors"
	"net/netip"
)

// Kind distinguishes the entry types the appliance serves. It mirrors the manifest's
// kinds deliberately: anything else on the exporting filesystem — devices, sockets,
// fifos — is never described, so this server never has to answer for one.
type Kind uint8

const (
	// KindFile is a regular file.
	KindFile Kind = iota
	// KindDir is a directory.
	KindDir
)

// String renders a kind for logs and errors.
func (k Kind) String() string {
	if k == KindDir {
		return "dir"
	}
	return "file"
}

// Entry is everything the server needs to answer for one object. It is deliberately
// flat and free of paths: a filehandle carries a share and a file id, and nothing in
// the serving path reconstructs a path from either.
type Entry struct {
	// FileID is assigned by the exporting side and is stable across renames, which is
	// what lets a filehandle survive both a rename at the peer and a restart here.
	FileID uint64
	// Parent is the containing directory's file id, used to answer "..". It is zero,
	// or equal to FileID, at an export root.
	Parent uint64
	Name   string
	Kind   Kind
	Size   int64
	// MTime is a Unix time in seconds. NFSv3 has no room for a time before the epoch
	// and clients cope badly with one, so a negative value is reported as the epoch.
	MTime int64
}

// Backend is the index this server reads. Implementations are expected to answer
// everything except [Backend.ReadAt] from local disk; ReadAt is the only method allowed
// to touch the network, and it is the only one given a context.
//
// Implementations must be safe for concurrent use: the server calls them from one
// goroutine per in-flight request.
type Backend interface {
	// Root returns the export root for a share name given to MNT.
	Root(share string) (Entry, error)
	// Lookup resolves one name within a directory. A name that is not present must
	// return ErrNotFound, which becomes NFS3ERR_NOENT.
	Lookup(share string, parent uint64, name string) (Entry, error)
	// Get returns an entry by file id. ErrNotFound here means the object behind a
	// filehandle has gone, which becomes NFS3ERR_STALE rather than NFS3ERR_NOENT.
	Get(share string, fileID uint64) (Entry, error)
	// ReadDir returns up to max children of dir, starting after cookie, along with
	// the cookie to resume from and whether the directory ended. Cookie zero means
	// the start of the directory. Cookies are opaque to this package.
	ReadDir(share string, dir uint64, cookie uint64, max int) (ents []Entry, next uint64, eof bool, err error)
	// ReadAt fills p from off, returning the number of bytes read. It follows
	// io.ReaderAt: a short read at the end of a file may be reported with io.EOF, and
	// the server treats that as end-of-file rather than as an error.
	ReadAt(ctx context.Context, share string, fileID uint64, p []byte, off int64) (int, error)
	// Handle returns the persistent filehandle for a file id, at most MaxHandleSize
	// bytes.
	Handle(share string, fileID uint64) ([]byte, error)
	// Resolve reverses Handle. A handle this backend did not issue, or one whose
	// share key has since been rotated, must return an error; every failure here is
	// reported to the client as NFS3ERR_STALE.
	Resolve(handle []byte) (share string, fileID uint64, err error)
	// Allowed reports whether a client address may mount a share. It is consulted on
	// MNT and again on every operation, so a revoked grant takes effect on the next
	// request rather than at the next mount.
	Allowed(share string, client netip.Addr) bool
}

// Errors a Backend returns. Anything else is reported to the client as NFS3ERR_IO,
// which is the honest answer for "the index or the peer failed and we do not know why".
var (
	// ErrNotFound means the named object is not in the index.
	ErrNotFound = errors.New("nfsd: not found")
	// ErrStale means a filehandle no longer resolves, typically because the object
	// was removed at the peer or the share key was rotated.
	ErrStale = errors.New("nfsd: stale filehandle")
	// ErrDenied means the caller may not have this object.
	ErrDenied = errors.New("nfsd: denied")
)

// MaxHandleSize is the largest filehandle NFSv3 permits, from the nfs_fh3 definition in
// RFC 1813 §2.6. Our own handles are 32 bytes (docs/15 §4); the limit is enforced on
// both directions so a backend cannot produce a handle a client is unable to store.
const MaxHandleSize = 64
