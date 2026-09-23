// Package localfs is modelfsd's byte source: a read-only view of one local directory,
// implementing the nfsd.Backend interface.
//
// It replaces wgshare's index+peer backend (internal/lanserve, which joined a metadata
// index to a remote peer transport) with the trivial case the storage design calls for:
// the files are on local disk, so metadata is a stat and a read is a positional read of
// an *os.File. There is no peer, no metadata index and no manifest. Path containment is
// delegated to safepath (ported from wgshare), and bulk sequential reads go through the
// read-ahead cache in readahead.go.
package localfs

import (
	"context"
	"encoding/binary"
	"errors"
	"io/fs"
	"net/netip"
	"os"
	"path"
	"sort"
	"sync"
	"time"

	"github.com/inchix/vllm_manager/storage/modelfsd/internal/nfsd"
	"github.com/inchix/vllm_manager/storage/modelfsd/internal/safepath"
	"golang.org/x/sys/unix"
)

// rootID is the file id of the export root. Ids are handed out from rootID+1 upward.
const rootID uint64 = 1

// handleSize is the length of a filehandle this backend issues: a four-byte generation
// tag followed by the eight-byte file id.
const handleSize = 12

// Config configures a Backend.
type Config struct {
	// ExportDir is the absolute path of the directory to serve, read-only.
	ExportDir string
	// Share is the mount path clients spell in their mount line and the name the
	// backend answers MNT for, e.g. "export/llm_models" (no leading slash).
	Share string
	// Allow is the set of client networks permitted to mount and read. An empty set
	// permits every client — the CLI requires at least one, so this is a permissive
	// default only for embedding and tests.
	Allow []netip.Prefix
	// Readahead is the read-ahead window in bytes. Zero disables prefetch.
	Readahead int64
	// ReadTimeout bounds one physical read; zero uses the package default.
	ReadTimeout time.Duration
	// OnRead, if set, is called with the byte count of every successful READ, for
	// metrics. It must not block.
	OnRead func(n int)
}

// Backend serves one local directory read-only. It is safe for concurrent use.
type Backend struct {
	share string
	root  *safepath.Root
	allow []netip.Prefix
	gen   uint32
	cache *readCache
	onRead func(int)

	mu     sync.Mutex
	byID   map[uint64]string // file id -> path relative to the export root ("" is root)
	byPath map[string]uint64
	parent map[uint64]uint64 // file id -> parent's file id
	nextID uint64

	filesMu sync.Mutex
	files   map[uint64]*os.File // open handles, cached for the file's lifetime
}

// New opens the export directory and builds a Backend.
func New(cfg Config) (*Backend, error) {
	root, err := safepath.OpenRoot(cfg.Share, cfg.ExportDir)
	if err != nil {
		return nil, err
	}
	b := &Backend{
		share:  cfg.Share,
		root:   root,
		allow:  cfg.Allow,
		gen:    uint32(time.Now().Unix()),
		onRead: cfg.OnRead,
		byID:   map[uint64]string{rootID: ""},
		byPath: map[string]uint64{"": rootID},
		parent: map[uint64]uint64{rootID: rootID},
		nextID: rootID + 1,
		files:  make(map[uint64]*os.File),
	}
	b.cache = newReadCache(b.readChunk, cfg.Readahead, cfg.ReadTimeout)
	return b, nil
}

// Close releases the export root handle and every cached open file.
func (b *Backend) Close() error {
	b.filesMu.Lock()
	for _, f := range b.files {
		_ = f.Close()
	}
	b.files = map[uint64]*os.File{}
	b.filesMu.Unlock()
	return b.root.Close()
}

// assign returns the id for rel, allocating one on first sight. Caller holds b.mu.
func (b *Backend) assign(rel string, parentID uint64) uint64 {
	if id, ok := b.byPath[rel]; ok {
		return id
	}
	id := b.nextID
	b.nextID++
	b.byID[id] = rel
	b.byPath[rel] = id
	b.parent[id] = parentID
	return id
}

func (b *Backend) relFor(id uint64) (string, uint64, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	rel, ok := b.byID[id]
	return rel, b.parent[id], ok
}

// join builds a child's relative path. The empty string is the root, so its children have
// no leading separator.
func join(dirRel, name string) string {
	if dirRel == "" {
		return name
	}
	return dirRel + "/" + name
}

// statEntry stats rel and builds the nfsd.Entry for it.
func (b *Backend) statEntry(rel string, id, parentID uint64) (nfsd.Entry, error) {
	fi, err := b.root.Stat(rel, safepath.ModeBeneath)
	if err != nil {
		return nfsd.Entry{}, mapErr(err)
	}
	kind := nfsd.KindFile
	switch {
	case fi.IsDir:
		kind = nfsd.KindDir
	case fi.Mode.IsRegular():
		kind = nfsd.KindFile
	default:
		// Only regular files and directories are ever exported; anything else —
		// device, socket, fifo — is reported as absent rather than described.
		return nfsd.Entry{}, nfsd.ErrNotFound
	}
	name := path.Base(rel)
	if rel == "" {
		name = b.share
	}
	return nfsd.Entry{
		FileID: id,
		Parent: parentID,
		Name:   name,
		Kind:   kind,
		Size:   fi.Size,
		MTime:  fi.ModTime.Unix(),
	}, nil
}

// mapErr translates a safepath/filesystem error into the sentinels nfsd distinguishes. A
// containment refusal is reported as "not found": the client learns nothing about what is
// outside the export.
func mapErr(err error) error {
	switch {
	case err == nil:
		return nil
	case errors.Is(err, safepath.ErrEscape),
		errors.Is(err, safepath.ErrSymlink),
		errors.Is(err, safepath.ErrInvalid):
		return nfsd.ErrNotFound
	case errors.Is(err, fs.ErrNotExist),
		errors.Is(err, unix.ENOENT),
		errors.Is(err, unix.ENOTDIR):
		return nfsd.ErrNotFound
	}
	return err
}

// Root returns the export root for a share name given to MNT.
func (b *Backend) Root(share string) (nfsd.Entry, error) {
	if share != b.share {
		return nfsd.Entry{}, nfsd.ErrNotFound
	}
	return b.statEntry("", rootID, rootID)
}

// Lookup resolves one name within a directory.
func (b *Backend) Lookup(share string, parentFileID uint64, name string) (nfsd.Entry, error) {
	if share != b.share {
		return nfsd.Entry{}, nfsd.ErrNotFound
	}
	prel, _, ok := b.relFor(parentFileID)
	if !ok {
		return nfsd.Entry{}, nfsd.ErrStale
	}
	childRel := join(prel, name)
	b.mu.Lock()
	id := b.assign(childRel, parentFileID)
	b.mu.Unlock()
	return b.statEntry(childRel, id, parentFileID)
}

// Get returns an entry by file id.
func (b *Backend) Get(share string, fileID uint64) (nfsd.Entry, error) {
	if share != b.share {
		return nfsd.Entry{}, nfsd.ErrStale
	}
	rel, parentID, ok := b.relFor(fileID)
	if !ok {
		return nfsd.Entry{}, nfsd.ErrStale
	}
	return b.statEntry(rel, fileID, parentID)
}

// ReadDir returns up to max children of dir, ordered by name, starting after cookie.
// The cookie is the number of entries already consumed from the sorted listing, so a
// client may replay any entry's cookie and resume immediately after it.
func (b *Backend) ReadDir(share string, dir uint64, cookie uint64, max int) (ents []nfsd.Entry, next uint64, eof bool, err error) {
	if share != b.share {
		return nil, 0, false, nfsd.ErrStale
	}
	drel, _, ok := b.relFor(dir)
	if !ok {
		return nil, 0, false, nfsd.ErrStale
	}
	raw, err := b.root.ReadDir(drel, safepath.ModeBeneath)
	if err != nil {
		return nil, 0, false, mapErr(err)
	}
	names := make([]string, 0, len(raw))
	for _, e := range raw {
		names = append(names, e.Name)
	}
	sort.Strings(names)

	start := int(cookie)
	if start >= len(names) {
		return nil, cookie, true, nil
	}
	end := start + max
	if max <= 0 || end > len(names) {
		end = len(names)
	}

	out := make([]nfsd.Entry, 0, end-start)
	for i := start; i < end; i++ {
		crel := join(drel, names[i])
		b.mu.Lock()
		id := b.assign(crel, dir)
		b.mu.Unlock()
		ent, statErr := b.statEntry(crel, id, dir)
		if statErr != nil {
			// The name was in the listing a moment ago but will not stat now, or is
			// a kind we do not export. Return a minimal entry rather than dropping
			// it: dropping would leave the cookie sequence with a hole the client
			// cannot resume across.
			ent = nfsd.Entry{FileID: id, Parent: dir, Name: names[i], Kind: nfsd.KindFile}
		}
		out = append(out, ent)
	}
	return out, uint64(end), end >= len(names), nil
}

// ReadAt fills p from off, via the read-ahead cache.
func (b *Backend) ReadAt(ctx context.Context, share string, fileID uint64, p []byte, off int64) (int, error) {
	if share != b.share {
		return 0, nfsd.ErrStale
	}
	if _, _, ok := b.relFor(fileID); !ok {
		return 0, nfsd.ErrStale
	}
	n, err := b.cache.ReadAt(ctx, fileID, p, off)
	if n > 0 && b.onRead != nil {
		b.onRead(n)
	}
	return n, err
}

// readChunk is the read-ahead cache's source: a positional read of the file behind a file
// id, using a cached open handle.
func (b *Backend) readChunk(fileID uint64, p []byte, off int64) (int, error) {
	f, err := b.openFile(fileID)
	if err != nil {
		return 0, err
	}
	return f.ReadAt(p, off)
}

// openFile returns a cached read-only handle for a file id, opening it under the contained
// export root on first use. Model weights are immutable, so a handle is kept open for the
// file's lifetime.
func (b *Backend) openFile(fileID uint64) (*os.File, error) {
	b.filesMu.Lock()
	if f, ok := b.files[fileID]; ok {
		b.filesMu.Unlock()
		return f, nil
	}
	b.filesMu.Unlock()

	rel, _, ok := b.relFor(fileID)
	if !ok {
		return nil, nfsd.ErrStale
	}
	f, err := b.root.OpenFile(rel, safepath.ModeBeneath)
	if err != nil {
		return nil, mapErr(err)
	}

	b.filesMu.Lock()
	defer b.filesMu.Unlock()
	if existing, ok := b.files[fileID]; ok {
		// Another goroutine won the race; keep theirs and drop ours.
		_ = f.Close()
		return existing, nil
	}
	b.files[fileID] = f
	return f, nil
}

// Handle returns the persistent filehandle for a file id: a generation tag plus the id.
func (b *Backend) Handle(share string, fileID uint64) ([]byte, error) {
	if share != b.share {
		return nil, nfsd.ErrStale
	}
	h := make([]byte, handleSize)
	binary.BigEndian.PutUint32(h[0:4], b.gen)
	binary.BigEndian.PutUint64(h[4:12], fileID)
	return h, nil
}

// Resolve reverses Handle. A handle from another run (different generation) or one this
// backend never issued is reported as stale.
func (b *Backend) Resolve(handle []byte) (share string, fileID uint64, err error) {
	if len(handle) != handleSize {
		return "", 0, nfsd.ErrStale
	}
	if binary.BigEndian.Uint32(handle[0:4]) != b.gen {
		return "", 0, nfsd.ErrStale
	}
	id := binary.BigEndian.Uint64(handle[4:12])
	b.mu.Lock()
	_, ok := b.byID[id]
	b.mu.Unlock()
	if !ok {
		return "", 0, nfsd.ErrStale
	}
	return b.share, id, nil
}

// Allowed reports whether a client may mount and read. An empty allow list permits
// everyone; otherwise the client's address must fall in one of the permitted networks.
func (b *Backend) Allowed(share string, client netip.Addr) bool {
	if share != b.share {
		return false
	}
	if len(b.allow) == 0 {
		return true
	}
	client = client.Unmap()
	for _, p := range b.allow {
		if p.Contains(client) {
			return true
		}
	}
	return false
}
