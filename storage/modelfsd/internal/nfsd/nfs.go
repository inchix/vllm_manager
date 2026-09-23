package nfsd

import (
	"context"
	"errors"
	"hash/fnv"
	"io"
	"math"
	"net/netip"
	"strings"
)

// NFS version 3, RFC 1813, read-only.
//
// The eleven procedures a read-only export needs are implemented; every mutating
// procedure returns NFS3ERR_ROFS with the correct failure body for its result type, so a
// client learns why it failed rather than seeing a short reply it cannot parse.
//
// READDIRPLUS is the procedure that matters. A media scanner walking a library issues a
// READDIR and then one GETATTR per entry; READDIRPLUS returns the attributes with the
// entries, which is the difference between one round trip per directory and one per
// file. Everything it answers with comes from the index, so it costs no network at all.

// NFS program numbers, from RFC 1813 §2.
const (
	nfsProgram = 100003
	nfsVersion = 3
)

// NFSv3 procedure numbers.
const (
	nfsProcNull        = 0
	nfsProcGetAttr     = 1
	nfsProcSetAttr     = 2
	nfsProcLookup      = 3
	nfsProcAccess      = 4
	nfsProcReadlink    = 5
	nfsProcRead        = 6
	nfsProcWrite       = 7
	nfsProcCreate      = 8
	nfsProcMkdir       = 9
	nfsProcSymlink     = 10
	nfsProcMknod       = 11
	nfsProcRemove      = 12
	nfsProcRmdir       = 13
	nfsProcRename      = 14
	nfsProcLink        = 15
	nfsProcReaddir     = 16
	nfsProcReaddirPlus = 17
	nfsProcFSStat      = 18
	nfsProcFSInfo      = 19
	nfsProcPathConf    = 20
	nfsProcCommit      = 21
)

// nfsstat3 values this server issues. The full enumeration is much longer; the ones
// absent here describe conditions a read-only export cannot reach.
const (
	nfs3OK             = 0
	nfs3ErrNoEnt       = 2
	nfs3ErrIO          = 5
	nfs3ErrAcces       = 13
	nfs3ErrNotDir      = 20
	nfs3ErrIsDir       = 21
	nfs3ErrInval       = 22
	nfs3ErrROFS        = 30
	nfs3ErrNameTooLong = 63
	nfs3ErrStale       = 70
	nfs3ErrBadCookie   = 10003
	nfs3ErrTooSmall    = 10005
	nfs3ErrServerFault = 10006
)

// ftype3 values. Only regular files and directories are ever exported, because only
// those two kinds reach the index in the first place.
const (
	nf3Reg = 1
	nf3Dir = 2
)

// ACCESS3 permission bits, RFC 1813 §3.3.4.
const (
	access3Read    = 0x0001
	access3Lookup  = 0x0002
	access3Modify  = 0x0004
	access3Extend  = 0x0008
	access3Delete  = 0x0010
	access3Execute = 0x0020
)

// FSINFO properties, RFC 1813 §3.3.19. A read-only export supports neither hard links
// nor symbolic links and cannot set times, so only homogeneity is claimed.
const fsf3Homogeneous = 0x0008

// Sizes and limits for the reply builder.
const (
	// fattr3Size is the encoded size of an fattr3: twenty-one four-byte words.
	fattr3Size = 84
	// maxNameLen matches the name_max reported by PATHCONF.
	maxNameLen = 255
	// maxWireName is how long a name may be before the decoder gives up on it.
	// filename3 is unbounded in the XDR, and a client that sends an over-long name
	// deserves NFS3ERR_NAMETOOLONG rather than GARBAGE_ARGS — the first tells it what
	// it did wrong. The generous margin above name_max exists only so that the
	// decoder still has a limit.
	maxWireName = 4096
	// maxReadSize bounds a single READ, and therefore the largest buffer one
	// in-flight request can hold. It is reported as rtmax so clients stay under it.
	maxReadSize = 512 << 10
	// maxReaddirCount bounds what a client can ask a single READDIR to return,
	// whatever it claims it can handle.
	maxReaddirCount = 512 << 10
	// maxReaddirEntries is a belt-and-braces cap on a single listing. The byte budget
	// normally stops the loop long before this.
	maxReaddirEntries = 4096
	// readdirFixedCost is the reply overhead a listing must leave room for: status,
	// the directory's post_op_attr, the cookie verifier, the list terminator and the
	// eof flag.
	readdirFixedCost = 4 + 4 + fattr3Size + 8 + 4 + 4
)

// Cookie encoding. NFSv3 cookies are opaque uint64s that a client may replay from any
// entry it saw, so every entry needs a cookie of its own that resumes exactly after it.
//
// The backend's cookies address its own children only, and this server also emits "."
// and "..", which the index knows nothing about. So the two synthetic entries take the
// two reserved values below and a backend cookie c is offered to clients as c+3.
const (
	cookieStart  = 0
	cookieDot    = 1
	cookieDotDot = 2
	cookieBase   = 3
)

// nfsService answers the NFS program.
type nfsService struct {
	backend Backend
}

// program describes the NFS program for the RPC dispatcher.
func (s *nfsService) program() *Program {
	return &Program{
		Number: nfsProgram,
		Low:    nfsVersion,
		High:   nfsVersion,
		Procs: map[uint32]Procedure{
			nfsProcNull:        s.null,
			nfsProcGetAttr:     s.getAttr,
			nfsProcLookup:      s.lookup,
			nfsProcAccess:      s.access,
			nfsProcReadlink:    s.readlink,
			nfsProcRead:        s.read,
			nfsProcReaddir:     s.readdir,
			nfsProcReaddirPlus: s.readdirPlus,
			nfsProcFSStat:      s.fsStat,
			nfsProcFSInfo:      s.fsInfo,
			nfsProcPathConf:    s.pathConf,

			// Declined. Each returns NFS3ERR_ROFS with the failure body its
			// result type defines, without decoding the arguments: there is
			// nothing we would do with them, and not parsing them keeps those
			// parsers off the attack surface entirely. The refusal is the same
			// whatever the arguments were, so it tells an unauthorised caller
			// nothing it did not already know.
			nfsProcSetAttr: readOnly(encodeWCCData),
			nfsProcWrite:   readOnly(encodeWCCData),
			nfsProcCreate:  readOnly(encodeWCCData),
			nfsProcMkdir:   readOnly(encodeWCCData),
			nfsProcSymlink: readOnly(encodeWCCData),
			nfsProcMknod:   readOnly(encodeWCCData),
			nfsProcRemove:  readOnly(encodeWCCData),
			nfsProcRmdir:   readOnly(encodeWCCData),
			nfsProcCommit:  readOnly(encodeWCCData),
			nfsProcRename:  readOnly(encodeRenameFail),
			nfsProcLink:    readOnly(encodeLinkFail),
		},
	}
}

// readOnly builds a procedure that refuses with NFS3ERR_ROFS, encoding the failure body
// the caller's result type requires.
func readOnly(body func(*Encoder)) Procedure {
	return func(_ context.Context, _ *Call, e *Encoder) error {
		e.Uint32(nfs3ErrROFS)
		body(e)
		return nil
	}
}

// encodeWCCData writes an empty wcc_data: neither before nor after attributes. Nothing
// changed, so there is nothing to describe.
func encodeWCCData(e *Encoder) {
	e.Bool(false) // pre_op_attr
	e.Bool(false) // post_op_attr
}

// encodeRenameFail writes RENAME3res's failure body: two wcc_data.
func encodeRenameFail(e *Encoder) {
	encodeWCCData(e)
	encodeWCCData(e)
}

// encodeLinkFail writes LINK3res's failure body: post_op_attr then wcc_data.
func encodeLinkFail(e *Encoder) {
	e.Bool(false)
	encodeWCCData(e)
}

// object is what a filehandle resolved to, with the client's access already checked.
type object struct {
	share string
	fsid  uint64
	entry Entry
}

// fsidFor derives a stable filesystem id from a share name. It has to be stable across
// restarts — a client that sees the fsid change treats it as a different filesystem —
// and the share name is the only thing available here that is.
func fsidFor(share string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(share))
	return h.Sum64()
}

// resolve turns a filehandle into an object, applying the grant check.
//
// Authorisation is re-checked on every operation rather than only at MNT, which is what
// makes a revoked grant take effect on the next request instead of at the next mount.
func (s *nfsService) resolve(fh []byte, client netip.Addr) (object, uint32) {
	share, id, err := s.backend.Resolve(fh)
	if err != nil {
		// A handle we did not issue, or one whose share key has been rotated. Both
		// are stale from the client's point of view, and neither is worth
		// distinguishing for it.
		return object{}, nfs3ErrStale
	}
	if !s.backend.Allowed(share, client) {
		return object{}, nfs3ErrAcces
	}
	ent, err := s.backend.Get(share, id)
	if err != nil {
		return object{}, handleStatus(err)
	}
	return object{share: share, fsid: fsidFor(share), entry: ent}, nfs3OK
}

// handleStatus maps a backend failure behind a filehandle. A missing object here means
// the file went away at the peer, which is exactly what NFS3ERR_STALE describes.
func handleStatus(err error) uint32 {
	switch {
	case err == nil:
		return nfs3OK
	case errors.Is(err, ErrNotFound), errors.Is(err, ErrStale):
		return nfs3ErrStale
	case errors.Is(err, ErrDenied):
		return nfs3ErrAcces
	}
	return nfs3ErrIO
}

// nameStatus maps a backend failure for a name within a directory, where a miss is an
// ordinary negative lookup rather than a stale handle.
func nameStatus(err error) uint32 {
	switch {
	case err == nil:
		return nfs3OK
	case errors.Is(err, ErrNotFound):
		return nfs3ErrNoEnt
	case errors.Is(err, ErrStale):
		return nfs3ErrStale
	case errors.Is(err, ErrDenied):
		return nfs3ErrAcces
	}
	return nfs3ErrIO
}

// encodeFattr writes an fattr3.
//
// Ownership is squashed to root and permissions are fixed: the index carries no
// ownership, the export is read-only, and inventing per-file modes from a peer's
// metadata would give clients a permission model this server does not actually enforce.
func encodeFattr(e *Encoder, fsid uint64, ent Entry) {
	size := uint64(0)
	if ent.Size > 0 {
		size = uint64(ent.Size)
	}
	ftype, mode, nlink := uint32(nf3Reg), uint32(0o444), uint32(1)
	if ent.Kind == KindDir {
		ftype, mode, nlink = nf3Dir, 0o555, 2
		// A directory's size is meaningless here, but zero makes some clients
		// treat it as empty before they have listed it.
		size = 4096
	}
	e.Uint32(ftype)
	e.Uint32(mode)
	e.Uint32(nlink)
	e.Uint32(0) // uid
	e.Uint32(0) // gid
	e.Uint64(size)
	e.Uint64(uint64(pad4096(size))) // used
	e.Uint32(0)                     // rdev specdata1
	e.Uint32(0)                     // rdev specdata2
	e.Uint64(fsid)
	e.Uint64(ent.FileID)
	t := clampTime(ent.MTime)
	for i := 0; i < 3; i++ { // atime, mtime, ctime
		e.Uint32(t)
		e.Uint32(0)
	}
}

// pad4096 rounds a size up to a whole number of 4 KiB blocks, for the fattr3 "used"
// field. The index does not know the peer's allocation, so the apparent size rounded up
// is the closest honest answer.
func pad4096(size uint64) uint64 {
	if size > math.MaxUint64-4095 {
		return size
	}
	return (size + 4095) &^ 4095
}

// encodePostOpAttr writes a post_op_attr, present or absent. Absent is always legal, so
// anything we could not look up is simply omitted rather than guessed at.
func encodePostOpAttr(e *Encoder, fsid uint64, ent *Entry) {
	if ent == nil {
		e.Bool(false)
		return
	}
	e.Bool(true)
	encodeFattr(e, fsid, *ent)
}

// decodeHandle reads an nfs_fh3.
func decodeHandle(d *Decoder) ([]byte, error) { return d.Opaque(MaxHandleSize) }

// nameCheck validates a filename argument before it reaches the backend.
func nameCheck(name string) uint32 {
	switch {
	case name == "":
		return nfs3ErrInval
	case len(name) > maxNameLen:
		return nfs3ErrNameTooLong
	case strings.ContainsAny(name, "/\x00"):
		// A name is one component. Anything carrying a separator or a NUL is a
		// client trying to address something the protocol does not let it address.
		return nfs3ErrInval
	}
	return nfs3OK
}

// null is the procedure every RPC program must answer, used by clients to check the
// server is alive.
func (s *nfsService) null(_ context.Context, _ *Call, _ *Encoder) error { return nil }

// getAttr returns an object's attributes.
func (s *nfsService) getAttr(_ context.Context, c *Call, e *Encoder) error {
	fh, err := decodeHandle(c.Args)
	if err != nil {
		return err
	}
	obj, st := s.resolve(fh, c.Client)
	e.Uint32(st)
	if st != nfs3OK {
		return nil
	}
	encodeFattr(e, obj.fsid, obj.entry)
	return nil
}

// lookup resolves one name in a directory.
func (s *nfsService) lookup(_ context.Context, c *Call, e *Encoder) error {
	fh, err := decodeHandle(c.Args)
	if err != nil {
		return err
	}
	name, err := c.Args.String(maxWireName)
	if err != nil {
		return err
	}
	dir, st := s.resolve(fh, c.Client)
	if st != nfs3OK {
		e.Uint32(st)
		e.Bool(false) // dir_attributes
		return nil
	}
	fail := func(st uint32) error {
		e.Uint32(st)
		encodePostOpAttr(e, dir.fsid, &dir.entry)
		return nil
	}
	if dir.entry.Kind != KindDir {
		return fail(nfs3ErrNotDir)
	}
	if bad := nameCheck(name); bad != nfs3OK {
		return fail(bad)
	}

	// "." and ".." are answered from the directory entry itself. The index holds
	// children, not these two, and a client that walks upwards would otherwise get a
	// negative lookup for a directory it is standing in.
	var target Entry
	switch name {
	case ".":
		target = dir.entry
	case "..":
		target = s.parentOf(dir)
	default:
		target, err = s.backend.Lookup(dir.share, dir.entry.FileID, name)
		if err != nil {
			return fail(nameStatus(err))
		}
	}
	handle, err := s.backend.Handle(dir.share, target.FileID)
	if err != nil || len(handle) > MaxHandleSize {
		return fail(nfs3ErrServerFault)
	}
	e.Uint32(nfs3OK)
	e.Opaque(handle)
	encodePostOpAttr(e, dir.fsid, &target)
	encodePostOpAttr(e, dir.fsid, &dir.entry)
	return nil
}

// parentOf returns the entry ".." names. At an export root it is the root itself, which
// is what stops a client walking out of the export by asking.
func (s *nfsService) parentOf(dir object) Entry {
	if dir.entry.Parent == 0 || dir.entry.Parent == dir.entry.FileID {
		return dir.entry
	}
	parent, err := s.backend.Get(dir.share, dir.entry.Parent)
	if err != nil {
		// The parent is not in the index — the export root's own parent, or a tree
		// caught mid-update. Standing still is better than failing the lookup.
		return dir.entry
	}
	return parent
}

// access reports which of the requested operations the client may perform.
//
// The answer is the same for every client that got this far, because access is decided
// by the grant table at resolve time, not per object. What it does say is that nothing
// may be modified, which is how a client learns the export is read-only before it tries.
func (s *nfsService) access(_ context.Context, c *Call, e *Encoder) error {
	fh, err := decodeHandle(c.Args)
	if err != nil {
		return err
	}
	want, err := c.Args.Uint32()
	if err != nil {
		return err
	}
	obj, st := s.resolve(fh, c.Client)
	if st != nfs3OK {
		e.Uint32(st)
		e.Bool(false)
		return nil
	}
	granted := uint32(access3Read)
	if obj.entry.Kind == KindDir {
		granted |= access3Lookup | access3Execute
	}
	granted &^= access3Modify | access3Extend | access3Delete
	e.Uint32(nfs3OK)
	encodePostOpAttr(e, obj.fsid, &obj.entry)
	e.Uint32(want & granted)
	return nil
}

// readlink always fails: the index describes files and directories only, so nothing
// this server exports is a symbolic link, and NFS3ERR_INVAL is what RFC 1813 §3.3.5
// specifies for a READLINK of something that is not one.
func (s *nfsService) readlink(_ context.Context, c *Call, e *Encoder) error {
	fh, err := decodeHandle(c.Args)
	if err != nil {
		return err
	}
	obj, st := s.resolve(fh, c.Client)
	if st != nfs3OK {
		e.Uint32(st)
		e.Bool(false)
		return nil
	}
	e.Uint32(nfs3ErrInval)
	encodePostOpAttr(e, obj.fsid, &obj.entry)
	return nil
}

// read returns a range of a file.
func (s *nfsService) read(ctx context.Context, c *Call, e *Encoder) error {
	fh, err := decodeHandle(c.Args)
	if err != nil {
		return err
	}
	off, err := c.Args.Uint64()
	if err != nil {
		return err
	}
	count, err := c.Args.Uint32()
	if err != nil {
		return err
	}
	obj, st := s.resolve(fh, c.Client)
	if st != nfs3OK {
		e.Uint32(st)
		e.Bool(false)
		return nil
	}
	fail := func(st uint32) error {
		e.Uint32(st)
		encodePostOpAttr(e, obj.fsid, &obj.entry)
		return nil
	}
	if obj.entry.Kind != KindFile {
		return fail(nfs3ErrIsDir)
	}
	if off > math.MaxInt64 {
		return fail(nfs3ErrInval)
	}

	size := obj.entry.Size
	if size < 0 {
		size = 0
	}
	// Clamp before allocating: the request is bounded by rtmax, and by what the file
	// can actually supply, so a client asking for half a megabyte of a one-byte file
	// costs one byte.
	want := int64(count)
	if want > maxReadSize {
		want = maxReadSize
	}
	if avail := size - int64(off); want > avail {
		want = avail
	}
	if want <= 0 {
		// Either the offset is at or past the end — not an error, it is how a
		// client reading forwards discovers where a file stops — or the client
		// asked for nothing, which is not the same thing and must not be reported
		// as the end of the file.
		e.Uint32(nfs3OK)
		encodePostOpAttr(e, obj.fsid, &obj.entry)
		e.Uint32(0)
		e.Bool(int64(off) >= size)
		e.Opaque(nil)
		return nil
	}

	buf := make([]byte, want)
	n, err := s.backend.ReadAt(ctx, obj.share, obj.entry.FileID, buf, int64(off))
	if n < 0 || n > len(buf) {
		return fail(nfs3ErrServerFault)
	}
	if err != nil && !errors.Is(err, io.EOF) {
		// A slow or dead peer becomes NFS3ERR_IO promptly rather than a hung
		// client; see docs/15 §6.
		return fail(handleStatus(err))
	}
	eof := errors.Is(err, io.EOF) || int64(off)+int64(n) >= size
	e.Uint32(nfs3OK)
	encodePostOpAttr(e, obj.fsid, &obj.entry)
	e.Uint32(uint32(n))
	e.Bool(eof)
	e.Opaque(buf[:n])
	return nil
}

// fsStat reports capacity. The index does not know the peer's filesystem size, and free
// space on a read-only export is genuinely zero; the nominal total exists only so that
// df prints something a person can read rather than a filesystem of size zero.
func (s *nfsService) fsStat(_ context.Context, c *Call, e *Encoder) error {
	fh, err := decodeHandle(c.Args)
	if err != nil {
		return err
	}
	obj, st := s.resolve(fh, c.Client)
	if st != nfs3OK {
		e.Uint32(st)
		e.Bool(false)
		return nil
	}
	e.Uint32(nfs3OK)
	encodePostOpAttr(e, obj.fsid, &obj.entry)
	e.Uint64(1 << 44) // tbytes, nominal
	e.Uint64(0)       // fbytes
	e.Uint64(0)       // abytes
	e.Uint64(1 << 24) // tfiles, nominal
	e.Uint64(0)       // ffiles
	e.Uint64(0)       // afiles
	e.Uint32(0)       // invarsec: the index changes when the peer does
	return nil
}

// fsInfo reports the transfer sizes and properties a client should plan around.
func (s *nfsService) fsInfo(_ context.Context, c *Call, e *Encoder) error {
	fh, err := decodeHandle(c.Args)
	if err != nil {
		return err
	}
	obj, st := s.resolve(fh, c.Client)
	if st != nfs3OK {
		e.Uint32(st)
		e.Bool(false)
		return nil
	}
	e.Uint32(nfs3OK)
	encodePostOpAttr(e, obj.fsid, &obj.entry)
	e.Uint32(maxReadSize) // rtmax
	e.Uint32(maxReadSize) // rtpref
	e.Uint32(4096)        // rtmult
	// The write sizes mirror the read sizes rather than being zero. Writes are
	// refused, so these numbers describe nothing that will happen; a zero wtmult,
	// however, is a division hazard in clients that compute a block count from it.
	e.Uint32(maxReadSize) // wtmax
	e.Uint32(maxReadSize) // wtpref
	e.Uint32(4096)        // wtmult
	e.Uint32(32768)       // dtpref
	e.Uint64(math.MaxInt64)
	e.Uint32(0) // time_delta seconds
	e.Uint32(1) // time_delta nanoseconds
	e.Uint32(fsf3Homogeneous)
	return nil
}

// pathConf reports the static limits of the export.
func (s *nfsService) pathConf(_ context.Context, c *Call, e *Encoder) error {
	fh, err := decodeHandle(c.Args)
	if err != nil {
		return err
	}
	obj, st := s.resolve(fh, c.Client)
	if st != nfs3OK {
		e.Uint32(st)
		e.Bool(false)
		return nil
	}
	e.Uint32(nfs3OK)
	encodePostOpAttr(e, obj.fsid, &obj.entry)
	e.Uint32(1)          // linkmax: no hard links
	e.Uint32(maxNameLen) // name_max
	e.Bool(true)         // no_trunc: an over-long name is refused, not silently cut
	e.Bool(true)         // chown_restricted
	e.Bool(false)        // case_insensitive
	e.Bool(true)         // case_preserving
	return nil
}

// dirEntry is one prepared directory entry. It is built before anything is encoded
// because a listing has to know what it will cost before it commits to it: NFSv3 gives
// the client a byte budget, and overrunning it is a protocol violation rather than a
// truncated reply.
type dirEntry struct {
	fileID uint64
	name   string
	cookie uint64
	attr   *Entry
	handle []byte
}

// dirCost is the encoded size of the entry's directory information: the presence flag,
// file id, name and cookie. READDIRPLUS's dircount is measured against exactly this.
func (de *dirEntry) dirCost() int {
	return 4 + 8 + opaqueSize(len(de.name)) + 8
}

// plusCost is the encoded size of the entry including its attributes and filehandle.
func (de *dirEntry) plusCost() int {
	n := de.dirCost() + 4 // post_op_attr presence
	if de.attr != nil {
		n += fattr3Size
	}
	n += 4 // post_op_fh3 presence
	if de.handle != nil {
		n += opaqueSize(len(de.handle))
	}
	return n
}

// listing accumulates entries under two byte budgets.
type listing struct {
	entries     []dirEntry
	eof         bool
	plus        bool
	dirBudget   int
	totalBudget int
	dirUsed     int
	totalUsed   int
}

// add appends an entry if both budgets allow it, reporting whether it fitted.
func (l *listing) add(de dirEntry) bool {
	dc := de.dirCost()
	tc := dc
	if l.plus {
		tc = de.plusCost()
	}
	if l.dirUsed+dc > l.dirBudget || l.totalUsed+tc > l.totalBudget {
		return false
	}
	l.dirUsed += dc
	l.totalUsed += tc
	l.entries = append(l.entries, de)
	return true
}

// wireCookie maps a backend cookie onto the value handed to the client, leaving room
// for the two synthetic entries.
func wireCookie(backend uint64) (uint64, bool) {
	if backend > math.MaxUint64-cookieBase {
		return 0, false
	}
	return backend + cookieBase, true
}

// collect builds a listing of dir starting after start.
//
// Children are fetched one at a time. The backend returns a single resume cookie for a
// batch, but NFSv3 lets a client replay the cookie of any entry it saw — the Linux
// client does exactly that after a seek — so each entry has to be given the cookie that
// resumes immediately after it, and asking for one at a time is the only way this
// interface can supply that. The calls are answered from local disk, and READDIRPLUS
// still collapses the N GETATTRs that would otherwise follow, which is where the round
// trips actually were.
func (s *nfsService) collect(dir object, start uint64, plus bool, dirBudget, totalBudget int) (*listing, uint32) {
	l := &listing{plus: plus, dirBudget: dirBudget, totalBudget: totalBudget}

	synthetic := func(id uint64, name string, cookie uint64, attr *Entry) dirEntry {
		de := dirEntry{fileID: id, name: name, cookie: cookie}
		if plus {
			de.attr = attr
			if h, err := s.backend.Handle(dir.share, id); err == nil && len(h) <= MaxHandleSize {
				de.handle = h
			}
		}
		return de
	}

	if start == cookieStart {
		self := dir.entry
		if !l.add(synthetic(self.FileID, ".", cookieDot, &self)) {
			return l, nfs3OK
		}
	}
	if start == cookieStart || start == cookieDot {
		parent := s.parentOf(dir)
		if !l.add(synthetic(parent.FileID, "..", cookieDotDot, &parent)) {
			return l, nfs3OK
		}
	}

	cookie := uint64(0)
	if start >= cookieBase {
		cookie = start - cookieBase
	}
	for i := 0; i < maxReaddirEntries; i++ {
		ents, next, atEnd, err := s.backend.ReadDir(dir.share, dir.entry.FileID, cookie, 1)
		if err != nil {
			// The directory itself resolved a moment ago, so a miss here is the
			// cookie, not the directory.
			if errors.Is(err, ErrNotFound) {
				return nil, nfs3ErrBadCookie
			}
			return nil, handleStatus(err)
		}
		if len(ents) == 0 {
			l.eof = true
			return l, nfs3OK
		}
		if len(ents) > 1 {
			// We asked for one. More than one leaves entries we cannot address
			// with a cookie, and silently dropping them would give the client a
			// listing with holes in it.
			return nil, nfs3ErrServerFault
		}
		child := ents[0]
		wire, ok := wireCookie(next)
		if !ok {
			return nil, nfs3ErrServerFault
		}
		de := dirEntry{fileID: child.FileID, name: child.Name, cookie: wire}
		if plus {
			attr := child
			de.attr = &attr
			if h, err := s.backend.Handle(dir.share, child.FileID); err == nil && len(h) <= MaxHandleSize {
				de.handle = h
			}
		}
		if !l.add(de) {
			return l, nfs3OK
		}
		cookie = next
		if atEnd {
			l.eof = true
			return l, nfs3OK
		}
	}
	return l, nfs3OK
}

// readdirBudget turns a client's declared count into a byte budget for entries, or
// reports that it left no room for even the smallest one.
func readdirBudget(count uint32) (int, bool) {
	if count > maxReaddirCount {
		count = maxReaddirCount
	}
	budget := int(count) - readdirFixedCost
	// The smallest possible entry: presence flag, file id, a one-byte name and a
	// cookie. Anything less than that is a client asking for the impossible.
	const smallest = 4 + 8 + 4 + 4 + 8
	if budget < smallest {
		return 0, false
	}
	return budget, true
}

// readdir lists a directory's names.
func (s *nfsService) readdir(_ context.Context, c *Call, e *Encoder) error {
	fh, err := decodeHandle(c.Args)
	if err != nil {
		return err
	}
	cookie, err := c.Args.Uint64()
	if err != nil {
		return err
	}
	// The cookie verifier is decoded and ignored. This server always issues a zero
	// verifier, so there is nothing to compare against; verifying would mean failing
	// listings every time the index caught up with the peer, which is a worse answer
	// than a listing that reflects the tree as it now is.
	if _, err := c.Args.Fixed(8); err != nil {
		return err
	}
	count, err := c.Args.Uint32()
	if err != nil {
		return err
	}

	dir, ok := s.resolveDir(fh, c, e)
	if !ok {
		return nil
	}
	budget, ok := readdirBudget(count)
	if !ok {
		e.Uint32(nfs3ErrTooSmall)
		encodePostOpAttr(e, dir.fsid, &dir.entry)
		return nil
	}
	l, st := s.collect(dir, cookie, false, budget, budget)
	if st != nfs3OK {
		e.Uint32(st)
		encodePostOpAttr(e, dir.fsid, &dir.entry)
		return nil
	}
	if len(l.entries) == 0 && !l.eof {
		e.Uint32(nfs3ErrTooSmall)
		encodePostOpAttr(e, dir.fsid, &dir.entry)
		return nil
	}

	e.Uint32(nfs3OK)
	encodePostOpAttr(e, dir.fsid, &dir.entry)
	e.Fixed(make([]byte, 8)) // cookie verifier
	for i := range l.entries {
		de := &l.entries[i]
		e.Bool(true)
		e.Uint64(de.fileID)
		e.String(de.name)
		e.Uint64(de.cookie)
	}
	e.Bool(false)
	e.Bool(l.eof)
	return nil
}

// readdirPlus lists a directory with attributes and filehandles attached.
//
// This is the procedure the whole index exists for: a scanner's readdir followed by one
// getattr per entry becomes a single call, answered entirely from local disk.
func (s *nfsService) readdirPlus(_ context.Context, c *Call, e *Encoder) error {
	fh, err := decodeHandle(c.Args)
	if err != nil {
		return err
	}
	cookie, err := c.Args.Uint64()
	if err != nil {
		return err
	}
	if _, err := c.Args.Fixed(8); err != nil {
		return err
	}
	dircount, err := c.Args.Uint32()
	if err != nil {
		return err
	}
	maxcount, err := c.Args.Uint32()
	if err != nil {
		return err
	}

	dir, ok := s.resolveDir(fh, c, e)
	if !ok {
		return nil
	}
	total, ok := readdirBudget(maxcount)
	if !ok {
		e.Uint32(nfs3ErrTooSmall)
		encodePostOpAttr(e, dir.fsid, &dir.entry)
		return nil
	}
	// dircount limits the names and cookies alone, maxcount the whole reply, so the
	// listing is held to whichever binds first. A dircount of zero asks for nothing
	// at all, which no client means; it is read as "no separate limit".
	dirBudget := int(dircount)
	if dirBudget > total || dirBudget == 0 {
		dirBudget = total
	}
	l, st := s.collect(dir, cookie, true, dirBudget, total)
	if st != nfs3OK {
		e.Uint32(st)
		encodePostOpAttr(e, dir.fsid, &dir.entry)
		return nil
	}
	if len(l.entries) == 0 && !l.eof {
		e.Uint32(nfs3ErrTooSmall)
		encodePostOpAttr(e, dir.fsid, &dir.entry)
		return nil
	}

	e.Uint32(nfs3OK)
	encodePostOpAttr(e, dir.fsid, &dir.entry)
	e.Fixed(make([]byte, 8))
	for i := range l.entries {
		de := &l.entries[i]
		e.Bool(true)
		e.Uint64(de.fileID)
		e.String(de.name)
		e.Uint64(de.cookie)
		encodePostOpAttr(e, dir.fsid, de.attr)
		if de.handle == nil {
			e.Bool(false)
		} else {
			e.Bool(true)
			e.Opaque(de.handle)
		}
	}
	e.Bool(false)
	e.Bool(l.eof)
	return nil
}

// resolveDir resolves a directory filehandle, writing the failure reply itself when it
// cannot. The second result reports whether the caller should carry on.
func (s *nfsService) resolveDir(fh []byte, c *Call, e *Encoder) (object, bool) {
	dir, st := s.resolve(fh, c.Client)
	if st != nfs3OK {
		e.Uint32(st)
		e.Bool(false)
		return object{}, false
	}
	if dir.entry.Kind != KindDir {
		e.Uint32(nfs3ErrNotDir)
		encodePostOpAttr(e, dir.fsid, &dir.entry)
		return object{}, false
	}
	return dir, true
}
