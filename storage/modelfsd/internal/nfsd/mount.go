package nfsd

import (
	"context"
	"errors"
	"net/netip"
	"strings"
	"sync"
)

// MOUNT version 3, RFC 1813 appendix I.
//
// Handled in this process, on the same port as NFS itself. The standing objection to
// NFSv3 is that it drags in rpcbind, rpc.mountd and rpc.statd as separate network
// services; owning the implementation removes that objection rather than answering it,
// because none of those services exist here. NLM, statd and quota are not offered at
// all — a read-only export has no locks to arbitrate and no state to recover.

// MOUNT program numbers.
const (
	mountProgram = 100005
	mountVersion = 3
)

// MOUNTv3 procedure numbers.
const (
	mountProcNull    = 0
	mountProcMnt     = 1
	mountProcDump    = 2
	mountProcUmnt    = 3
	mountProcUmntAll = 4
	mountProcExport  = 5
)

// mountstat3 values, RFC 1813 appendix I §5.1.
const (
	mnt3OK             = 0
	mnt3ErrNoEnt       = 2
	mnt3ErrAcces       = 13
	mnt3ErrNotDir      = 20
	mnt3ErrInval       = 22
	mnt3ErrNameTooLong = 63
	mnt3ErrServerFault = 10006
)

// Wire limits from the MOUNT protocol definition.
const (
	maxDirPath  = 1024
	maxHostName = 255
	// maxMountRecords bounds the table DUMP reports from. The table is a courtesy —
	// nothing depends on it, and UMNT is advisory — so a client that mounts endlessly
	// stops being recorded rather than being allowed to grow the server's memory.
	maxMountRecords = 1024
)

// mountRecord is one entry in the table DUMP reports.
type mountRecord struct {
	host string
	dir  string
}

// mountService answers the MOUNT program.
type mountService struct {
	backend Backend
	// exports is what EXPORT advertises. It is a display list only: a name here that
	// the client has no grant for is filtered out, and a grant is what MNT actually
	// checks.
	exports []string

	mu     sync.Mutex
	mounts map[mountRecord]struct{}
}

func newMountService(b Backend, exports []string) *mountService {
	return &mountService{
		backend: b,
		exports: exports,
		mounts:  make(map[mountRecord]struct{}),
	}
}

// program describes the MOUNT program for the RPC dispatcher.
func (s *mountService) program() *Program {
	return &Program{
		Number: mountProgram,
		Low:    mountVersion,
		High:   mountVersion,
		Procs: map[uint32]Procedure{
			mountProcNull:    s.null,
			mountProcMnt:     s.mnt,
			mountProcDump:    s.dump,
			mountProcUmnt:    s.umnt,
			mountProcUmntAll: s.umntAll,
			mountProcExport:  s.export,
		},
	}
}

// shareFromPath extracts a share name from a MOUNT dirpath.
//
// modelfsd has exactly one export, and its share name is the canonical mount path the
// operator configured — which is normally a multi-segment path such as
// "/export/llm_models". So the dirpath is normalised (surrounding slashes trimmed) and
// kept whole, including interior separators; only the pieces that could name something
// outside the namespace — an empty, ".", ".." or NUL-bearing component — are refused.
// There is no per-component path resolution here: the whole normalised string is compared
// against the one configured export by the backend, which is the point.
func shareFromPath(dirpath string) (string, uint32) {
	if len(dirpath) > maxDirPath {
		return "", mnt3ErrNameTooLong
	}
	if strings.ContainsRune(dirpath, '\x00') {
		return "", mnt3ErrInval
	}
	name := strings.Trim(dirpath, "/")
	if name == "" {
		return "", mnt3ErrInval
	}
	for _, comp := range strings.Split(name, "/") {
		switch comp {
		case "", ".", "..":
			return "", mnt3ErrNotDir
		}
		if len(comp) > maxNameLen {
			return "", mnt3ErrNameTooLong
		}
	}
	return name, mnt3OK
}

// exportPath renders a share name the way a client's mount line spells it.
func exportPath(share string) string { return "/" + share }

// null answers the liveness probe.
func (s *mountService) null(_ context.Context, _ *Call, _ *Encoder) error { return nil }

// mnt returns the root filehandle for a share, or refuses the client.
func (s *mountService) mnt(_ context.Context, c *Call, e *Encoder) error {
	dirpath, err := c.Args.String(maxDirPath)
	if err != nil {
		return err
	}
	share, st := shareFromPath(dirpath)
	if st != mnt3OK {
		e.Uint32(st)
		return nil
	}
	// The allowlist is consulted before the index, so a client that may not have this
	// share cannot learn whether it exists.
	if !s.backend.Allowed(share, c.Client) {
		e.Uint32(mnt3ErrAcces)
		return nil
	}
	root, err := s.backend.Root(share)
	if err != nil {
		e.Uint32(mountStatus(err))
		return nil
	}
	if root.Kind != KindDir {
		e.Uint32(mnt3ErrNotDir)
		return nil
	}
	handle, err := s.backend.Handle(share, root.FileID)
	if err != nil || len(handle) == 0 || len(handle) > MaxHandleSize {
		e.Uint32(mnt3ErrServerFault)
		return nil
	}

	s.record(c.Client, exportPath(share))
	e.Uint32(mnt3OK)
	e.Opaque(handle)
	// The flavours we accept. Both are squashed to nobody; AUTH_SYS is advertised
	// only because a client that is offered nothing but AUTH_NULL sometimes refuses
	// to mount at all.
	e.Uint32(2)
	e.Uint32(authFlavourNull)
	e.Uint32(authFlavourSys)
	return nil
}

// mountStatus maps a backend failure onto a MOUNT status.
func mountStatus(err error) uint32 {
	switch {
	case err == nil:
		return mnt3OK
	case errors.Is(err, ErrNotFound), errors.Is(err, ErrStale):
		return mnt3ErrNoEnt
	case errors.Is(err, ErrDenied):
		return mnt3ErrAcces
	}
	return mnt3ErrServerFault
}

// record notes a successful mount, up to the table's cap.
func (s *mountService) record(client netip.Addr, dir string) {
	host := hostName(client)
	s.mu.Lock()
	defer s.mu.Unlock()
	rec := mountRecord{host: host, dir: dir}
	if _, ok := s.mounts[rec]; ok {
		return
	}
	if len(s.mounts) >= maxMountRecords {
		return
	}
	s.mounts[rec] = struct{}{}
}

// hostName renders a client address for the mount table.
func hostName(client netip.Addr) string {
	if !client.IsValid() {
		return "unknown"
	}
	h := client.String()
	if len(h) > maxHostName {
		h = h[:maxHostName]
	}
	return h
}

// dump lists what this server believes is mounted.
//
// It is advisory and always has been: UMNT is a courtesy a client may never send, and a
// client that reboots never sends it. Nothing in the serving path consults this table,
// so an entry that outlives its mount is untidy rather than wrong.
func (s *mountService) dump(_ context.Context, _ *Call, e *Encoder) error {
	s.mu.Lock()
	records := make([]mountRecord, 0, len(s.mounts))
	for rec := range s.mounts {
		records = append(records, rec)
	}
	s.mu.Unlock()

	// mountlist is a linked list: each element is preceded by a present flag, and the
	// list ends with an absent one.
	for _, rec := range records {
		e.Bool(true)
		e.String(rec.host)
		e.String(rec.dir)
	}
	e.Bool(false)
	return nil
}

// umnt forgets one mount.
func (s *mountService) umnt(_ context.Context, c *Call, _ *Encoder) error {
	dirpath, err := c.Args.String(maxDirPath)
	if err != nil {
		return err
	}
	share, st := shareFromPath(dirpath)
	if st != mnt3OK {
		// UMNT returns void, so there is nothing to report a bad path with. The
		// client is unmounting either way.
		return nil
	}
	rec := mountRecord{host: hostName(c.Client), dir: exportPath(share)}
	s.mu.Lock()
	delete(s.mounts, rec)
	s.mu.Unlock()
	return nil
}

// umntAll forgets every mount this client holds.
//
// Only this client's records are dropped. The procedure's name invites dropping the
// whole table, and a server that did so would let any host on the LAN erase everyone
// else's entries.
func (s *mountService) umntAll(_ context.Context, c *Call, _ *Encoder) error {
	host := hostName(c.Client)
	s.mu.Lock()
	for rec := range s.mounts {
		if rec.host == host {
			delete(s.mounts, rec)
		}
	}
	s.mu.Unlock()
	return nil
}

// export lists the shares this client may mount.
//
// Filtered by the same allowlist MNT uses, so EXPORT cannot be used to enumerate the
// names of shares the caller was never granted.
func (s *mountService) export(_ context.Context, c *Call, e *Encoder) error {
	for _, share := range s.exports {
		if !s.backend.Allowed(share, c.Client) {
			continue
		}
		e.Bool(true)
		e.String(exportPath(share))
		// The group list names the hosts allowed to mount. We publish none: the
		// allowlist is the authority and reproducing it here would be a second
		// copy of it for an operator to find disagreeing with the first.
		e.Bool(false)
	}
	e.Bool(false)
	return nil
}
