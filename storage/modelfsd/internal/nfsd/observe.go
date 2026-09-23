package nfsd

import "time"

// Observation reporting, for metrics and for nothing else.
//
// A function field rather than an interface, and one call site rather than sprinkled through
// the procedures: the dispatcher already knows the program, the procedure and how the reply
// ended, so it is the only place that needs to say anything. Nothing here changes behaviour
// — an Observer that panics would take a connection with it, and a slow one delays a reply,
// so keep it to arithmetic.
//
// It carries no file handles, no paths and no client addresses. What a household's machines
// read is theirs; see docs/23-metrics.md §4.

// Event describes one served RPC.
type Event struct {
	// Program is the RPC program number: 100003 for NFS, 100005 for MOUNT.
	Program uint32
	// Proc is the procedure number, and Name is its name — "READ", "MNT" — or the
	// number as a string if this server does not know it.
	Proc uint32
	Name string
	// Status is how the reply ended: "ok", "garbage_args", "system_err",
	// "prog_unavail", "prog_mismatch", "proc_unavail", or "auth_error". These are the
	// RPC-level outcomes; a procedure that answers NFS3ERR_NOENT has status "ok",
	// because the server did answer.
	Status string
	// Elapsed is how long the procedure took.
	Elapsed time.Duration
}

// Observer receives one Event per served RPC. Nil does nothing.
type Observer func(Event)

// Status values, named so a caller can compare rather than spell.
const (
	StatusOK           = "ok"
	StatusGarbageArgs  = "garbage_args"
	StatusSystemErr    = "system_err"
	StatusProgUnavail  = "prog_unavail"
	StatusProgMismatch = "prog_mismatch"
	StatusProcUnavail  = "proc_unavail"
	StatusAuthError    = "auth_error"
)

// procNames gives every procedure a name, because "wgshare_nfs_requests_total{proc="6"}"
// is a metric nobody can read at three in the morning.
var procNames = map[uint32]map[uint32]string{
	nfsProgram: {
		nfsProcNull: "NULL", nfsProcGetAttr: "GETATTR", nfsProcSetAttr: "SETATTR",
		nfsProcLookup: "LOOKUP", nfsProcAccess: "ACCESS", nfsProcReadlink: "READLINK",
		nfsProcRead: "READ", nfsProcWrite: "WRITE", nfsProcCreate: "CREATE",
		nfsProcMkdir: "MKDIR", nfsProcSymlink: "SYMLINK", nfsProcMknod: "MKNOD",
		nfsProcRemove: "REMOVE", nfsProcRmdir: "RMDIR", nfsProcRename: "RENAME",
		nfsProcLink: "LINK", nfsProcReaddir: "READDIR", nfsProcReaddirPlus: "READDIRPLUS",
		nfsProcFSStat: "FSSTAT", nfsProcFSInfo: "FSINFO", nfsProcPathConf: "PATHCONF",
		nfsProcCommit: "COMMIT",
	},
	mountProgram: {
		mountProcNull: "NULL", mountProcMnt: "MNT", mountProcDump: "DUMP",
		mountProcUmnt: "UMNT", mountProcUmntAll: "UMNTALL", mountProcExport: "EXPORT",
	},
}

// ProcName names a procedure, or returns "" when this server does not serve it.
func ProcName(program, proc uint32) string {
	return procNames[program][proc]
}
