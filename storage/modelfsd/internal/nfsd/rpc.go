package nfsd

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/netip"
	"slices"
	"strconv"
	"time"
)

// ONC RPC version 2 over TCP (RFC 5531), including the record marking of §11.
//
// Only the server half is here, and only what a read-only file server needs: there is
// no client, no rpcbind registration and no authentication flavour beyond the two a
// LAN NFS client will actually offer. AUTH_SYS credentials are parsed for
// well-formedness and then discarded — identity is squashed, and authorisation is
// decided by the client's address against the grant table, never by a uid the client
// asserted about itself.

// RPC message types and reply statuses, from RFC 5531 §9.
const (
	msgTypeCall  = 0
	msgTypeReply = 1

	replyAccepted = 0
	replyDenied   = 1

	// accept_stat
	acceptSuccess      = 0
	acceptProgUnavail  = 1
	acceptProgMismatch = 2
	acceptProcUnavail  = 3
	acceptGarbageArgs  = 4
	acceptSystemErr    = 5

	// reject_stat
	rejectRPCMismatch = 0
	rejectAuthError   = 1

	// auth_stat: only the two refusals this server ever issues.
	authBadCred = 1
	authTooWeak = 5

	// Authentication flavours (RFC 5531 §8).
	authFlavourNull = 0
	authFlavourSys  = 1

	rpcVersion = 2
)

// Wire limits. An opaque_auth body is capped at 400 bytes by RFC 5531 §9; the rest are
// ours, chosen so that a hostile client cannot make the server allocate on its say-so.
const (
	maxAuthBody    = 400
	maxMachineName = 255
	maxAuxGroups   = 16
	// maxFragments bounds a record split across many fragments. Without it a client
	// can hold a connection open indefinitely by sending zero-length non-final
	// fragments, which costs nothing to send and never terminates.
	maxFragments = 1024
	// lastFragment is the high bit of a record marking header.
	lastFragment = 0x8000_0000
	fragmentMask = 0x7fff_ffff
)

// Record framing errors.
var (
	// ErrRecordTooLarge means a record, or one of its fragments, exceeded the
	// configured maximum message size.
	ErrRecordTooLarge = errors.New("nfsd: rpc: record exceeds maximum message size")
	// ErrTooManyFragments means a record was split across more fragments than a
	// legitimate client would use.
	ErrTooManyFragments = errors.New("nfsd: rpc: too many record fragments")
)

// errNoReply means a message was so malformed that no reply can be addressed to it —
// there is no usable transaction id, or it was not a call at all. The connection is the
// only thing left to act on.
var errNoReply = errors.New("nfsd: rpc: unanswerable message")

// readRecord reassembles one RPC record from its fragments (RFC 5531 §11).
//
// max bounds the whole record, not each fragment, so a client cannot evade the limit by
// splitting. The buffer is grown as fragments arrive rather than allocated from a
// declared total, because no such total is declared: record marking gives the length of
// the fragment in hand and nothing more.
func readRecord(r io.Reader, max int) ([]byte, error) {
	var (
		hdr  [4]byte
		out  []byte
		frag int
	)
	for {
		if frag++; frag > maxFragments {
			return nil, ErrTooManyFragments
		}
		if _, err := io.ReadFull(r, hdr[:]); err != nil {
			return nil, err
		}
		mark := uint32(hdr[0])<<24 | uint32(hdr[1])<<16 | uint32(hdr[2])<<8 | uint32(hdr[3])
		size := int(mark & fragmentMask)
		last := mark&lastFragment != 0
		if size > max || len(out)+size > max {
			return nil, fmt.Errorf("%w: %d bytes", ErrRecordTooLarge, len(out)+size)
		}
		if size > 0 {
			start := len(out)
			out = slices.Grow(out, size)[:start+size]
			if _, err := io.ReadFull(r, out[start:]); err != nil {
				return nil, err
			}
		}
		if last {
			return out, nil
		}
	}
}

// writeRecord frames a reply as a single final fragment. Splitting a reply would be
// legal and buys nothing: the whole message is already in memory by the time it is
// written.
func writeRecord(w io.Writer, payload []byte) error {
	if len(payload) > fragmentMask {
		return fmt.Errorf("%w: reply of %d bytes", ErrRecordTooLarge, len(payload))
	}
	mark := uint32(len(payload)) | lastFragment
	hdr := [4]byte{byte(mark >> 24), byte(mark >> 16), byte(mark >> 8), byte(mark)}
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err := w.Write(payload)
	return err
}

// Credentials is what a client claimed about itself in an AUTH_SYS credential.
//
// It is recorded for logging and nothing else. The whole point of squashing is that no
// decision anywhere in this package consults these fields; they are kept because "which
// uid did the client think it was" is the first question asked when a client behaves
// oddly, and it is unrecoverable after the fact.
type Credentials struct {
	// Flavour is the authentication flavour the client offered.
	Flavour uint32
	// UID, GID, Machine and Groups are the AUTH_SYS fields, empty for AUTH_NULL.
	UID     uint32
	GID     uint32
	Machine string
	Groups  []uint32
}

// Call is one decoded RPC CALL, with its arguments left undecoded for the procedure
// that will handle them.
type Call struct {
	// XID is the client's transaction id, which the reply must carry back.
	XID uint32
	// Prog, Vers and Proc name what the client asked for.
	Prog uint32
	Vers uint32
	Proc uint32
	// Cred is what the client claimed about itself, for logs only.
	Cred Credentials
	// Client is the address the call arrived from, and the only thing
	// authorisation is decided by.
	Client netip.Addr
	// Args decodes the procedure's arguments, positioned just past the header.
	Args *Decoder
}

// Procedure serves one RPC procedure, encoding its results into enc.
//
// A procedure reports protocol-level failures — a status in the reply body — by
// encoding them; it returns an error only when the arguments could not be decoded, or
// when the server itself failed. Returning a decoding error produces GARBAGE_ARGS and
// anything else produces SYSTEM_ERR, so a procedure must not have written to enc before
// returning one.
type Procedure func(ctx context.Context, c *Call, enc *Encoder) error

// Program is one RPC program: a number, the version range it answers for, and its
// procedures.
type Program struct {
	// Number is the RPC program number.
	Number uint32
	// Low and High bound the versions this program answers for. A call outside
	// the range is refused with PROG_MISMATCH naming them.
	Low  uint32
	High uint32
	// Procs maps procedure numbers to their implementations. A number absent from
	// the map is refused with PROC_UNAVAIL.
	Procs map[uint32]Procedure
}

// dispatcher routes calls to programs by number.
type dispatcher struct {
	programs map[uint32]*Program
	// observe is called once per served call, for metrics. Set before serving and
	// not changed afterwards.
	observe Observer
}

func newDispatcher(programs ...*Program) *dispatcher {
	d := &dispatcher{programs: make(map[uint32]*Program, len(programs))}
	for _, p := range programs {
		d.programs[p.Number] = p
	}
	return d
}

// decodeAuth decodes one opaque_auth. The body of an AUTH_SYS credential is parsed to
// confirm it is well formed and then thrown away.
func decodeAuth(d *Decoder) (Credentials, error) {
	flavour, err := d.Uint32()
	if err != nil {
		return Credentials{}, err
	}
	body, err := d.Opaque(maxAuthBody)
	if err != nil {
		return Credentials{}, err
	}
	cred := Credentials{Flavour: flavour}
	if flavour != authFlavourSys {
		return cred, nil
	}
	// AUTH_SYS body, RFC 5531 §8.2: stamp, machine name, uid, gid, aux gids.
	bd := NewDecoder(body)
	if _, err := bd.Uint32(); err != nil {
		return cred, err
	}
	if cred.Machine, err = bd.String(maxMachineName); err != nil {
		return cred, err
	}
	if cred.UID, err = bd.Uint32(); err != nil {
		return cred, err
	}
	if cred.GID, err = bd.Uint32(); err != nil {
		return cred, err
	}
	if cred.Groups, err = DecodeArray(bd, maxAuxGroups, (*Decoder).Uint32); err != nil {
		return cred, err
	}
	return cred, nil
}

// dispatch decodes one call and encodes the whole reply, including the RPC header.
//
// It returns an error only when no reply can be sent at all. Every other outcome —
// unknown program, unknown procedure, undecodable arguments, refused credentials — is a
// reply, because a client that gets silence learns nothing and retries forever.
func (dp *dispatcher) dispatch(ctx context.Context, msg []byte, client netip.Addr, enc *Encoder) error {
	d := NewDecoder(msg)
	xid, err := d.Uint32()
	if err != nil {
		return errNoReply
	}
	mtype, err := d.Uint32()
	if err != nil {
		return errNoReply
	}
	if mtype != msgTypeCall {
		// A REPLY arriving at a server is either a confused client or a probe.
		// There is nothing to answer.
		return errNoReply
	}
	vers, err := d.Uint32()
	if err != nil {
		encodeGarbageArgs(enc, xid)
		return nil
	}
	if vers != rpcVersion {
		encodeRPCMismatch(enc, xid, rpcVersion, rpcVersion)
		return nil
	}

	c := Call{XID: xid, Client: client}
	if c.Prog, err = d.Uint32(); err != nil {
		encodeGarbageArgs(enc, xid)
		return nil
	}
	if c.Vers, err = d.Uint32(); err != nil {
		encodeGarbageArgs(enc, xid)
		return nil
	}
	if c.Proc, err = d.Uint32(); err != nil {
		encodeGarbageArgs(enc, xid)
		return nil
	}
	cred, err := decodeAuth(d)
	if err != nil {
		encodeAuthError(enc, xid, authBadCred)
		return nil
	}
	// The verifier is decoded to keep the argument stream aligned. For AUTH_NULL and
	// AUTH_SYS it carries nothing, and a client that sends something in it is not
	// thereby saying anything we would act on.
	if _, err := decodeAuth(d); err != nil {
		encodeAuthError(enc, xid, authBadCred)
		return nil
	}
	switch cred.Flavour {
	case authFlavourNull, authFlavourSys:
	default:
		// Not a refusal of the client, but of the flavour: we offer exactly two,
		// and advertise both in the MOUNT reply.
		encodeAuthError(enc, xid, authTooWeak)
		return nil
	}
	c.Cred = cred
	c.Args = d

	started := time.Now()
	report := func(status string) {
		if dp.observe == nil {
			return
		}
		name := ProcName(c.Prog, c.Proc)
		if name == "" {
			name = strconv.FormatUint(uint64(c.Proc), 10)
		}
		dp.observe(Event{Program: c.Prog, Proc: c.Proc, Name: name,
			Status: status, Elapsed: time.Since(started)})
	}

	prog, ok := dp.programs[c.Prog]
	if !ok {
		encodeAcceptedHeader(enc, xid, acceptProgUnavail)
		report(StatusProgUnavail)
		return nil
	}
	if c.Vers < prog.Low || c.Vers > prog.High {
		encodeAcceptedHeader(enc, xid, acceptProgMismatch)
		enc.Uint32(prog.Low)
		enc.Uint32(prog.High)
		report(StatusProgMismatch)
		return nil
	}
	proc, ok := prog.Procs[c.Proc]
	if !ok {
		encodeAcceptedHeader(enc, xid, acceptProcUnavail)
		report(StatusProcUnavail)
		return nil
	}

	encodeAcceptedHeader(enc, xid, acceptSuccess)
	if err := proc(ctx, &c, enc); err != nil {
		// The procedure may have written part of a reply before discovering the
		// arguments were short, so the body is discarded and the header rewritten.
		enc.Truncate(0)
		if isGarbage(err) {
			encodeGarbageArgs(enc, xid)
			report(StatusGarbageArgs)
		} else {
			encodeAcceptedHeader(enc, xid, acceptSystemErr)
			report(StatusSystemErr)
		}
		return nil
	}
	report(StatusOK)
	return nil
}

// isGarbage reports whether an error from a procedure means the client's arguments were
// malformed, as opposed to the server failing.
func isGarbage(err error) bool {
	return errors.Is(err, ErrTruncated) || errors.Is(err, ErrTooLong) || errors.Is(err, ErrBadBool)
}

// encodeAcceptedHeader writes a MSG_ACCEPTED reply header with a null verifier. The
// verifier is AUTH_NONE with an empty body for every flavour we accept.
func encodeAcceptedHeader(e *Encoder, xid uint32, stat uint32) {
	e.Uint32(xid)
	e.Uint32(msgTypeReply)
	e.Uint32(replyAccepted)
	e.Uint32(authFlavourNull)
	e.Uint32(0)
	e.Uint32(stat)
}

// encodeGarbageArgs writes the reply for arguments that could not be decoded.
func encodeGarbageArgs(e *Encoder, xid uint32) {
	e.Truncate(0)
	encodeAcceptedHeader(e, xid, acceptGarbageArgs)
}

// encodeAuthError writes a MSG_DENIED reply rejecting the credentials.
func encodeAuthError(e *Encoder, xid uint32, stat uint32) {
	e.Truncate(0)
	e.Uint32(xid)
	e.Uint32(msgTypeReply)
	e.Uint32(replyDenied)
	e.Uint32(rejectAuthError)
	e.Uint32(stat)
}

// encodeRPCMismatch writes a MSG_DENIED reply for an RPC version we do not speak.
func encodeRPCMismatch(e *Encoder, xid uint32, low, high uint32) {
	e.Truncate(0)
	e.Uint32(xid)
	e.Uint32(msgTypeReply)
	e.Uint32(replyDenied)
	e.Uint32(rejectRPCMismatch)
	e.Uint32(low)
	e.Uint32(high)
}
