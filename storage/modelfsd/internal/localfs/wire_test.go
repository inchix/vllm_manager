package localfs_test

import (
	"context"
	"encoding/binary"
	"io"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/inchix/vllm_manager/storage/modelfsd/internal/localfs"
	"github.com/inchix/vllm_manager/storage/modelfsd/internal/nfsd"
)

// Program and procedure numbers from RFC 1813, repeated here so the test drives the server
// purely over the wire.
const (
	nfsProg   = 100003
	nfsVers   = 3
	mountProg = 100005
	mountVers = 3

	procNFSNull   = 0
	procNFSWrite  = 7
	procMountMnt  = 1
	nfs3ErrROFS   = 30
	mnt3OK        = 0
	replyAccepted = 0
	acceptSuccess = 0
)

// startServer runs a modelfsd NFS server over a real TCP socket backed by a one-file
// export, and returns a connected client.
func startServer(t *testing.T) net.Conn {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "weights.bin"), []byte("hello weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	b, err := localfs.New(localfs.Config{ExportDir: dir, Share: share})
	if err != nil {
		t.Fatal(err)
	}
	srv, err := nfsd.New(nfsd.Config{Backend: b, Exports: []string{share}})
	if err != nil {
		t.Fatal(err)
	}
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = srv.Serve(ctx, l) }()

	conn, err := net.Dial("tcp", l.Addr().String())
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = conn.Close()
		cancel()
		_ = l.Close()
		_ = b.Close()
	})
	return conn
}

// encodeCall builds an ONC-RPC v2 CALL with AUTH_NULL credentials and the given raw args.
func encodeCall(xid, prog, vers, proc uint32, args []byte) []byte {
	e := nfsd.NewEncoder(64)
	e.Uint32(xid)
	e.Uint32(0) // msg type: CALL
	e.Uint32(2) // RPC version 2
	e.Uint32(prog)
	e.Uint32(vers)
	e.Uint32(proc)
	e.Uint32(0) // cred flavour AUTH_NULL
	e.Uint32(0) // cred length
	e.Uint32(0) // verf flavour AUTH_NULL
	e.Uint32(0) // verf length
	e.Fixed(args)
	return e.Bytes()
}

func roundTrip(t *testing.T, conn net.Conn, payload []byte) *nfsd.Decoder {
	t.Helper()
	_ = conn.SetDeadline(time.Now().Add(5 * time.Second))

	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(payload))|0x80000000)
	if _, err := conn.Write(hdr[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := conn.Write(payload); err != nil {
		t.Fatal(err)
	}

	if _, err := io.ReadFull(conn, hdr[:]); err != nil {
		t.Fatal(err)
	}
	mark := binary.BigEndian.Uint32(hdr[:])
	n := mark & 0x7fffffff
	buf := make([]byte, n)
	if _, err := io.ReadFull(conn, buf); err != nil {
		t.Fatal(err)
	}
	return nfsd.NewDecoder(buf)
}

// replyHeader consumes the RPC reply header and asserts the call was accepted with
// success, returning the decoder positioned at the procedure result.
func replyHeader(t *testing.T, d *nfsd.Decoder, wantXID uint32) *nfsd.Decoder {
	t.Helper()
	must := func(v uint32, err error) uint32 {
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	if xid := must(d.Uint32()); xid != wantXID {
		t.Fatalf("reply xid = %d, want %d", xid, wantXID)
	}
	if mt := must(d.Uint32()); mt != 1 {
		t.Fatalf("reply msg type = %d, want 1 (REPLY)", mt)
	}
	if rs := must(d.Uint32()); rs != replyAccepted {
		t.Fatalf("reply stat = %d, want accepted", rs)
	}
	_ = must(d.Uint32()) // verf flavour
	vl := must(d.Uint32())
	if vl > 0 {
		if _, err := d.Fixed(int(vl)); err != nil {
			t.Fatal(err)
		}
	}
	if as := must(d.Uint32()); as != acceptSuccess {
		t.Fatalf("accept stat = %d, want success", as)
	}
	return d
}

// TestWireNullAndWriteROFS drives the server over TCP: NULL succeeds (liveness), and a
// WRITE is refused with NFS3ERR_ROFS — the read-only guarantee, end to end.
func TestWireNullAndWriteROFS(t *testing.T) {
	conn := startServer(t)

	// NULL: accepted, empty body.
	replyHeader(t, roundTrip(t, conn, encodeCall(1, nfsProg, nfsVers, procNFSNull, nil)), 1)

	// WRITE: the read-only handler refuses without decoding args, so empty args suffice.
	body := replyHeader(t, roundTrip(t, conn, encodeCall(2, nfsProg, nfsVers, procNFSWrite, nil)), 2)
	status, err := body.Uint32()
	if err != nil {
		t.Fatal(err)
	}
	if status != nfs3ErrROFS {
		t.Fatalf("WRITE status = %d, want %d (NFS3ERR_ROFS)", status, nfs3ErrROFS)
	}
}

// TestWireMount drives a MOUNTv3 MNT for the configured export and asserts a root
// filehandle comes back.
func TestWireMount(t *testing.T) {
	conn := startServer(t)

	e := nfsd.NewEncoder(64)
	e.String("/" + share)
	body := replyHeader(t, roundTrip(t, conn, encodeCall(3, mountProg, mountVers, procMountMnt, e.Bytes())), 3)

	status, err := body.Uint32()
	if err != nil {
		t.Fatal(err)
	}
	if status != mnt3OK {
		t.Fatalf("MNT status = %d, want %d (MNT3_OK)", status, mnt3OK)
	}
	handle, err := body.Opaque(64)
	if err != nil {
		t.Fatal(err)
	}
	if len(handle) == 0 {
		t.Fatal("MNT returned an empty filehandle")
	}
}
