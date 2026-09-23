package localfs_test

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/inchix/vllm_manager/storage/modelfsd/internal/localfs"
	"github.com/inchix/vllm_manager/storage/modelfsd/internal/nfsd"
)

const share = "export/llm_models"

// buildExport writes a small model-repo-shaped tree and returns a backend over it, plus
// the bytes of the large file for comparison.
func buildExport(t *testing.T) (*localfs.Backend, []byte) {
	t.Helper()
	dir := t.TempDir()

	// A file larger than one read-ahead chunk (256 KiB), with a recognisable pattern so
	// a read at any offset can be checked.
	big := make([]byte, 700*1024)
	for i := range big {
		big[i] = byte(i*7 + 3)
	}
	if err := os.WriteFile(filepath.Join(dir, "weights.bin"), big, 0o600); err != nil {
		t.Fatal(err)
	}
	sub := filepath.Join(dir, "config")
	if err := os.Mkdir(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(sub, "params.json"), []byte(`{"n":1}`), 0o600); err != nil {
		t.Fatal(err)
	}

	b, err := localfs.New(localfs.Config{ExportDir: dir, Share: share})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = b.Close() })
	return b, big
}

func TestRootAndLookup(t *testing.T) {
	b, _ := buildExport(t)

	root, err := b.Root(share)
	if err != nil {
		t.Fatalf("Root: %v", err)
	}
	if root.Kind != nfsd.KindDir {
		t.Fatalf("root kind = %v, want dir", root.Kind)
	}

	// The wrong share name is not found.
	if _, err := b.Root("other/path"); !errors.Is(err, nfsd.ErrNotFound) {
		t.Errorf("Root(other) = %v, want ErrNotFound", err)
	}

	f, err := b.Lookup(share, root.FileID, "weights.bin")
	if err != nil {
		t.Fatalf("Lookup(weights.bin): %v", err)
	}
	if f.Kind != nfsd.KindFile || f.Size != 700*1024 {
		t.Errorf("weights.bin entry = %+v, want file of 716800 bytes", f)
	}
	// Get by the id returns the same object.
	if got, err := b.Get(share, f.FileID); err != nil || got.Size != f.Size {
		t.Errorf("Get(%d) = %+v, %v", f.FileID, got, err)
	}
	// A missing name is a negative lookup.
	if _, err := b.Lookup(share, root.FileID, "nope"); !errors.Is(err, nfsd.ErrNotFound) {
		t.Errorf("Lookup(nope) = %v, want ErrNotFound", err)
	}
}

// TestLookupCannotEscape asserts a name that tries to leave the export is reported as
// absent, never resolved.
func TestLookupCannotEscape(t *testing.T) {
	b, _ := buildExport(t)
	root, _ := b.Root(share)
	for _, name := range []string{"..", "../..", "/etc"} {
		if _, err := b.Lookup(share, root.FileID, name); !errors.Is(err, nfsd.ErrNotFound) {
			t.Errorf("Lookup(%q) = %v, want ErrNotFound", name, err)
		}
	}
}

func TestReadDir(t *testing.T) {
	b, _ := buildExport(t)
	root, _ := b.Root(share)

	// The server always asks one entry at a time, replaying the cookie. Mirror that.
	var names []string
	var cookie uint64
	for i := 0; i < 100; i++ {
		ents, next, eof, err := b.ReadDir(share, root.FileID, cookie, 1)
		if err != nil {
			t.Fatalf("ReadDir: %v", err)
		}
		for _, e := range ents {
			names = append(names, e.Name)
		}
		cookie = next
		if eof {
			break
		}
	}
	if len(names) != 2 || names[0] != "config" || names[1] != "weights.bin" {
		t.Errorf("listing = %v, want [config weights.bin] in order", names)
	}
}

// TestReadBytes is the core read path: bytes read back through the read-ahead cache must
// match the file exactly, across and within chunk boundaries.
func TestReadBytes(t *testing.T) {
	b, big := buildExport(t)
	root, _ := b.Root(share)
	f, err := b.Lookup(share, root.FileID, "weights.bin")
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	cases := []struct {
		off  int64
		want int
	}{
		{0, 100},              // start of first chunk
		{250 * 1024, 20 * 1024}, // spanning the first chunk boundary
		{699 * 1024, 4096},    // last chunk, running into EOF
	}
	for _, tc := range cases {
		buf := make([]byte, tc.want)
		n, err := b.ReadAt(ctx, share, f.FileID, buf, tc.off)
		if err != nil && !errors.Is(err, io.EOF) {
			t.Fatalf("ReadAt(off=%d): %v", tc.off, err)
		}
		end := int(tc.off) + n
		if end > len(big) {
			t.Fatalf("read past end: off=%d n=%d", tc.off, n)
		}
		for i := 0; i < n; i++ {
			if buf[i] != big[int(tc.off)+i] {
				t.Fatalf("byte %d at off %d = %d, want %d", i, tc.off, buf[i], big[int(tc.off)+i])
			}
		}
	}

	// A read wholly past the end returns EOF and no bytes.
	buf := make([]byte, 16)
	n, err := b.ReadAt(ctx, share, f.FileID, buf, int64(len(big)))
	if n != 0 || !errors.Is(err, io.EOF) {
		t.Errorf("ReadAt at EOF = %d, %v; want 0, EOF", n, err)
	}
}

// TestHandleRoundTrip asserts a filehandle resolves back to its file id, and a handle from
// a different generation is stale.
func TestHandleRoundTrip(t *testing.T) {
	b, _ := buildExport(t)
	root, _ := b.Root(share)
	f, _ := b.Lookup(share, root.FileID, "weights.bin")

	h, err := b.Handle(share, f.FileID)
	if err != nil {
		t.Fatal(err)
	}
	gotShare, id, err := b.Resolve(h)
	if err != nil || gotShare != share || id != f.FileID {
		t.Fatalf("Resolve = %q, %d, %v; want %q, %d, nil", gotShare, id, err, share, f.FileID)
	}
	// A tampered generation is refused.
	h[0] ^= 0xff
	if _, _, err := b.Resolve(h); !errors.Is(err, nfsd.ErrStale) {
		t.Errorf("Resolve(bad gen) = %v, want ErrStale", err)
	}
}
