package safepath_test

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/inchix/vllm_manager/storage/modelfsd/internal/safepath"
)

// TestValidateRelRejectsDotDot asserts the containment primitive refuses "..", absolute
// paths and NUL bytes before any resolution is attempted.
func TestValidateRelRejectsDotDot(t *testing.T) {
	escapes := []string{"..", "../etc/passwd", "a/../../b", "sub/.."}
	for _, rel := range escapes {
		if _, err := safepath.ValidateRel(rel); !errors.Is(err, safepath.ErrEscape) {
			t.Errorf("ValidateRel(%q): want ErrEscape, got %v", rel, err)
		}
	}
	invalid := []string{"/etc/passwd", "a\x00b", "a//b", "./x"}
	for _, rel := range invalid {
		if _, err := safepath.ValidateRel(rel); err == nil {
			t.Errorf("ValidateRel(%q): want error, got nil", rel)
		}
	}
	// A legitimate nested path is accepted.
	if parts, err := safepath.ValidateRel("a/b/c"); err != nil || len(parts) != 3 {
		t.Errorf("ValidateRel(a/b/c) = %v, %v; want 3 parts, nil", parts, err)
	}
}

// TestStatEscapeRefused asserts a resolution that would leave the root, via ".." or via a
// symlink pointing outside, is refused rather than followed.
func TestStatEscapeRefused(t *testing.T) {
	outside := t.TempDir()
	secret := filepath.Join(outside, "secret.txt")
	if err := os.WriteFile(secret, []byte("top secret"), 0o600); err != nil {
		t.Fatal(err)
	}

	export := t.TempDir()
	if err := os.WriteFile(filepath.Join(export, "model.bin"), []byte("weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	// A symlink inside the export that points outside it.
	if err := os.Symlink(secret, filepath.Join(export, "escape")); err != nil {
		t.Fatal(err)
	}

	root, err := safepath.OpenRoot("test", export)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()

	// The real file resolves.
	if _, err := root.Stat("model.bin", safepath.ModeBeneath); err != nil {
		t.Fatalf("Stat(model.bin): %v", err)
	}
	// ".." is refused.
	if _, err := root.Stat("..", safepath.ModeBeneath); !errors.Is(err, safepath.ErrEscape) {
		t.Errorf("Stat(..): want ErrEscape, got %v", err)
	}
	// The escaping symlink is refused, even in beneath mode which follows contained
	// symlinks.
	if _, err := root.Stat("escape", safepath.ModeBeneath); !errors.Is(err, safepath.ErrEscape) {
		t.Errorf("Stat(escape symlink): want ErrEscape, got %v", err)
	}
}
