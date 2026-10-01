package mlxbench

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// writeIdentityFixture creates a small complete report and its required files.
func writeIdentityFixture(t *testing.T, path string, identity runIdentity) {
	t.Helper()
	data, err := json.Marshal(map[string]any{"complete": true, "identity": identity})
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	base := path[:len(path)-len(filepath.Ext(path))]
	for _, suffix := range []string{".f32", "-probabilities.json.gz", "-chunks.jsonl"} {
		if err = os.WriteFile(base+suffix, []byte("saved result"), 0600); err != nil {
			t.Fatal(err)
		}
	}
}

// TestResumeIdentity rejects changes to inputs and measurement policy while
// preserving the old report. Identical inputs and settings resume successfully.
func TestResumeIdentity(t *testing.T) {
	c := Case{Name: "baseline", Context: 4096, Chunk: 512, HeadBatch: 32}
	config := Config{DecodeTokens: 32}
	prompt := Prompt{Prompt: "the new story", PreviousPrompt: "the old story", Prefix: "the ", PromptTokens: 10}
	original := identityFor(c, config, prompt, "weights", "executable", "device")
	path := filepath.Join(t.TempDir(), "baseline.json")
	writeIdentityFixture(t, path, original)
	before, _ := os.ReadFile(path)
	if done, err := completedResult(path, original); !done || err != nil {
		t.Fatalf("identical run did not resume: %v", err)
	}
	for name, change := range map[string]func(*runIdentity){
		"weights":            func(i *runIdentity) { i.Checkpoint = "new weights" },
		"prompt":             func(i *runIdentity) { i.Prompt = textDigest("another story") },
		"previous prompt":    func(i *runIdentity) { i.PreviousPrompt = textDigest("other old story") },
		"protected prefix":   func(i *runIdentity) { i.Prefix = textDigest("header") },
		"warmup":             func(i *runIdentity) { i.WarmupFull = true },
		"rest":               func(i *runIdentity) { i.RestSeconds++ },
		"cooldown":           func(i *runIdentity) { i.CoolSeconds++ },
		"reference":          func(i *runIdentity) { i.ReferenceCase = "different" },
		"executable":         func(i *runIdentity) { i.Executable = "changed executable" },
		"device":             func(i *runIdentity) { i.Device = "other GPU" },
		"decode":             func(i *runIdentity) { i.DecodeTokens++ },
		"case":               func(i *runIdentity) { i.Case.Chunk = 256 },
		"kernel environment": func(i *runIdentity) { i.KernelEnvironment = map[string]string{"TINYOAI_QMM_FAST_LOADS": "1"} },
	} {
		t.Run(name, func(t *testing.T) {
			changed := original
			change(&changed)
			if done, err := completedResult(path, changed); done || err == nil {
				t.Fatal("resumed incompatible report")
			}
			after, _ := os.ReadFile(path)
			if string(after) != string(before) {
				t.Fatal("changed old report")
			}
		})
	}
}

// TestResumeRequiresIdentityAndArtifacts rejects legacy, partial, and corrupt
// reports rather than treating them as measurements for a new run.
func TestResumeRequiresIdentityAndArtifacts(t *testing.T) {
	path := filepath.Join(t.TempDir(), "case.json")
	identity := runIdentity{Version: 1}
	if done, err := completedResult(path, identity); done || err != nil {
		t.Fatalf("missing report: %v", err)
	}
	for _, data := range []string{`{"complete":true}`, `{"complete":false}`, `{`} {
		if err := os.WriteFile(path, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
		if done, err := completedResult(path, identity); done || err == nil {
			t.Fatalf("accepted %s", data)
		}
	}
	writeIdentityFixture(t, path, identity)
	if err := os.Remove(filepath.Join(filepath.Dir(path), "case.f32")); err != nil {
		t.Fatal(err)
	}
	if done, err := completedResult(path, identity); done || err == nil {
		t.Fatal("accepted missing logits")
	}
}

// TestCheckpointIdentity hashes contents rather than paths or file sizes and
// observes cancellation before reading large model files.
func TestCheckpointIdentity(t *testing.T) {
	dir := t.TempDir()
	names := []string{"config.json", "tokenizer.json", "model.safetensors"}
	for _, name := range names {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("original"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	original, err := checkpointDigest(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	moved := t.TempDir()
	for _, name := range names {
		if err := os.WriteFile(filepath.Join(moved, name), []byte("original"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	if same, err := checkpointDigest(context.Background(), moved); err != nil || same != original {
		t.Fatalf("moving files changed identity: %v", err)
	}
	for _, name := range names {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte("modified"), 0600); err != nil {
			t.Fatal(err)
		}
		if changed, err := checkpointDigest(context.Background(), dir); err != nil || changed == original {
			t.Fatalf("missed same-size change in %s: %v", name, err)
		}
		if err := os.WriteFile(path, []byte("original"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := checkpointDigest(ctx, dir); err != context.Canceled {
		t.Fatalf("ignored cancellation: %v", err)
	}
}
