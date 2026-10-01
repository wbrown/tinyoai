package mlxbench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
)

// runIdentity binds a report to its inputs, executable, device, and scheduling
// policy. Checkpoint directories may move; their contents must remain identical.
// Version changes invalidate reports when measurement semantics change.
type runIdentity struct {
	Version           int               `json:"version"`
	Case              Case              `json:"case"`
	DecodeTokens      int               `json:"decode_tokens"`
	Checkpoint        string            `json:"checkpoint_sha256"`
	Executable        string            `json:"executable_sha256"`
	Device            string            `json:"device"`
	Platform          string            `json:"platform"`
	Prompt            string            `json:"prompt_sha256"`
	PreviousPrompt    string            `json:"previous_prompt_sha256"`
	Prefix            string            `json:"prefix_sha256"`
	PromptTokens      int               `json:"prompt_tokens"`
	WarmupFull        bool              `json:"warmup_full"`
	RestSeconds       int               `json:"rest_seconds"`
	CoolSeconds       int               `json:"cool_seconds"`
	ReferenceCase     string            `json:"reference_case"`
	KernelEnvironment map[string]string `json:"kernel_environment"`
}

// identityFor records effective inputs after exact-length prompt preparation
// and kernel overrides, before warm-up or measurement begins.
func identityFor(c Case, config Config, prompt Prompt, checkpoint, executable, device string) runIdentity {
	env := make(map[string]string)
	for _, key := range []string{"TINYOAI_QMM_BM", "TINYOAI_QMM_BN", "TINYOAI_QMM_BK", "TINYOAI_QMM_FAST_LOADS", "MLX_ENABLE_TF32", "MLX_METAL_FAST_SYNCH", "MLX_METAL_PREWARM"} {
		if value, ok := os.LookupEnv(key); ok {
			env[key] = value
		}
	}
	return runIdentity{Version: 1, Case: c, DecodeTokens: config.DecodeTokens, Checkpoint: checkpoint, Executable: executable,
		Device: device, Platform: runtime.GOOS + "/" + runtime.GOARCH, Prompt: textDigest(prompt.Prompt), PreviousPrompt: textDigest(prompt.PreviousPrompt),
		Prefix: textDigest(prompt.Prefix), PromptTokens: prompt.PromptTokens, WarmupFull: config.WarmupFull, RestSeconds: config.RestSeconds,
		CoolSeconds: config.CoolSeconds, ReferenceCase: config.ReferenceCase, KernelEnvironment: env}
}

// textDigest fingerprints exact input bytes, preserving whitespace and token boundaries.
func textDigest(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])
}

// fileDigest hashes large inputs with bounded memory and checks cancellation
// between reads. A file changed during hashing cannot establish an identity.
func fileDigest(ctx context.Context, path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	before, err := f.Stat()
	if err != nil {
		return "", err
	}
	h := sha256.New()
	buffer := make([]byte, 1<<20)
	for {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		n, err := f.Read(buffer)
		h.Write(buffer[:n])
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
	}
	after, err := f.Stat()
	if err != nil {
		return "", err
	}
	if before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) {
		return "", fmt.Errorf("input changed while hashing %s", path)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// checkpointDigest binds configuration, tokenization, and every safetensors
// shard used by the native loader. File names delimit the content digests.
func checkpointDigest(ctx context.Context, dir string) (string, error) {
	shards, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if err != nil {
		return "", err
	}
	if len(shards) == 0 {
		return "", fmt.Errorf("no safetensors weights in %s", dir)
	}
	paths := append([]string{filepath.Join(dir, "config.json"), filepath.Join(dir, "tokenizer.json")}, shards...)
	records := make([][2]string, 0, len(paths))
	for _, path := range paths {
		digest, err := fileDigest(ctx, path)
		if err != nil {
			return "", err
		}
		records = append(records, [2]string{filepath.Base(path), digest})
	}
	data, _ := json.Marshal(records)
	return textDigest(string(data)), nil
}

// completedResult accepts only reports whose complete identity still matches.
// Older reports remain untouched; callers must use a new output directory for
// changed inputs or reports written before identity tracking was introduced.
func completedResult(path string, want runIdentity) (bool, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	var saved struct {
		Complete bool         `json:"complete"`
		Identity *runIdentity `json:"identity"`
	}
	if err = json.Unmarshal(data, &saved); err != nil {
		return false, fmt.Errorf("invalid existing result %s: %w", path, err)
	}
	expected, _ := json.Marshal(want)
	actual, _ := json.Marshal(saved.Identity)
	if !saved.Complete || saved.Identity == nil || string(expected) != string(actual) {
		return false, fmt.Errorf("incompatible existing result %s: model, prompt, executable, device, or run settings changed (or identity is missing); use a new output directory", path)
	}
	base := path[:len(path)-len(filepath.Ext(path))]
	for _, suffix := range []string{".f32", "-probabilities.json.gz", "-chunks.jsonl"} {
		info, err := os.Stat(base + suffix)
		if err != nil {
			return false, fmt.Errorf("incomplete existing result %s: %w", path, err)
		}
		if !info.Mode().IsRegular() || info.Size() == 0 {
			return false, fmt.Errorf("incomplete existing result %s: empty or invalid %s", path, suffix)
		}
	}
	return true, nil
}
