//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestMLXPrefixReplacementResume replaces a long prefix, interrupts and
// resumes preparation, then compares cached logits and greedy continuation
// with fresh evaluation. It optionally persists reuse and timing measurements.
func TestMLXPrefixReplacementResume(t *testing.T) {
	dir := os.Getenv("TINYOAI_MLX_DIR")
	if dir == "" {
		t.Skip("set TINYOAI_MLX_DIR for Metal validation")
	}
	m, err := LoadMLXNative(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	// Replace a retained long prefix with a shorter input, then cancel and
	// resume evaluation to verify that completed chunks remain reusable.
	prefix := "A recorded observation.\n"
	old := prefix + strings.Repeat("The lamp was still burning in the window. ", 600)
	rolled := prefix + strings.Repeat("A different wind was blowing across the empty road. ", 480)
	opts := GenerateOptions{ContextLength: 8192, MaxTokens: 16, KVBits: 8, Logprobs: 12, PromptLogprobs: true}
	if _, err := m.Prefill(old, opts); err != nil {
		t.Fatal(err)
	}
	retained := len(m.tokenizer.encode(rolled))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	opts.Context = ctx
	opts.OnProgress = func(_ string, n int) {
		if n >= 1024 {
			cancel()
		}
	}
	started := time.Now()
	_, err = m.Prefill(rolled, opts)
	if !errors.Is(err, context.Canceled) {
		t.Fatal("did not cancel", err)
	}
	completed := len(m.tokens)
	if completed < 1024 {
		t.Fatal("lost completed KV", completed)
	}
	opts.Context, opts.OnProgress = context.Background(), nil
	prepared, err := m.Prefill(rolled, opts)
	if err != nil || prepared.CachedPromptTokens < completed-1 || prepared.CompletionTokens != 0 {
		t.Fatal(prepared, err)
	}
	resumeSeconds := time.Since(started).Seconds()
	ids := m.tokenizer.encode(rolled)
	logits := func() []float32 {
		t.Helper()
		z, e := m.native.Forward(context.Background(), len(ids)-1, ids[len(ids)-1:], true)
		if e != nil {
			t.Fatal(e)
		}
		return z
	}
	warmLogits := logits()
	warm, err := m.Generate(rolled, opts)
	if err != nil || warm.CachedPromptTokens < retained-2 {
		t.Fatal(warm, err)
	}
	m.tokens = nil
	if _, err = m.Prefill(rolled, opts); err != nil {
		t.Fatal(err)
	}
	coldLogits := logits()
	kl := distributionKL(coldLogits, warmLogits)
	if math.IsNaN(kl) || kl > 0.0001 || argmax(coldLogits) != argmax(warmLogits) {
		t.Fatal("resumed/cold mismatch", kl)
	}
	cold, err := m.Generate(rolled, opts)
	if err != nil || cold.Text != warm.Text {
		t.Fatal("resumption changed greedy continuation", cold.Text, warm.Text, err)
	}
	report := map[string]any{"old_prompt_tokens": len(m.tokenizer.encode(old)), "retained_prompt_tokens": retained, "headroom": 8192 - retained, "completed_at_cancel": completed, "reused_after_cancel": prepared.CachedPromptTokens, "generation_cached_tokens": warm.CachedPromptTokens, "resume_seconds": resumeSeconds, "cold_vs_resumed_kl": kl, "argmax_matches": true, "greedy_tokens": warm.CompletionTokens, "greedy_text_matches": true}
	data, _ := json.MarshalIndent(report, "", "  ")
	t.Log(string(data))
	if out := os.Getenv("TINYOAI_PREFIX_RESUME_OUTPUT"); out != "" {
		if err := os.MkdirAll(out, 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(out, "native-validation.json"), data, 0644); err != nil {
			t.Fatal(err)
		}
	}
}
