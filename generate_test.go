package tinyoai

import (
	"strings"
	"testing"
)

// TestGenerate exercises the full forward pass on the embedded model and proves
// the tokenizer encodes ordinary TinyStories-style prompts and that decoding
// produces real, deterministic completions.
func TestGenerate(t *testing.T) {
	m, err := Default()
	if err != nil {
		t.Fatalf("Default: %v", err)
	}

	res, err := m.Generate("Once upon a time there was a little", GenerateOptions{
		MaxTokens:   32,
		Temperature: 0.8,
		Seed:        1,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	t.Logf("completion: %q (finish=%s, prompt=%d, completion=%d)",
		res.Text, res.FinishReason, res.PromptTokens, res.CompletionTokens)

	if res.PromptTokens == 0 {
		t.Error("expected non-zero prompt tokens")
	}
	if res.CompletionTokens == 0 || strings.TrimSpace(res.Text) == "" {
		t.Error("expected a non-empty completion")
	}
	if res.CompletionTokens > 32 {
		t.Errorf("completion exceeded MaxTokens: %d", res.CompletionTokens)
	}

	// Same seed must reproduce the same text; the forward pass is deterministic.
	res2, err := m.Generate("Once upon a time there was a little", GenerateOptions{
		MaxTokens: 32, Temperature: 0.8, Seed: 1,
	})
	if err != nil {
		t.Fatalf("Generate (repeat): %v", err)
	}
	if res2.Text != res.Text {
		t.Errorf("same seed produced different text:\n  %q\n  %q", res.Text, res2.Text)
	}
}

// TestGenerateGreedyEndToEnd checks greedy decoding (temperature 0) and that
// MaxTokens bounds the completion, yielding a "length" finish.
func TestGenerateGreedy(t *testing.T) {
	m, err := Default()
	if err != nil {
		t.Fatalf("Default: %v", err)
	}
	res, err := m.Generate("The dog", GenerateOptions{MaxTokens: 8, Temperature: 0})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	t.Logf("greedy: %q (finish=%s)", res.Text, res.FinishReason)
	if res.CompletionTokens != 8 || res.FinishReason != "length" {
		t.Errorf("expected 8 tokens and finish=length, got %d / %s", res.CompletionTokens, res.FinishReason)
	}
}
