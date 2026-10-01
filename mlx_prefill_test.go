package tinyoai

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
)

// TestMLXPrefillCancellationRetainsChunksAndNeverSamples checks that prefill
// emits no completion, keeps completed chunks on cancellation, and resumes
// into the same generation as a cold request.
func TestMLXPrefillCancellationRetainsChunksAndNeverSamples(t *testing.T) {
	m, err := newMLX("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	m.config.MaxPositionEmbeddings = 8192
	f := &contextForwarder{vocab: m.config.VocabSize}
	m.native = f
	prompt := strings.Repeat("A story by the sea.\n", 80)
	ctx, cancel := context.WithCancel(context.Background())
	opts := GenerateOptions{Context: ctx, MaxTokens: 8, ContextLength: 8192, OnToken: func(string) { t.Error("prefill sampled text") }, OnProgress: func(phase string, n int) {
		if phase != "prefill" {
			t.Error("prefill decoded")
		}
		if n >= 512 {
			cancel()
		}
	}}
	if _, err := m.Prefill(prompt, opts); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if len(m.tokens) != 512 || !reflect.DeepEqual(m.tokens, f.tokens) {
		t.Fatal("lost completed chunk", len(m.tokens), len(f.tokens))
	}
	opts.Context, opts.OnProgress = context.Background(), nil
	result, err := m.Prefill(prompt, opts)
	if err != nil || result.CachedPromptTokens != 512 || result.CompletionTokens != 0 || result.Text != "" || result.FinishReason != "prefill" {
		t.Fatal(result, err)
	}
	opts.OnToken = nil
	warm, err := m.Generate(prompt, opts)
	if err != nil || warm.CachedPromptTokens != result.PromptTokens-1 || warm.CompletionTokens != 8 {
		t.Fatal(warm, err)
	}
	m.tokens = nil
	cold, err := m.Generate(prompt, opts)
	if err != nil || cold.Text != warm.Text {
		t.Fatal("prefill changed generation", cold, warm, err)
	}
}
