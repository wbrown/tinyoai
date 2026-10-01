package tinyoai

import (
	"context"
	"reflect"
	"strings"
	"testing"
)

// The fake keeps tokens as its KV state so a stale prefix after a reset is an
// error. It exercises full-length prompt preparation without loading weights.
type contextForwarder struct {
	tokens []int
	limit  int
	vocab  int
	resets int
	bits   int
}

// SetKVBits emulates cache invalidation when the fake backend changes
// precision.
func (f *contextForwarder) SetKVBits(bits int) (bool, error) {
	if f.bits == bits {
		return false, nil
	}
	f.bits, f.tokens = bits, nil
	return true, nil
}

// LimitCache records the selected window and clears fake KV when it exceeds
// that window.
func (f *contextForwarder) LimitCache(limit int) (bool, error) {
	f.limit = limit
	if len(f.tokens) > limit {
		f.tokens = nil
		f.resets++
		return true, nil
	}
	return false, nil
}

// Forward records tokens as fake KV and optionally returns deterministic
// logits. Slicing at prefix exposes attempts to reuse invalidated cache
// positions.
func (f *contextForwarder) Forward(_ context.Context, prefix int, ids []int, logits bool) ([]float32, error) {
	f.tokens = append(f.tokens[:prefix], ids...)
	if !logits {
		return nil, nil
	}
	row := make([]float32, f.vocab)
	row[10] = 1
	return row, nil
}

// Close satisfies the fake backend lifecycle without owning external
// resources.
func (f *contextForwarder) Close() error { return nil }

// TestMLXContextSelectionTruncatesAndResetsPrefix checks 4K/8K prompt
// budgeting, protected metadata, progress positions, and cache invalidation
// when the window shrinks.
func TestMLXContextSelectionTruncatesAndResetsPrefix(t *testing.T) {
	m, err := newMLX("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	m.config.MaxPositionEmbeddings = 8192
	f := &contextForwarder{vocab: m.config.VocabSize}
	m.native = f
	header := "[ Title: Test ]\n"
	prompt := header + strings.Repeat("The sea was quiet. ", 3000)
	for _, limit := range []int{8192, 4096, 8192} {
		last := 0
		opts := GenerateOptions{ContextLength: limit, MaxTokens: 32, TruncatePrompt: true, PromptPrefix: header,
			OnProgress: func(_ string, position int) { last = position }}
		result, err := m.Generate(prompt, opts)
		if err != nil {
			t.Fatal(err)
		}
		if result.PromptTokens != limit-31 || result.CompletionTokens != 32 || len(f.tokens) != limit || last != limit {
			t.Fatalf("limit %d: %+v, cache=%d, progress=%d", limit, result, len(f.tokens), last)
		}
		prefix := m.tokenizer.encode(header)
		if !reflect.DeepEqual(f.tokens[:len(prefix)], prefix) {
			t.Fatal("metadata or BOS lost")
		}
		full := m.tokenizer.encode(prompt)
		if !reflect.DeepEqual(f.tokens[len(prefix):result.PromptTokens], full[len(full)-(result.PromptTokens-len(prefix)):]) {
			t.Fatal("latest story tokens lost")
		}
		if limit == 4096 && (f.resets != 1 || result.CachedPromptTokens != 0) {
			t.Fatal("larger cache was reused after lowering context")
		}
	}
	for _, limit := range []int{-1, 8193} {
		if _, err := m.Generate("Hi", GenerateOptions{ContextLength: limit}); err == nil {
			t.Fatalf("accepted context %d", limit)
		}
	}
}

// TestCPUContextSelectionReleasesLargerCache checks that CPU cache capacity
// follows a smaller selected window instead of retaining its former
// allocation.
func TestCPUContextSelectionReleasesLargerCache(t *testing.T) {
	m, err := LoadStableLM("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	prompt := strings.Repeat("A quiet sea. ", 50)
	for _, limit := range []int{64, 32, 64} {
		got, err := m.Generate(prompt, GenerateOptions{ContextLength: limit, MaxTokens: 4, TruncatePrompt: true})
		if err != nil {
			t.Fatal(err)
		}
		if got.PromptTokens != limit-3 {
			t.Fatalf("limit %d: %+v", limit, got)
		}
		saved := <-m.cache
		capacity := len(saved.state.att) / m.config.NumAttentionHeads
		m.cache <- saved
		if capacity != limit {
			t.Fatalf("retained cache %d, want %d", capacity, limit)
		}
	}
}

// TestMLXKVPrecisionSwitchInvalidatesPrefix checks that a precision change
// clears reuse, unchanged precision retains reuse, and unsupported bit widths
// are rejected.
func TestMLXKVPrecisionSwitchInvalidatesPrefix(t *testing.T) {
	m, err := newMLX("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	f := &contextForwarder{vocab: m.config.VocabSize}
	m.native = f
	for _, bits := range []int{8, 16, 8} {
		result, err := m.Generate("A quiet sea.", GenerateOptions{KVBits: bits, MaxTokens: 2})
		if err != nil {
			t.Fatal(err)
		}
		if result.CachedPromptTokens != 0 {
			t.Fatal("precision switch reused stale KV")
		}
		result, err = m.Generate("A quiet sea.", GenerateOptions{KVBits: bits, MaxTokens: 2})
		if err != nil {
			t.Fatal(err)
		}
		if result.CachedPromptTokens == 0 {
			t.Fatal("unchanged precision lost prefix reuse")
		}
	}
	for _, bits := range []int{4, -1, 32} {
		if _, err = m.Generate("Hi", GenerateOptions{KVBits: bits, MaxTokens: 2}); err == nil {
			t.Fatalf("accepted KV bits %d", bits)
		}
	}
}
