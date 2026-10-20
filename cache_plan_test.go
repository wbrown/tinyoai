package tinyoai

import (
	"errors"
	"reflect"
	"slices"
	"testing"
)

// Exercise the public entry points with a token-only forwarder. No weights or
// neural inference are used: the assertions concern cache planning and dispatch.
func TestMLXCachePlanEntryPoints(t *testing.T) {
	m, err := newMLX("testdata/stablelm")
	if err != nil {
		t.Fatal(err)
	}
	f := &contextForwarder{vocab: m.config.VocabSize}
	m.native = f
	var plans []CachePlan
	opts := GenerateOptions{MaxTokens: 1, OnCachePlan: func(p CachePlan) error {
		plans = append(plans, p)
		return nil
	}}
	prompt := "The sea was quiet."
	first, err := m.Prefill(prompt, opts)
	if err != nil || len(plans) != 1 || plans[0].ReusedTokens != 0 || plans[0].PrefillTokens != first.PromptTokens {
		t.Fatal(plans, first, err)
	}
	next, err := m.Generate(prompt, opts)
	if err != nil || len(plans) != 2 || plans[1].ReusedTokens != next.CachedPromptTokens ||
		plans[1].PrefillTokens != next.PromptTokens-next.CachedPromptTokens {
		t.Fatal(plans, next, err)
	}
	opts.KVBits = 8
	_, err = m.Prefill(prompt, opts)
	if err != nil || len(plans) != 3 || plans[2].ReusedTokens != 0 || !slices.Contains(plans[2].Reasons, "kv-precision-changed") {
		t.Fatal(plans, err)
	}
	before := append([]int(nil), f.tokens...)
	stop := errors.New("caller declined inference")
	opts.OnCachePlan = func(CachePlan) error { return stop }
	if _, err = m.Generate("Another scene.", opts); !errors.Is(err, stop) || !reflect.DeepEqual(before, f.tokens) {
		t.Fatal("callback did not prevent forwards", err)
	}
}

func TestCachePlanMatchesProbabilityReuse(t *testing.T) {
	for _, width := range []int{0, 1, 12, 128} {
		for _, rowCount := range []int{0, 2, 4} {
			for _, prompt := range [][]int{{1, 2, 3, 4}, {1, 2}, {1, 9, 3, 4}, {1, 2, 3, 4, 5}} {
				resident := []int{1, 2, 3, 4}
				opts := GenerateOptions{Logprobs: 12, PromptLogprobs: true}
				p := cachePlan(resident, prompt, rowCount, width, opts)
				rows, count := make([]logprobRow, rowCount), width
				actual := prepareProbabilityCache(reusablePrefix(resident, prompt), &rows, &count, opts)
				if p.ReusedTokens != actual || p.PrefillTokens != len(prompt)-actual {
					t.Fatal(p, actual)
				}
			}
		}
	}
}

func TestLlamaMLXRejectsOversizedPromptBeforeDispatch(t *testing.T) {
	// Loading tokenizer/configuration is sufficient; no native backend is
	// attached, so an accidental dispatch would fail this test.
	model, err := Default()
	if err != nil {
		t.Fatal(err)
	}
	m := &LlamaMLX{model: model}
	if _, err = m.Prefill("A story.", GenerateOptions{ContextLength: 1}); err == nil {
		t.Fatal("silently shortened the prompt")
	}
	if _, err = m.Generate("", GenerateOptions{Logprobs: int(model.config.VocabSize) + 1}); err == nil {
		t.Fatal("accepted alternatives beyond the vocabulary")
	}
}
