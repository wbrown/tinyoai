package tinyoai

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

type apiGenerator struct{ options GenerateOptions }

// Generate records options and emits a deterministic completion with optional
// probabilities for HTTP adapter tests.
func (g *apiGenerator) Generate(_ string, opts GenerateOptions) (GenerateResult, error) {
	g.options = opts
	rows := []TokenLogprob{{TokenProbability: TokenProbability{ID: 1, Text: "hello", Logprob: -1}, Top: []TokenProbability{{ID: 2, Text: "hi", Logprob: -2}}}}
	if opts.Logprobs > 0 && opts.OnLogprobs != nil {
		opts.OnLogprobs(ProbabilityEvent{Phase: "completion", Tokens: rows})
	}
	if opts.OnToken != nil {
		opts.OnToken("hello")
	}
	result := GenerateResult{Text: "hello", CompletionTokens: 1, FinishReason: "length"}
	if opts.Logprobs > 0 {
		result.Logprobs = rows
	}
	return result, nil
}

// TestCoreCompletionAndStandardProbabilities verifies standard text-completion
// fields and streaming logprobs while keeping application extensions and
// tokenizer routes disabled by default.
func TestCoreCompletionAndStandardProbabilities(t *testing.T) {
	for _, stream := range []bool{false, true} {
		model := &apiGenerator{}
		raw, _ := json.Marshal(map[string]any{"prompt": "hello", "stream": stream, "logprobs": 3, "top_p": .8, "presence_penalty": .3, "top_k": 25, "prompt_logprobs": true, "context_length": 8192})
		w := httptest.NewRecorder()
		NewServer(model).ServeHTTP(w, httptest.NewRequest("POST", "/v1/completions", strings.NewReader(string(raw))))
		if w.Code != 200 || !strings.Contains(w.Body.String(), `"top_logprobs":[{"hi":-2}]`) {
			t.Fatal(w.Code, w.Body.String())
		}
		if strings.Contains(w.Body.String(), "token_probabilities") || model.options.PromptLogprobs || model.options.ContextLength != 0 || model.options.Sampling.TopK != 0 {
			t.Fatal("application extensions enabled implicitly")
		}
		if model.options.Sampling.TopP != .8 || model.options.Sampling.PresencePenalty != .3 {
			t.Fatal("standard sampling fields lost")
		}
		if stream && !strings.HasSuffix(w.Body.String(), "data: [DONE]\n\n") {
			t.Fatal("unfinished stream")
		}
	}
	w := httptest.NewRecorder()
	NewServer(&apiGenerator{}).ServeHTTP(w, httptest.NewRequest("GET", "/v1/tokenizer", nil))
	if w.Code != 404 {
		t.Fatal("tokenizer route enabled implicitly")
	}
}

// TestCoreCompletionValidation rejects malformed JSON, unsupported prompt
// shapes, invalid limits, and invalid sampling values before generation.
func TestCoreCompletionValidation(t *testing.T) {
	for _, body := range []string{`{"prompt":[3]}`, `{"max_tokens":0}`, `{"stop":3}`, `{} {}`, `{"logprobs":65}`, `{"top_p":0}`} {
		w := httptest.NewRecorder()
		NewServer(&apiGenerator{}).ServeHTTP(w, httptest.NewRequest("POST", "/v1/completions", strings.NewReader(body)))
		if w.Code != 400 {
			t.Fatal(body, w.Code)
		}
	}
}
