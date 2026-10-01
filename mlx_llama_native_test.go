//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// tinyMLX opens the embedded model on Metal and registers cleanup when the
// caller explicitly enables native integration tests.
func tinyMLX(t *testing.T) *LlamaMLX {
	t.Helper()
	if os.Getenv("TINYOAI_TEST_METAL") != "1" {
		t.Skip("set TINYOAI_TEST_METAL=1 on a Metal-capable Mac")
	}
	m, err := DefaultMLX()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := m.Close(); err != nil {
			t.Error(err)
		}
	})
	return m
}

// requireLlamaFP32 skips parity checks unless TF32 was disabled at process
// startup, avoiding an arithmetic-mode mismatch with the CPU reference.
func requireLlamaFP32(t *testing.T) {
	t.Helper()
	if os.Getenv("MLX_ENABLE_TF32") != "0" {
		t.Skip("set MLX_ENABLE_TF32=0 at process startup for float32 parity checks")
	}
}

// TestLlamaMLXForwardParity compares every native vocabulary row with
// sequential CPU inference through batched prefill and greedy decoding,
// including the final context position. Both traces and divergence statistics
// are retained.
func TestLlamaMLXForwardParity(t *testing.T) {
	requireLlamaFP32(t)
	for _, tc := range []struct {
		name              string
		prompt, generated int
	}{{"short", 128, 256}, {"boundary", 480, 32}} {
		t.Run(tc.name, func(t *testing.T) {
			m := tinyMLX(t)
			cpu := m.model
			ids, err := cpu.bpeEncode(strings.Repeat("Once upon a time, a little girl lived in a house near the forest. ", 100))
			if err != nil {
				t.Fatal(err)
			}
			ids = append([]int{1}, ids...)[:tc.prompt]
			out := os.Getenv("TINYOAI_LLAMA_OUTPUT")
			if out == "" {
				out = t.TempDir()
			}
			if err := os.MkdirAll(out, 0755); err != nil {
				t.Fatal(err)
			}
			wantTrace := newLogitTrace(t, filepath.Join(out, tc.name+"-go.safetensors"), 512)
			gotTrace := newLogitTrace(t, filepath.Join(out, tc.name+"-mlx.safetensors"), 512)
			s := cpu.newRunState()
			matches, rows := 0, 0
			var maxKL, meanKL, maxError float64
			check := func(got []float32) {
				t.Helper()
				want := s.logits
				kl := distributionKL(want, got)
				maxKL = math.Max(maxKL, kl)
				meanKL += kl
				for i, v := range got {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatal("nonfinite logit")
					}
					maxError = math.Max(maxError, math.Abs(float64(v-want[i])))
				}
				if argmax(want) == argmax(got) {
					matches++
				}
				wantTrace.append(t, want)
				gotTrace.append(t, got)
				rows++
				if kl > 0.000001 || math.IsNaN(kl) || argmax(want) != argmax(got) {
					t.Fatalf("position %d: argmax %d / %d, KL %.12f, max error %.9f", rows-1, argmax(want), argmax(got), kl, maxError)
				}
			}
			for pos := 0; pos < len(ids); {
				end := min(pos+64, len(ids))
				got, err := m.native.Forward(context.Background(), pos, ids[pos:end], true)
				if err != nil {
					t.Fatal(err)
				}
				if len(got) != (end-pos)*512 {
					t.Fatal("wrong logits shape")
				}
				for i := pos; i < end; i++ {
					cpu.transformer(int32(ids[i]), int32(i), s)
					check(got[(i-pos)*512 : (i-pos+1)*512])
				}
				pos = end
			}
			generated := []int{}
			for i := 0; i < tc.generated; i++ {
				token := int(argmax(s.logits))
				generated = append(generated, token)
				pos := tc.prompt + i
				got, err := m.native.Forward(context.Background(), pos, []int{token}, false)
				if err != nil {
					t.Fatal(err)
				}
				cpu.transformer(int32(token), int32(pos), s)
				check(got)
			}
			report := map[string]any{"prompt_ids": ids, "generated_ids": generated, "rows": rows, "argmax_matches": matches, "mean_kl": meanKL / float64(rows), "max_kl": maxKL, "max_abs_error": maxError, "last_position": rows - 1}
			data, _ := json.MarshalIndent(report, "", "  ")
			if err := os.WriteFile(filepath.Join(out, tc.name+"-report.json"), data, 0644); err != nil {
				t.Fatal(err)
			}
			t.Logf("%d/%d argmax; mean KL %.12f, max KL %.12f, max error %.9f", matches, rows, meanKL/float64(rows), maxKL, maxError)
		})
	}
}

// TestLlamaMLXGenerate compares native and CPU text, streaming, stop handling,
// sampling, and context-exhaustion accounting.
func TestLlamaMLXGenerate(t *testing.T) {
	requireLlamaFP32(t)
	m := tinyMLX(t)
	for _, prompt := range []string{"Once upon a time", "Lily went to the forest.", "", " A little dog"} {
		for _, temperature := range []float64{0, 0.8} {
			opts := GenerateOptions{MaxTokens: 64, Temperature: temperature, Seed: 42}
			want, err := m.model.Generate(prompt, opts)
			if err != nil {
				t.Fatal(err)
			}
			var streamed strings.Builder
			opts.OnToken = func(s string) { streamed.WriteString(s) }
			got, err := m.Generate(prompt, opts)
			if err != nil {
				t.Fatal(err)
			}
			if got.Text != want.Text || got.PromptTokens != want.PromptTokens || got.CompletionTokens != want.CompletionTokens || got.FinishReason != want.FinishReason || streamed.String() != got.Text {
				t.Fatalf("prompt %q temperature %v: got %+v want %+v", prompt, temperature, got, want)
			}
		}
	}
	opts := GenerateOptions{MaxTokens: 128, Stop: []string{" the"}}
	want, _ := m.model.Generate("Once upon a time", opts)
	got, err := m.Generate("Once upon a time", opts)
	if err != nil || got.Text != want.Text || got.FinishReason != want.FinishReason {
		t.Fatalf("stop: %+v %v, want %+v", got, err, want)
	}
	for _, prompt := range []string{"The cat", strings.Repeat("The cat ", 100)} {
		opts := GenerateOptions{ContextLength: 32}
		want, err := m.model.Generate(prompt, opts)
		if err != nil {
			t.Fatal(err)
		}
		got, err := m.Generate(prompt, opts)
		if err != nil || got.Text != want.Text || got.CompletionTokens != want.CompletionTokens || got.FinishReason != want.FinishReason {
			t.Fatalf("context exhaustion: %+v %v, want %+v", got, err, want)
		}
	}
}

// TestLlamaMLXCacheAndProbabilities verifies prompt and completion byte spans,
// probability values against CPU forwards, edited-prefix reuse, context
// shrinkage, and resumption after cancellation.
func TestLlamaMLXCacheAndProbabilities(t *testing.T) {
	requireLlamaFP32(t)
	m := tinyMLX(t)
	prompt := strings.Repeat("Once upon a time, a little girl lived in a house. ", 12)
	opts := GenerateOptions{MaxTokens: 8, Logprobs: 8, PromptLogprobs: true}
	prepared, err := m.Prefill(prompt, opts)
	if err != nil {
		t.Fatal(err)
	}
	if prepared.CompletionTokens != 0 || prepared.Text != "" || prepared.FinishReason != "prefill" {
		t.Fatalf("prefill: %+v", prepared)
	}
	got, err := m.Generate(prompt, opts)
	if err != nil {
		t.Fatal(err)
	}
	if got.CachedPromptTokens == 0 || len(got.PromptLogprobs) != got.PromptTokens || len(got.Logprobs) != got.CompletionTokens {
		t.Fatalf("cache/logprob shape: %+v", got)
	}
	for _, r := range got.PromptLogprobs {
		if r.End > len(prompt) || prompt[r.Start:r.End] != r.Text || len(r.Top) != 8 || r.Logprob > 0 {
			t.Fatalf("prompt record: %+v", r)
		}
	}
	for _, r := range got.Logprobs {
		if r.End > len(got.Text) || got.Text[r.Start:r.End] != r.Text || len(r.Top) != 8 || r.Logprob > 0 {
			t.Fatalf("completion record: %+v", r)
		}
	}
	// Check prompt probabilities against independent sequential CPU forwards,
	// including reuse with a smaller requested alternative count.
	s := m.model.newRunState()
	previous := 1
	for i, r := range got.PromptLogprobs {
		m.model.transformer(int32(previous), int32(i), s)
		want := summarizeLogits(s.logits, r.ID, 8)
		if math.Abs(r.Logprob-want.Chosen) > 0.0001 {
			t.Fatalf("prompt probability %d: %v / %v", i, r.Logprob, want.Chosen)
		}
		for j, alt := range r.Top {
			if alt.ID != want.IDs[j] || math.Abs(alt.Logprob-want.Values[j]) > 0.0001 {
				t.Fatalf("prompt alternative %d/%d: %+v", i, j, alt)
			}
		}
		previous = r.ID
	}
	smaller := opts
	smaller.Logprobs = 3
	reused, err := m.Prefill(prompt, smaller)
	if err != nil || reused.CachedPromptTokens == 0 {
		t.Fatalf("probability reuse: %+v %v", reused, err)
	}
	for _, r := range reused.PromptLogprobs {
		if len(r.Top) != 3 {
			t.Fatalf("retained the wrong alternative count: %d", len(r.Top))
		}
	}
	// Edit across the cache's 256-position growth boundary and compare cold/warm.
	for _, p := range []string{prompt, prompt[:len(prompt)/2] + "A dog came home.", "The cat"} {
		warm, err := m.Generate(p, opts)
		if err != nil {
			t.Fatal(err)
		}
		cold := tinyMLX(t)
		want, err := cold.Generate(p, opts)
		if err != nil {
			t.Fatal(err)
		}
		if warm.Text != want.Text {
			t.Fatal("edited prefix changed greedy output")
		}
		for i, r := range warm.Logprobs {
			if math.Abs(r.Logprob-want.Logprobs[i].Logprob) > 0.0001 {
				t.Fatal("stale cached probabilities")
			}
		}
	}
	opts.ContextLength = 32
	opts.PromptLogprobs = false
	res, err := m.Generate("The cat", opts)
	if err != nil || res.CachedPromptTokens != 0 {
		t.Fatalf("context shrink: %+v %v", res, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	_, err = m.Prefill(prompt, GenerateOptions{Context: ctx, OnProgress: func(phase string, _ int) {
		if phase == "prefill" {
			cancel()
		}
	}})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
	res, err = m.Generate(prompt, GenerateOptions{MaxTokens: 4})
	if err != nil || res.CachedPromptTokens == 0 {
		t.Fatalf("resume: %+v %v", res, err)
	}
}

// TestLlamaMLXServerAndLifecycle exercises HTTP probability output, concurrent
// requests, queued cancellation, unsupported options, legacy loading, and
// idempotent closure.
func TestLlamaMLXServerAndLifecycle(t *testing.T) {
	m := tinyMLX(t)
	s := NewServer(m)
	for _, stream := range []bool{false, true} {
		body, _ := json.Marshal(map[string]any{"prompt": "Once upon a time", "max_tokens": 8, "temperature": 0, "stream": stream, "logprobs": 3, "prompt_logprobs": true})
		w := httptest.NewRecorder()
		s.ServeHTTP(w, httptest.NewRequest("POST", "/v1/completions", bytes.NewReader(body)))
		if w.Code != 200 || !strings.Contains(w.Body.String(), "top_logprobs") {
			t.Fatalf("HTTP: %d %s", w.Code, w.Body.String())
		}
		if stream && !strings.Contains(w.Body.String(), "[DONE]") {
			t.Fatal("incomplete SSE")
		}
	}
	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := m.Generate("The cat", GenerateOptions{MaxTokens: 2}); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	// A canceled caller can leave the queue without acquiring the model.
	<-m.gate
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := m.Generate("The cat", GenerateOptions{Context: ctx})
	m.gate <- struct{}{}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("queued cancellation: %v", err)
	}
	if _, err := m.Generate("The cat", GenerateOptions{KVBits: 8}); err == nil {
		t.Fatal("accepted unsupported Q8 tiny-head cache")
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Generate("The cat", GenerateOptions{MaxTokens: 2}); err == nil {
		t.Fatal("generation after close")
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadLlamaMLX(bytes.NewReader(defaultCheckpoint), bytes.NewReader(defaultTokenizer))
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()
	if loaded.Config() != m.Config() {
		t.Fatal("legacy loading changed dimensions")
	}
}
