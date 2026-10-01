//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	"encoding/json"
	"fmt"
	mx "github.com/wbrown/tinyoai/internal/mlx"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// TestMLXRecordedProbabilities checks next-token alignment, byte spans,
// full-vocabulary normalization, cached score reuse, and branch isolation.
// Capturing probabilities must not change the one-token sampling logits.
func TestMLXRecordedProbabilities(t *testing.T) {
	dir := os.Getenv("TINYOAI_MLX_DIR")
	if dir == "" {
		t.Skip("set TINYOAI_MLX_DIR")
	}
	m, err := LoadMLXNative(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	n := m.native.(*nativeMLX)
	if _, err = n.SetKVBits(8); err != nil {
		t.Fatal(err)
	}
	prompt := "[ Title: A test ]\nThe café was quiet. She opened the door."
	ids := m.tokenizer.encode(prompt)
	targets := append(append([]int{}, ids[1:]...), 0)
	logits, rows, err := n.ForwardWithLogprobs(context.Background(), 0, ids, targets, 12)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != len(ids) {
		t.Fatal(len(rows), len(ids))
	}
	// Teacher-forced single-position forwards independently verify both the
	// next-token alignment and normalized probabilities from the batched head.
	for i := range ids {
		reference, e := n.Forward(context.Background(), i, ids[i:i+1], true)
		if e != nil {
			t.Fatal(e)
		}
		want := summarizeLogits(reference, targets[i], 12)
		if math.Abs(want.Chosen-rows[i].Chosen) > 0.08 {
			t.Fatalf("position %d chosen %.6f vs %.6f", i, rows[i].Chosen, want.Chosen)
		}
		for j, id := range rows[i].IDs {
			p := summarizeLogits(reference, id, 1).Chosen
			if math.Abs(p-rows[i].Values[j]) > 0.08 {
				t.Fatalf("position %d top id %d: %.6f vs %.6f", i, id, rows[i].Values[j], p)
			}
		}
	}
	// Same prefill shape must preserve sampling logits exactly, even though
	// prompt scores use a batched vocabulary projection.
	plain, err := n.Forward(context.Background(), 0, ids, true)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(plain, logits) {
		t.Fatalf("capture changed sampling logits: KL %.12f", distributionKL(plain, logits))
	}
	opts := GenerateOptions{MaxTokens: 8, Temperature: 0, KVBits: 8, Logprobs: 12, PromptLogprobs: true}
	first, err := m.Generate(prompt, opts)
	if err != nil {
		t.Fatal(err)
	}
	again, err := m.Generate(prompt, opts)
	if err != nil {
		t.Fatal(err)
	}
	if first.Text != again.Text || again.CachedPromptTokens < len(ids)-2 {
		t.Fatal("cache reuse", first, again)
	}
	for i, r := range first.PromptLogprobs {
		if math.Abs(r.Logprob-again.PromptLogprobs[i].Logprob) > 0.08 {
			t.Fatal("reused prompt score", r, again.PromptLogprobs[i])
		}
	}
	for _, r := range first.PromptLogprobs {
		if prompt[r.Start:r.End] != r.Text {
			t.Fatal("prompt span", r)
		}
	}
	for _, r := range first.Logprobs {
		if r.Text != "" && first.Text[r.Start:r.End] != r.Text {
			t.Fatal("completion span", r)
		}
	}
	// Expansion uses only an existing prefix, and cannot contaminate subsequent generation.
	rec := again.PromptLogprobs[len(again.PromptLogprobs)-1]
	var candidate TokenProbability
	for _, p := range rec.Top {
		if p.Text != "" {
			candidate = p
			break
		}
	}
	tokensBefore, _ := json.Marshal(m.tokens)
	rowsBefore, _ := json.Marshal(m.probRows)
	offsetBefore := n.offset
	branches, e := m.OpenBranches(context.Background(), BranchRequest{Position: rec.Position, Prefix: rec.Prefix, Tokens: []int{candidate.ID}})
	if e != nil {
		t.Fatal(e)
	}
	defer branches.Close()
	if scores := branches.InitialLogprobs(); len(scores) != 1 || scores[0] != candidate.Logprob {
		t.Fatal("initial branch score", scores)
	}
	if _, e = branches.Forward(context.Background(), []int{candidate.ID}); e != nil {
		t.Fatal(e)
	}
	if e = branches.Close(); e != nil {
		t.Fatal(e)
	}
	tokensAfter, _ := json.Marshal(m.tokens)
	rowsAfter, _ := json.Marshal(m.probRows)
	if string(tokensBefore) != string(tokensAfter) || string(rowsBefore) != string(rowsAfter) || n.offset != offsetBefore {
		t.Fatal("scoring changed the resident document")
	}
	if _, e = m.OpenBranches(context.Background(), BranchRequest{Position: rec.Position, Prefix: "stale", Tokens: []int{candidate.ID}}); e == nil {
		t.Fatal("stale context accepted")
	}
	opts.Logprobs, opts.PromptLogprobs = 0, false
	without, err := m.Generate(prompt, opts)
	if err != nil {
		t.Fatal(err)
	}
	if without.Text != first.Text {
		t.Fatal("sampling changed", without.Text, first.Text)
	}
}

// TestMLXLogprobMeasurement saves complete probability records and paired
// timing and memory measurements with and without prompt-score capture at
// several context lengths.
func TestMLXLogprobMeasurement(t *testing.T) {
	out := os.Getenv("TINYOAI_LOGPROB_OUTPUT")
	dir := os.Getenv("TINYOAI_MLX_DIR")
	if out == "" || dir == "" {
		t.Skip("set model and output paths")
	}
	if err := os.MkdirAll(out, 0755); err != nil {
		t.Fatal(err)
	}
	m, err := LoadMLXNative(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	data, err := os.ReadFile("testdata/long-prompt.txt")
	if err != nil {
		t.Fatal(err)
	}
	prompt := strings.Repeat(string(data), 2)
	var reports []map[string]any
	for _, length := range []int{512, 4096, 8192} {
		for _, count := range []int{0, 12} {
			for run := 0; run < 2; run++ {
				m.tokens = nil
				m.probRows = nil
				if err := mx.Run(func() { m.native.(*nativeMLX).reset(); mx.ClearCache(); mx.ResetMemory() }); err != nil {
					t.Fatal(err)
				}
				start := time.Now()
				prefill := 0.0
				opts := GenerateOptions{ContextLength: length, MaxTokens: 16, KVBits: 8, Temperature: 0, TruncatePrompt: true, Logprobs: count, PromptLogprobs: count > 0, OnProgress: func(phase string, _ int) {
					if phase == "prefill" {
						prefill = time.Since(start).Seconds()
					}
				}}
				result, err := m.Generate(prompt, opts)
				if err != nil {
					t.Fatal(err)
				}
				elapsed := time.Since(start).Seconds()
				var active, peak uint64
				if err := mx.Run(func() { active, peak = mx.Memory() }); err != nil {
					t.Fatal(err)
				}
				record := map[string]any{"context": length, "top": count, "run": run, "prefill_seconds": prefill, "decode_tps": float64(result.CompletionTokens-1) / (elapsed - prefill), "active_mlx_bytes": active, "peak_mlx_bytes": peak, "result": result}
				name := fmt.Sprintf("context-%d-top%d-run%d.json", length, count, run)
				b, _ := json.Marshal(record)
				if err := os.WriteFile(filepath.Join(out, name), b, 0644); err != nil {
					t.Fatal(err)
				}
				delete(record, "result")
				reports = append(reports, record)
				t.Logf("context %d top %d run %d: prefill %.3fs decode %.2ft/s peak %d", length, count, run, prefill, record["decode_tps"], peak)
			}
		}
	}
	b, _ := json.MarshalIndent(reports, "", "  ")
	if err := os.WriteFile(filepath.Join(out, "timings.json"), b, 0644); err != nil {
		t.Fatal(err)
	}
}

// TestMLXProbabilityReduction checks native top-ID selection and normalized
// log probabilities against the Go full-vocabulary calculation.
func TestMLXProbabilityReduction(t *testing.T) {
	if os.Getenv("TINYOAI_MLX_DIR") == "" {
		t.Skip("requires Metal")
	}
	err := mx.Run(func() {
		c := mx.New()
		defer c.Close()
		a := &mx.Arena{Context: c}
		defer a.Free()
		z := a.Reshape(a.Cast(a.Tokens([]int{1000, 1002, 1001, 999, 1003, 1004, 1000, 1002, 1003, 1001}), mx.Float32), 1, 2, 5)
		indices := a.TopIndices(z, 2)
		probs := a.Contiguous(a.Subtract(a.TakeAlong(z, indices), a.LogSumExp(z)))
		ids := a.Contiguous(a.Cast(indices, mx.Float32))
		mx.Eval(probs, ids)
		ii, pp := ids.Floats(), probs.Floats()
		want := []map[int]bool{{4: true, 1: true}, {0: true, 3: true}}
		for row := 0; row < 2; row++ {
			logits := [][]float32{{1000, 1002, 1001, 999, 1003}, {1004, 1000, 1002, 1003, 1001}}[row]
			for j := 0; j < 2; j++ {
				k := row*2 + j
				id := int(ii[k])
				if !want[row][id] {
					t.Errorf("wrong top ID %d", id)
				}
				delete(want[row], id)
				if math.Abs(float64(pp[k])-summarizeLogits(logits, id, 2).Chosen) > 0.0001 {
					t.Errorf("wrong probability %.8f", pp[k])
				}
			}
		}
	})
	if err != nil {
		t.Fatal(err)
	}
}
