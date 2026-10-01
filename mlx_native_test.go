//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	mx "github.com/wbrown/tinyoai/internal/mlx"
)

// TestMLXNativeParity is an opt-in comparison with saved MLX traces on short
// and boundary prompts. It saves full logits, checks rewinding across cache
// growth, and optionally compares a free-running greedy trajectory with the
// Python worker.
func TestMLXNativeParity(t *testing.T) {
	dir, root := os.Getenv("TINYOAI_MLX_DIR"), os.Getenv("TINYOAI_REFERENCE_ROOT")
	if dir == "" || root == "" {
		t.Skip("set TINYOAI_MLX_DIR and TINYOAI_REFERENCE_ROOT")
	}
	out := os.Getenv("TINYOAI_MLX_OUTPUT")
	if out == "" {
		out = t.TempDir()
	}
	if err := os.MkdirAll(out, 0755); err != nil {
		t.Fatal(err)
	}
	for _, which := range []string{"short", "long"} {
		t.Run(which, func(t *testing.T) {
			refDir := "greedy-parity"
			if which == "long" {
				refDir = "greedy-boundary"
			}
			data, err := os.ReadFile(filepath.Join(root, refDir, "reference.json"))
			if err != nil {
				t.Fatal(err)
			}
			var ref struct {
				Prompt    []int `json:"prompt_ids"`
				Generated []int `json:"generated_ids"`
			}
			if err = json.Unmarshal(data, &ref); err != nil {
				t.Fatal(err)
			}
			saved, err := os.ReadFile(filepath.Join(root, "mlx-q8-"+which+"-native-attention.f32"))
			if err != nil {
				t.Fatal(err)
			}
			if len(saved) != len(ref.Generated)*65536*4 {
				t.Fatal("saved trace shape mismatch")
			}
			m, err := LoadMLXNative(dir)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			n := m.native.(*nativeMLX)
			run := func(prefix int, ids []int, logits bool) []float32 {
				t.Helper()
				r, e := n.Forward(context.Background(), prefix, ids, logits)
				if e != nil {
					t.Fatal(e)
				}
				return r
			}
			trace := newLogitTrace(t, filepath.Join(out, which+"-logits.safetensors"), 65536)
			start := time.Now()
			var row []float32
			for pos := 0; pos < len(ref.Prompt); {
				end := min(pos+512, len(ref.Prompt))
				row = run(pos, ref.Prompt[pos:end], end == len(ref.Prompt))
				pos = end
			}
			prefill := time.Since(start).Seconds()
			var meanKL, maxKL, maxAbs, decode float64
			matches := 0
			exact := true
			perStep := make([]float64, len(ref.Generated))
			for step, token := range ref.Generated {
				trace.append(t, row)
				want := make([]float32, 65536)
				for i := range want {
					want[i] = math.Float32frombits(binary.LittleEndian.Uint32(saved[(step*65536+i)*4:]))
					diff := math.Abs(float64(row[i] - want[i]))
					maxAbs = math.Max(maxAbs, diff)
					if diff != 0 {
						exact = false
					}
				}
				kl := distributionKL(want, row)
				perStep[step] = kl
				meanKL += kl
				maxKL = math.Max(maxKL, kl)
				if argmax(want) == argmax(row) {
					matches++
				}
				if math.IsNaN(kl) || kl > 0.0000001 || argmax(want) != argmax(row) {
					t.Fatalf("step %d: KL %.12f, argmax %d vs %d, max error %.9f", step, kl, argmax(row), argmax(want), maxAbs)
				}
				if step+1 < len(ref.Generated) {
					start = time.Now()
					row = run(len(ref.Prompt)+step, []int{token}, true)
					decode += time.Since(start).Seconds()
				}
			}
			report := map[string]any{"prompt_tokens": len(ref.Prompt), "output_tokens": len(ref.Generated), "argmax_matches": matches, "bit_identical": exact, "mean_kl": meanKL / float64(len(ref.Generated)), "max_kl": maxKL, "max_abs_error": maxAbs, "per_step_kl": perStep, "prefill_seconds": prefill, "decode_tps": float64(len(ref.Generated)-1) / decode}
			b, _ := json.MarshalIndent(report, "", "  ")
			if err = os.WriteFile(filepath.Join(out, which+"-report.json"), b, 0644); err != nil {
				t.Fatal(err)
			}
			t.Logf("%s: %d/%d matches, bit-identical %v, KL %.12f / %.12f; prefill %.3fs, decode %.2f tokens/s", which, matches, len(ref.Generated), exact, meanKL/float64(len(ref.Generated)), maxKL, prefill, float64(len(ref.Generated)-1)/decode)
			// Rewind across an allocation boundary and compare against a fresh
			// prefix, exposing stale KV reads after generating and editing a story.
			reused := run(127, ref.Prompt[127:257], true)
			_ = mx.Run(func() { n.reset() })
			fresh := run(0, ref.Prompt[:257], true)
			if kl := distributionKL(fresh, reused); kl > 0.0000001 || argmax(fresh) != argmax(reused) {
				t.Fatalf("rewind KL %.12f", kl)
			}
			if python := os.Getenv("TINYOAI_MLX_PYTHON"); python != "" {
				_ = mx.Run(func() { n.reset() })
				p, e := LoadMLX(python, dir)
				if e != nil {
					t.Fatal(e)
				}
				defer p.Close()
				nativeTrace := newLogitTrace(t, filepath.Join(out, which+"-greedy-native.safetensors"), 65536)
				pythonTrace := newLogitTrace(t, filepath.Join(out, which+"-greedy-python.safetensors"), 65536)
				var pr []float32
				for pos := 0; pos < len(ref.Prompt); {
					end := min(pos+512, len(ref.Prompt))
					last := end == len(ref.Prompt)
					row = run(pos, ref.Prompt[pos:end], last)
					pr, e = p.exchange(context.Background(), &mlxRequest{pos, ref.Prompt[pos:end], last})
					if e != nil {
						t.Fatal(e)
					}
					pos = end
				}
				ids := make([]int, 0, len(ref.Generated))
				kls := make([]float64, 0, len(ref.Generated))
				for step := range ref.Generated {
					nativeTrace.append(t, row)
					pythonTrace.append(t, pr)
					kl := distributionKL(pr, row)
					kls = append(kls, kl)
					token := int(argmax(row))
					ids = append(ids, token)
					if math.IsNaN(kl) || kl > 0.0000001 || argmax(pr) != argmax(row) {
						t.Fatalf("free-running step %d KL %.12f", step, kl)
					}
					if step+1 < len(ref.Generated) {
						pos := len(ref.Prompt) + step
						row = run(pos, []int{token}, true)
						pr, e = p.exchange(context.Background(), &mlxRequest{pos, []int{token}, true})
						if e != nil {
							t.Fatal(e)
						}
					}
				}
				b, _ := json.MarshalIndent(map[string]any{"generated_ids": ids, "per_step_kl": kls, "argmax_matches": len(ids)}, "", "  ")
				if e = os.WriteFile(filepath.Join(out, which+"-greedy.json"), b, 0644); e != nil {
					t.Fatal(e)
				}
			}
		})
	}
}

// TestMLXNativeErrorRecovery verifies that a C API failure becomes a Go error
// and leaves the process usable for a later native operation.
func TestMLXNativeErrorRecovery(t *testing.T) {
	if os.Getenv("TINYOAI_MLX_DIR") == "" {
		t.Skip("requires Metal")
	}
	var c *mx.Context
	if err := mx.Run(func() { c = mx.New() }); err != nil {
		t.Fatal(err)
	}
	defer mx.Run(func() { c.Close() })
	err := mx.Run(func() { a := &mx.Arena{Context: c}; defer a.Free(); a.Reshape(a.Zeros(mx.Float16, 2, 3), 7) })
	if err == nil {
		t.Fatal("expected recoverable native error")
	}
	if err = mx.Run(func() { a := &mx.Arena{Context: c}; defer a.Free(); mx.Eval(a.Zeros(mx.Float16, 2, 3)) }); err != nil {
		t.Fatal(err)
	}
}

// TestMLXNativeLifecycle checks prefix reuse, queued and streaming
// cancellation, recovery, closed-model rejection, and release of native
// allocations.
func TestMLXNativeLifecycle(t *testing.T) {
	dir := os.Getenv("TINYOAI_MLX_DIR")
	if dir == "" {
		t.Skip("set TINYOAI_MLX_DIR")
	}
	var baseline, after uint64
	if err := mx.Run(func() { baseline, _ = mx.Memory() }); err != nil {
		t.Fatal(err)
	}
	m, err := LoadMLXNative(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompt := "[ Author: Emilie J.Author; Title: The Title; Tags: steamy; Genre: romance ]\nThe candle burned."
	opts := GenerateOptions{MaxTokens: 5, Temperature: 0, Seed: 42}
	a, err := m.Generate(prompt, opts)
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.Generate(prompt, opts)
	if err != nil {
		t.Fatal(err)
	}
	if a.Text != b.Text || b.CachedPromptTokens != b.PromptTokens-1 {
		t.Fatalf("prefix reuse: %#v vs %#v", a, b)
	}
	// Cancel a queued request without taking the model's gate.
	<-m.gate
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	canceled := opts
	canceled.Context = ctx
	_, err = m.Generate(prompt, canceled)
	m.gate <- struct{}{}
	if !errors.Is(err, context.Canceled) {
		t.Fatal("queued cancellation", err)
	}
	// Cancel during streaming; subsequent requests must still reproduce output.
	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()
	canceled.Context = ctx
	canceled.OnToken = func(string) { cancel() }
	_, err = m.Generate(prompt, canceled)
	if !errors.Is(err, context.Canceled) {
		t.Fatal("stream cancellation", err)
	}
	b, err = m.Generate(prompt, opts)
	if err != nil || a.Text != b.Text {
		t.Fatal("recovery", b, err)
	}
	if err = m.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = m.Generate(prompt, opts); err == nil {
		t.Fatal("generation after close succeeded")
	}
	if err = mx.Run(func() { after, _ = mx.Memory() }); err != nil {
		t.Fatal(err)
	}
	if after > baseline+16<<20 {
		t.Fatalf("native allocations leaked: before %d, after %d", baseline, after)
	}
	t.Logf("MLX active bytes before load %d, after close %d", baseline, after)
}

// TestMLXNativeSamplingParity compares fixed-seed native and Python-worker
// completions with the same Go sampler on cold and reused prefixes.
func TestMLXNativeSamplingParity(t *testing.T) {
	dir, python := os.Getenv("TINYOAI_MLX_DIR"), os.Getenv("TINYOAI_MLX_PYTHON")
	if dir == "" || python == "" {
		t.Skip("requires native and Python MLX")
	}
	n, err := LoadMLXNative(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer n.Close()
	p, err := LoadMLX(python, dir)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	header := "[ Author: Emilie J.Author; Title: The Title; Tags: steamy; Genre: romance ]\n"
	options := GenerateOptions{MaxTokens: 75, Temperature: 1, Seed: 42, PromptPrefix: header, TruncatePrompt: true,
		Sampling: &SamplingOptions{TopK: 25, TopP: 1, TFS: .925, RepetitionPenalty: 1.9, RepetitionRange: 768, RepetitionSlope: 1, PresencePenalty: .001, FrequencyPenalty: .0025}}
	for i := 0; i < 2; i++ {
		a, err := n.Generate(header+"The candle burned.", options)
		if err != nil {
			t.Fatal(err)
		}
		b, err := p.Generate(header+"The candle burned.", options)
		if err != nil {
			t.Fatal(err)
		}
		if a.Text != b.Text {
			t.Fatalf("run %d, caches %d/%d, native %q; Python %q", i, a.CachedPromptTokens, b.CachedPromptTokens, a.Text, b.Text)
		}
		t.Logf("run %d: fixed-seed text identical, cached %d/%d", i, a.CachedPromptTokens, b.CachedPromptTokens)
	}
}
