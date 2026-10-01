//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	mx "github.com/wbrown/tinyoai/internal/mlx"
)

// TestMLXKVProbe measures FP16 and Q8 cache accuracy, speed, and memory on the
// same weights. FP16 generates the reference history; Q8 follows that history
// to isolate cache quantization. Full logits and per-step timings are saved.
func TestMLXKVProbe(t *testing.T) {
	dir, root, out := os.Getenv("TINYOAI_MLX_DIR"), os.Getenv("TINYOAI_REFERENCE_ROOT"), os.Getenv("TINYOAI_KV_OUTPUT")
	if dir == "" || root == "" || out == "" {
		t.Skip("opt-in KV measurement")
	}
	if err := os.MkdirAll(out, 0755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"short", "long"} {
		t.Run(name, func(t *testing.T) {
			folder := "greedy-parity"
			if name == "long" {
				folder = "greedy-boundary"
			}
			var input struct {
				Prompt    []int `json:"prompt_ids"`
				Generated []int `json:"generated_ids"`
			}
			data, err := os.ReadFile(filepath.Join(root, folder, "reference.json"))
			if err != nil {
				t.Fatal(err)
			}
			if err = json.Unmarshal(data, &input); err != nil {
				t.Fatal(err)
			}
			m, err := LoadMLXNative(dir)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			n := m.native.(*nativeMLX)
			var baseline [][]float32
			var generated []int
			var reports []map[string]any
			for _, bits := range []int{16, 8} {
				if _, err = n.SetKVBits(bits); err != nil {
					t.Fatal(err)
				}
				run := func(prefix int, ids []int, logits bool) []float32 {
					t.Helper()
					row, err := n.Forward(context.Background(), prefix, ids, logits)
					if err != nil {
						t.Fatal(err)
					}
					return row
				}
				run(0, input.Prompt[:32], true)
				run(32, input.Prompt[32:33], true)
				if err = mx.Run(func() { n.reset(); mx.ResetMemory() }); err != nil {
					t.Fatal(err)
				}
				file := name + "-fp16"
				if bits == 8 {
					file = name + "-q8"
				}
				trace := newLogitTrace(t, filepath.Join(out, file+".safetensors"), n.config.VocabSize)
				var row []float32
				start := time.Now()
				var chunks []float64
				for pos := 0; pos < len(input.Prompt); {
					end := min(pos+512, len(input.Prompt))
					before := time.Now()
					row = run(pos, input.Prompt[pos:end], end == len(input.Prompt))
					chunks = append(chunks, time.Since(before).Seconds())
					pos = end
				}
				prefill := time.Since(start).Seconds()
				var decode, kls []float64
				matches, greedyPrefix := 0, 0
				prefixMatches := true
				var maxAbs float64
				var first []float32
				for step := range input.Generated {
					if step == 0 {
						first = append([]float32(nil), row...)
					}
					for _, v := range row {
						if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
							t.Fatal("non-finite logit")
						}
					}
					trace.append(t, row)
					pick := int(argmax(row))
					if bits == 16 {
						baseline = append(baseline, row)
						generated = append(generated, pick)
					} else {
						kl := distributionKL(baseline[step], row)
						if math.IsNaN(kl) || math.IsInf(kl, 0) {
							t.Fatal("non-finite KL")
						}
						kls = append(kls, kl)
						if pick == generated[step] {
							matches++
						} else {
							prefixMatches = false
						}
						if prefixMatches {
							greedyPrefix++
						}
						for i, v := range row {
							maxAbs = math.Max(maxAbs, math.Abs(float64(v-baseline[step][i])))
						}
					}
					if step+1 < len(input.Generated) {
						before := time.Now()
						row = run(len(input.Prompt)+step, generated[step:step+1], true)
						decode = append(decode, time.Since(before).Seconds())
					}
				}
				if err = trace.file.Sync(); err != nil {
					t.Fatal(err)
				}
				if n.offset != len(input.Prompt)+len(input.Generated)-1 {
					t.Fatal("wrong cache boundary")
				}
				var active, peak, cache, allocated uint64
				if err = mx.Run(func() {
					active, peak = mx.Memory()
					cache = mx.CacheMemory()
					for _, layer := range n.layers {
						allocated += layer.keys.Bytes() + layer.values.Bytes()
					}
				}); err != nil {
					t.Fatal(err)
				}
				// Rewind must keep packed KV aligned and overwrite the old suffix.
				reused := run(len(input.Prompt)-1, input.Prompt[len(input.Prompt)-1:], true)
				reuseKL := distributionKL(first, reused)
				if reuseKL > 0.0001 || math.IsNaN(reuseKL) {
					t.Fatalf("rewind KL too large: %.9f", reuseKL)
				}
				var seconds, mean, maxKL float64
				for _, d := range decode {
					seconds += d
				}
				for _, kl := range kls {
					mean += kl
					maxKL = math.Max(maxKL, kl)
				}
				if len(kls) > 0 {
					mean /= float64(len(kls))
				}
				r := map[string]any{"kv_bits": bits, "weight_bits": n.bits, "prompt_tokens": len(input.Prompt), "output_tokens": len(input.Generated), "final_cached_tokens": len(input.Prompt) + len(input.Generated) - 1,
					"prefill_seconds": prefill, "prefill_chunks": chunks, "decode_seconds": seconds, "decode_steps": decode, "decode_tps": float64(len(decode)) / seconds,
					"active_mlx_bytes": active, "peak_mlx_bytes": peak, "cache_mlx_bytes": cache, "kv_allocated_bytes": allocated, "reuse_kl": reuseKL}
				if bits == 8 {
					r["argmax_matches"] = matches
					r["greedy_prefix_matches"] = greedyPrefix
					r["mean_kl"] = mean
					r["max_kl"] = maxKL
					r["per_step_kl"] = kls
					r["max_logit_diff"] = maxAbs
				}
				reports = append(reports, r)
				encoded, _ := json.MarshalIndent(map[string]any{"prompt_ids": input.Prompt, "fp16_greedy_ids": generated, "runs": reports}, "", "  ")
				if err = os.WriteFile(filepath.Join(out, name+".json"), encoded, 0644); err != nil {
					t.Fatal(err)
				}
				t.Logf("%s KV%d: prefill %.2fs decode %.2f t/s KV %.3f GB peak %.3f GB matches %d/%d mean KL %.9f max KL %.9f", name, bits, prefill, float64(len(decode))/seconds, float64(allocated)/1e9, float64(peak)/1e9, matches, len(input.Generated), mean, maxKL)
			}
			if err = m.Close(); err != nil {
				t.Fatal(err)
			}
			if err = mx.Run(func() {
				active, _ := mx.Memory()
				if active != 0 {
					t.Errorf("retained %d bytes after close", active)
				}
			}); err != nil {
				t.Fatal(err)
			}
		})
	}
}
