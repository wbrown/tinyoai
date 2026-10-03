//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	mx "github.com/wbrown/tinyoai/internal/mlx"
)

// TestMLXLoRAReference checks a supplied MLX-LM trace and saves matched native
// timings. Both implementations must use the same base, FP16 adapter, FP16 KV,
// 256-token prefill chunks, and one-token decoding. All vocabulary entries are
// compared, not just the sampled token. Input histories are held fixed.
func TestMLXLoRAReference(t *testing.T) {
	model, adapter, reference := os.Getenv("TINYOAI_MLX_DIR"), os.Getenv("TINYOAI_LORA_DIR"), os.Getenv("TINYOAI_LORA_REFERENCE")
	if model == "" || adapter == "" || reference == "" {
		t.Skip("set TINYOAI_MLX_DIR, TINYOAI_LORA_DIR, TINYOAI_LORA_REFERENCE")
	}
	data, err := os.ReadFile(filepath.Join(reference, "reference.json"))
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
	saved, err := os.ReadFile(filepath.Join(reference, "reference.f32"))
	if err != nil {
		t.Fatal(err)
	}
	m, err := LoadMLXNative(model)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	n := m.native.(*nativeMLX)
	if len(ref.Prompt) < 32 || len(ref.Generated) < 2 || len(saved) != len(ref.Generated)*n.config.VocabSize*4 {
		t.Fatal("invalid reference shape")
	}
	ctx := context.Background()
	run := func(prefix int, ids []int, want bool) []float32 {
		t.Helper()
		row, e := n.Forward(ctx, prefix, ids, want)
		if e != nil {
			t.Fatal(e)
		}
		return row
	}
	var reports []map[string]any
	for _, enabled := range []bool{false, true} {
		if enabled {
			if _, err = m.LoadLoRA(ctx, adapter, 1); err != nil {
				t.Fatal(err)
			}
		}
		// Warm the actual prefill/decode projection shapes before timing.
		warm := min(256, len(ref.Prompt))
		run(0, ref.Prompt[:warm], true)
		run(warm, ref.Generated[:1], true)
		if err = mx.Run(func() { n.reset(); mx.ResetMemory() }); err != nil {
			t.Fatal(err)
		}
		started := time.Now()
		var row []float32
		for pos := 0; pos < len(ref.Prompt); {
			end := min(pos+256, len(ref.Prompt))
			row = run(pos, ref.Prompt[pos:end], end == len(ref.Prompt))
			pos = end
		}
		prefill := time.Since(started).Seconds()
		var decode, meanKL, maxKL, maxAbs float64
		matches, exact := 0, true
		var steps []float64
		var trace *logitTrace
		if enabled {
			trace = newLogitTrace(t, filepath.Join(reference, "native.safetensors"), n.config.VocabSize)
		}
		for step, token := range ref.Generated {
			if enabled {
				trace.append(t, row)
				want := make([]float32, len(row))
				for i, v := range row {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatal("non-finite logits")
					}
					want[i] = math.Float32frombits(binary.LittleEndian.Uint32(saved[(step*len(row)+i)*4:]))
					delta := math.Abs(float64(v) - float64(want[i]))
					maxAbs = math.Max(maxAbs, delta)
					exact = exact && delta == 0
				}
				kl := distributionKL(want, row)
				meanKL += kl
				maxKL = math.Max(maxKL, kl)
				steps = append(steps, kl)
				if argmax(row) == argmax(want) {
					matches++
				}
			}
			if step+1 < len(ref.Generated) {
				started = time.Now()
				row = run(len(ref.Prompt)+step, []int{token}, true)
				decode += time.Since(started).Seconds()
			}
		}
		var active, peak uint64
		if err = mx.Run(func() { active, peak = mx.Memory() }); err != nil {
			t.Fatal(err)
		}
		report := map[string]any{"adapter": enabled, "prompt_tokens": len(ref.Prompt), "output_tokens": len(ref.Generated), "prefill_seconds": prefill, "prefill_tps": float64(len(ref.Prompt)) / prefill, "decode_seconds": decode, "decode_tps": float64(len(ref.Generated)-1) / decode, "active_bytes": active, "peak_bytes": peak, "adapter_info": n.loRAInfo(), "compared_to_reference": enabled}
		if enabled {
			report["argmax_matches"], report["bit_identical"] = matches, exact
			report["mean_kl"], report["max_kl"] = meanKL/float64(len(ref.Generated)), maxKL
			report["max_abs_error"], report["per_step_kl"] = maxAbs, steps
		}
		reports = append(reports, report)
		data, err := json.MarshalIndent(reports, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(reference, "native-report.json"), append(data, '\n'), 0600); err != nil {
			t.Fatal(err)
		}
		t.Logf("adapter=%v prefill %.2f t/s decode %.2f t/s", enabled, float64(len(ref.Prompt))/prefill, float64(len(ref.Generated)-1)/decode)
		if enabled {
			t.Logf("matches=%d/%d KL %.12f / %.12f exact=%v", matches, len(ref.Generated), meanKL/float64(len(ref.Generated)), maxKL, exact)
		}
		if enabled && (matches != len(ref.Generated) || maxKL > 0.000001 || math.IsNaN(meanKL)) {
			t.Fatal("native adapter differs from MLX-LM; see saved full logits")
		}
	}
	// The ordinary full-vocabulary head, captured probabilities, and batched
	// word branches must all see the same adapter. Compare identical prefixes.
	ids := ref.Prompt[:32]
	plain := run(0, ids, true)
	targets := append(append([]int{}, ids[1:]...), ref.Generated[0])
	with, _, err := n.ForwardWithLogprobs(ctx, 0, ids, targets, 8)
	if err != nil || !reflect.DeepEqual(plain, with) {
		t.Fatal("probability capture changed adapter logits", err)
	}
	for _, bits := range []int{16, 8} {
		if _, err = n.SetKVBits(bits); err != nil {
			t.Fatal(err)
		}
		run(0, ids, true)
		branches, e := n.NewBranches(len(ids), 2)
		if e != nil {
			t.Fatal(e)
		}
		seeds := ref.Generated[:2]
		rows, e := branches.Forward(ctx, seeds)
		if e != nil {
			t.Fatal(e)
		}
		if e = branches.Close(); e != nil {
			t.Fatal(e)
		}
		for i, token := range seeds {
			same := run(len(ids), []int{token}, true)
			if kl := distributionKL(same, rows[i]); math.IsNaN(kl) || kl > 0.0001 {
				t.Fatalf("KV%d branch%d KL %.12f", bits, i, kl)
			}
		}
	}
}
