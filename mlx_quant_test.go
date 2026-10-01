//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	mx "github.com/wbrown/tinyoai/internal/mlx"
)

// TestMLXQuantProbe collects an explicitly requested quantization assessment.
// Accuracy is computed separately from these full-vocabulary saved traces;
// lossy quantizations are measurements, not expected to match CPU argmax.
func TestMLXQuantProbe(t *testing.T) {
	model, reference, output := os.Getenv("TINYOAI_MLX_DIR"), os.Getenv("TINYOAI_QUANT_REFERENCE"), os.Getenv("TINYOAI_QUANT_OUTPUT")
	if model == "" || reference == "" || output == "" {
		t.Skip("opt-in quantization measurement")
	}
	var ref struct {
		Prompt    []int `json:"prompt_ids"`
		Generated []int `json:"generated_ids"`
	}
	data, err := os.ReadFile(reference)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	if len(ref.Prompt) < 32 || len(ref.Generated) < 2 {
		t.Fatal("invalid reference")
	}
	if err = os.MkdirAll(output, 0755); err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	m, err := LoadMLXNative(model)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	load := time.Since(started).Seconds()
	n := m.native.(*nativeMLX)
	run := func(prefix int, ids []int, logits bool) []float32 {
		t.Helper()
		row, e := n.Forward(context.Background(), prefix, ids, logits)
		if e != nil {
			t.Fatal(e)
		}
		return row
	}
	// Warm both prefill and one-token decode before any recorded trial.
	run(0, ref.Prompt[:32], true)
	run(32, ref.Prompt[32:33], true)
	var trials []map[string]any
	for trial := 0; trial < 3; trial++ {
		if err = mx.Run(func() { n.reset(); mx.ResetMemory() }); err != nil {
			t.Fatal(err)
		}
		runtime.GC()
		trace := newLogitTrace(t, filepath.Join(output, []string{"trial-1.safetensors", "trial-2.safetensors", "trial-3.safetensors"}[trial]), 65536)
		var row []float32
		started = time.Now()
		for pos := 0; pos < len(ref.Prompt); {
			end := min(pos+512, len(ref.Prompt))
			row = run(pos, ref.Prompt[pos:end], end == len(ref.Prompt))
			pos = end
		}
		prefill := time.Since(started).Seconds()
		var decode []float64
		var picks []int
		var active, peak, cache, maxTotal, kv uint64
		memory := func() {
			if err := mx.Run(func() { active, peak = mx.Memory(); cache = mx.CacheMemory() }); err != nil {
				t.Fatal(err)
			}
			maxTotal = max(maxTotal, active+cache)
		}
		memory()
		for step, token := range ref.Generated {
			for _, v := range row {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatal("non-finite logits")
				}
			}
			trace.append(t, row)
			picks = append(picks, int(argmax(row)))
			if step+1 < len(ref.Generated) {
				started = time.Now()
				row = run(len(ref.Prompt)+step, []int{token}, true)
				decode = append(decode, time.Since(started).Seconds())
				memory()
			}
		}
		if err = trace.file.Sync(); err != nil {
			t.Fatal(err)
		}
		if n.offset != len(ref.Prompt)+len(ref.Generated)-1 {
			t.Fatal("wrong cache boundary")
		}
		if err = mx.Run(func() {
			for _, l := range n.layers {
				for _, a := range []mlxKV{l.keys, l.values} {
					kv += a.Bytes()
				}
			}
		}); err != nil {
			t.Fatal(err)
		}
		seconds := 0.0
		for _, d := range decode {
			seconds += d
		}
		rss, _ := peakRSS()
		trials = append(trials, map[string]any{"prefill_seconds": prefill, "prefill_tps": float64(len(ref.Prompt)) / prefill, "decode_tps": float64(len(decode)) / seconds, "decode_seconds": seconds, "decode_steps": decode, "argmax": picks, "mlx_active_bytes": active, "mlx_peak_bytes": peak, "mlx_cache_bytes": cache, "max_observed_active_plus_cache_bytes": maxTotal, "kv_allocated_bytes": kv, "process_peak_rss_bytes": rss})
		report := map[string]any{"bits": n.bits, "group_size": n.group, "engine": "Go/native MLX", "prompt_tokens": len(ref.Prompt), "output_tokens": len(ref.Generated), "batch": 512, "cache_dtype": "float16", "load_seconds": load, "trials": trials}
		data, _ := json.MarshalIndent(report, "", "  ")
		if err = os.WriteFile(filepath.Join(output, "report.json"), data, 0644); err != nil {
			t.Fatal(err)
		}
		t.Logf("bits=%d trial=%d prefill=%.3fs decode=%.2f t/s peak=%.3f GB", n.bits, trial+1, prefill, float64(len(decode))/seconds, float64(peak)/1e9)
	}
}
