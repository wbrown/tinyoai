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

// TestMLXMatchedProbe runs one isolated, opt-in measurement. The driver rotates
// model order between rounds. Full logits and timings are saved after inference.
func TestMLXMatchedProbe(t *testing.T) {
	model, reference, output := os.Getenv("TINYOAI_MLX_DIR"), os.Getenv("TINYOAI_MATCHED_REFERENCE"), os.Getenv("TINYOAI_MATCHED_OUTPUT")
	if model == "" || reference == "" || output == "" {
		t.Skip("opt-in matched MLX measurement")
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
	if len(ref.Prompt) < 32 || len(ref.Generated) < 8 {
		t.Fatal("invalid history")
	}
	if err = os.MkdirAll(output, 0755); err != nil {
		t.Fatal(err)
	}
	memory := func() map[string]uint64 {
		var active, peak, cache uint64
		if err := mx.Run(func() { active, peak = mx.Memory(); cache = mx.CacheMemory() }); err != nil {
			t.Fatal(err)
		}
		return map[string]uint64{"active_bytes": active, "peak_bytes": peak, "cache_bytes": cache}
	}
	started := time.Now()
	m, err := LoadMLXNative(model)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	load := time.Since(started).Seconds()
	loaded := memory()
	n := m.native.(*nativeMLX)
	run := func(prefix int, ids []int, logits bool) []float32 {
		t.Helper()
		row, e := n.Forward(context.Background(), prefix, ids, logits)
		if e != nil {
			t.Fatal(e)
		}
		return row
	}
	prefill := func(record bool) ([]float32, []float64, []map[string]uint64) {
		var row []float32
		var steps []float64
		var samples []map[string]uint64
		for pos := 0; pos < len(ref.Prompt); {
			end := min(pos+512, len(ref.Prompt))
			start := time.Now()
			row = run(pos, ref.Prompt[pos:end], end == len(ref.Prompt))
			elapsed := time.Since(start).Seconds()
			if record {
				steps = append(steps, elapsed)
				samples = append(samples, memory())
			}
			pos = end
		}
		return row, steps, samples
	}
	// Warm the actual prompt shape and decode kernels. Retain no prompt cache
	// across the warmup boundary; all measured prefill starts from an empty KV.
	started = time.Now()
	prefill(false)
	warmPrefill := time.Since(started).Seconds()
	for i := 0; i < 8; i++ {
		run(len(ref.Prompt)+i, ref.Generated[i:i+1], true)
	}
	warm := time.Since(started).Seconds()
	if err = mx.Run(func() { n.reset(); mx.ResetMemory() }); err != nil {
		t.Fatal(err)
	}
	runtime.GC()
	started = time.Now()
	row, chunks, samples := prefill(true)
	prefillWall := time.Since(started).Seconds()
	rows := make([][]float32, 0, len(ref.Generated))
	rows = append(rows, row)
	decode := make([]float64, 0, len(ref.Generated)-1)
	started = time.Now()
	for step := 0; step+1 < len(ref.Generated); step++ {
		start := time.Now()
		row = run(len(ref.Prompt)+step, ref.Generated[step:step+1], true)
		decode = append(decode, time.Since(start).Seconds())
		rows = append(rows, row)
		samples = append(samples, memory())
	}
	decodeWall := time.Since(started).Seconds()
	inferenceMemory := memory()
	if n.offset != len(ref.Prompt)+len(ref.Generated)-1 {
		t.Fatal("wrong cache boundary")
	}
	var kv uint64
	if err = mx.Run(func() {
		for _, l := range n.layers {
			for _, a := range []mlxKV{l.keys, l.values} {
				kv += a.Bytes()
			}
		}
	}); err != nil {
		t.Fatal(err)
	}
	var heap runtime.MemStats
	runtime.ReadMemStats(&heap)
	rss, rssOK := peakRSS()
	// Repeating the same prompt reuses its prefix and recomputes its final
	// token. Save the numerical drift relative to a fresh prefill.
	started = time.Now()
	reused := run(len(ref.Prompt)-1, ref.Prompt[len(ref.Prompt)-1:], true)
	reuseSeconds := time.Since(started).Seconds()
	reuseKL := distributionKL(rows[0], reused)
	reuseMatch := argmax(rows[0]) == argmax(reused)
	if err = m.Close(); err != nil {
		t.Fatal(err)
	}
	closed := memory()
	if closed["active_bytes"] != 0 {
		t.Fatalf("MLX allocations after close: %d", closed["active_bytes"])
	}
	trace := newLogitTrace(t, filepath.Join(output, "logits.safetensors"), n.config.VocabSize)
	picks := make([]int, 0, len(rows))
	for _, r := range rows {
		if len(r) != n.config.VocabSize {
			t.Fatal("wrong logits shape")
		}
		for _, v := range r {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatal("non-finite logits")
			}
		}
		trace.append(t, r)
		picks = append(picks, int(argmax(r)))
	}
	if err = trace.file.Sync(); err != nil {
		t.Fatal(err)
	}
	reusedTrace := newLogitTrace(t, filepath.Join(output, "cached-logits.safetensors"), n.config.VocabSize)
	reusedTrace.append(t, reused)
	if err = reusedTrace.file.Sync(); err != nil {
		t.Fatal(err)
	}
	sum := func(xs []float64) float64 {
		var s float64
		for _, x := range xs {
			s += x
		}
		return s
	}
	pp, dt := sum(chunks), sum(decode)
	report := map[string]any{
		"engine": "Go/native MLX", "bits": n.bits, "group_size": n.group, "model": model,
		"prompt_tokens": len(ref.Prompt), "output_tokens": len(ref.Generated), "decode_forward_tokens": len(decode),
		"batch": 512, "kv_dtype": "float16", "allocator_cache_limit_bytes": 256 << 20,
		"load_seconds": load, "loaded_memory": loaded, "warmup_seconds": warm, "first_request_prefill_seconds": warmPrefill,
		"prefill_seconds": pp, "prefill_tps": float64(len(ref.Prompt)) / pp, "prefill_wall_seconds": prefillWall, "prefill_chunks_seconds": chunks,
		"decode_seconds": dt, "decode_tps": float64(len(decode)) / dt, "decode_wall_seconds": decodeWall, "decode_steps_seconds": decode,
		"memory_samples": samples, "inference_memory": inferenceMemory, "kv_allocated_bytes": kv, "go_heap_allocated_bytes": heap.HeapAlloc,
		"process_peak_rss_bytes": rss, "process_peak_rss_available": rssOK, "closed_memory": closed,
		"cached_prompt_first_prediction_seconds": reuseSeconds, "cached_prompt_vs_prefill_kl": reuseKL, "cached_prompt_vs_prefill_argmax_matches": reuseMatch,
		"argmax": picks,
	}
	encoded, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(output, "report.json"), encoded, 0644); err != nil {
		t.Fatal(err)
	}
	t.Logf("bits=%d prompt=%d load=%.3fs prefill=%.3fs decode=%.2f t/s peak=%.3f GB", n.bits, len(ref.Prompt), load, pp, float64(len(decode))/dt, float64(inferenceMemory["peak_bytes"])/1e9)
}
