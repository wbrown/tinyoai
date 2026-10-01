//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	mx "github.com/wbrown/tinyoai/internal/mlx"
)

// branchKL computes full-vocabulary KL(reference || actual) from equal-length
// logit rows for branch comparisons.
func branchKL(ref, got []float32) float64 {
	r := summarizeLogits(ref, 0, 0)
	g := summarizeLogits(got, 0, 0)
	rz, gz := float64(ref[0])-r.Chosen, float64(got[0])-g.Chosen
	kl := 0.0
	for i, v := range ref {
		lp := float64(v) - rz
		kl += math.Exp(lp) * (lp - (float64(got[i]) - gz))
	}
	return kl
}

// TestMLXBranchIsolation checks that FP16 and Q8 branches cannot alter
// document KV, then compares batched branches with separate forwards on the
// same prefix.
func TestMLXBranchIsolation(t *testing.T) {
	dir := os.Getenv("TINYOAI_MLX_DIR")
	if dir == "" {
		t.Skip("requires native Clio")
	}
	m, err := LoadMLXNative(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	n := m.native.(*nativeMLX)
	ctx := context.Background()
	ids := m.tokenizer.encode(strings.Repeat("The rain tapped against the window as she unfolded the letter. ", 60))[:512]
	for _, bits := range []int{8, 16} {
		if _, err = n.SetKVBits(bits); err != nil {
			t.Fatal(err)
		}
		ref, err := n.Forward(ctx, 0, ids, true)
		if err != nil {
			t.Fatal(err)
		}
		b, err := n.NewBranches(128, 4)
		if err != nil {
			t.Fatal(err)
		}
		for step := 0; step < 3; step++ {
			if _, err = b.Forward(ctx, []int{ids[128], ids[129], ids[130], ids[131]}); err != nil {
				t.Fatal(err)
			}
		}
		if n.offset != 512 {
			t.Fatal("branch changed main offset")
		}
		canceled, cancel := context.WithCancel(ctx)
		cancel()
		if _, err = b.Forward(canceled, []int{1, 2, 3, 4}); err == nil {
			t.Fatal("ignored cancellation")
		}
		if err = b.Close(); err != nil {
			t.Fatal(err)
		}
		got, err := n.Forward(ctx, 511, ids[511:], true)
		if err != nil {
			t.Fatal(err)
		}
		// Compare against the same one-token shape, independent of prefill kernels.
		same, err := n.Forward(ctx, 511, ids[511:], true)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, same) || branchKL(ref, got) > 0.001 {
			t.Fatalf("KV%d document cache changed", bits)
		}
		// Strong exact check: branches at an earlier word must leave all later KV intact.
		b, err = n.NewBranches(128, 4)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = b.Forward(ctx, []int{1, 2, 3, 4}); err != nil {
			t.Fatal(err)
		}
		_ = b.Close()
		after, err := n.Forward(ctx, 511, ids[511:], true)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(same, after) {
			t.Fatalf("KV%d branches altered document logits", bits)
		}
		// Batched versus separate branches, on exactly the same prefix.
		b, _ = n.NewBranches(512, 4)
		seeds := []int{ids[128], ids[129], ids[130], ids[131]}
		rows, err := b.Forward(ctx, seeds)
		if err != nil {
			t.Fatal(err)
		}
		_ = b.Close()
		for i, id := range seeds {
			row, err := n.Forward(ctx, 512, []int{id}, true)
			if err != nil {
				t.Fatal(err)
			}
			kl := branchKL(row, rows[i])
			t.Logf("KV%d branch%d KL %.9f argmax %d/%d", bits, i, kl, argmax(row), argmax(rows[i]))
			if kl > 0.001 {
				t.Fatalf("branch distribution differs: %.9f", kl)
			}
		}
	}
}

// TestMLXBatchedBranchMeasurement measures paired serial and batched suffix
// expansion, including full-vocabulary transfers and greedy selection. Loading
// and prefill are excluded; forced histories isolate distribution drift, and
// closing branches must release suffix memory.
func TestMLXBatchedBranchMeasurement(t *testing.T) {
	dir, out := os.Getenv("TINYOAI_MLX_DIR"), os.Getenv("TINYOAI_BATCH_OUTPUT")
	if dir == "" || out == "" {
		t.Skip("set TINYOAI_MLX_DIR and TINYOAI_BATCH_OUTPUT")
	}
	if err := os.MkdirAll(out, 0755); err != nil {
		t.Fatal(err)
	}
	m, err := LoadMLXNative(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	n := m.native.(*nativeMLX)
	ctx := context.Background()
	if _, err = n.SetKVBits(8); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("testdata/long-prompt.txt")
	if err != nil {
		t.Fatal(err)
	}
	ids := m.tokenizer.encode(strings.Repeat(string(data), 2))
	var reports []map[string]any
	for _, prefix := range []int{512, 4096, 8188} {
		if err = mx.Run(func() { n.reset(); mx.ClearCache() }); err != nil {
			t.Fatal(err)
		}
		var logits []float32
		for pos := 0; pos < prefix; {
			end := min(pos+512, prefix)
			logits, err = n.Forward(ctx, pos, ids[pos:end], end == prefix)
			if err != nil {
				t.Fatal(err)
			}
			pos = end
		}
		seeds := summarizeLogits(logits, 0, 12).IDs
		warm, _ := n.NewBranches(prefix, 12)
		if _, err = warm.Forward(ctx, seeds); err != nil {
			t.Fatal(err)
		}
		_ = warm.Close()
		if _, err = n.Forward(ctx, prefix, []int{seeds[0]}, true); err != nil {
			t.Fatal(err)
		}
		for repeat := 0; repeat < 3; repeat++ {
			for _, count := range []int{4, 8, 12} {
				for _, depth := range []int{1, 2, 3} {
					paths := make([][]int, count)
					logprobs := make([][]float64, count)
					baseline := make([][][]float32, count)
					start := time.Now()
					for i, seed := range seeds[:count] {
						paths[i] = []int{seed}
						token := seed
						for step := 0; step < depth; step++ {
							row, e := n.Forward(ctx, prefix+step, []int{token}, true)
							if e != nil {
								t.Fatal(e)
							}
							baseline[i] = append(baseline[i], row)
							token = int(argmax(row))
							paths[i] = append(paths[i], token)
							logprobs[i] = append(logprobs[i], summarizeLogits(row, token, 0).Chosen)
						}
					}
					serial := time.Since(start).Seconds()
					var active, peak, after, suffix uint64
					_ = mx.Run(func() { mx.ResetMemory(); active, _ = mx.Memory() })
					b, _ := n.NewBranches(prefix, count)
					tokens := slices.Clone(seeds[:count])
					agree := 0
					maxKL := 0.0
					var steps [][]int
					// Teacher-force the serial paths so each full distribution has identical conditioning.
					start = time.Now()
					for step := 0; step < depth; step++ {
						rows, e := b.Forward(ctx, tokens)
						if e != nil {
							t.Fatal(e)
						}
						next := make([]int, count)
						for i, row := range rows {
							next[i] = int(argmax(row))
							_ = summarizeLogits(row, next[i], 0)
						}
						steps = append(steps, next)
						for i := range tokens {
							tokens[i] = paths[i][step+1]
						}
						// Exclude KL checking from the measured compute/selection latency.
						elapsed := time.Since(start)
						for i, row := range rows {
							if next[i] == paths[i][step+1] {
								agree++
							}
							maxKL = max(maxKL, branchKL(baseline[i][step], row))
						}
						start = time.Now().Add(-elapsed)
					}
					batched := time.Since(start).Seconds()
					_ = mx.Run(func() {
						_, peak = mx.Memory()
						for _, l := range b.(*nativeBranches).layers {
							suffix += l.keys.Bytes() + l.values.Bytes()
						}
					})
					_ = b.Close()
					_ = mx.Run(func() { after, _ = mx.Memory() })
					report := map[string]any{"prefix": prefix, "branches": count, "depth": depth, "repeat": repeat, "serial_seconds": serial, "batch_seconds": batched, "speedup": serial / batched, "argmax_agree": agree, "comparisons": count * depth, "max_kl": maxKL, "serial_paths": paths, "serial_logprobs": logprobs, "batch_next_ids": steps, "active_before": active, "active_after": after, "peak_extra": peak - active, "suffix_bytes": suffix}
					reports = append(reports, report)
					encoded, _ := json.MarshalIndent(reports, "", "  ")
					if err = os.WriteFile(filepath.Join(out, "runs.json"), encoded, 0644); err != nil {
						t.Fatal(err)
					}
					t.Logf("prefix%d N%d depth%d serial %.3fs batch %.3fs %.2fx KL %.9f argmax %d/%d extra %.1fMiB suffix %.1fMiB", prefix, count, depth, serial, batched, serial/batched, maxKL, agree, count*depth, float64(peak-active)/(1<<20), float64(suffix)/(1<<20))
					if maxKL > 0.001 {
						t.Fatalf("KL too large: %g", maxKL)
					}
					if after != active {
						t.Fatalf("branch memory retained: before %d after %d", active, after)
					}
				}
			}
		}
	}
}
