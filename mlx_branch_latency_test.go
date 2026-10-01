//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	mx "github.com/wbrown/tinyoai/internal/mlx"
)

// TestMLXGreedyBranchLatency is an opt-in measurement of serial greedy
// suffixes from resident prefixes. It excludes prefill and saves individual
// pass timings for fixed token counts.
func TestMLXGreedyBranchLatency(t *testing.T) {
	dir, out := os.Getenv("TINYOAI_MLX_DIR"), os.Getenv("TINYOAI_BRANCH_OUTPUT")
	if dir == "" || out == "" {
		t.Skip("set TINYOAI_MLX_DIR and TINYOAI_BRANCH_OUTPUT")
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
	if _, err := n.SetKVBits(8); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("testdata/long-prompt.txt")
	if err != nil {
		t.Fatal(err)
	}
	ids := m.tokenizer.encode(strings.Repeat(string(data), 2))
	if len(ids) < 8188 {
		t.Fatal("prompt fixture too short")
	}
	forward := func(prefix, token int) []float32 {
		t.Helper()
		row, err := n.Forward(context.Background(), prefix, []int{token}, true)
		if err != nil {
			t.Fatal(err)
		}
		return row
	}
	var reports []map[string]any
	for _, prefix := range []int{512, 4096, 8188} {
		if err := mx.Run(func() { n.reset(); mx.ClearCache() }); err != nil {
			t.Fatal(err)
		}
		prefillStart := time.Now()
		var logits []float32
		for pos := 0; pos < prefix; {
			end := min(pos+512, prefix)
			logits, err = n.Forward(context.Background(), pos, ids[pos:end], end == prefix)
			if err != nil {
				t.Fatal(err)
			}
			pos = end
		}
		prefill := time.Since(prefillStart).Seconds()
		seeds := summarizeLogits(logits, int(argmax(logits)), 12)
		// Warm one-token kernels and allocation before timing branch requests.
		token := seeds.IDs[0]
		for i := 0; i < 3; i++ {
			token = int(argmax(forward(prefix+i, token)))
		}
		for repeat := 0; repeat < 3; repeat++ {
			for _, count := range []int{4, 8, 12} {
				for _, depth := range []int{1, 2} {
					start := time.Now()
					var branches []map[string]any
					for _, seed := range seeds.IDs[:count] {
						branchStart := time.Now()
						token := seed
						path := []int{seed}
						var passes []float64
						var logprobs []float64
						for step := 0; step < depth; step++ {
							passStart := time.Now()
							row := forward(prefix+step, token)
							token = int(argmax(row))
							logprobs = append(logprobs, summarizeLogits(row, token, 0).Chosen)
							passes = append(passes, time.Since(passStart).Seconds())
							path = append(path, token)
						}
						branches = append(branches, map[string]any{"ids": path, "continuation_logprobs": logprobs,
							"seconds": time.Since(branchStart).Seconds(), "pass_seconds": passes})
					}
					seconds := time.Since(start).Seconds()
					r := map[string]any{"cached_prefix": prefix, "top_n": count, "extra_tokens": depth,
						"repeat": repeat, "seconds": seconds, "prefill_seconds_excluded": prefill,
						"branches": branches, "weight_bits": 6, "kv_bits": 8}
					reports = append(reports, r)
					encoded, _ := json.MarshalIndent(r, "", "  ")
					name := fmt.Sprintf("prefix-%d-n%d-depth%d-run%d.json", prefix, count, depth, repeat)
					if err := os.WriteFile(filepath.Join(out, name), encoded, 0644); err != nil {
						t.Fatal(err)
					}
					t.Logf("prefix=%d N=%d extra=%d run=%d %.3fs", prefix, count, depth, repeat, seconds)
				}
			}
		}
	}
	encoded, _ := json.MarshalIndent(reports, "", "  ")
	if err := os.WriteFile(filepath.Join(out, "runs.json"), encoded, 0644); err != nil {
		t.Fatal(err)
	}
}
