//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"github.com/wbrown/tinyoai/internal/mlxbench"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	mx "github.com/wbrown/tinyoai/internal/mlx"
)

// TestMLXPrefillFinalistParity saves full-vocabulary comparisons for 512+256
// and 8129+64 matched-history trajectories, reaching KV position 8191. Exact
// optimizations must preserve every logit; lossy arithmetic experiments report
// argmax and KL for assessment.
func TestMLXPrefillFinalistParity(t *testing.T) {
	dir, out, configPath := os.Getenv("TINYOAI_MLX_DIR"), os.Getenv("TINYOAI_PREFILL_PARITY_OUTPUT"), os.Getenv("TINYOAI_PREFILL_CANDIDATE_FILE")
	if dir == "" || out == "" || configPath == "" {
		t.Skip("set model, candidate, and parity output")
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	var candidate MLXPrefillCase
	if err = json.Unmarshal(data, &candidate); err != nil {
		t.Fatal(err)
	}
	if candidate.Chunk == 0 {
		candidate.Chunk = 512
	}
	if candidate.HeadBatch == 0 {
		candidate.HeadBatch = 32
	}
	if err = os.MkdirAll(out, 0700); err != nil {
		t.Fatal(err)
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
	var story strings.Builder
	for i := 0; i < 900; i++ {
		fmt.Fprintf(&story, "On day %d, Mara visited the lighthouse and wrote about the weather in her notebook.\n", i)
	}
	allIDs := m.tokenizer.encode(story.String())
	var summaries []map[string]any
	for _, shape := range [][2]int{{512, 256}, {8129, 64}} {
		length, steps := shape[0], shape[1]
		ids := allIDs[:length]
		var reference []float32
		var forced []int
		for _, kind := range []string{"baseline", "candidate"} {
			if err := mx.Run(func() { n.reset(); mx.ResetMemory() }); err != nil {
				t.Fatal(err)
			}
			c := MLXPrefillCase{Chunk: 512, HeadBatch: 32}
			if kind == "candidate" {
				c = candidate
			}
			n.probabilityBatch, n.topKBlock, n.reserveTokens = c.HeadBatch, c.TopKBlock, 0
			n.fusedProjections, n.densePrefill = c.FusedProjections, c.DensePrefill
			if c.FusedProjections {
				if err := n.packProjections(); err != nil {
					t.Fatal(err)
				}
			}
			if c.ReserveKV {
				n.reserveTokens = ((length + 255) / 256) * 256
			}
			var logits []float32
			for pos := 0; pos < length; {
				end := min(pos+c.Chunk, length)
				targets := make([]int, end-pos)
				copy(targets, allIDs[pos+1:end+1])
				var rows []logprobRow
				logits, rows, err = n.ForwardWithLogprobs(context.Background(), pos, ids[pos:end], targets, 12)
				if err != nil || len(rows) != end-pos {
					t.Fatal("incomplete prefill", err)
				}
				pos = end
			}
			var saved []float32
			agree, meanKL, maxKL := 0, 0.0, 0.0
			identical := true
			for step := 0; step < steps; step++ {
				id := mlxbench.Argmax(logits)
				if kind == "baseline" {
					forced = append(forced, id)
				} else {
					ref := reference[step*m.config.VocabSize : (step+1)*m.config.VocabSize]
					kl := mlxbench.KL(ref, logits)
					meanKL += kl
					maxKL = max(maxKL, kl)
					identical = identical && reflect.DeepEqual(ref, logits)
				}
				if id == forced[step] {
					agree++
				}
				saved = append(saved, logits...)
				if step+1 < steps {
					logits, err = n.Forward(context.Background(), length+step, []int{forced[step]}, true)
					if err != nil {
						t.Fatal(err)
					}
				}
			}
			f, err := os.Create(filepath.Join(out, fmt.Sprintf("%s-%d-%d.f32", kind, length, steps)))
			if err != nil {
				t.Fatal(err)
			}
			err = binary.Write(f, binary.LittleEndian, saved)
			closeErr := f.Close()
			if err != nil {
				t.Fatal(err)
			}
			if closeErr != nil {
				t.Fatal(closeErr)
			}
			if kind == "baseline" {
				reference = saved
				continue
			}
			r := map[string]any{"prompt": length, "steps": steps, "last_kv_position": n.offset - 1, "argmax_agreement": agree, "mean_kl": meanKL / float64(steps), "max_kl": maxKL, "bit_identical": identical, "candidate": candidate, "prompt_ids": ids, "forced_tokens": forced}
			summaries = append(summaries, r)
			t.Logf("%d+%d: %d/%d, mean KL %.12f max KL %.12f, bit identical %v, last KV position %d", length, steps, agree, steps, meanKL/float64(steps), maxKL, identical, n.offset-1)
			if length == 8129 && n.offset != 8192 {
				t.Error("did not exercise last cache position")
			}
			if err := mlxbench.WriteJSON(filepath.Join(out, "summary.json"), summaries); err != nil {
				t.Fatal(err)
			}
			if !c.FusedProjections && !c.DensePrefill && !identical {
				t.Error("exact optimization changed logits")
			}
		}
	}
}
