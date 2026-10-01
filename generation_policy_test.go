package tinyoai

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// TestBackendsRejectInvalidGeneration checks validation before any backend can
// dereference weights, allocate context, or start its worker.
func TestBackendsRejectInvalidGeneration(t *testing.T) {
	backends := map[string]Generator{"llama CPU": &Model{}, "StableLM CPU": &StableLM{}, "MLX": &MLX{}, "Llama MLX": &LlamaMLX{}}
	for name, backend := range backends {
		for _, opts := range []GenerateOptions{{MaxTokens: -1}, {Temperature: -1}, {Temperature: math.NaN()}, {Temperature: math.Inf(1)}, {Temperature: math.Inf(-1)}} {
			if _, err := backend.Generate("hello", opts); err == nil {
				t.Errorf("%s accepted %+v", name, opts)
			}
		}
	}
}

// TestLegacyTokenizerLengthValidation loads a real checkpoint with corrupt
// tokenizer metadata and requires errors rather than panics or large allocations.
func TestLegacyTokenizerLengthValidation(t *testing.T) {
	for _, tc := range []struct {
		name    string
		maximum uint32
		length  int32
	}{
		{"negative", 16, -1}, {"above declared maximum", 16, 17}, {"huge entry", 16, math.MaxInt32}, {"huge maximum", math.MaxUint32, 1}, {"zero maximum", 0, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var tok bytes.Buffer
			for _, v := range []any{tc.maximum, float32(0), tc.length} {
				if err := binary.Write(&tok, binary.LittleEndian, v); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := LoadModel(bytes.NewReader(defaultCheckpoint), &tok); err == nil {
				t.Fatal("accepted invalid tokenizer length")
			}
		})
	}
}

// TestProbabilityCacheReuse checks edits, changed shortlist widths, and capture
// toggling without needing native hardware. Rows before the cut remain intact.
func TestProbabilityCacheReuse(t *testing.T) {
	for _, tc := range []struct {
		name                   string
		resident, prompt       []int
		rows, count, wantCount int
		capture                bool
		want                   int
	}{
		{"identical", []int{1, 2, 3, 4}, []int{1, 2, 3, 4}, 4, 12, 12, true, 2},
		{"edit", []int{1, 2, 3, 4}, []int{1, 2, 8, 9}, 4, 12, 12, true, 1},
		{"wider shortlist", []int{1, 2, 3, 4}, []int{1, 2, 3, 4}, 4, 4, 12, true, 0},
		{"incomplete rows", []int{1, 2, 3, 4}, []int{1, 2, 3, 4}, 2, 12, 12, true, 0},
		{"disable capture", []int{1, 2, 3, 4}, []int{1, 2, 3, 4}, 4, 12, 0, false, 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rows := make([]logprobRow, tc.rows)
			for i := range rows {
				rows[i].Chosen = float64(i + 1)
			}
			count := tc.count
			prefix := prepareProbabilityCache(reusablePrefix(tc.resident, tc.prompt), &rows, &count, GenerateOptions{PromptLogprobs: tc.capture, Logprobs: tc.wantCount})
			if prefix != tc.want || count != tc.wantCount {
				t.Fatalf("prefix=%d count=%d", prefix, count)
			}
			if !tc.capture {
				if rows != nil {
					t.Fatal("retained disabled probabilities")
				}
				return
			}
			if len(rows) != prefix {
				t.Fatal("stale suffix rows")
			}
			for i, row := range rows {
				if row.Chosen != float64(i+1) {
					t.Fatal("reused rows changed")
				}
			}
		})
	}
}
