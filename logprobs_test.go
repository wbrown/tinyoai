package tinyoai

import (
	"math"
	"math/rand"
	"reflect"
	"sort"
	"testing"
)

// TestProbabilitySummary checks stable full-vocabulary normalization and
// deterministic token-ID ordering for tied alternatives.
func TestProbabilitySummary(t *testing.T) {
	r := summarizeLogits([]float32{1000, 1002, 1002, 999}, 3, 2)
	z := 1002 + math.Log(2+math.Exp(-2)+math.Exp(-3))
	if !reflect.DeepEqual(r.IDs, []int{1, 2}) || math.Abs(r.Chosen-(999-z)) > 1e-12 {
		t.Fatal(r)
	}
	if r.Values[0] != r.Values[1] {
		t.Fatal(r)
	}
}

// TestProbabilityByteSpans checks raw UTF-8 byte offsets before and after
// dropping the oldest prompt tokens.
func TestProbabilityByteSpans(t *testing.T) {
	tok := &nerdstashTokenizer{pieces: []string{"<bos>", "Hello", "▁café", "!", "lost", "▁end"}, special: map[int]bool{0: true}}
	for _, tc := range []struct {
		prompt  string
		ids     []int
		dropped int
		want    [][2]int
	}{
		{"Hello café!", []int{0, 1, 2, 3}, 0, [][2]int{{0, 0}, {0, 5}, {5, 11}, {11, 12}}},
		{"Hello lost end", []int{0, 5}, 2, [][2]int{{0, 0}, {10, 14}}},
	} {
		got := tok.promptSpans(tc.prompt, "", tc.ids, tc.dropped)
		if !reflect.DeepEqual(got, tc.want) {
			t.Fatal(got, tc.want)
		}
	}
}

// TestProbabilitySummaryMatchesFullVocabulary uses an independent full-sort
// oracle to check shortlisted IDs and exact normalization for ties, extreme
// values, and a 65,536-token vocabulary.
func TestProbabilitySummaryMatchesFullVocabulary(t *testing.T) {
	rng := rand.New(rand.NewSource(73))
	for _, size := range []int{1, 63, 65536} {
		for _, pattern := range []string{"random", "ties", "ascending", "descending", "extreme"} {
			values := make([]float32, size)
			for i := range values {
				switch pattern {
				case "random":
					values[i] = float32(rng.NormFloat64() * 5)
				case "ties":
					values[i] = float32(i % 7)
				case "ascending":
					values[i] = float32(i)
				case "descending":
					values[i] = -float32(i)
				case "extreme":
					values[i] = float32((i%3)-1) * math.MaxFloat32
				}
			}
			ids := make([]int, size)
			best := float64(values[0])
			for i, v := range values {
				ids[i] = i
				best = math.Max(best, float64(v))
			}
			sort.Slice(ids, func(i, j int) bool {
				return values[ids[i]] > values[ids[j]] || values[ids[i]] == values[ids[j]] && ids[i] < ids[j]
			})
			total := 0.0
			for _, v := range values {
				total += math.Exp(float64(v) - best)
			}
			z := best + math.Log(total)
			for _, count := range []int{0, 1, 12, 64} {
				got := summarizeLogits(values, size-1, count)
				wantIDs := ids[:min(count, size)]
				if !reflect.DeepEqual(got.IDs, wantIDs) || math.Float64bits(got.Chosen) != math.Float64bits(float64(values[size-1])-z) {
					t.Fatalf("size=%d pattern=%s count=%d: IDs or chosen differ", size, pattern, count)
				}
				for i, id := range wantIDs {
					if math.Float64bits(got.Values[i]) != math.Float64bits(float64(values[id])-z) {
						t.Fatalf("top logprob differs")
					}
				}
			}
		}
	}
}
