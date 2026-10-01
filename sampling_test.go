package tinyoai

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sort"
	"testing"
)

// TestProbabilityPenalties checks repetition, presence, and frequency
// penalties and their invariance to a constant logit offset.
func TestProbabilityPenalties(t *testing.T) {
	o := DefaultSampling()
	o.RepetitionPenalty, o.PresencePenalty, o.FrequencyPenalty = 2, .1, .2
	o.RepetitionRange = 3
	l := penalizedLogits([]float32{0, 0, 0, 0}, []int{3, 1, 2, 1}, o)
	for i, want := range []float64{0, -math.Log(2) - .1 - .4, -math.Log(2) - .1 - .2, 0} {
		if math.Abs(l[i]-want) > 1e-12 {
			t.Fatalf("token %d: %g != %g", i, l[i], want)
		}
	}
	// Unlike sign-dependent raw-logit multiplication, probability penalties
	// must produce the same distribution after a constant logit offset.
	shifted := penalizedLogits([]float32{20, 20, 20, 20}, []int{3, 1, 2, 1}, o)
	for i := range l {
		if math.Abs(shifted[i]-l[i]-20) > 1e-12 {
			t.Fatal("offset changed penalty")
		}
	}
}

// TestRepetitionSlopeAndRange checks the recency curve, range boundaries, and
// placement of short histories within the configured window.
func TestRepetitionSlopeAndRange(t *testing.T) {
	if repetitionWeight(0, 5, 1) != 2 || repetitionWeight(2, 5, 1) != 1 || repetitionWeight(4, 5, 1) != 0 {
		t.Fatal("slope endpoints")
	}
	if repetitionWeight(1, 5, 3.33) <= repetitionWeight(1, 5, 1) {
		t.Fatal("slope does not favor recent half")
	}
	o := DefaultSampling()
	o.RepetitionPenalty = math.E
	o.RepetitionSlope = 1
	o.RepetitionRange = 5
	l := penalizedLogits(make([]float32, 5), []int{4, 3, 2, 1, 0}, o)
	for i, want := range []float64{-2, -1.5, -1, -.5, 0} {
		if math.Abs(l[i]-want) > 1e-12 {
			t.Fatal(l)
		}
	}
	// Short prompts retain the recent end of the configured penalty window.
	l = penalizedLogits(make([]float32, 5), []int{1, 0}, o)
	if math.Abs(l[1]+1.5) > 1e-12 {
		t.Fatal(l)
	}
}

// TestSamplingFiltersAndSeed exercises top-k, nucleus, and tail-free filtering
// and deterministic sampling from equal seeds.
func TestSamplingFiltersAndSeed(t *testing.T) {
	o := DefaultSampling()
	o.TopK = 1
	options := GenerateOptions{Temperature: 1, Sampling: &o}
	rng := rand.New(rand.NewSource(7))
	for i := 0; i < 100; i++ {
		if got := sampleWithOptions([]float32{1, 2, 3}, nil, options, rng); got != 2 {
			t.Fatal(got)
		}
	}
	o.TopK = 0
	o.TopP = .5
	for i := 0; i < 100; i++ {
		if got := sampleWithOptions([]float32{0, 0, 5}, nil, options, rng); got != 2 {
			t.Fatal(got)
		}
	}
	o.TopP = 1
	o.TFS = .5
	// This sharp curvature cuts the tail, leaving only the first candidate.
	for i := 0; i < 100; i++ {
		if got := sampleWithOptions([]float32{5, 0, -1, -2}, nil, options, rng); got != 0 {
			t.Fatal(got)
		}
	}
	o.TFS = 1
	o.TopK = 2
	a, b := rand.New(rand.NewSource(99)), rand.New(rand.NewSource(99))
	for i := 0; i < 100; i++ {
		x, y := sampleWithOptions([]float32{1, 2, 3}, nil, options, a), sampleWithOptions([]float32{1, 2, 3}, nil, options, b)
		if x != y || x == 0 {
			t.Fatalf("seed/top-k: %d %d", x, y)
		}
	}
}

// TestTrimPromptReservesGeneration checks that truncation keeps BOS and recent
// tokens while reserving the correct number of forwarded generation positions.
func TestTrimPromptReservesGeneration(t *testing.T) {
	got := trimPrompt([]int{2, 10, 11, 12, 13, 14}, 8, 5)
	if len(got) != 4 || got[0] != 2 || got[1] != 12 || got[3] != 14 {
		t.Fatal(got)
	}
	if len(got)+5-1 != 8 {
		t.Fatal("off-by-one at context boundary")
	}
}

// TestSamplingCandidatesMatchFullSort compares shortlist selection against an
// independent stable ordering across vocabulary sizes, ties, and input orders.
func TestSamplingCandidatesMatchFullSort(t *testing.T) {
	rng := rand.New(rand.NewSource(1701))
	for _, size := range []int{1, 25, 257, 65536} {
		for _, pattern := range []string{"random", "ties", "ascending", "descending"} {
			values := make([]float64, size)
			for i := range values {
				switch pattern {
				case "random":
					values[i] = rng.NormFloat64()
				case "ties":
					values[i] = float64(i % 7)
				case "ascending":
					values[i] = float64(i)
				case "descending":
					values[i] = -float64(i)
				}
			}
			ref := make([]candidate, size)
			for i, v := range values {
				ref[i] = candidate{i, v}
			}
			sort.Slice(ref, func(i, j int) bool {
				if ref[i].p == ref[j].p {
					return ref[i].id < ref[j].id
				}
				return ref[i].p > ref[j].p
			})
			for _, k := range []int{0, 1, 12, 25, 64, 256, 257, 65536} {
				want := ref
				if k > 0 && k < len(want) {
					want = want[:k]
				}
				if got := samplingCandidates(values, k); !slices.Equal(got, want) {
					t.Fatalf("size=%d pattern=%s k=%d shortlist differs", size, pattern, k)
				}
			}
		}
	}
}

// BenchmarkSamplingCandidates compares full-vocabulary sorting with a small
// top-k shortlist over 65,536 logits.
func BenchmarkSamplingCandidates(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	values := make([]float64, 65536)
	for i := range values {
		values[i] = rng.NormFloat64()
	}
	for _, k := range []int{0, 25} {
		b.Run(fmt.Sprint(k), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				samplingCandidates(values, k)
			}
		})
	}
}
