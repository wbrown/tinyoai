package tinyoai

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// SamplingOptions controls filtering and probability-space penalties. Start
// with DefaultSampling before changing fields; its zero value is not valid.
// Nil Sampling in GenerateOptions preserves the temperature-only sampler.
type SamplingOptions struct {
	// TopK keeps this many highest-scoring candidates; zero keeps all
	// candidates.
	TopK int `json:"top_k"`
	// TopP keeps the smallest nucleus reaching this probability mass; one
	// disables the filter.
	TopP float64 `json:"top_p"`
	// TFS is the tail-free second-difference cutoff in (0, 1]; one disables it.
	TFS float64 `json:"tail_free_sampling"`
	// RepetitionPenalty divides repeated-token probability by this factor raised
	// to its recency weight before normalization; one disables it.
	RepetitionPenalty float64 `json:"repetition_penalty"`
	// RepetitionRange limits penalty history to this many recent tokens; zero
	// uses all history.
	RepetitionRange int `json:"repetition_penalty_range"`
	// RepetitionSlope shapes recency weighting of all three penalties; zero
	// gives uniform weight.
	RepetitionSlope float64 `json:"repetition_penalty_slope"`
	// PresencePenalty subtracts this amount times the most recent occurrence's
	// recency weight once per token in the retained history.
	PresencePenalty float64 `json:"presence_penalty"`
	// FrequencyPenalty subtracts this amount times each occurrence's recency
	// weight throughout the retained history.
	FrequencyPenalty float64 `json:"frequency_penalty"`
}

// DefaultSampling returns valid neutral settings: no top-k truncation, full
// top-p and tail-free mass, and no repetition or additive penalties.
func DefaultSampling() SamplingOptions {
	return SamplingOptions{TopP: 1, TFS: 1, RepetitionPenalty: 1}
}

// Validate checks that every sampling control is finite and within the
// supported range. A zero-value SamplingOptions is not neutral; initialize
// with DefaultSampling before overriding fields.
func (o SamplingOptions) Validate() error {
	for _, v := range []float64{o.TopP, o.TFS, o.RepetitionPenalty, o.RepetitionSlope, o.PresencePenalty, o.FrequencyPenalty} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("sampling values must be finite")
		}
	}
	if o.TopK < 0 || o.TopK > 65536 || o.TopP <= 0 || o.TopP > 1 || o.TFS <= 0 || o.TFS > 1 ||
		o.RepetitionPenalty < 1 || o.RepetitionPenalty > 16 || o.RepetitionRange < 0 || o.RepetitionRange > 8192 ||
		o.RepetitionSlope < 0 || o.RepetitionSlope > 10 || math.Abs(o.PresencePenalty) > 16 || math.Abs(o.FrequencyPenalty) > 16 {
		return fmt.Errorf("sampling value out of range")
	}
	return nil
}

// repetitionWeight follows the documented zero-to-double recency curve. The
// whole configured window anchors the curve, even when the story is shorter.
// These sampling operations do not reproduce NovelAI's private implementation exactly.
func repetitionWeight(age, window int, slope float64) float64 {
	if slope == 0 || window <= 1 {
		return 1
	}
	x := 1 - 2*float64(age)/float64(window-1)
	return 1 + slope*x/(1+math.Abs(x)*(slope-1))
}

// penalizedLogits copies raw scores to float64 and applies recency-weighted
// frequency, presence, and logarithmic repetition penalties. Frequency counts
// every occurrence; presence and repetition use the most recent occurrence of
// each ID. The raw logits are unchanged.
func penalizedLogits(logits []float32, history []int, o SamplingOptions) []float64 {
	l := make([]float64, len(logits))
	for i, v := range logits {
		l[i] = float64(v)
	}
	window := o.RepetitionRange
	if window == 0 {
		window = len(history)
	}
	seen := make(map[int]bool)
	for age := 0; age < min(window, len(history)); age++ {
		id := history[len(history)-1-age]
		if id < 0 || id >= len(l) {
			continue
		}
		weight := repetitionWeight(age, window, o.RepetitionSlope)
		l[id] -= o.FrequencyPenalty * weight
		if !seen[id] {
			// A probability-space repetition penalty is an additive log penalty,
			// invariant to the arbitrary constant offset of the model's logits.
			l[id] -= (math.Log(o.RepetitionPenalty) + o.PresencePenalty) * weight
			seen[id] = true
		}
	}
	return l
}

type candidate struct {
	id int
	p  float64
}

// samplingCandidates returns scores in descending order, breaking ties by
// ascending token ID and optionally retaining only k entries. A bounded
// insertion shortlist for small k preserves the full sort's ordering, which
// also preserves seeded sampling sums.
func samplingCandidates(logits []float64, k int) []candidate {
	if k > 0 && k <= 256 && k < len(logits) {
		out := make([]candidate, 0, k+1)
		for id, p := range logits {
			if len(out) == k && p <= out[k-1].p {
				continue
			}
			at := sort.Search(len(out), func(i int) bool { return p > out[i].p })
			out = append(out, candidate{})
			copy(out[at+1:], out[at:len(out)-1])
			out[at] = candidate{id, p}
			if len(out) > k {
				out = out[:k]
			}
		}
		return out
	}
	out := make([]candidate, len(logits))
	for id, p := range logits {
		out[id] = candidate{id, p}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].p == out[j].p {
			return out[i].id < out[j].id
		}
		return out[i].p > out[j].p
	})
	if k > 0 && k < len(out) {
		out = out[:k]
	}
	return out
}

// sampleWithOptions selects the next token using validated controls. With
// explicit sampling settings, it applies penalties before greedy selection or
// temperature/top-k/top-p/tail-free sampling. Nil Sampling delegates to the
// legacy sampler, which may overwrite logits; probability callers must capture
// raw scores first.
func sampleWithOptions(logits []float32, history []int, opts GenerateOptions, rng *rand.Rand) int32 {
	if opts.Sampling == nil {
		return sampleNext(logits, opts.Temperature, rng)
	}
	l := penalizedLogits(logits, history, *opts.Sampling)
	best := 0
	for i := range l {
		if l[i] > l[best] {
			best = i
		}
	}
	if opts.Temperature == 0 {
		return int32(best)
	}
	c := samplingCandidates(l, opts.Sampling.TopK)
	maxLogit, total := c[0].p, 0.0
	for i := range c {
		c[i].p = math.Exp((c[i].p - maxLogit) / opts.Temperature)
		total += c[i].p
	}
	// Fresh Coffee order: temperature, top-k, top-p, then tail-free.
	if opts.Sampling.TopP < 1 {
		cumulative := 0.0
		for i := range c {
			cumulative += c[i].p
			if cumulative >= total*opts.Sampling.TopP {
				c = c[:i+1]
				break
			}
		}
	}
	if opts.Sampling.TFS < 1 && len(c) > 2 {
		curvature := make([]float64, len(c)-2)
		sum := 0.0
		for i := range curvature {
			curvature[i] = math.Abs(c[i].p - 2*c[i+1].p + c[i+2].p)
			sum += curvature[i]
		}
		if sum > 1e-12 {
			acc := 0.0
			for i, v := range curvature {
				acc += v
				if acc > opts.Sampling.TFS*sum {
					c = c[:i+1] // keep at least the highest-probability token
					break
				}
			}
		}
	}
	total = 0
	for _, v := range c {
		total += v.p
	}
	draw := rng.Float64() * total
	for _, v := range c {
		draw -= v.p
		if draw <= 0 {
			return int32(v.id)
		}
	}
	return int32(c[len(c)-1].id)
}
