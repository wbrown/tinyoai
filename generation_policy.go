package tinyoai

import (
	"fmt"
	"math"
)

// validateGeneration checks controls shared by every backend. Zero MaxTokens
// means use the remaining context; HTTP adapters apply their own defaults.
// Model-specific capabilities and context limits are checked by each backend.
func validateGeneration(opts GenerateOptions) error {
	if opts.MaxTokens < 0 || opts.Temperature < 0 || math.IsNaN(opts.Temperature) || math.IsInf(opts.Temperature, 0) {
		return fmt.Errorf("max_tokens must be nonnegative and temperature must be finite and nonnegative")
	}
	if opts.Logprobs < 0 || opts.Logprobs > 64 || opts.PromptLogprobs && opts.Logprobs == 0 {
		return fmt.Errorf("logprobs must be 0–64, and positive when collecting prompt probabilities")
	}
	if opts.Sampling != nil {
		return opts.Sampling.Validate()
	}
	return nil
}

// reusablePrefix leaves the final prompt token for evaluation to recover its
// next-token logits. Both inputs include BOS and only resident contains KV.
func reusablePrefix(resident, prompt []int) int {
	prefix := 0
	for prefix < min(len(resident), len(prompt)-1) && resident[prefix] == prompt[prefix] {
		prefix++
	}
	return prefix
}

// prepareProbabilityCache adjusts reuse for prompt scoring and discards stale
// suffix rows. The last reused position predicts the first changed token, so
// scoring revisits it. A wider shortlist requires fresh probability rows.
func prepareProbabilityCache(prefix int, rows *[]logprobRow, count *int, opts GenerateOptions) int {
	if !opts.PromptLogprobs {
		*rows, *count = nil, 0
		return prefix
	}
	if len(*rows) < prefix || *count < opts.Logprobs {
		prefix = 0
	}
	if prefix > 0 {
		prefix--
	}
	*rows = (*rows)[:min(prefix, len(*rows))]
	*count = opts.Logprobs
	return prefix
}
