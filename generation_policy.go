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
	if opts.Logprobs < 0 || opts.PromptLogprobs && opts.Logprobs == 0 {
		return fmt.Errorf("logprobs must be nonnegative, and positive when collecting prompt probabilities")
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

// CachePlan records the backend's actual reuse decision before any forward.
// CommonTokens includes the terminal prompt token; ReusedTokens accounts for
// logits alignment and probability-row requirements. Reasons are cumulative.
type CachePlan struct {
	ResidentTokens   int      `json:"resident_tokens"`
	PromptTokens     int      `json:"prompt_tokens"`
	CommonTokens     int      `json:"common_tokens"`
	ReusedTokens     int      `json:"reused_tokens"`
	PrefillTokens    int      `json:"prefill_tokens"`
	ProbabilityRows  int      `json:"probability_rows"`
	ProbabilityWidth int      `json:"probability_width"`
	Reasons          []string `json:"reasons"`
}

func cachePlan(resident, prompt []int, rows, width int, opts GenerateOptions) CachePlan {
	common := 0
	for common < min(len(resident), len(prompt)) && resident[common] == prompt[common] {
		common++
	}
	p := CachePlan{ResidentTokens: len(resident), PromptTokens: len(prompt), CommonTokens: common, ProbabilityRows: rows, ProbabilityWidth: width, Reasons: []string{}}
	if len(resident) == 0 {
		p.Reasons = append(p.Reasons, "no-resident-tokens")
	} else if common < min(len(resident), len(prompt)) {
		p.Reasons = append(p.Reasons, "prompt-prefix-changed")
	} else if common < len(prompt) {
		p.Reasons = append(p.Reasons, "prompt-appended")
	}
	prefix := reusablePrefix(resident, prompt)
	if prefix < common {
		p.Reasons = append(p.Reasons, "terminal-logits-recompute")
	}
	if opts.PromptLogprobs {
		if rows < prefix {
			p.Reasons = append(p.Reasons, "missing-prompt-probability-rows")
		}
		if width < opts.Logprobs {
			p.Reasons = append(p.Reasons, "probability-shortlist-expanded")
		}
		if rows < prefix || width < opts.Logprobs {
			prefix = 0
		} else if prefix > 0 {
			prefix--
			p.Reasons = append(p.Reasons, "probability-boundary-recompute")
		}
	}
	p.ReusedTokens = prefix
	p.PrefillTokens = len(prompt) - prefix
	return p
}
