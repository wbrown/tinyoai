package tinyoai

import (
	"fmt"
	"strings"
)

// contextLength resolves zero to the model limit and rejects negative or
// oversized requests. A positive result is the total position budget,
// including prompt and generation.
func contextLength(modelLimit, requested int) (int, error) {
	if requested < 0 || requested > modelLimit {
		return 0, fmt.Errorf("context_length must be between 1 and %d, or 0 for the model default", modelLimit)
	}
	if requested == 0 {
		return modelLimit, nil
	}
	return requested, nil
}

// preparePrompt tokenizes a StableLM prompt and returns retained IDs, the
// number dropped, and any policy error. Whole-prompt BPE is preserved when it
// fits. Truncation retains BOS, an independently encoded protected prefix, and
// the newest suffix tokens.
//
// Ordinary truncation reserves MaxTokens-1 additional forward positions
// because the last sampled token need not be evaluated. A client-prepared
// prompt instead must match ExpectedPromptTokens and fit the stricter
// prompt-plus-output reservation; it is never silently truncated.
func preparePrompt(prompt string, opts GenerateOptions, limit int, encode func(string) []int) ([]int, int, error) {
	if !strings.HasPrefix(prompt, opts.PromptPrefix) {
		return nil, 0, fmt.Errorf("prompt_prefix must be at the beginning of prompt")
	}
	ids := encode(prompt)
	if opts.ExpectedPromptTokens > 0 {
		if len(ids) != opts.ExpectedPromptTokens {
			return nil, 0, fmt.Errorf("prepared context token count changed: client counted %d, model counted %d", opts.ExpectedPromptTokens, len(ids))
		}
		if opts.TruncatePrompt || len(ids)+max(1, opts.MaxTokens) > limit {
			return nil, 0, fmt.Errorf("prepared context and requested output exceed the %d-token context", limit)
		}
		return ids, 0, nil
	}
	keep := max(1, limit-max(1, opts.MaxTokens)+1)
	if !opts.TruncatePrompt || len(ids) <= keep {
		return ids, 0, nil
	}
	if opts.PromptPrefix == "" {
		trimmed := trimPrompt(ids, limit, opts.MaxTokens)
		return trimmed, len(ids) - len(trimmed), nil
	}
	prefix := encode(opts.PromptPrefix) // includes the same BOS as the full prompt
	if len(prefix) > keep {
		return nil, 0, fmt.Errorf("prompt_prefix needs %d tokens but only %d fit with the requested output", len(prefix), keep)
	}
	tail := keep - len(prefix)
	return append(prefix, ids[len(ids)-tail:]...), len(ids) - keep, nil
}

// trimPrompt keeps BOS and the newest tokens that fit the requested
// forward-position budget. Generating output tokens requires output-1
// additional forwards, because the final sampled token need not be evaluated.
// If no trimming is needed, the input slice is returned unchanged.
func trimPrompt(ids []int, limit, output int) []int {
	keep := max(1, limit-max(1, output)+1)
	if len(ids) <= keep {
		return ids
	}
	return append([]int{ids[0]}, ids[len(ids)-keep+1:]...)
}
