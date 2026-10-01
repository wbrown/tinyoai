package tinyoai

import (
	"context"
	"fmt"
	"io"
	"math/rand"
	"strings"
)

// LlamaMLX runs legacy llama2.c models on Metal, storing weights, activations
// and KV in float32. Set MLX_ENABLE_TF32=0 before starting the process for
// closest numerical agreement with Model; MLX otherwise permits reduced
// precision matrix multiplication on supported hardware.
// It shares Model's tokenizer and sampler, including its
// BOS stop sentinel and prompt-token accounting. Concurrent requests serialize.
type LlamaMLX struct {
	model             *Model
	native            llamaMLXForwarder
	gate              chan struct{}
	tokens            []int
	probRows          []logprobRow
	probCount, window int
	closed            bool
}

type llamaMLXForwarder interface {
	// Forward evaluates a suffix and returns either every vocabulary row or
	// only the last one. The caller owns the model gate.
	Forward(context.Context, int, []int, bool) ([]float32, error)
	// Reset discards KV while leaving weights resident.
	Reset() error
	// Close releases weights, KV, and native streams.
	Close() error
}

// DefaultMLX uploads the embedded stories260K model to MLX. Each call owns an
// independent GPU cache; call Close when finished. No downloads or Python.
func DefaultMLX() (*LlamaMLX, error) {
	m, err := Default()
	if err != nil {
		return nil, err
	}
	return newLlamaMLX(m)
}

// LoadLlamaMLX loads the same legacy checkpoint/tokenizer streams as LoadModel.
// Native MLX must be enabled at build time, as for DefaultMLX.
func LoadLlamaMLX(checkpoint, tokenizer io.Reader) (*LlamaMLX, error) {
	m, err := LoadModel(checkpoint, tokenizer)
	if err != nil {
		return nil, err
	}
	return newLlamaMLX(m)
}

// Config returns a copy of the legacy checkpoint dimensions.
func (m *LlamaMLX) Config() Config { return m.model.Config() }

// Close waits for exclusive access and releases native arrays and cached
// tokens. Repeated calls are harmless.
func (m *LlamaMLX) Close() error {
	<-m.gate
	defer func() { m.gate <- struct{}{} }()
	if m.closed {
		return nil
	}
	m.closed = true
	m.tokens, m.probRows = nil, nil
	return m.native.Close()
}

// Generate completes text with legacy Llama tokenization and sampling on
// native MLX. Requests serialize and reuse matching prefix KV; callbacks run
// synchronously while the model gate is held.
func (m *LlamaMLX) Generate(prompt string, opts GenerateOptions) (GenerateResult, error) {
	return m.generate(prompt, opts, false)
}

// Prefill prepares reusable KV and optional prompt probabilities without sampling.
func (m *LlamaMLX) Prefill(prompt string, opts GenerateOptions) (GenerateResult, error) {
	return m.generate(prompt, opts, true)
}

// probability turns a saved distribution into token alternatives and byte
// offsets. Prompt pieces use raw vocabulary text; completion pieces apply the
// legacy BOS whitespace rule.
func (m *LlamaMLX) probability(row logprobRow, id, previous, position, start int, prefix []int, prompt bool, count int) TokenLogprob {
	piece := func(token int) string {
		if token == 1 {
			return ""
		}
		if prompt {
			return m.model.vocab[token]
		}
		return m.model.tokenPiece(int32(previous), int32(token))
	}
	text := piece(id)
	r := TokenLogprob{TokenProbability: TokenProbability{ID: id, Text: text, Logprob: row.Chosen},
		Position: position, Prefix: prefixFingerprint(prefix), Start: start, End: start + len(text)}
	for i, token := range row.IDs[:min(count, len(row.IDs))] {
		r.Top = append(r.Top, TokenProbability{ID: token, Text: piece(token), Logprob: row.Values[i]})
	}
	return r
}

// generate preserves Model's BOS sentinel and token accounting while adding
// prefix reuse and optional probability capture. A smaller context releases
// old KV. Probability requests recompute the row predicting an edited token,
// and failed forwards invalidate the recorded prefix.
func (m *LlamaMLX) generate(prompt string, opts GenerateOptions, prefillOnly bool) (GenerateResult, error) {
	if err := validateGeneration(opts); err != nil {
		return GenerateResult{}, err
	}
	if opts.KVBits != 0 {
		return GenerateResult{}, fmt.Errorf("Llama MLX uses float32 KV; kv_bits is not supported")
	}
	if opts.TruncatePrompt || opts.PromptPrefix != "" || opts.TokenizerID != "" {
		return GenerateResult{}, fmt.Errorf("Llama MLX expects a complete prompt; TruncatePrompt, PromptPrefix, and TokenizerID are not supported")
	}
	window, err := contextLength(int(m.model.config.SeqLen), opts.ContextLength)
	if err != nil {
		return GenerateResult{}, err
	}
	promptTokens, err := m.model.bpeEncode(prompt)
	if err != nil {
		return GenerateResult{}, err
	}
	if opts.ExpectedPromptTokens < 0 || opts.ExpectedPromptTokens > 0 && opts.ExpectedPromptTokens != len(promptTokens) {
		return GenerateResult{}, fmt.Errorf("prepared prompt token count does not match Llama tokenizer")
	}
	ctx := opts.Context
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return GenerateResult{}, ctx.Err()
	case <-m.gate:
	}
	defer func() { m.gate <- struct{}{} }()
	if m.closed {
		return GenerateResult{}, fmt.Errorf("Llama MLX backend is closed")
	}
	if err := ctx.Err(); err != nil {
		return GenerateResult{}, err
	}
	if window < m.window {
		if err := m.native.Reset(); err != nil {
			return GenerateResult{}, err
		}
		m.tokens, m.probRows = nil, nil
	}
	m.window = window
	ids := append([]int{1}, promptTokens...)
	// Match Model.Generate when a prompt exhausts the context: no completion.
	ids = ids[:min(len(ids), window)]
	prefix := reusablePrefix(m.tokens, ids)
	prefix = prepareProbabilityCache(prefix, &m.probRows, &m.probCount, opts)
	m.tokens = m.tokens[:prefix]
	var logits []float32
	for pos := prefix; pos < len(ids); {
		if err := ctx.Err(); err != nil {
			return GenerateResult{}, err
		}
		end := min(pos+64, len(ids))
		values, err := m.native.Forward(ctx, pos, ids[pos:end], opts.PromptLogprobs)
		if err != nil {
			m.tokens, m.probRows = nil, nil
			return GenerateResult{}, err
		}
		vocab := int(m.model.config.VocabSize)
		if opts.PromptLogprobs {
			for i := pos; i < end; i++ {
				next := 0
				if i+1 < len(ids) {
					next = ids[i+1]
				}
				m.probRows = append(m.probRows, summarizeLogits(values[(i-pos)*vocab:(i-pos+1)*vocab], next, opts.Logprobs))
			}
		}
		logits = values[len(values)-vocab:]
		m.tokens = append(m.tokens, ids[pos:end]...)
		pos = end
		if opts.OnProgress != nil {
			opts.OnProgress("prefill", pos)
		}
	}
	if err := ctx.Err(); err != nil {
		return GenerateResult{}, err
	}
	result := GenerateResult{PromptTokens: len(promptTokens), CachedPromptTokens: max(0, prefix-1), FinishReason: "length"}
	if opts.PromptLogprobs {
		start := 0
		for i := 1; i < len(ids); i++ {
			r := m.probability(m.probRows[i-1], ids[i], ids[i-1], i, start, ids[:i], true, opts.Logprobs)
			result.PromptLogprobs = append(result.PromptLogprobs, r)
			start = r.End
		}
		if opts.OnLogprobs != nil {
			opts.OnLogprobs(ProbabilityEvent{"prompt", result.PromptLogprobs})
		}
	}
	if prefillOnly {
		result.FinishReason = "prefill"
		return result, nil
	}
	if len(promptTokens) >= window {
		return result, nil
	}
	rng := rand.New(rand.NewSource(opts.Seed))
	history := append([]int(nil), ids...)
	var out strings.Builder
	for pos := len(ids) - 1; pos < window; pos++ {
		if err := ctx.Err(); err != nil {
			return GenerateResult{}, err
		}
		working := logits
		if opts.Logprobs > 0 && opts.Sampling == nil {
			working = append([]float32(nil), logits...)
		}
		next := int(sampleWithOptions(working, history, opts, rng))
		if next == 1 {
			result.FinishReason = "stop"
			break
		}
		if opts.Logprobs > 0 {
			row := summarizeLogits(logits, next, opts.Logprobs)
			r := m.probability(row, next, history[len(history)-1], len(history), out.Len(), history, false, opts.Logprobs)
			result.Logprobs = append(result.Logprobs, r)
			if opts.OnLogprobs != nil {
				opts.OnLogprobs(ProbabilityEvent{"completion", []TokenLogprob{r}})
			}
			if len(m.probRows) == len(m.tokens) {
				m.probRows[len(m.probRows)-1] = row
			}
		}
		piece := m.model.tokenPiece(int32(history[len(history)-1]), int32(next))
		history = append(history, next)
		out.WriteString(piece)
		if opts.OnToken != nil {
			opts.OnToken(piece)
		}
		result.CompletionTokens++
		if matchesStop(out.String(), opts.Stop) {
			result.FinishReason = "stop"
			break
		}
		if opts.MaxTokens > 0 && result.CompletionTokens >= opts.MaxTokens || pos+1 == window {
			break
		}
		if err := ctx.Err(); err != nil {
			return GenerateResult{}, err
		}
		logits, err = m.native.Forward(ctx, len(m.tokens), []int{next}, false)
		if err != nil {
			m.tokens, m.probRows = nil, nil
			return GenerateResult{}, err
		}
		m.tokens = append(m.tokens, next)
		if opts.PromptLogprobs {
			m.probRows = append(m.probRows, logprobRow{})
		}
		if opts.OnProgress != nil {
			opts.OnProgress("decode", len(m.tokens))
		}
	}
	result.Text = out.String()
	return result, nil
}
