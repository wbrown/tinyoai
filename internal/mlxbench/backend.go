package mlxbench

import (
	"context"
	"github.com/wbrown/tinyoai/tokenizerinfo"
	"io"
)

// Backend exposes measurements and controls without exposing native array
// ownership. A runner owns one backend exclusively until Close.
type Backend interface {
	// Tokenizer describes encoding and the context limit.
	Tokenizer() tokenizerinfo.Spec
	// Encode returns IDs including BOS.
	Encode(string) []int
	// Decode renders IDs with the checkpoint decoder.
	Decode([]int) string
	// Configure selects experimental forward settings and KV precision.
	Configure(Case, int) error
	// Reset discards tokens, probability rows, and native KV.
	Reset() error
	// Reserve selects retained KV capacity for subsequent forwards.
	Reserve(int)
	// Capture selects one chunk for a Metal trace.
	Capture(string, int)
	// Observe installs synchronous callbacks; nil disables them.
	Observe(func(Chunk), func(Phase))
	// Prefill evaluates with probability capture and returns reused tokens.
	Prefill(context.Context, string, PrefillOptions) (int, error)
	// Warmup evaluates one chunk without retaining Go prefix state.
	Warmup(context.Context, []int, []int) error
	// Forward evaluates forced tokens and returns the last vocabulary row.
	Forward(context.Context, int, []int) ([]float32, error)
	// CacheSize reports completed tokens and saved probability rows.
	CacheSize() (int, int)
	// WriteProbabilities encodes saved rows to a caller-owned writer.
	WriteProbabilities(io.Writer) error
	// WeightBits reports checkpoint precision.
	WeightBits() int
	// Matrix evaluates a named projection with deterministic test inputs.
	Matrix(context.Context, string, int, int) ([]float32, error)
	// Close releases the model and native resources.
	Close() error
}

// PrefillOptions carries only the generation policy exercised by the runner.
// Probability capture is always enabled with twelve retained alternatives.
type PrefillOptions struct {
	// ContextLength sets the total position budget.
	ContextLength int
	// MaxTokens reserves output positions when preparing the prompt.
	MaxTokens int
	// KVBits selects cache precision.
	KVBits int
	// Prefix protects leading prompt text.
	Prefix string
	// OnProgress receives completed prefill chunks.
	OnProgress func(string, int)
}
