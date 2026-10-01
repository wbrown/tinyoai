// Package mlxbench runs optional MLX experiments independently of inference policy.
package mlxbench

import "github.com/wbrown/tinyoai/tokenizerinfo"

// Chunk separates the transformer evaluation from vocabulary scoring.
// Timings include graph construction and blocking evaluation; no extra GPU
// synchronization is inserted into the ordinary forward pass.
type Chunk struct {
	// Prefix is the number of cached positions before this chunk.
	Prefix int `json:"prefix"`
	// Tokens is the number of newly evaluated input tokens.
	Tokens int `json:"tokens"`
	// TransformerSeconds measures graph construction and evaluation through
	// final hidden states or requested logits.
	TransformerSeconds float64 `json:"transformer_seconds"`
	// ProbabilitySeconds measures subsequent probability scoring and transfer;
	// without capture this is only post-evaluation bookkeeping.
	ProbabilitySeconds float64 `json:"probability_seconds"`
	// TotalSeconds measures the complete chunk up to the diagnostic callback.
	TotalSeconds float64 `json:"total_seconds"`
	// Logits is the last vocabulary row; observers must not modify it.
	Logits []float32 `json:"-"`
}

// Case selects one diagnostic configuration. Experimental knobs are
// opt-in and do not change the ordinary generation defaults.
type Case struct {
	// QMMFastLoads selects "stock" or "wide" patched kernel loads; empty
	// preserves the launch setting.
	QMMFastLoads string `json:"qmm_fast_loads,omitempty"`
	// Name is a unique filename-safe identifier for persisted results.
	Name string `json:"name"`
	// Context is the total position budget supplied to prompt preparation.
	Context int `json:"context"`
	// Chunk is the transformer prefill batch size, from 1 to 2048 tokens.
	Chunk int `json:"chunk"`
	// HeadBatch is the probability projection batch size, from 1 to 512 tokens.
	HeadBatch int `json:"head_batch"`
	// Rollover passes a prefix-replacement increment to the caller's prompt
	// policy; zero selects a fresh-prompt scenario.
	Rollover int `json:"rollover,omitempty"`
	// TopKBlock is the vocabulary block size for hierarchical selection; zero
	// uses full-vocabulary selection.
	TopKBlock int `json:"topk_block,omitempty"`
	// ReserveKV reserves prompt capacity before measuring instead of relying
	// only on incremental growth.
	ReserveKV bool `json:"reserve_kv,omitempty"`
	// FusedProjections combines Q/K/V and gate/up output rows into packed
	// projections.
	FusedProjections bool `json:"fused_projections,omitempty"`
	// DensePrefill temporarily dequantizes sufficiently large projections for
	// dense multiplication.
	DensePrefill bool `json:"dense_prefill,omitempty"`
	// AllocatorCacheMB caps idle allocator storage in MiB; zero selects the
	// platform default.
	AllocatorCacheMB int `json:"allocator_cache_mb,omitempty"`
	// CancelAfterMS requests cancellation this many milliseconds after the first
	// completed chunk, then measures resumption; zero disables it.
	CancelAfterMS int `json:"cancel_after_ms,omitempty"`
	// Model selects an entry in Config.Models; empty uses the
	// default directory.
	Model string `json:"model,omitempty"`
	// KVBits selects 8- or 16-bit KV; zero uses the diagnostic default of 8.
	KVBits int `json:"kv_bits,omitempty"`
	// CaptureTrace records a Metal capture and requires capture support enabled
	// before launch.
	CaptureTrace bool `json:"capture_trace,omitempty"`
	// CapturePrefix is the chunk-aligned cached position at which capture
	// starts.
	CapturePrefix int `json:"capture_prefix,omitempty"`
	// LayerBatch forces evaluation after this many layers; zero leaves the graph
	// unsegmented.
	LayerBatch int `json:"layer_batch,omitempty"`
	// ProfilePhases forces per-layer phase boundaries; these intrusive timings
	// need an uninstrumented control.
	ProfilePhases bool `json:"profile_phases,omitempty"`
	// QMMTile requires a runtime patched with the TINYOAI_QMM_* diagnostic knobs.
	// It applies to the legacy Metal QMM path, not M5 NAX kernels or decode.
	QMMTile string `json:"qmm_tile,omitempty"`
	// ExactPromptTokens selects an exact round-tripping prefix length; zero
	// leaves sizing to the prompt adapter.
	ExactPromptTokens int `json:"exact_prompt_tokens,omitempty"`
	// OutputRoot tests evaluation rooted only in requested outputs when present,
	// rather than also listing all KV arrays.
	OutputRoot bool `json:"output_root,omitempty"`
}

// Prompt is a diagnostic input prepared by the caller's context policy.
type Prompt struct {
	// Prompt is the final text to evaluate.
	Prompt string
	// Prefix is the protected leading text, if any.
	Prefix         string
	PreviousPrompt string // Optional retained prefix to seed before measuring Prompt.
	// PromptTokens is the adapter's token count including BOS, used for
	// reservation and capture bounds.
	PromptTokens int
}

// Config configures a dedicated, resumable diagnostic run with
// caller-owned prompt policy and output storage.
type Config struct {
	// PreparePrompt builds input from tokenizer metadata, a case, and the decode
	// length; the caller selects the input text and truncation policy.
	PreparePrompt func(tokenizerinfo.Spec, Case, int) (Prompt, error) `json:"-"`
	// MatrixScreen runs only the exact kernel-load guard comparison.
	MatrixScreen bool `json:"matrix_screen,omitempty"`
	// Cases runs in order, skipping compatible completed reports.
	Cases []Case `json:"cases"`
	// DecodeTokens is the number of logit decisions to compare, from 2 to 256;
	// zero selects 32.
	DecodeTokens int `json:"decode_tokens"`
	// Empty Model selects modelDir. Only one checkpoint is resident at a time.
	Models map[string]string `json:"models,omitempty"`
	// ReferenceCase names the case allowed to establish a missing reference;
	// empty requires the stock baseline settings.
	ReferenceCase string `json:"reference_case,omitempty"`
	// WarmupFull warms the entire prompt before a fresh measurement; otherwise
	// only the chunk shape is warmed.
	WarmupFull bool `json:"warmup_full,omitempty"`
	// RestSeconds pauses between warm-up and measurement using a cancellable
	// timer.
	RestSeconds int `json:"rest_seconds,omitempty"`
	// Wait for nominal thermal state before each case, bounded by this duration.
	CoolSeconds int `json:"cool_seconds"`
	// MemoryWarnings optionally returns the application's cumulative
	// memory-warning count, sampled before and after prefill.
	MemoryWarnings func() int `json:"-"`
	// Progress receives synchronous diagnostic status messages; nil discards
	// them.
	Progress func(string) `json:"-"`
}

// Phase is an intrusive diagnostic: each phase forces evaluation and
// releases dead handles. It includes dispatch/synchronization overhead, so its
// timings must be compared with an ordinary uninstrumented control.
type Phase struct {
	// Prefix is the number of cached positions before the chunk.
	Prefix int `json:"prefix"`
	// Tokens is the chunk length.
	Tokens int `json:"tokens"`
	// Layer is the zero-based transformer layer.
	Layer int `json:"layer"`
	// Name identifies the forced evaluation boundary.
	Name string `json:"name"`
	// Seconds includes graph construction, dispatch, and synchronization since
	// the previous boundary.
	Seconds float64 `json:"seconds"`
}
