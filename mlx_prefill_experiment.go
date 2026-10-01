//go:build mlx && darwin && arm64 && cgo

package tinyoai

import (
	"context"
	"encoding/json"
	mx "github.com/wbrown/tinyoai/internal/mlx"
	"github.com/wbrown/tinyoai/internal/mlxbench"
	"github.com/wbrown/tinyoai/tokenizerinfo"
	"io"
	"strings"
)

// MLXPrefillChunk records transformer and probability timing for one chunk.
type MLXPrefillChunk = mlxbench.Chunk

// MLXPrefillPhase records an intrusive per-layer evaluation boundary.
type MLXPrefillPhase = mlxbench.Phase

// MLXPrefillCase selects one diagnostic configuration.
type MLXPrefillCase = mlxbench.Case

// MLXPrefillPrompt is input supplied by the caller's context policy.
type MLXPrefillPrompt = mlxbench.Prompt

// MLXPrefillExperiments configures a dedicated diagnostic run and its inputs.
type MLXPrefillExperiments = mlxbench.Config

// RunMLXPrefillExperiments runs resumable prefill diagnostics in a dedicated
// process. The internal benchmark package owns reports, input identity,
// telemetry, and scheduling. Kernel overrides are restored on return.
func RunMLXPrefillExperiments(ctx context.Context, modelDir, out string, config MLXPrefillExperiments) error {
	return mlxbench.Run(ctx, modelDir, out, config, loadMLXBenchmark)
}

// mlxBenchmark lends controls to a runner that exclusively owns its model.
// Native handles stay inside the engine; reports only receive Go data.
type mlxBenchmark struct {
	m *MLX
	n *nativeMLX
}

// loadMLXBenchmark opens a private native backend for diagnostics.
func loadMLXBenchmark(dir string) (mlxbench.Backend, error) {
	m, err := LoadMLXNative(dir)
	if err != nil {
		return nil, err
	}
	return &mlxBenchmark{m: m, n: m.native.(*nativeMLX)}, nil
}

// Tokenizer returns the checkpoint's immutable tokenizer specification.
func (b *mlxBenchmark) Tokenizer() tokenizerinfo.Spec { return b.m.ContextTokenizer() }

// Encode returns checkpoint token IDs including BOS.
func (b *mlxBenchmark) Encode(text string) []int { return b.m.tokenizer.encode(text) }

// Decode renders IDs using the same byte-fallback rules as generation.
func (b *mlxBenchmark) Decode(ids []int) string {
	d := nerdstashDecoder{tokenizer: b.m.tokenizer}
	var text strings.Builder
	for _, id := range ids {
		text.WriteString(d.add(id))
	}
	text.WriteString(d.flush())
	return text.String()
}

// Configure applies native experiment settings while no request is active.
func (b *mlxBenchmark) Configure(c mlxbench.Case, bits int) error {
	n := b.n
	if _, err := n.SetKVBits(bits); err != nil {
		return err
	}
	b.m.prefillChunk, n.probabilityBatch = c.Chunk, c.HeadBatch
	n.layerBatch, n.outputRoot = c.LayerBatch, c.OutputRoot
	n.topKBlock, n.reserveTokens = c.TopKBlock, 0
	n.fusedProjections, n.densePrefill = c.FusedProjections, c.DensePrefill
	n.capturePath = ""
	if c.FusedProjections {
		return n.packProjections()
	}
	return nil
}

// Reset discards retained prompt state and native KV between measurements.
func (b *mlxBenchmark) Reset() error {
	b.m.tokens, b.m.probRows, b.m.probCount = nil, nil, 0
	return mx.Run(b.n.reset)
}

// Reserve sets a capacity reservation before measured prefill.
func (b *mlxBenchmark) Reserve(tokens int) { b.n.reserveTokens = tokens }

// Capture selects the path and prefix of a single Metal capture.
func (b *mlxBenchmark) Capture(path string, prefix int) {
	b.n.capturePath, b.n.capturePrefix = path, prefix
}

// Observe installs synchronous instrumentation; callbacks must not reenter MLX
// because evaluation holds the native lock.
func (b *mlxBenchmark) Observe(chunk func(mlxbench.Chunk), phase func(mlxbench.Phase)) {
	b.n.onPrefillChunk, b.n.onPrefillPhase = chunk, phase
}

// Prefill uses the ordinary generation policy with twelve prompt alternatives.
func (b *mlxBenchmark) Prefill(ctx context.Context, prompt string, o mlxbench.PrefillOptions) (int, error) {
	r, err := b.m.Prefill(prompt, GenerateOptions{Context: ctx, ContextLength: o.ContextLength, MaxTokens: o.MaxTokens,
		KVBits: o.KVBits, Logprobs: 12, PromptLogprobs: true, PromptPrefix: o.Prefix, OnProgress: o.OnProgress})
	return r.CachedPromptTokens, err
}

// Warmup evaluates only the chosen chunk and probability-head shapes.
func (b *mlxBenchmark) Warmup(ctx context.Context, ids, targets []int) error {
	_, _, err := b.n.ForwardWithLogprobs(ctx, 0, ids, targets, 12)
	return err
}

// Forward evaluates forced tokens independently of the sampling loop.
func (b *mlxBenchmark) Forward(ctx context.Context, prefix int, ids []int) ([]float32, error) {
	return b.n.Forward(ctx, prefix, ids, true)
}

// CacheSize reports only fully committed Go prefix state.
func (b *mlxBenchmark) CacheSize() (int, int) { return len(b.m.tokens), len(b.m.probRows) }

// WriteProbabilities serializes saved rows without lending mutable cache slices.
func (b *mlxBenchmark) WriteProbabilities(w io.Writer) error {
	return json.NewEncoder(w).Encode(b.m.probRows)
}

// WeightBits reports the checkpoint's weight precision.
func (b *mlxBenchmark) WeightBits() int { return b.n.bits }

// Close releases the exclusively owned backend.
func (b *mlxBenchmark) Close() error { return b.m.Close() }
