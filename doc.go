// Package tinyoai runs small language-model inference and exposes an optional
// OpenAI-compatible HTTP interface. Its default model is embedded, runs in pure
// Go, and needs neither network access nor external dependencies. Larger local
// StableLM checkpoints and an optional native MLX backend use the same
// generation interface.
//
// # Reading the program
//
// Start with [Default], [GenerateOptions], and [Model.Generate] in embed.go and
// model.go. Together they show the complete inference cycle: encode a prompt,
// evaluate its tokens, choose a next token, emit text, and repeat. The compact
// legacy Llama implementation is the baseline explanation of that cycle.
//
// Read stablelm.go next for the second architecture. Its layer normalizes the
// input once, computes attention and the gated MLP from that same input, and
// adds both outputs to the residual. Llama instead applies attention and MLP
// sequentially, with a separate RMS normalization before each. StableLM also
// uses partial split-half rotary embeddings and a separate output matrix;
// Llama rotates adjacent pairs using tables stored in the checkpoint and ties
// its output matrix to the embedding table. These are architectural differences,
// not interchangeable optimizations.
//
// The remaining files answer progressively narrower questions:
//
//   - prompt.go and the tokenizer package establish which tokens are evaluated.
//   - prefix_cache.go and kv_cache.go describe CPU cache reuse and storage.
//   - prefill.go, matmul.go, dot.go, and attention.go explain batched and
//     one-token arithmetic; scalar and SIMD files implement the same boundaries.
//   - safetensors.go, gguf.go, and quant_k.go interpret local weight containers.
//   - sampling.go and logprobs.go turn logits into choices and observations.
//   - mlx.go and mlx_native.go preserve the generation cycle while delegating
//     tensor evaluation to MLX. mlx_llama.go does the same for legacy Llama.
//   - sessions.go and mlx_sessions.go explain how independent generation
//     sessions retain a prefix without copying its tensor storage immediately.
//   - lora.go and mlx_lora.go attach low-rank corrections to a session and
//     invalidate activations when its effective weights change.
//   - server.go, completions.go, and models.go adapt generation to HTTP.
//     servercmd/lora.go adds optional controls for registered local adapters.
//
// # A forward pass predicts the following token
//
// Evaluating a token at position p writes that position's keys and values and
// produces logits for position p+1. Prefill evaluates known prompt tokens;
// decoding samples from the last logits and feeds the selected token into the
// next pass. Consequently N output choices need only N-1 additional forwards
// after prefill. The last returned token need not occupy a KV slot yet.
//
// Log probabilities describe the unmodified model distribution at temperature
// one, before repetition penalties, temperature, or filtering. Retaining only
// the top alternatives does not restrict normalization: all vocabulary logits
// contribute. Prompt score row p therefore describes prompt token p+1.
//
// # Cache ownership and cancellation
//
// [Model] has immutable weights and allocates independent state for each call.
// [StableLM], [MLX], and [LlamaMLX] instead retain one token prefix and KV cache
// per instance. Their request gates serialize generation, prefill, and lifecycle
// operations. Callbacks run synchronously; a callback must not call back into
// the same gated model. Separate registered models have separate request gates.
//
// A recorded prefix contains only fully evaluated positions. CPU StableLM
// commits completed prefill batches; native MLX returns an admitted chunk and
// its probability rows even if cancellation arrives during GPU evaluation.
// The caller records that work before checking cancellation at the next
// boundary. A partially computed or failed suffix is never advertised as valid.
// The Python worker uses a different boundary: cancellation terminates it and
// discards its cache so unread protocol bytes cannot contaminate the next call.
//
// Reuse is based on identical token IDs, not identical character counts. An edit
// can change BPE segmentation before the edited character. The last prompt token
// is evaluated again to recover logits; prompt-probability capture also revisits
// the row predicting the first changed token. Rewinding changes the valid length,
// while capacity can remain allocated. Attention must see only the valid prefix.
// Changing KV precision or reducing an oversized retained context invalidates
// the corresponding cache. Loading, unloading, or changing the strength of an
// adapter also invalidates KV and saved probabilities: the same tokens now
// pass through different effective weights.
//
// CPU StableLM KV is [layer][KV head][position][feature]. Legacy CPU Llama KV
// is [layer][position][KV head][feature]. Native arrays add a batch axis and use
// [batch, KV head, position, feature] per layer. A capacity change changes the
// head stride in head-major storage, so growing KV requires preserving each
// head's prefix separately. Tests exercise growth, rewind, and final positions.
// [StableLM.SaveCache] persists only valid CPU positions with a model identity
// and checksum; native MLX caches are currently memory-only.
//
// # Independent sessions and adapter state
//
// Native StableLM [MLX.Fork] snapshots a session's retained prefix, probability
// records, and adapter selection. The child has its own request gate and can
// generate after the snapshot releases the parent's gate. [Brancher] instead
// holds the parent gate for a temporary batch of suffix explorations; closing
// that branch session lets generation on the parent resume.
//
// A fork retains its own native handles to shared base weights, adapter tensors,
// and KV. The Go token and probability slices are copied. MLX can copy a layer's
// full shared KV buffer on a later write, so a diverged fork may retain another
// full cache. The caller must close each [GenerationSession]; closing a parent
// or child leaves the other's handles valid. Native submission still shares the
// process-wide MLX lock, so separate sessions do not promise parallel GPU work.
//
// [LoRAController] changes a session's projections without modifying base
// weights. Each adapted projection adds a pair of low-rank matrix products,
// scaled by strength times alpha/rank. Zero strength bypasses the correction.
// Native StableLM stores adapter tensors in FP16 alongside dense or quantized
// base weights; CPU, native Llama, and Python inference do not support adapters.
//
// Replacement is prepared before it is committed. The loader validates and
// evaluates new tensors while retaining the old adapter and cache. On success,
// it replaces the adapter and clears native KV before Go discards its token and
// probability records. Failure before commit preserves the old state. Changes
// affect only the owning session; a fork with a different adapter must rebuild
// its own KV. See docs/lora.md for format limits and a library example.
//
// # Precision and native lifetime
//
// CPU StableLM retains packed weights and computes float32 activations. Scalar
// projections accumulate in float64; SIMD projections combine short float32
// reductions in float64. Batched evaluation and GPU kernels can use different
// reduction orders, so correctness is assessed through matched histories,
// greedy choices, and full-vocabulary divergence as well as local kernel tests.
// Weight quantization and KV quantization are separate sources of approximation.
//
// Native MLX arrays are lazy graphs with explicit reference ownership. The
// internal/mlx arena owns temporary handles; a retained weight or cache handle
// must be released separately. Releasing graph-building handles before
// evaluation lets MLX reuse buffers. A process-wide MLX lock protects native
// error handling and command submission; it is not reentrant. OS telemetry is
// sampled independently. See docs/mlx-native.md for build and runtime details.
//
// # Serving and extending
//
// [Generator] is the minimal serving contract. [Prefiller], [TokenizerProvider],
// [Brancher], [Forker], and [LoRAController] describe optional capabilities.
// Native StableLM supports forks and LoRA; Python-backed [MLX] values expose
// these methods but return unsupported-backend errors.
// [NewModelServer] gives generators explicit model IDs. [WithCompletionExtension]
// adds text-completion request and response policy while the server retains
// routing, cancellation, and streaming. Additional HTTP handlers compose with
// the server through net/http; extensions are optional for every backend.
//
// The standalone command can register local adapters and expose /v1/adapters.
// Its cancellable gate keeps selection metadata consistent with model changes;
// the model gate also serializes those changes with inference. Selection applies
// to a resident model across requests, so clients cannot bind an adapter change
// and a later completion into one atomic request through this endpoint.
//
// The HTTP interface is deliberately a subset: chat messages are flattened into
// text, roles do not add a template, and tool execution is not implemented.
// Legacy Llama and StableLM also differ in BOS/EOS accounting and stop-text
// emission; consult [Model.Generate] and [StableLM.Generate] when comparing them.
//
// # Verification as part of the explanation
//
// Ordinary tests use the embedded model and small independent fixtures. Tests
// next to each subsystem state the invariant they check. Full-checkpoint and
// Metal comparisons are opt-in and retain logits and measurement reports.
// Free-running parity detects trajectory divergence; forced-history comparisons
// isolate quantization error after a differing token choice. These answer
// different questions and should not be conflated.
//
// This reading order follows Knuth's emphasis on explaining a program to a
// human reader, while keeping ordinary Go source and its toolchain. It is not
// a WEB/CWEB document with a separate weaving or tangling step. Maintenance
// conventions and verification commands are in docs/reading-code.md.
package tinyoai
