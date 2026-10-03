# Inference features

The [feature matrix](../README.md#inference-features) lists support by backend. CPU StableLM provides KV reuse, batched prefill, and disk persistence. MLX backends expose prefill-only calls. Probability capture requires native MLX; independent generation forks and LoRA require native StableLM MLX.

## KV during decoding and between requests

A KV cache stores attention keys and values for evaluated tokens. All four engines use one. Each new token attends to cached positions without reevaluating earlier tokens through the transformer. Attention still reads the history, so longer context can increase decode cost.

CPU `Model` allocates private activation and KV buffers per `Generate` call, allowing concurrent requests against immutable weights. Each call starts empty.

`StableLM`, `LlamaMLX`, and native `MLX` retain one prefix per instance and serialize requests to it. Keep the instance alive to reuse that cache.

A retained-cache request tokenizes the full prompt and reuses the longest identical token prefix. Repeats, appends, and edits reuse positions before the first changed token. BPE segmentation can change at an edit boundary, so the engine compares token IDs.

The final prompt token is reevaluated to obtain next-token logits. Native prompt-probability capture may also need to restore missing score rows.

Once a prefill chunk completes, every position in it has valid KV. An edit inside that chunk can reuse its unchanged prefix. An interrupted, incomplete chunk is excluded from reuse.

Generated tokens become reusable after their forward passes complete. The final sampled token may not yet have KV.

KV depends on preceding tokens and absolute position. Deleting text or shifting a suffix requires reevaluating that suffix; an unchanged leading prefix can remain cached. The engine has no sliding-window KV transform or per-session cache pool.

KV and saved probabilities also depend on the effective model weights. Loading, unloading, or changing a LoRA's strength clears that session's prefix and probability records, even when prompt tokens remain identical.

`GenerateResult.CachedPromptTokens` counts reused prompt positions. HTTP usage includes `prompt_tokens_details.cached_tokens` when positive. For streaming usage, set `stream_options.include_usage`. StableLM counts include BOS; legacy Llama counts exclude it.

## Cache precision, capacity, and memory

KV precision is independent of weight precision. CPU StableLM defaults to float32 KV. Select float16 with `LoadStableLMWithOptions(path, StableLMOptions{KVCache: "f16"})` or `-kv-cache f16`.

Computation stays float32 and uses a reusable layer workspace. Conversion overhead can reduce throughput despite the smaller cache.

For native StableLM, `GenerateOptions.KVBits` accepts 8 for affine group-64 KV or 16 for float16. Zero selects FP16 on every request. Changing precision discards the retained prefix.

Legacy Llama uses float32 KV on both backends. CPU engines reject nonzero `KVBits`; CPU StableLM precision is selected at load time.

Dense KV requires approximately `2 × layers × KV heads × head width × allocated positions × bytes per element`. At 8192 Clio positions, that is 4.81 GiB in float32 or 2.41 GiB in float16, excluding workspaces and runtime overhead.

Native 8-bit KV includes group scales and biases. Prefill reconstructs one layer for attention; no dense copy is retained permanently.

CPU StableLM grows capacity in powers of two up to the model limit. Native caches grow in 256-position blocks. Rewind shortens the valid prefix but usually retains capacity. A smaller context setting can release an oversized cache.

Legacy CPU Llama allocates for the checkpoint's full context per request. Lowering `ContextLength` does not shrink that allocation.

`mlxruntime.Memory` reports active, peak, and idle native buffers separately from Go's heap. `ClearCache` frees idle buffers. `ResetPeakMemory` also clears idle buffers before resetting the peak. Neither releases live weights or KV; `Close` releases the model's retained arrays. See [memory accounting](backend-comparison.md#native-mlx-observations).

## Independent generation sessions

Native StableLM implements `Forker`. `Fork(ctx)` snapshots the currently retained prefix and returns a `GenerationSession` with independent token history, saved probability records, adapter selection, and request gate. Fork creation waits for the parent's active work; that wait can cancel.

The child retains separate native handles to shared base weights, adapter tensors, and KV. Go token and probability slices are copied immediately. A later write to shared KV can copy a layer's full cache, so a diverged fork may retain another complete KV allocation. Budget for that growth using the cache sizes above.

Pass the full desired prompt to the child's `Generate`, with the same KV precision to preserve reuse. Parent and child can then accept independent requests, though native command submission still uses the process-wide MLX lock. Each session remains usable after the other closes. Call `Close` on every fork to release its retained handles.

`OpenBranches` holds the parent's generation gate while exploring a batch of short suffixes. `Fork` releases the parent gate after the snapshot and returns a normal generator for ongoing use. CPU and native Llama do not support forks. Python-backed `MLX` values expose `Fork` but return an unsupported-backend error.

## Per-session LoRA adapters

Native StableLM implements `LoRAController`: `LoadLoRA` attaches one ordinary PEFT adapter, `SetLoRAScale` changes its strength, and `LoRAInfo` reports its identity and retained tensor bytes. CPU, native Llama, and Python inference do not support adapter execution.

Adapters add low-rank corrections to attention and MLP projections while leaving the base weights unchanged. Zero strength bypasses the correction but retains the adapter tensors. Loading an empty directory unloads them. Successful changes invalidate only the receiving session's cache; failed loads preserve its previous adapter and prefix.

A fork inherits the parent's adapter and strength, sharing immutable adapter storage. Either session can then select a different adapter or strength. Changing the child requires fresh prefill there because its inherited KV was computed with different effective weights. Closing or changing one session leaves the other's adapter usable.

See [native LoRA inference](lora.md) for supported tensor formats, scaling, replacement semantics, and a fork-and-select example. Adapter selection does not supply a prompt template or change stop strings.

## Persisting CPU StableLM KV

`StableLM.SaveCache(path)` saves completed token IDs and KV. `LoadCache(path)` checks checkpoint identity, precision, and checksum before replacing the resident cache. Failed loads leave it intact.

Both methods serialize with generation. Call them after a request returns, never from a callback holding the same model gate.

Snapshots contain prompt-derived data and use mode 0600. Saving writes a temporary file and atomically renames it. The standalone server enables persistence with:

```sh
go run ./cmd/tinyoai -model-dir /path/to/checkpoint \
  -kv-cache f32 -cache-file ./prefix.kv
```

The server restores the snapshot at startup and saves on Ctrl+C or SIGTERM. Abrupt termination loses work since the last save. Restore still requires reading the weights and snapshot, but skips cached prefill. Legacy CPU Llama and native MLX do not persist KV to disk.

## Prefill, cancellation, and concurrency

CPU StableLM prefill batches contain up to 128 tokens; native Llama uses 64 and native StableLM uses 512. All decode one token at a time. Each batch belongs to one prompt; independent requests are not combined.

`LlamaMLX` and `MLX` implement `Prefiller`, preparing a prefix without sampling. Success returns `FinishReason: "prefill"` and zero completion tokens.

CPU `Model` and `StableLM` have no public `Prefill` method. CPU StableLM batches prefill inside `Generate`. `MaxTokens: 0` generates to the context limit; it does not request prefill only.

Pass cancellation through `GenerateOptions.Context`; HTTP adapters use the request context. CPU StableLM checks between layers and forwards and retains completed batches. Native MLX finishes the current forward chunk and records its KV and scores before stopping.

Legacy CPU Llama checks between token iterations and retains no state afterward. Cancellation returns an error, even if callbacks have already delivered text.

Queued requests can cancel while waiting for a model gate. Later requests resume from the remaining valid prefix. Native MLX also serializes command submission through a process-wide lock.

Callbacks are synchronous and must not reenter a gated model. MLX backends emit `OnProgress` after completed prefill and decode forwards; CPU engines do not.

## Prompt limits and preparation

`ContextLength` sets the total position budget; zero uses the checkpoint limit. `MaxTokens` bounds output within that budget. StableLM rejects oversized prompts unless truncation is enabled.

`TruncatePrompt` keeps BOS and the newest tokens that fit the output reservation. `PromptPrefix` protects leading text. Truncation uses token boundaries.

StableLM's `ContextTokenizer` exposes immutable assets and the context limit. Supply `ExpectedPromptTokens` and `TokenizerID` to validate a prepared prompt. It must fit its output reservation and is never silently truncated.

Native Llama checks a positive expected count excluding BOS, but rejects protected-prefix truncation and tokenizer identity options. Legacy CPU Llama ignores preparation fields and stops at the context limit.

## Sampling and streaming

All backends use the Go sampler. Temperature zero selects greedily; positive temperature samples with the request seed. Start extended settings from `DefaultSampling()` because the zero value is invalid:

```go
sampling := tinyoai.DefaultSampling()
sampling.TopK = 40
sampling.TopP = 0.95
sampling.TFS = 0.95
sampling.RepetitionPenalty = 1.1
sampling.RepetitionRange = 512
sampling.RepetitionSlope = 1
sampling.PresencePenalty = 0.1
sampling.FrequencyPenalty = 0.05
opts := tinyoai.GenerateOptions{
    MaxTokens:   64,
    Temperature: 0.8,
    Seed:        1,
    Sampling:    &sampling,
}
```

Penalties apply first. At temperature zero, the sampler chooses the best penalized score. Otherwise it applies temperature, top-k, top-p, and tail-free filtering before drawing a token.

All backends reject negative output limits and negative or non-finite temperatures before inference. For direct generation calls, `MaxTokens: 0` uses the remaining context. HTTP adapters supply their documented defaults.

A fixed seed does not guarantee identical text across weight precisions or prefill shapes: their arithmetic can produce different probabilities.

`OnToken` emits decoded text, which may span multiple vocabulary tokens. StableLM buffers incomplete UTF-8 and possible stop prefixes, withholding matched stop text. Legacy Llama detects stop text after emitting it, so it remains in the result. HTTP streaming preserves these behaviors.

## Probability capture and token branches

Native Llama and StableLM accept `Logprobs` from 1 to 64 alternatives per token. `PromptLogprobs` additionally scores prompt tokens and requires positive `Logprobs`. Results include IDs, text, byte offsets, and scores; `OnLogprobs` emits events.

Scores describe the full-vocabulary distribution at temperature one, before penalties or filters. The retained shortlist is not renormalized. CPU engines reject probability requests.

Prompt scoring adds output projections and retained score rows. Enabling it after unscored prefill, or requesting more alternatives than were saved, can force reevaluation. Byte-fallback tokens can have empty display text while retaining IDs and scores.

Native StableLM implements `Brancher`. `OpenBranches` validates a saved prefix fingerprint and seed tokens from a retained probability row. Each `Forward` advances one token per branch and returns full-vocabulary logits.

Branches share a read-only prefix and own their suffix storage. Close the session to release that storage and the model gate before generating again. Native Llama and CPU backends do not support branches.

## Library features and the base HTTP API

Text completions accept `temperature`, `top_p`, presence/frequency penalties, `seed`, `stop`, and backend-supported completion `logprobs`. The legacy chat adapter ignores `top_p`. Neither endpoint supplies chat templates or tool execution.

Both endpoints limit request bodies to 4 MiB and share SSE delivery. Every frame has a 30-second write deadline on supported transports. Write or flush failures cancel generation; failed opening frames never start inference. Middleware can expose transport capabilities through `Unwrap() http.ResponseWriter`.

Per-request KV precision, prompt logprobs, protected-prefix truncation, top-k, tail-free sampling, and repetition controls require a `CompletionExtension` to expose them as JSON fields.

Prefill-only calls, tokenizer assets, token branches, and fork creation need caller-provided HTTP routes. A library caller can also register a fork under its own model ID with `NewModelServer`. See [extension hooks](../README.md#optional-application-extensions).

The standalone command enables optional `GET` and `POST /v1/adapters` controls when started with registered `-lora name=directory` flags. Clients select registered names rather than paths. Selection applies to the resident model across completion requests; it is not a standard OpenAI request field. Both the endpoint queue and the model queue allow cancellation. See [server controls](lora.md#server-controls).

## Optional Python MLX worker

`LoadMLX` starts a persistent Python StableLM worker. It supports prefix reuse, 512-token prefill, `Prefill`, shared sampling, and progress callbacks. Native probability capture, selectable KV precision, token branches, generation forks, and LoRA controls are unavailable.

Cancelling an active exchange terminates the worker and discards its cache. The next request restarts it. See [Python worker setup](mlx-native.md#optional-python-worker).
