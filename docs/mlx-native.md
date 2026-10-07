# Native MLX inference

Native MLX runs the embedded Llama and local MLX-format StableLM checkpoints on Apple Silicon. Go calls MLX's C API in-process; runtime Python is unnecessary. The default build remains pure Go.

MLX uses the Metal GPU, not the Neural Engine. Go manages layers, tokenization, sampling, requests, and prefix reuse. See the [CPU/MLX measurements](backend-comparison.md) and [server commands](../README.md#running-with-native-mlx).

The [inference benchmark report](inference-benchmarks.md) adds M3 Ultra, M5, M4, and A17 Pro timings, KV precision and memory, cancellation, token branches, and batched offline evaluation. Its portable JSON preserves individual results and measurement settings.

The [feature guide](inference-features.md) covers cache validity, cancellation, sampling, probability capture, token branches, and independent generation forks.

StableLM also supports [native LoRA adapters](lora.md), with one adapter per session, adjustable strength, and optional server controls for switching without reloading base weights.

## Embedded tiny Llama

Run the embedded stories260K model without a model directory:

```sh
MLX_ENABLE_TF32=0 ./tinyoai -mlx -addr 127.0.0.1:8080
```

The API advertises `stories260K` with chat/text completions and SSE. Weights and tokenizer are embedded; native libraries and shaders must accompany the binary. `scripts/build-mlx.sh` builds and packages them.

Use `DefaultMLX()` or `LoadLlamaMLX(checkpoint, tokenizer)` for legacy llama2.c weights. Pass the result to `NewServer` or call `Generate`. `Prefill` prepares a prefix without sampling. Call `Close` when finished. Unsupported builds return an error.

Llama uses RMSNorm, sequential residuals, grouped-query attention, SwiGLU, and adjacent-pair rotary tables from the checkpoint. Weights, activations, and KV are float32.

Set `MLX_ENABLE_TF32=0` before startup for closest agreement with CPU arithmetic. MLX can otherwise use reduced-precision float32 matrix multiplication. This is a process-wide setting. Reduction order still differs, so logits need not be bit-identical.

The Llama cache grows in 256-position blocks and reuses matching prefixes. Prefill batches contain up to 64 tokens. Generation supports streaming, cancellation/resumption, context limits, seeded sampling, and logprobs.

The 512-token context needs 640 KiB of FP32 KV, excluding runtime and temporary buffers. Quantized KV and disk persistence are unsupported. The 8-wide heads do not fit group-64 quantization. GPU dispatch overhead can outweigh inference work for this small model.

With the native build environment configured, run the opt-in Metal validation:

```sh
TINYOAI_TEST_METAL=1 MLX_ENABLE_TF32=0 \
  TINYOAI_LLAMA_OUTPUT=/path/to/test-artifacts \
  go test -tags mlx -run '^TestLlamaMLX' -v .
```

This checks all 512 logits against sequential Go inference at 128+256 and 480+32 positions, including the final cache slot. It saves both traces and JSON reports. Tests also cover generation, edits, cancellation, probabilities, and HTTP/SSE. They use embedded weights.

## Build and run

Run the build on an Apple Silicon Mac with Go, CMake 3.25 or newer, and Xcode providing macOS SDK 26.2 or newer. Install Xcode's Metal Toolchain component too; the standalone Command Line Tools are insufficient for compiling shaders. The first build needs network access to fetch dependencies and, if absent, Go 1.27.1.

The script downloads pinned MLX 0.32.2 and MLX-C sources, builds the SDK, and packages the API executable, libraries, shaders, licenses, and checksums. It uses Go 1.27.1 with experimental SIMD and targets macOS 26.2 or later. Neither the native build nor the packaged server needs Python.

```sh
./scripts/build-mlx.sh
cd dist/mlx
MLX_ENABLE_TF32=0 ./tinyoai -mlx
# Or load an MLX-format StableLM checkpoint:
./tinyoai -mlx -model-dir /path/to/model
```

The default output is `dist/mlx/`. To rebuild after changing Go code, choose a new output directory with `./scripts/build-mlx.sh ./dist/mlx-next`. Existing releases are not overwritten. Download additional checkpoints separately.

Sources and incremental native builds stay under `.build/mlx/`, which Git ignores. The installed runtime SDK is `.build/mlx/runtime-sdk/`; the static C API shim and its headers are in `.build/mlx/sdk/`. Repeated builds reuse both. Set `BUILD_JOBS=8` to change the default four compiler jobs. Interrupted builds can be rerun unless they already wrote the release executable; in that case, use a new output directory.

The script honors `DEVELOPER_DIR`. If the selected tools lack Metal, it tries `/Applications/Xcode.app/Contents/Developer` without changing the system's Xcode selection. For another installation, run `DEVELOPER_DIR=/path/to/Xcode.app/Contents/Developer ./scripts/build-mlx.sh`.

To supply an existing MLX 0.32.2 SDK, keep the two-argument form:

```sh
./scripts/build-mlx.sh /path/to/mlx-sdk ./dist/mlx-custom
```

That SDK must contain headers, `libmlx.dylib`, `libjaccl.dylib`, `mlx.metallib`, and `share/cmake/MLX/`. The script builds the matching C API shim against it. Modified dependency checkouts are never reset; move them aside or supply a separately built SDK when experimenting with MLX changes.

Keep `lib/` beside the executable. StableLM accepts FP16 or affine group-64 2/3/4/5/6/8-bit MLX weights with float16 activations. GGUF is supported by the CPU loader only.

## Preparing StableLM weights

Native MLX expects FP16 weights or packed affine weights with FP16 scales, biases, and normalization tensors. Convert the original BF16 safetensors download from [Running Clio](../README.md#running-clio) into a separate directory. These commands use MLX-LM 0.31.3 and MLX 0.32.2 on Apple Silicon:

```sh
python3 -m venv .venv-convert
.venv-convert/bin/python -m pip install 'mlx==0.32.2' 'mlx-lm==0.31.3'
.venv-convert/bin/python -m mlx_lm.convert \
  --hf-path ./models/clio --mlx-path ./models/clio-mlx-q8 \
  --dtype float16 -q --q-bits 8 --q-group-size 64 --q-mode affine
cp ./models/clio/tokenizer.json ./models/clio-mlx-q8/tokenizer.json
```

Use a separate output directory for each precision. Change `--q-bits` to 2, 3, 4, 5, or 6 for other supported formats. For FP16, omit `-q` and the three `--q-*` options. Keep `--dtype float16` and copy the original tokenizer.

Python is needed for conversion only. MLX Q5/Q6/Q8 are affine group-64 formats, distinct from GGUF quantizers.

## Library use and cache lifetime

Build callers with cgo, `-tags mlx`, and the SDK include/link flags from the build script. Load once, reuse the model, and close it when finished:

```go
m, err := tinyoai.LoadMLXNative("./models/clio-mlx-q8")
if err != nil {
    log.Fatal(err)
}
defer m.Close()

result, err := m.Generate("Once upon a time", tinyoai.GenerateOptions{
    MaxTokens:   64,
    Temperature: 0,
    KVBits:      8,
    OnToken:     func(text string) { fmt.Print(text) },
})
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.CompletionTokens, result.CachedPromptTokens)
```

`KVBits: 8` selects quantized StableLM KV. Zero selects FP16, including after an 8-bit request, so pass the same precision to preserve a prefix. Weight precision is set by the checkpoint.

`Prefill` prepares KV without generating. `Logprobs` scores completions; `PromptLogprobs` also scores prompt tokens. Scoring adds work beyond the comparison's forward timings.

Each model serializes requests. MLX command submission also uses a process-wide lock. Cancellation preserves completed chunks and probability rows. Synchronous callbacks must not reenter the same model.

`Close` releases weights and KV. `mlxruntime` reports active, peak, and idle allocations separately from Go's heap. Native KV is memory-only. See [cache lifetime](inference-features.md#cache-precision-capacity-and-memory).

Native StableLM also implements `Forker`. `Fork(ctx)` returns a `GenerationSession` with independent token history, probability records, KV position, adapter selection, and request gate. Base weights, adapter tensors, and initial KV share storage through separately retained handles. Closing either session leaves the other usable.

Forking is unavailable for CPU, Python, and native Llama; the Python-backed `MLX.Fork` returns an unsupported-backend error. The [LoRA guide](lora.md#selecting-an-adapter-on-a-fork) shows how to change a child's adapter while preserving the parent's state. Changing the child's effective weights requires rebuilding its KV.

Forking itself does not copy tensor storage. Subsequent writes to shared KV can copy a layer's complete cache, so a long-lived fork can retain another full KV allocation. It is not a suffix-only branch. Call `Close` on each fork when no longer needed. `TestMLXForkIsolation`, enabled with `TINYOAI_MLX_DIR`, checks both KV precisions, cache growth, rewind, and parent/child lifetimes on Metal.

## Optional Python worker

`LoadMLX(python, directory)` or `-mlx-python /path/to/python` starts a persistent MLX-LM worker. Install the versions listed in `scripts/mlx_worker.py` in that interpreter. Go handles tokenization, sampling, and HTTP; the worker evaluates local weights. This path requires Python and MLX at runtime, but no cgo link or `mlx` build tag.

The worker supports generation and prefix reuse, but lacks probability capture, configurable KV precision, token branches, generation forks, and LoRA controls. Cancellation during an exchange kills the worker and discards its prefix. The next request restarts it.

## Implementation and validation

`internal/mlx` wraps tensor operations with explicit array ownership and scoped cleanup. Serialized streams let Go goroutines move between OS threads. Native exceptions become Go errors. Cancellation takes effect after the current chunk, at most 512 prefill tokens, preserving completed work.

StableLM follows MLX-LM's parallel residuals, shared LayerNorm, partial RoPE, and compiled SwiGLU. KV capacity grows in 256-position blocks. Sampling and tokenization share Go code with the CPU backend.

Prompt scoring uses 64-token output batches. Hierarchical top-k first selects candidates within 1024-entry vocabulary blocks, then selects the requested alternatives. Decode retains its one-token output shape. Cancellation completes score rows for the current chunk so its KV remains reusable.

`RunMLXPrefillExperiments` tests batch sizes, top-k blocks, KV reservation, projection packing, temporary dense weights, allocator limits, and cancellation/resumption. Callers supply input through `PreparePrompt`. Cases save logits, score rows, timing, memory, and reuse results in the selected output directory.

The runner lives in `internal/mlxbench`; a small engine adapter supplies inference and instrumentation. The public entry point and configuration types remain unchanged. Checkpoint hashing, loading, and warm-up run outside the measured prefill interval.

Resumption checks checkpoint contents, prepared prompt and prefix, executable, GPU, kernel overrides, and run settings. Required result files must still exist. Changed inputs or reports without an identity require a new output directory; existing data is never overwritten by a resume mismatch. Use a new directory when replacing external MLX libraries or shaders.

Default inference uses 512-token transformer chunks, 64-token scoring batches, and blocked top-k. Other experiment settings are opt-in.

`QMMTile` and `QMMFastLoads` require an MLX patch implementing `TINYOAI_QMM_BM`, `TINYOAI_QMM_BN`, `TINYOAI_QMM_BK`, and `TINYOAI_QMM_FAST_LOADS`. Stock MLX ignores these variables. The runner restores their prior values on return.

Eight-bit KV uses affine groups of 64 with FP16 scales and biases. Prefill dequantizes each layer for scaled-dot-product attention. Decode uses packed K/V matmuls and precise softmax, following [MLX-LM quantized attention](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/base.py). No dense copy is retained permanently.

`TestMLXKVProbe` compares FP16 and 8-bit KV on matching histories at 512+256 and 8129+64. Set `TINYOAI_MLX_DIR`, `TINYOAI_REFERENCE_ROOT`, and `TINYOAI_KV_OUTPUT`. The FP16 run supplies the greedy history; the Q8 run follows it.

Reports include every logit, argmax agreement, KL, first differing decision, KV allocation, and timings. GPU tests also cover cache growth, rewind, and precision changes without reloading weights.

For `TestMLXNativeParity`, set `TINYOAI_MLX_DIR` to the converted model, `TINYOAI_REFERENCE_ROOT` to references, and `TINYOAI_MLX_OUTPUT` to a new artifact directory. It checks 65,536-way distributions at 512+256 and 8129+64, including position 8191 and cache rewind.

Set `TINYOAI_MLX_PYTHON` to include Python-worker greedy comparisons. Reports retain logits, per-step KL, IDs, and timings.

Compare sampling with identical cache histories. Batched prefill and a cached one-token forward round differently in FP16, which can change seeded output. Native and Python paths agree for matching cold and warm histories.
