# tinyoai

A pure-Go, OpenAI-compatible `/v1/chat/completions` server with a tiny embedded language model. The default uses only the standard library and a ~1MB model. It runs offline without cgo or an API key.

Use it to test OpenAI-compatible clients, such as `github.com/wbrown/openai`, against transformer inference, token accounting, and SSE streaming.

tinyoai also runs [Clio](#running-clio), NovelAI's 3-billion-parameter prose-completion model with an 8192-token context. Clio supports the Go CPU engine and an optional [native MLX backend](#running-with-native-mlx) for Apple Silicon GPUs. MLX also runs the embedded tiny Llama model and needs no Python at runtime.

## Reading the implementation

Start with [`doc.go`](doc.go), also available through `go doc .`. It explains tokenization, inference, cache updates, sampling, and serving. [Reading and maintaining tinyoai](docs/reading-code.md) describes the documentation conventions and checks.

## The model

The embedded model is Andrej Karpathy's **stories260K**, a ~260K-parameter Llama 2 trained on TinyStories for unit tests. It has 8 query heads and 4 key/value heads. The weights (`assets/stories260K.bin`, ~1MB) and tokenizer (`assets/tok512.bin`, ~6KB) are included with `go:embed`.

## CPU and MLX backends

The default CPU build is portable and uses only Go's standard library. Native MLX calls the MLX C API in the Go process; distribute its libraries and Metal shaders with the executable. Go handles tokenization, sampling, cache policy, cancellation, and HTTP for both backends.

| Capability | Go CPU | Native MLX |
| --- | --- | --- |
| Build and platform | Standard library; portable scalar default, optional Go SIMD | Apple Silicon, cgo, `-tags mlx`, MLX libraries and Metal shaders |
| Embedded Llama | `Default()`; legacy llama2.c checkpoints through `LoadModel` | `DefaultMLX()`; same checkpoints through `LoadLlamaMLX` |
| StableLM weights | Original BF16/F16/F32 safetensors or supported GGUF formats | Converted FP16 or affine group-64 2/3/4/5/6/8-bit MLX safetensors |
| Arithmetic | Float32 activations; StableLM projections use higher-precision accumulation | Float32 for legacy Llama; float16 activations for StableLM |
| StableLM KV storage | Float32 default, optional float16 | Float16 default, optional affine 8-bit |
| Prefix reuse | StableLM retains one reusable prefix; legacy Llama allocates per request | Both architectures retain one reusable prefix per instance; StableLM can fork sessions |
| Cache persistence | StableLM `SaveCache` / `LoadCache` and `-cache-file` | Memory only |
| HTTP | Chat/text completions, SSE, cancellation, token usage | Same endpoints and framing |

These Clio measurements used the same Apple M5 with 32 GiB memory in separate sessions. CPU rows are single eight-worker Go SIMD observations; MLX rows are medians of five trials. All start with empty KV and exclude model loading. MLX kernels were warmed first.

CPU used float32 KV; MLX used float16 KV. GGUF Q8_0 and MLX Q8 use different quantizers, and BF16 and FP16 use different weight representations. These forward timings exclude HTTP overhead and do not establish a controlled CPU/GPU speedup.

| Backend and weights | Prompt + output | Prefill seconds | Decode tokens/s |
| --- | --- | ---: | ---: |
| Go SIMD, GGUF Q8_0 | 512 + 256 | 26.656 | 2.61 |
| Native MLX, affine Q8 | 512 + 256 | 0.348 | 33.98 |
| Go SIMD, BF16 | 8129 + 64 | 614.494 | 1.65 |
| Native MLX, FP16 | 8129 + 64 | 6.660 | 14.54 |

See the [backend comparison](docs/backend-comparison.md) for FP16, Q8, Q6, and Q5 results at four context lengths, including memory and accuracy. GPU dispatch overhead can outweigh inference work for the tiny embedded model. The [native MLX guide](docs/mlx-native.md) covers setup and library use.

The newer [inference measurements](docs/inference-benchmarks.md) cover M3 Ultra, M5, M4, and A17 Pro with Q6 weights, Q8 KV, and prompt probabilities enabled. The M3 Ultra measured 1.517-second prefill and 114.80 decode tokens/s for a 4043-token prompt, and 3.293 seconds and 106.01 tokens/s for 8160 tokens. That report also covers KV memory savings, prefix reuse, cancellation, suffix branches, lower-bit weights, and batched offline evaluation.

## Inference features

Both CPU engines use KV during generation. Llama CPU allocates private KV for each request. StableLM CPU retains it across requests, reuses the longest unchanged token prefix, and supports disk snapshots.

| Feature | Llama CPU (`Model`) | StableLM CPU (`StableLM`) | Llama native MLX (`LlamaMLX`) | StableLM native MLX (`MLX`) |
| --- | --- | --- | --- | --- |
| KV during generation | Yes | Yes | Yes | Yes |
| Reuse across requests and edits | No | Yes | Yes | Yes |
| Prefill batch size | One token | Up to 128 tokens | Up to 64 tokens | Up to 512 tokens |
| Public prefill-only call | No | No | `Prefill` | `Prefill` |
| KV storage precision | Float32 | Float32 default; float16 at load time | Float32 | Float16 default; 8-bit per request |
| Disk KV snapshots | No | `SaveCache` / `LoadCache` | No | No |
| Streaming and cancellation | Yes | Yes | Yes | Yes |
| Extended sampling controls | Yes | Yes | Yes | Yes |
| Prompt/completion logprobs | No | No | Yes | Yes |
| Batched isolated token branches | No | No | No | `OpenBranches` |
| Independent generation forks | No | No | No | [`Fork`](docs/inference-features.md#independent-generation-sessions) |
| Ordinary PEFT LoRA adapters | No | No | No | [`LoadLoRA` / `SetLoRAScale`](docs/lora.md) |
| Protected-prefix prompt truncation | No | Yes | No | Yes |
| Progress callbacks | No | No | `OnProgress` | `OnProgress` |

Extended sampling includes temperature, top-k, top-p, tail-free sampling, and repetition, presence, and frequency penalties.

For prefix reuse, keep the model instance and submit the full updated prompt. Each instance retains one prefix. Native StableLM can fork that prefix into an independent session with shared base weights. `CachedPromptTokens` reports reused positions; HTTP usage exposes `prompt_tokens_details.cached_tokens` when nonzero.

The [inference feature guide](docs/inference-features.md) covers cache validity, memory, cancellation, sampling, probabilities, token branches, generation forks, LoRA, and HTTP access. It also lists the Python worker's limitations.

## Library use

```go
m, err := tinyoai.Default() // embedded stories260K
if err != nil {
    log.Fatal(err)
}

res, err := m.Generate("Once upon a time", tinyoai.GenerateOptions{
    MaxTokens:   64,
    Temperature: 0.8,
    Seed:        1, // reproducible
})
fmt.Println(res.Text, res.FinishReason, res.PromptTokens, res.CompletionTokens)
```

Load your own legacy llama2.c checkpoint with `LoadModel(checkpoint, tokenizer)`.

For native MLX, use `DefaultMLX()` or `LoadLlamaMLX(checkpoint, tokenizer)`, then `Close()` when finished. See [tiny Llama on MLX](docs/mlx-native.md#embedded-tiny-llama) for runtime setup and CPU/GPU precision settings.

For Clio on CPU, use `LoadStableLM`; see [Running Clio](#running-clio).

## As a server

In tests:

```go
s, err := tinyoai.NewDefaultServer()
if err != nil {
    t.Fatal(err)
}
srv := httptest.NewServer(s)
conv.SetEndpoint(srv.URL + "/v1/chat/completions")
```

Or as a standalone binary:

```bash
go run ./cmd/tinyoai -addr :8080
curl localhost:8080/v1/chat/completions -d '{
  "model": "stories260K",
  "messages": [{"role": "user", "content": "Once upon a time"}],
  "max_completion_tokens": 64
}'
```

The server supports streaming (`"stream": true`) with OpenAI-style `chat.completion.chunk` SSE events and a trailing usage chunk when `stream_options.include_usage` is set.

## Running with native MLX

On Apple Silicon, one command downloads and builds the pinned MLX 0.32.2 SDK, then packages the executable, native libraries, and shaders. Install Go, CMake, and Xcode with its Metal Toolchain component first. The [build guide](docs/mlx-native.md#build-and-run) covers prerequisites and using an existing SDK.

```sh
./scripts/build-mlx.sh
MLX_ENABLE_TF32=0 ./dist/mlx/tinyoai -mlx -addr 127.0.0.1:8080
```

That command serves the embedded `stories260K` model. For the larger StableLM model, first [convert the original checkpoint](docs/mlx-native.md#preparing-stablelm-weights), then select its directory:

```sh
./dist/mlx/tinyoai -mlx -model-dir ./models/clio-mlx-q8 \
  -addr 127.0.0.1:8080
```

The server advertises this model as `clio-accel`. To expose a CPU model alongside it, add `-cpu-model`:

```sh
./dist/mlx/tinyoai -mlx -model-dir ./models/clio-mlx-q8 \
  -cpu-model ./models/clio/clio-v1-legacy-Q8_0.gguf \
  -addr 127.0.0.1:8080
```

In another terminal, list the models and stream a completion:

```sh
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"clio-accel","prompt":"Once upon a time","max_tokens":64,"temperature":0,"stream":true}'
```

Use `"model":"clio-cpu"` to select CPU inference. Each model loads its own weights and retains its own prefix cache.

`-threads`, `-kv-cache`, and `-cache-file` configure the CPU backend. Native StableLM KV precision uses `GenerateOptions.KVBits` in library code or an installed completion extension.

## Optional application extensions

Install `WithCompletionExtension` to decode extra request fields, decorate results, or encode probability events. The server handles model selection, cancellation, and SSE framing. Add routes with `http.ServeMux`; `SelectModel` provides the completion endpoints' model routing and errors.

`Prefiller`, `TokenizerProvider`, `Brancher`, `Forker`, and `LoRAController` are optional interfaces. Close each token-branch session to release the parent's generation gate. Generation forks have their own gates; close each fork to release its cache and shared-weight references.

Native StableLM supports independent forks and per-session LoRA. The standalone command can expose registered adapters through the optional [`/v1/adapters` controls](docs/lora.md#server-controls). Adapter selection applies to subsequent requests on that model; it is not a completion parameter.

The `tokenizer` package provides Nerdstash encoding; `tokenizerinfo` describes its format and identity. `mlxruntime` provides native allocator statistics and cache release. `servercmd.Main` supplies model loading and HTTP lifecycle for commands with custom handlers.

## Running Clio

Clio uses a StableLM variant with shared LayerNorm, parallel attention/MLP residuals, split-half RoPE over 25% of each head, an untied output head, and Nerdstash BPE tokenization. `LoadStableLM` implements this architecture with the Go standard library.

Download the base model once (about 6.1 GB). For example, with Hugging Face's optional `hf` CLI:

```bash
hf download NovelAI/clio-v1-legacy \
  --revision 0ba2fa2dd57fe4b93f729c413c7fa80fc0cbae11 \
  --include config.json tokenizer.json model.safetensors.index.json 'model-*.safetensors' \
  --local-dir ./models/clio

go run ./cmd/tinyoai -model-dir ./models/clio -addr 127.0.0.1:8080
```

The CPU loader accepts the published BF16 safetensors shards directly and runs offline. With no `-model-dir`, the server uses the embedded tiny model.

The published Q8_0 GGUF reduces weight storage to about 3.24 GB. Keep the original `tokenizer.json` beside the GGUF to retain Clio's added-token whitespace behavior. No conversion is needed:

```bash
hf download NovelAI/clio-v1-legacy \
  --revision 1146e79da57e1f35d00e248bd5933fb57b27250c \
  --include clio-v1-legacy-Q8_0.gguf tokenizer.json \
  --local-dir ./models/clio

go run ./cmd/tinyoai -model-dir ./models/clio/clio-v1-legacy-Q8_0.gguf
```

The GGUF v3 loader supports F32, F16, BF16, Q8_0, Q6_K, Q5_K, and Q5_1 tensors. Published alternatives include `clio-v1-legacy-Q6_K.gguf` (2.64 GB) and `clio-v1-legacy-Q5_K_M.gguf` (2.26 GB).

The Q5 file mixes tensor formats. Some 7552-wide down projections use Q5_1 because their width does not fit a 256-weight K block.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "clio-cpu",
    "messages": [{"role": "user", "content": "Once upon a time"}],
    "temperature": 0,
    "max_completion_tokens": 16,
    "stream": true
  }'
```

Library use:

```go
m, err := tinyoai.LoadStableLM("./models/clio")
if err != nil {
    log.Fatal(err)
}
res, err := m.Generate("Once upon a time", tinyoai.GenerateOptions{
    MaxTokens: 16, Temperature: 0,
})
// Or pass m to tinyoai.NewServer(m).
```

The CPU engine retains weights in their file precision and uses float32 activations. Scalar matrix-vector products accumulate in float64; SIMD combines short float32 reductions in float64. RoPE preserves float32 rounding at the reference implementation's boundaries.

Float32 KV uses about 0.60 MiB per position, reaching 4.81 GiB at 8192 positions. Capacity grows in powers of two. Allow additional memory for the tokenizer, runtime, and scratch buffers.

CPU matrix and attention workers are bounded by `GOMAXPROCS`. `-threads N` sets that limit; zero keeps the runtime default or environment setting. Each attention head has independent scratch and output storage, preserving its reduction order.

CPU prefill evaluates 128-token batches; decode uses matrix-vector kernels. A loaded StableLM model retains one prefix and serializes requests to it. Completed prefill batches survive cancellation. See [cache reuse](docs/inference-features.md#kv-during-decoding-and-between-requests) for edit and tokenization behavior.

Use `-cache-file ./story.kv` to restore a CPU prefix at startup and save it on Ctrl+C or SIGTERM. An abrupt kill loses unsaved work. The library provides `SaveCache` and `LoadCache`; see [snapshot format and behavior](docs/inference-features.md#persisting-cpu-stablelm-kv).

With `-kv-cache f16`, retained KV falls from 4.81 to 2.41 GiB at full context. A reusable float32 layer workspace adds 0.17 GiB. Float32 remains the default; FP16 storage adds rounding error and conversion work. Library callers use `LoadStableLMWithOptions(path, StableLMOptions{KVCache: "f16"})`.

FP16 KV preserved every argmax in 512+256 Q8 and 8129+64 BF16 comparisons, with maximum KL below 1e-6 against the same weights and float32 KV. At 8K, peak RSS fell from 11.10 to 8.76 GB, while decode fell from 1.65 to 0.80 tok/s, mostly due to cache conversion.

For example, after building the SIMD server:

```bash
./tinyoai -model-dir ./models/clio/clio-v1-legacy-Q8_0.gguf \
  -threads 8 -kv-cache f16 -cache-file ./story.kv -addr 127.0.0.1:8080
```

Measure worker counts on the target machine; memory bandwidth and scheduling overhead limit scaling.

The CPU loader supports the documented Clio-compatible StableLM variant and tokenizer settings. It does not support NovelAI modules or LoRA. The native StableLM MLX backend supports [ordinary PEFT LoRA adapters](docs/lora.md).

These saved CPU validation timings used eight SIMD workers on an Apple M5 with 32 GiB memory. Other workloads were active. Decode rates exclude distribution checks and trace writes; peak RSS includes the test harness.

| Build | Prompt + output | Prefill seconds | Decode tok/s | Peak RSS (GB) |
| --- | --- | ---: | ---: | ---: |
| Previous Q8, batch 64 | 512 + 256 | 40.62 | 2.03 | 3.97 |
| Current Q8, batch 128 | 512 + 256 | 26.66 | 2.61 | 4.04 |
| Current Q6_K, batch 128 | 512 + 256 | 24.80 | 0.79 | 3.34 |
| Current Q5_K_M, batch 128 | 512 + 256 | 24.01 | 0.95 | 2.96 |
| BF16, matrix tuning only | 8129 + 64 | 1107.68 | 1.36 | 11.10 |
| BF16, current attention | 8129 + 64 | 614.49 | 1.65 | 11.10 |

The saved Q8, Q6, Q5, and long BF16 runs matched every greedy decision from their exact-weight PyTorch CPU float32 references. The long run reached position 8191. Maximum KL was approximately 4e-12 for Q8 and 3.53e-10 for long BF16. Independent NumPy audits checked all 65,536 logits per step.

The attention change preserved saved logits bit for bit at both lengths. Its attention-only benchmark improved by about 2.8× at 512 positions and 2.6× at 8192.

A Q8 cache test with a 513-token prompt took 26.10 seconds cold and 0.23 seconds on repeat, reusing 512 positions with identical logits. Saving the 324 MB snapshot took 1.52 seconds; restoring it took 1.19 seconds. These times exclude model loading. CLI save/restart checks passed for both KV precisions.

Quantization drift against the original BF16 weights, evaluated on 256 identical reference prefixes:

| Weights | Argmax agreement | Mean KL | Maximum KL |
| --- | ---: | ---: | ---: |
| Q8_0 | 253/256 | 0.000127 | 0.000801 |
| Q6_K | 253/256 | 0.000789 | 0.004642 |
| Q5_K_M | 252/256 | 0.002392 | 0.013584 |

Q6 had the same argmax count as Q8 but larger KL. Its three differing decisions involved candidates within 1.2 percentage points in the original distribution. Each format separately matched its exact-weight reference 256/256. These results cover one trajectory.

Q8 with float32 KV decoded fastest among the quantized CPU configurations measured. Lower-bit weights and FP16 KV reduced storage.

An optional SIMD build uses Go 1.27's experimental, portable `simd` package:

```bash
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd CGO_ENABLED=0 go build ./cmd/tinyoai
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test ./...
```

SIMD covers BF16/F32/Q8_0 matrix-vector products, batched projections, and attention. BF16 and Q8 weights remain packed in memory. Q8 decode extracts bytes in vector registers on 128/256-bit targets; wider vectors use tiled decoding.

Q6_K, Q5_K, and Q5_1 use tiled decoding with SIMD dot products. F16 matrix-vector operations and emulated SIMD targets use the scalar kernel. Prefill uses shared unpacking and batching. Scalar and SIMD builds undergo separate greedy and KL checks.

The build constraint is `goexperiment.simd && go1.27`, with a scalar fallback. Default builds retain the existing Go version requirement and have no external dependencies. The SIMD API is experimental; this project uses Go 1.27.1. See the [Go SIMD release notes](https://go.dev/doc/go1.27#simd).

Run `go test -run TestNumericalBackend -v` in the build environment to print the selected backend and hardware vector width.

Clio prompt counts include BOS (ID 2). Generation stops on EOS (ID 3), a stop string, the token limit, or the context limit. EOS counts toward usage but is omitted from text. Stop strings are withheld across token boundaries, and incomplete UTF-8 is buffered.

Text completions support `top_p` and presence/frequency penalties. The legacy chat adapter accepts `top_p` but ignores it. Other controls use the library or a completion extension; see [sampling and streaming](docs/inference-features.md#sampling-and-streaming).

### Verification

`go test ./...` uses independent fixtures for a small random StableLM model. Tests cover loading, tokenization, rotary embeddings, cached logits, generation, HTTP/SSE, batching, cancellation, KV precision, and snapshot corruption. GGUF decoding is checked against independent fixtures.

Fixtures contain no NovelAI weights or tokenizer data. Regenerate them with `scripts/stablelm_reference.py` and its listed dependencies. Python is required only for reference generation.

For the full checkpoint:

```bash
python scripts/clio_reference.py ./models/clio /tmp/clio-reference.json
TINYOAI_CLIO_DIR=./models/clio \
TINYOAI_CLIO_REFERENCE=/tmp/clio-reference.json \
  go test -run TestClioCheckpoint -v -timeout 20m
```

The reference requires about 12 GB for float32 weights. It uses CPU float32, eager attention, and cached one-token evaluation for both prefill and decoding. Full-prompt GEMM and cached GEMV have different reduction orders. Batched Go prefill is compared with the cached reference through greedy agreement and full-distribution KL.

For a longer comparison, generate a greedy reference and check Go's free-running trajectory. Each step measures KL(reference || Go) at temperature 1 over all 65,536 logits:

```bash
python scripts/clio_greedy_reference.py ./models/clio /tmp/clio-greedy \
  --prompt-tokens 512 --new-tokens 256
TINYOAI_CLIO_DIR=./models/clio \
TINYOAI_GREEDY_REFERENCE=/tmp/clio-greedy \
TINYOAI_PREFILL_BATCHED=1 \
TINYOAI_PARITY_REPORT=/tmp/clio-greedy/go-report.json \
TINYOAI_PARITY_TRACE=/tmp/clio-greedy/go-logits.safetensors \
  go test -run TestClioGreedyParity -v -timeout 2h
```

Repeat with `--prompt-tokens 8129 --new-tokens 64` and a new output directory to exercise position 8191. These full-checkpoint tests are opt-in. The test feeds back Go's own argmax and fails on token divergence or per-step KL above `1e-7`.

Reports include KL, logit differences, timings, and the last forward position. Stage timings separate attention, KV writes, projections, MLP, normalization, rotary, embedding, and the output head.

Omit `TINYOAI_PREFILL_BATCHED` for one-token prefill. For GGUF, generate a reference with `--gguf ./models/clio/clio-v1-legacy-Q8_0.gguf` and `gguf==0.17.1`, then point `TINYOAI_CLIO_DIR` at that file. The reference uses llama.cpp's Python GGUF decoder and PyTorch CPU float32 inference. Substitute the matching filename for Q6_K or Q5_K_M.

Set `TINYOAI_QUANTIZATION_COMPARISON=1` with the original reference to measure quantization drift. It feeds reference tokens back into inference, ensuring identical histories, and records mismatches without failing on lossy-weight differences. Keep these reports separate from exact-weight parity checks. Use `TINYOAI_KV_CACHE=f16` in the same mode to compare KV precision.

Decode timings exclude validation and trace writes. Stage profiling runs only in this checkpoint test. Reports record Go heap statistics after loading, prefill, and decode, plus process peak RSS on macOS/Linux. These include test-harness allocations.

`TINYOAI_PARITY_TRACE` saves float32 logits to safetensors and updates the row count after each step, retaining partial results on failure. The Python reference saves logits, token IDs, text, and runtime metadata. Neither test downloads or modifies weights.

Recheck saved distributions without model weights or inference:

```bash
python scripts/compare_saved_logits.py /tmp/clio-greedy \
  --go-logits go-logits.safetensors --report /tmp/clio-greedy/offline-audit.json
```

The offline audit calculates KL in float64 using `expm1`/`log1p` and verifies output argmax IDs. Use separate trace and report filenames for scalar and SIMD runs.

## Scope

stories260K is intended for client protocol tests: requests, responses, streaming, and token accounting. It has no chat training, tool calling, or vision. The server joins message text into a prompt and discards roles.

Clio provides local prose completion with separately downloaded weights, batched prefill, streaming, and reusable KV. It uses the same plain-text chat adaptation. Native MLX runs both architectures on Apple Silicon and adds runtime dependencies. The default CPU build remains pure Go. Loaders support only the documented architectures and formats.

## License

MIT. Bundles MIT-licensed material from `tmc/go-llama2`, `karpathy/llama2.c`, and `karpathy/tinyllamas` — see `LICENSE`.

Clio's separately downloaded weights and tokenizer are published by NovelAI under GPL-2.0-only, as stated in its [model card](https://huggingface.co/NovelAI/clio-v1-legacy). They are not bundled with tinyoai. The synthetic test fixtures are generated by this repository's reference script.
