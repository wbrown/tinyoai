# Native MLX inference measurements

Saved Go/native MLX runs cover device throughput, KV precision, prefix reuse, cancellation, suffix branches, and batched evaluation. The [portable results](benchmarks/inference-2026-10.json) retain individual timings, measurement settings, and source-report hashes. The earlier [CPU/MLX comparison](backend-comparison.md) remains a separate FP16-KV, forward-only series.

## Prefill and decode across Apple Silicon

The October 1–3, 2026 runs use the same Clio affine group-64 Q6 weights and group-64 Q8 KV. Prefill includes top-12 prompt probabilities, using 512-token transformer chunks, 64-row scoring batches, and 1024-entry top-k blocks. Model loading and full-prompt warm-up are excluded. Each run starts with empty KV, records 32 output decisions, and times the 31 decode forwards after prefill.

Times and rates are medians of three runs. Peak process memory is the largest physical-footprint sample, taken every 100 ms; child processes are excluded. GB is decimal. The allocator's idle pool is capped at 256 MiB on macOS and 32 MiB on iOS, separately from the retained KV cache.

```text
| Device   | Prompt tokens | Prefill s | Prefill tok/s | Decode tok/s | Peak process GB |
|----------|---------------|-----------|---------------|--------------|-----------------|
| M3 Ultra | 4043          | 1.517     | 2664.6        | 114.80       | 4.608           |
| M5       | 4043          | 5.587     | 723.7         | 23.18        | 4.210           |
| M4       | 4043          | 9.993     | 404.6         | 28.59        | 4.037           |
| A17 Pro  | 4043          | 24.471    | 165.2         | 14.09        | 3.895           |
| M3 Ultra | 8160          | 3.293     | 2477.7        | 106.01       | 5.375           |
| M5       | 8160          | 11.510    | 708.9         | 24.17        | 5.097           |
| M4       | 8160          | 24.058    | 339.2         | 23.64        | 4.932           |
| A17 Pro  | 8160          | 61.299    | 133.1         | 11.42        | 4.949           |
```

The M3 Ultra is the 80-GPU-core, 512 GiB Mac Studio; the M5 is a 32 GiB MacBook Air. The M4 is an iPad Pro 13-inch, and the A17 Pro is an 8 GiB iPad mini. The saved OS versions are macOS 26.7.1, macOS 26.6.2, iPadOS 27.0, and iPadOS 27.0.1, respectively.

The engine snapshot is `f6928c30b252a5564bf178ac426ff089fbae6bf3`, with MLX 0.32.2. The Macs use the same preserved executables and precompiled Metal library; iOS uses JIT kernels with device-specific tuning. The JSON records the MLX and MLX-C revisions, runtime hashes, checkpoint identity, and prepared-prompt hashes.

All 480 recorded M3 Ultra decisions match their M5 reference. The M4 and A17 Pro each match all 288. Counts include repeated 32-decision histories across context, rebuild, and cancellation cases; they are not hundreds of distinct free-running continuation positions. Full-vocabulary KL compares the same Q6 checkpoint across devices, independently of weight-quantization error.

### Allocator cache size

Changing the idle buffer pool affects allocation reuse without changing model weights or KV precision. The two Macs were also measured at the same 32 MiB cap as the iPads:

```text
| Device   | Prompt tokens | Idle pool MiB | Prefill s | Decode tok/s |
|----------|---------------|---------------|-----------|--------------|
| M3 Ultra | 4043          | 32            | 1.467     | 122.59       |
| M3 Ultra | 4043          | 256           | 1.517     | 114.80       |
| M5       | 4043          | 32            | 8.086     | 16.43        |
| M5       | 4043          | 256           | 5.587     | 23.18        |
| M3 Ultra | 8160          | 32            | 3.155     | 110.40       |
| M3 Ultra | 8160          | 256           | 3.293     | 106.01       |
| M5       | 8160          | 32            | 18.122    | 13.16        |
| M5       | 8160          | 256           | 11.510    | 24.17        |
```

These observations show why allocator settings belong in a performance report. The larger cap helped the M5 in this series; it did not help the M3 Ultra. They do not imply a universal best cap.

## KV precision and memory

A separate September 29 comparison used Q6 weights on the A17 Pro, one instrumented run per case, a 32 MiB idle pool, and FP16 versus Q8 KV. Prompts of 4065 and 8161 tokens plus 32 decisions filled exactly 4096 and 8192 cached positions. The timings predate the device series above.

```text
| Context | KV   | Retained KV GB | End process GB | Peak process GB | Prefill s | Decode tok/s | Warnings |
|---------|------|----------------|----------------|-----------------|-----------|--------------|----------|
| 4096    | FP16 | 1.292          | 3.861          | 4.343           | 21.70     | 10.55        | 0        |
| 4096    | Q8   | 0.686          | 3.245          | 3.797           | 23.36     | 10.73        | 2        |
| 8192    | FP16 | 2.584          | 5.137          | 5.438           | 68.10     | 4.90         | 24       |
| 8192    | Q8   | 1.373          | 3.932          | 4.833           | 61.11     | 8.07         | 1        |
```

At 8192 positions, Q8 KV retains 1,211,105,280 fewer bytes than FP16, a 46.875% reduction. Peak process savings are smaller because prefill reconstructs one layer's dense KV for attention. There is no permanently retained FP16 copy. Warning counts describe those individual runs; both formats completed and produced identical 32-token text at each tested length.

The corresponding Mac comparison forced the FP16-cache token history through Q8 KV and compared all 65,536 logits. These are KL(FP16 KV || Q8 KV), with the same Q6 weights:

```text
| Prompt + decisions | Argmax  | Mean KL        | Maximum KL     |
|--------------------|---------|----------------|----------------|
| 512 + 256          | 255/256 | 0.000037751243 | 0.000327546488 |
| 8129 + 64          | 64/64   | 0.000024639329 | 0.000580654378 |
```

The short history has one changed argmax; this does not promise 256 identical freely generated tokens. Native library callers select `GenerateOptions.KVBits`; zero still means FP16. See [cache precision](inference-features.md#cache-precision-capacity-and-memory).

## Prefix reuse and cancellation

The October device series also rebuilt prepared prompts after keeping a 20-token prefix of an existing 8K cache. Each rebuild was measured once with the matched 32 MiB idle pool. A separate interrupted rebuild recorded cancellation and resumed from its completed KV and probability rows.

```text
| Device   | 7158-token rebuild s | 6123-token rebuild s | Cancel ms | KV at cancel | Reused on resume |
|----------|----------------------|----------------------|-----------|--------------|------------------|
| M3 Ultra | 2.887                | 2.376                | 120.3     | 1044         | 1043             |
| M5       | 14.773               | 7.918                | 819.9     | 1044         | 1043             |
| M4       | 19.488               | 16.089               | 1104.4    | 1044         | 1043             |
| A17 Pro  | 54.371               | 42.566               | 2361.7    | 1044         | 1043             |
```

Deleting a prefix-adjacent section shifts the later absolute positions, so only the unchanged leading tokens can be reused. The 7158-token retained prompt takes more prefill than the 6123-token prompt. Cancellation times are observed chunk-completion delays, not latency bounds. All devices retained 1044 KV positions and reused 1043 after boundary retokenization.

## Shared-prefix suffix branches

Eight candidate seeds were expanded from a prefix of 8189 tokens after filling the 8192-position cache. Timings include the Go expansion loop over native inference and exclude model loading, prefill, and dictionary lookup. Each depth limit of one, two, and three was repeated three times, but all candidates reached a boundary after one actual expansion step.

```text
| Device   | Serial ms | Batched ms | Extra active MiB | Max logprob delta |
|----------|-----------|------------|------------------|-------------------|
| M3 Ultra | 79.2      | 26.9       | 154.2            | 0.002396733       |
| M5       | 418.0     | 143.0      | 118.9            | 0.006156584       |
| M4       | 370.3     | 143.2      | 118.9            | 0.002396733       |
| A17 Pro  | 779.2     | 369.5      | 79.5             | 0.004762297       |
```

Medians cover those nine calls. Candidate IDs, text, token counts, and boundary flags agree between serial and batched expansion. The table reports the largest accumulated logprob difference and additional active MLX allocation. Active allocation returned to its starting value after every call. These measurements do not establish latency for three-step expansions. See [token branches](inference-features.md#probability-capture-and-token-branches).

## Batched independent evaluation

This section reports the development evaluator; its batching and training APIs have not landed on `main`. The October 4 Studio measurements use Q6 weights, a fixed LoRA checkpoint, FP16 KV, and 512-token prefill chunks. Each panel contains 32 requests: sixteen prompts of 129 tokens and sixteen of 1025 tokens, each generating 64 tokens. Times include prefill and decode, with model and adapter already loaded; they are medians of three runs.

```text
| Batch limit | Largest group | Greedy s | Sampled s | Greedy texts identical | Sampled texts identical |
|-------------|---------------|----------|-----------|------------------------|-------------------------|
| 1           | 1             | 21.094   | 21.175    | 32/32                  | 32/32                   |
| 4           | 4             | 11.621   | 11.672    | 29/32                  | 9/32                    |
| 8           | 8             | 10.145   | 10.222    | 28/32                  | 6/32                    |
| 16          | 16            | 9.674    | 9.756     | 30/32                  | 8/32                    |
| 32          | 16            | 9.670    | 9.757     | 30/32                  | 8/32                    |
```

A limit of 16 reduces the greedy panel from 21.094 to 9.674 seconds, about 2.18 times the throughput. The requested limit of 32 still ran groups of 16 because batching groups exact prompt lengths. The sampled panel uses temperature one, top-k 25, tail-free sampling 0.925, and fixed seeds; the complete penalty settings are saved in the JSON.

Identical seeds do not guarantee identical text when batch shapes change FP16 reductions. The exact-text counts above were the same in all three repetitions. A separate comparison on fixed token histories used four sequences with 16 decisions each, batch size four, and 256-token prefill chunks. All 64 argmax decisions agreed at each tested prompt length:

```text
| Prompt tokens | Argmax | Mean KL        | Maximum KL     |
|---------------|--------|----------------|----------------|
| 512           | 64/64  | 0.000015130701 | 0.000419108357 |
| 4096          | 64/64  | 0.000000110342 | 0.000004264883 |
| 8176          | 64/64  | 0.000000049885 | 0.000001778300 |
```

The 8176-token case reaches the final supported positions. These forced-history checks isolate numerical differences; the free-generation counts show their possible effect on trajectories.

In that development implementation, `TrainingEvaluationOptions.BatchSize` enables this offline evaluator path, with separate KV, random state, sampling history, and stopping conditions for each request. `Generate` and the HTTP server still serialize requests per model instance; they do not dynamically combine arriving requests into batches.

### Reusing prompts while scoring alternatives

For 32 saved recall fixtures, repeated fresh-cache scoring took 70.358 seconds. Retaining prompt KV and final logits reduced it to 31.063 seconds; batching alternative suffixes reduced it to 28.541 seconds. These are single aggregate measurements over the fixed panel, using 256-token prefill chunks.

Prefix reuse preserved token logprobs exactly. Batched suffixes preserved the winning offered answer in every case; their maximum token-logprob difference was 0.024020678. Greedy diagnostic text and top-five traces were checked separately and remained identical. This measures scoring cost, not an improvement in factual accuracy.

## Lower-bit weights

The earlier M5 affine group-64 sweep includes Q4, Q3, and Q2 as well as Q5, Q6, and Q8. Each checkpoint was quantized directly from the same FP16 conversion. The short history is 512 prompt tokens plus 256 decisions; the long history is 8129 plus 64. Every case uses FP16 KV.

```text
| Weights | Weight files GB | Short peak active GB | Long peak active GB | Short argmax | Long argmax |
|---------|-----------------|----------------------|---------------------|--------------|-------------|
| Q8      | 3.234           | 3.745                | 6.141               | 254/256      | 64/64       |
| Q6      | 2.473           | 3.041                | 5.456               | 249/256      | 64/64       |
| Q5      | 2.093           | 2.659                | 5.077               | 236/256      | 63/64       |
| Q4      | 1.712           | 2.287                | 4.694               | 201/256      | 62/64       |
| Q3      | 1.332           | 1.912                | 4.315               | 59/256       | 13/64       |
| Q2      | 0.952           | 1.620                | 3.934               | 0/256        | 0/64        |
```

Peak active allocation includes weights, KV, and temporaries; it excludes idle allocator buffers and other process memory. The [full numerical comparison](backend-comparison.md#numerical-agreement) includes mean and maximum KL against the original PyTorch CPU float32 reference. Q4 matches 201/256 short-history argmax decisions, compared with 249/256 for Q6. Q3 and Q2 lose substantially more agreement on these histories.

All 36 quantized Go traces in this sweep match their same-format Python MLX references bit for bit. That separates implementation parity from drift caused by the weight format. These affine bit widths are not equivalent to GGUF K-quants.

## Evidence and reproduction

[The JSON record](benchmarks/inference-2026-10.json) stores each series separately with individual runs, source hashes, and its measurement protocol. No inference was rerun for this documentation update. The device runs include prompt scoring and Q8 KV; the older [M5 backend comparison](backend-comparison.md#measurement-conditions) excludes prompt scoring and uses FP16 KV. Compare like settings before interpreting a timing change.

For new measurements, use the native [prefill experiment runner](mlx-native.md#implementation-and-validation) with `PreparePrompt`, or `TestMLXMatchedProbe` for forward-only reference histories. Keep the checkpoint, prepared tokens, warm-up, KV precision, scoring, allocator cap, runtime, and number of trials in the saved report. The development evaluator uses `EvaluateLoRA` and `TrainingEvaluationOptions`; the measured adapter and request-panel hashes are retained with this series.
