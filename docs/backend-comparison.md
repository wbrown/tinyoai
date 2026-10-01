# Comparing Go CPU and native MLX

The CPU backend uses only Go's standard library. Native MLX adds Apple Silicon GPU inference through a C API, requiring native libraries and shaders. See [backend capabilities](../README.md#cpu-and-mlx-backends) and [MLX setup](mlx-native.md).

The [feature guide](inference-features.md) covers cache reuse, persistence, cancellation, sampling, probabilities, and branching.

These measurements use the 3-billion-parameter Clio StableLM checkpoint. They do not predict stories260K performance, where GPU dispatch can cost more than the small matrix operations.

## What changes in the inference loop

Both StableLM backends share Go tokenization, sampling, and stop handling. CPU uses custom matrix and attention loops, optional SIMD, and worker parallelism. It prefills 128 tokens at a time with float32 activations.

Native MLX builds tensor graphs for Metal kernels, using 512-token prefill chunks and float16 activations. Both decode one token at a time.

Prefill reuses weights across many positions. Decode produces one position per forward and is more sensitive to weight traffic, KV length, and dispatch cost.

In these CPU runs, Q6_K and Q5_K_M saved storage but decoded more slowly than Q8_0. MLX lower-bit weights generally improved decode while prefill stayed similar. Results depend on the kernels and hardware.

Each StableLM model retains one prefix and serializes requests. MLX also has a process-wide submission lock. Neither backend combines concurrent requests into inference batches. CPU StableLM supports disk snapshots; native MLX keeps KV in memory.

## Measurement conditions

Hardware: Apple M5 with 32 GiB unified memory. CPU rows are individual Go 1.27.1 SIMD runs with eight workers, batch 128, and float32 KV.

MLX rows summarize five processes per configuration from September 28, 2026, using Go 1.27.1 and MLX 0.32.2. Settings were batch 512, float16 KV, and a 256 MiB idle allocator cache limit. Tables report median timings and throughput, plus maximum observed peak active allocation.

The CPU and MLX runs used separate source snapshots and sessions on an active desktop. Competing workloads and swapping affected accepted trials; decode ranges show the variation. Exact speedup ratios and close rankings need a controlled rerun.

GGUF Q8_0/Q6_K/Q5_K_M and MLX affine Q8/Q6/Q5 use different quantizers. Their weight representations are not equivalent.

All prefill starts with empty KV. MLX trials first warm the full prompt shape and eight decode forwards, then clear KV and idle buffers. CPU runs lack that matched warmup.

Timers exclude model loading, tokenization, sampling, prompt scoring, HTTP, and trace writes. MLX evaluation and logit transfer to Go are timed. Prefill produces the first output distribution: 256 decisions require 255 decode forwards; 64 require 63.

## CPU observations

Peak RSS includes the CPU test process and its validation allocations. GB below means decimal billions of bytes.

| Weights | Prompt + output | Prefill seconds | Prefill tokens/s | Decode tokens/s | Peak RSS GB |
| --- | --- | ---: | ---: | ---: | ---: |
| GGUF Q8_0 | 512 + 256 | 26.656 | 19.2 | 2.61 | 4.044 |
| GGUF Q6_K | 512 + 256 | 24.798 | 20.6 | 0.79 | 3.340 |
| GGUF Q5_K_M | 512 + 256 | 24.007 | 21.3 | 0.95 | 2.965 |
| BF16 | 8129 + 64 | 614.494 | 13.2 | 1.65 | 11.101 |

## Native MLX observations

MLX Q5/Q6/Q8 use affine group-64 weights converted from FP16. Every row uses FP16 KV. Q8 KV was not measured in this series.

| Weights | Prompt + output | Prefill seconds | Prefill tokens/s | Decode tokens/s | Decode range | Peak active MLX GB |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| FP16 | 512 + 256 | 0.351 | 1457.0 | 20.33 | 13.23–20.61 | 6.458 |
| Q8 | 512 + 256 | 0.348 | 1471.6 | 33.98 | 28.36–37.39 | 3.745 |
| Q6 | 512 + 256 | 0.368 | 1392.8 | 37.81 | 31.63–44.80 | 3.041 |
| Q5 | 512 + 256 | 0.375 | 1364.0 | 45.40 | 34.20–51.09 | 2.659 |
| FP16 | 2048 + 64 | 1.502 | 1363.7 | 18.66 | 14.23–19.20 | 6.987 |
| Q8 | 2048 + 64 | 1.601 | 1279.5 | 32.21 | 18.67–32.67 | 4.257 |
| Q6 | 2048 + 64 | 1.537 | 1332.3 | 37.70 | 28.37–38.46 | 3.571 |
| Q5 | 2048 + 64 | 1.597 | 1282.7 | 38.04 | 32.36–42.85 | 3.186 |
| FP16 | 4096 + 64 | 2.986 | 1371.7 | 17.29 | 10.26–17.64 | 7.725 |
| Q8 | 4096 + 64 | 3.170 | 1292.2 | 27.43 | 12.62–27.78 | 4.942 |
| Q6 | 4096 + 64 | 3.603 | 1136.8 | 30.75 | 17.29–32.04 | 4.272 |
| Q5 | 4096 + 64 | 3.327 | 1231.2 | 33.04 | 18.24–35.40 | 3.893 |
| FP16 | 8129 + 64 | 6.660 | 1220.5 | 14.54 | 7.42–14.86 | 8.936 |
| Q8 | 8129 + 64 | 7.031 | 1156.2 | 20.64 | 12.48–21.86 | 6.141 |
| Q6 | 8129 + 64 | 7.299 | 1113.7 | 21.31 | 14.44–23.96 | 5.456 |
| Q5 | 8129 + 64 | 7.274 | 1117.6 | 24.03 | 15.13–26.09 | 5.077 |

Peak active MLX includes live weights, KV, and temporaries. It excludes idle buffers, Go's heap, and other allocations. RSS does not account for all Metal unified-memory use, so the CPU and MLX memory columns cannot establish a total-RAM saving. All 80 trials returned active MLX allocations to zero after closing.

Dense KV at 8192 positions requires about 4.81 GiB in float32 or 2.41 GiB in float16, independent of weight precision. CPU FP16 KV also needs a float32 workspace. Native Q8 KV reduces retained storage with scales, biases, and packed values; it adds quantization error and conversion work. See [KV validation](mlx-native.md#implementation-and-validation).

## Numerical agreement

CPU Q8_0, Q6_K, and Q5_K_M each matched all 256 greedy decisions from an independently decoded, exact-weight PyTorch CPU float32 reference. BF16 matched all 64 decisions through position 8191. These checks validate inference against each format's weights; quantization drift requires comparison with the original checkpoint.

The MLX table compares against the original checkpoint's PyTorch CPU float32 reference in cached one-token mode. Every distribution uses the reference token history. KL(reference || MLX) covers all 65,536 entries at temperature one.

Results include FP16 arithmetic differences and weight quantization error. Argmax counts measure individual decisions; they do not measure agreement between freely generated passages.

| MLX weights | Prompt + output | Argmax agreement | Mean KL | Maximum KL |
| --- | --- | ---: | ---: | ---: |
| FP16 | 512 + 256 | 255/256 | 0.000003566763 | 0.000020547252 |
| Q8 | 512 + 256 | 254/256 | 0.000140222156 | 0.000932272132 |
| Q6 | 512 + 256 | 249/256 | 0.002880635091 | 0.024466845076 |
| Q5 | 512 + 256 | 236/256 | 0.009879625755 | 0.059590967449 |
| FP16 | 8129 + 64 | 64/64 | 0.000000868282 | 0.000029212052 |
| Q8 | 8129 + 64 | 64/64 | 0.000049726349 | 0.001460507242 |
| Q6 | 8129 + 64 | 64/64 | 0.000813665507 | 0.014327933753 |
| Q5 | 8129 + 64 | 63/64 | 0.001903803885 | 0.024004135237 |

All 40 short/long traces matched their same-precision Python MLX references bit for bit. The 2048- and 4096-token trials used slices of the long history and had no independent CPU reference, so only their performance appears here. Accuracy results cover fixed histories rather than broad model quality.

## Evidence and reproduction

[apple-m5.json](benchmarks/apple-m5.json) contains unrounded summaries, method metadata, and source-report hashes: four CPU observations, 16 MLX configurations from 80 trials, and eight accuracy summaries. Full reports and logits are retained separately.

Follow [CPU verification](../README.md#verification) to generate reference histories. With native build flags configured, measure one MLX trial with:

```sh
TINYOAI_MLX_DIR=/path/to/converted-model \
TINYOAI_MATCHED_REFERENCE=/path/to/reference.json \
TINYOAI_MATCHED_OUTPUT=/path/to/new-trial \
  go test -tags mlx -run '^TestMLXMatchedProbe$' -count=1 -v .
```

The history needs `prompt_ids` and `generated_ids`, as written by `scripts/clio_greedy_reference.py`. Use a fresh process and output directory per trial. Rotate model order and hold history, precision, scoring settings, and thermal conditions constant.

`TestMLXMatchedProbe` saves logits on reference histories for independent comparison. It does not test free-running greedy agreement. Measure actual generation or HTTP latency separately, including probability capture and cache reuse.
