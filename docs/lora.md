# Native LoRA inference

The native StableLM MLX backend can attach one ordinary PEFT LoRA to each generation session. Base weights remain FP16 or quantized; the adapter adds a low-rank correction during every adapted projection. No Python, weight merging, or new Metal kernels are required at runtime.

Adapters must be trained against the base checkpoint being loaded. Tensor names and dimensions are validated, and the adapter's declared base-model identifier is reported, but the loader cannot establish training provenance from matching shapes.

CPU, legacy Llama MLX, and Python inference do not support adapters. Python-backed `MLX` values expose the `LoRAController` methods but return an unsupported-backend error.

## Files and supported form

Put these files in an adapter directory, separate from the base checkpoint:

```text
adapter_config.json
adapter_model.safetensors
```

The configuration must specify `peft_type: "LORA"`, rank `r`, `lora_alpha`, and a list of target projections. Supported targets are `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. Each listed target must have A/B tensors in every layer, using PEFT names such as `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`.

The loader accepts F32, BF16, or F16 adapter tensors and stores them as FP16. It rejects missing or extra tensors, shape/rank mismatches, non-finite values, and values that would overflow FP16. Bias training, DoRA, rsLoRA, per-layer rank/alpha patterns, selective layer sets, saved replacement modules, and other non-default variants are rejected. Embedding and vocabulary-head adapters are not supported.

For an input `x`, ordinary base projection `xWᵀ`, and adapter matrices A and B:

```text
y = xWᵀ + strength × (alpha / rank) × ((xAᵀ)Bᵀ)
```

Strength defaults to 1 in the server. Zero bypasses the correction entirely. Negative strengths are allowed; the effective multiplier must fit FP16. Its scalar multiplication uses FP16 rounding, like the activations.

A rank-8 adapter across all seven projections of a 28-layer 2816-wide model occupies about 24 MB after conversion. Additional GPU dispatches can affect throughput despite the small matrices.

## Library use

```go
m, err := tinyoai.LoadMLXNative(modelDirectory)
if err != nil {
    return err
}
defer m.Close()

info, err := m.LoadLoRA(ctx, adapterDirectory, 1)
if err != nil {
    return err
}
fmt.Println(info.SHA256, info.Bytes)

_, err = m.SetLoRAScale(ctx, 0.5) // Retain adapter tensors; change strength.
if err != nil {
    return err
}
_, err = m.LoadLoRA(ctx, "", 0) // Unload and return to the base model.
return err
```

`LoRAInfo(ctx)` reports the combined SHA-256 of configuration and weight-file bytes, declared base model, rank, alpha, strength, projection count, and retained adapter bytes. Its SHA is empty when no adapter is loaded. Loading a directory again rereads it; `SetLoRAScale` reuses the resident tensors.

`Scale` reports the requested strength before multiplication by `Alpha/Rank`. `Projections` counts adapted base matrices across all layers. `Bytes` counts FP16 adapter tensor storage without allocator overhead or workspaces. Forks can share those tensors, so summing their reports can count the same storage more than once.

Calls serialize with generation, prefill, word branches, and close. The loader constructs and evaluates the new adapter before replacing the old one. A failed or cancelled load preserves the previous adapter and cache. Cancellation after a native commit does not undo that commit; inspect `LoRAInfo` if a client loses its response.

Every successful load, unload, or strength change clears that session's KV, token history, and saved probability rows. The next generation prefills again. Base weights stay resident. The first implementation does not persist native KV to disk.

Prefill, decoding, prompt logprobs, and batched word branches use the same corrected projection. Active adapters use the separate projection path even if the base matrices were packed for the fused-projection experiment. Pack before attaching an adapter; repacking while one is loaded is rejected.

### Selecting an adapter on a fork

`Fork` retains independent handles to the base weights, adapter tensors, and existing KV. Their tensor storage is initially shared. Changing either session's adapter invalidates only its own cache; a child can outlive its parent.

Given an unadapted native StableLM model `m`, attach an adapter to a child and generate there:

```go
child, err := m.Fork(ctx)
if err != nil {
    return err
}
defer child.Close()

controller, ok := child.(tinyoai.LoRAController)
if !ok {
    return fmt.Errorf("fork does not support LoRA")
}
if _, err := controller.LoadLoRA(ctx, adapterDirectory, 1); err != nil {
    return err
}

_, err = child.Generate(prompt, tinyoai.GenerateOptions{
    Context:   ctx,
    MaxTokens: 64,
    OnToken:   func(text string) { fmt.Print(text) },
})
return err
```

Pass the full prompt to `child.Generate`. Loading the adapter clears the child's inherited KV, so its first generation prefills again. The parent keeps its original cache and remains unadapted. A fork that retains the same adapter, strength, and KV precision can reuse its inherited prefix.

Writing to shared KV can copy a layer's complete cache. A long-lived fork may therefore retain another full KV allocation, even though base weights and unchanged adapter tensors remain shared. See [independent generation sessions](inference-features.md#independent-generation-sessions) for ownership and concurrency.

## Server controls

Register adapters when starting the native server:

```sh
./tinyoai -mlx -model-dir /path/to/base -addr 127.0.0.1:8080 \
  -lora instruct=/path/to/instruct \
  -lora another=/path/to/another \
  -lora-active instruct -lora-scale 1
```

`-lora` is repeatable. `-lora-active` defaults to `base`, so registration alone does not modify inference. Adapter flags require native StableLM. Registering any adapter enables the optional `/v1/adapters` endpoint around the existing API; without these flags, the endpoint is absent.

`GET /v1/adapters` lists registered names and current per-model adapter metadata. Change the resident selection with:

```sh
curl http://127.0.0.1:8080/v1/adapters \
  -H 'Content-Type: application/json' \
  -d '{"model":"clio-accel","adapter":"instruct","scale":0.5}'
```

Selecting the same name changes its strength without rereading tensors. Selecting another name loads that registered directory while retaining the base. Select `base` to unload. HTTP clients cannot supply filesystem paths.

Requests serialize through an endpoint gate that keeps selection names consistent with model state, then wait for the model's active inference. Cancellation interrupts either wait without changing the selection. Once native evaluation starts, it finishes before temporary arrays are released; cancellation does not undo a change that has already committed.

This is a session-wide experimental control, not a standard OpenAI completion field. Use a dedicated server for a controlled comparison: other clients can change the selection between completion requests. Register an alias or restart the server to reread a changed adapter file when its name is already selected.

Adapter selection does not change prompt formatting or stop strings. Clients must supply the format expected by each adapter through the existing completion API.

## Instruction-module prompt format

The Clio model's `special_instruct` adapter uses the [published instruction template](https://huggingface.co/NovelAI/clio-v1-legacy/blob/main/clio-special_instruct.jinja). Optional story text comes first. Each user instruction is enclosed in `{ ` and ` }`, followed by a newline. Assistant text ends with the literal string `{{}}`, followed by a newline.

For example, a second-turn prompt has this structure; the bracketed story and answer below are placeholders:

```text
[Story metadata and text]
{ What seems inconsistent in this passage? Quote the relevant lines. }
[Previous assistant answer]{{}}
{ Which of those inconsistencies might be intentional? }
```

Generation starts immediately after the last instruction's newline. Send `"stop": ["{{}}"]` in the completion request to stop at the answer terminator. When building the next prompt, append `{{}}` to the saved assistant answer even though the stop string was omitted from the returned text. Match the literal string rather than a fixed token ID, because its tokenization depends on context.

This template supplies no `Author:` or `Clio:` role labels, `[ Style: chat ]` tag, worked examples, or separate thinking phase. Any persona or additional instructions must be supplied explicitly by the client. The instruction adapter must also be loaded and active: writing braces in a prompt does not enable it.

## Validation

Pure-Go tests cover PEFT configuration, tensor coverage, numeric validity, cancellation, and server selection. `TestLoRAEndpointQueuedCancellation` checks that cancelled GET and POST requests return while an earlier request still waits for inference, leave selection unchanged, and allow later requests to proceed.

`TINYOAI_TEST_METAL=1 go test -tags mlx -run '^TestMLXLoRALifecycle$' -v .` uses the tiny checked-in fixture to verify atomic failure, packed-projection fallback, cache invalidation, zero-strength base restoration, fork isolation, and handle release.

The optional `TestMLXLoRAReference` uses `TINYOAI_MLX_DIR`, `TINYOAI_LORA_DIR`, and `TINYOAI_LORA_REFERENCE`. The reference directory contains `reference.json` with `prompt_ids` and `generated_ids`, plus little-endian float32 rows in `reference.f32`. Generate the reference with the same base checkpoint, FP16 adapter and KV, native FP16 Q/K attention, 256-token prefill chunks, and cached one-token decoding.

The test saves every native vocabulary row, checks argmax and full-distribution KL at every step, and records base/adapter timings and memory. It also compares probability capture and batched branches, including Q8 KV. Numerical parity validates adapter execution; it does not establish the quality of the resulting text.
