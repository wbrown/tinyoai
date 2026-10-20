# Native MLX memory headroom

The macOS allocator patch implements the maintainer's requirement to scale
reusable buffers to available machine memory and leave a 10% physical-RAM
margin. This is an idle-buffer retention policy, not a model, context, or workload
limit. Live tensors are never discarded to satisfy it.

The budget is `max(0, min(physical, managed + available) - ceil(physical / 10))`.
Managed memory includes live and idle cached Metal buffers. Available memory
uses Mach's free and explicitly purgeable pages. Inactive anonymous pages may
be the allocator's own unwired buffers: counting them again as available can
retain hundreds of GB while the OS compressor grows. Speculative pages are
already included in free pages.

The allocator samples VM counters at most every 100 ms and applies the sampled
budget on allocations, cache hits, and releases. A native maintenance thread
also trims unused buffers while GPU work is idle. The cadence controls telemetry
overhead, not how much requested work executes. All counter reads and cache
mutations use the allocator mutex; the worker stops before destruction. Explicit
caller cache settings can lower idle retention but cannot bypass machine
headroom. If telemetry fails, idle caching is disabled with a diagnostic until
a successful sample. Work whose live tensors exceed available memory still needs
a workload-level solution; this policy cannot guarantee an impossible reserve.

`scripts/build-mlx.sh` applies `memory-budget.patch` to a separate copy of the
pinned MLX source, keyed by patch contents, before building the SDK. It preserves
the upstream checkout and previous build sources. An explicitly supplied SDK
keeps its own native allocator; do not claim the headroom behavior without
checking that SDK. The native patch does not change the iOS allocator.

The standalone `memory_budget.h` is the same header embedded in the patch.
`memory_budget_test.cpp` exercises 8, 32 and 512 GiB machines, changing
availability, cache hits, oversized live sets and integer boundaries without
loading a model. Native tests in `internal/mlx/cache_test.go` check caller-policy
preservation, idle retention/reclamation and the actual machine budget. Run
those only on an Apple Silicon host with the matching patched SDK.

This contribution contains no trainer, checkpoint writer or evaluation worker.
