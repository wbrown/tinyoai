# Native Clio inference buffer reuse

Normal `Generate` now reserves KV storage for the caller's validated context
length before prefill. Previously, successive chunks enlarged and copied the
full cache, leaving obsolete allocation sizes in MLX's unused buffer pool.
The reservation changes physical allocation timing, not usable context length.
Selecting a smaller context still invalidates the larger cache as before.

For multi-token Q8 attention, dense K/V reconstruction now uses the retained
capacity as its physical shape. Attention receives a view of only the populated
prefix. This permits allocator reuse across prefill chunks and replay lengths;
spare positions and abandoned suffixes never become visible to attention.
Single-token packed Q8 attention and FP16 attention arithmetic are unchanged.
There is no additional persistent dense KV cache, new idle-cache ceiling, or
per-generation cache flush. A larger selected context reserves more live memory
up front and reconstructs more dense positions, in exchange for stable buffers.

An actual Clio CE18k Q6 check used a 7,531-token prompt, selected 8,192-token
context, Q8 KV and 12 saved prompt alternatives. Reserving KV reduced unused
cached memory from 11.27 GiB to 4.26 GiB; stable dense workspace reduced it to
0.73–0.77 GiB. Live memory with the reservation was 3.59 GiB. These are measured
single-run memory results, not universal latency or memory guarantees. Generated
text and all saved prompt probability rows matched exactly. Rewind, context
shrink/regrowth, and the app's model-release lifecycle were also checked.

`TestClioContextBufferReuse` is opt-in: set `TINYOAI_MLX_DIR` to actual Clio
weights and `TINYOAI_BUFFER_PROMPT` to a suitable text file. Run that test alone
with the native `mlx` build tag and configured MLX SDK. It compares the old dense
view with production defaults, asserts automatic reservation through `Generate`,
and checks exact probability/output parity across rewind and context changes.
Memory and timing are logged as diagnostics without arbitrary pass thresholds.
No model weights, private prompts, training/evaluation code or generated binaries
are included in this change.
