"""Private forward-pass worker for tinyoai. JSON commands, framed float32 logits.

Requires mlx==0.32.2 and mlx-lm==0.31.3 on Apple Silicon. The Go process owns
tokenization, sampling and HTTP; this worker never interprets generation settings.
"""
import contextlib
import json
import math
import sys
import traceback
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models import stablelm
from mlx_lm.utils import load_model


# Adapted from MLX-LM's StableLM attention, Copyright 2023-2024 Apple Inc.
# MIT license reproduced in docs/MLX-LICENSE.txt. Only the query/key float32
# casts are removed, as in the saved Clio native-attention parity validation.
def native_attention(self, x, mask=None, cache=None):
    """Evaluate StableLM attention with native-dtype queries and keys.

    Cache offsets supply absolute rotary positions; keeping this dtype matches
    the native Go backend rather than the upstream float32 query/key casts."""
    queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    batch, length, dim = queries.shape
    queries = queries.reshape(batch, length, self.num_heads, -1)
    keys = keys.reshape(batch, length, self.num_key_value_heads, -1)
    if self.qk_layernorm:
        queries, keys = self.q_layernorm(queries), self.k_layernorm(keys)
    queries, keys = queries.transpose(0, 2, 1, 3), keys.transpose(0, 2, 1, 3)
    values = values.reshape(batch, length, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
    if cache is not None:
        queries, keys = self.rope(queries, offset=cache.offset), self.rope(keys, offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
    else:
        queries, keys = self.rope(queries), self.rope(keys)
    output = stablelm.scaled_dot_product_attention(
        queries, keys, values, cache=cache, scale=math.sqrt(1 / queries.shape[-1]), mask=mask
    ).astype(values.dtype)
    return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


def reply(data=b"", error=None):
    """Write a JSON byte-count header and an optional little-endian float32 payload.

    Stdout is reserved for this protocol; diagnostics belong on stderr."""
    print(json.dumps({"bytes": len(data), "error": error}), flush=True)
    if data:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()


def main():
    """Load local MLX weights once and serve suffix forwards until stdin closes.

    Each request rewinds all layer caches to its supplied valid prefix. The Go
    caller owns sampling, cancellation, and worker restart after failure."""
    if not mx.metal.is_available():
        raise RuntimeError("MLX requires an available Apple Metal GPU")
    # Avoid retaining the multi-gigabyte scratch allocations of long prefills.
    mx.set_cache_limit(256 * 1024 * 1024)
    stablelm.Attention.__call__ = native_attention
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    with contextlib.redirect_stdout(sys.stderr):
        model, config = load_model(Path(sys.argv[1]), lazy=False, strict=True)
    cache = make_prompt_cache(model)
    reply()
    for line in sys.stdin:
        request = json.loads(line)
        prefix, tokens = request["prefix"], request["tokens"]
        if not tokens or prefix < 0 or prefix > cache[0].offset:
            raise ValueError("invalid cached prefix")
        if prefix + len(tokens) > config["max_position_embeddings"]:
            raise ValueError("context limit exceeded")
        for layer in cache:
            layer.trim(layer.offset - prefix)
        hidden = model.model(mx.array([tokens]), cache=cache)
        if request["logits"]:
            logits = model.lm_head(hidden[:, -1:, :])[:, -1, :].astype(mx.float32)
            mx.eval(logits, [c.state for c in cache])
            reply(np.asarray(logits, dtype="<f4").tobytes())
        else:
            mx.eval([c.state for c in cache])
            reply()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        reply(error=str(exc))
        sys.exit(1)
