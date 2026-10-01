"""Compare the actual Clio checkpoint with tinyoai's opt-in integration test.

python scripts/clio_reference.py /path/to/clio /tmp/clio-reference.json
TINYOAI_CLIO_DIR=/path/to/clio TINYOAI_CLIO_REFERENCE=/tmp/clio-reference.json \
    go test -run TestClioCheckpoint -v -timeout 20m

Uses cached, one-token-at-a-time CPU float32 inference, matching the Go loop.
Expands weights to float32, requiring ~12 GB
for reference weights. The Go backend retains the original ~6.1 GB BF16 data.
This script reads only local files and makes no downloads.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import transformers
from tokenizers import Tokenizer
from transformers import StableLmForCausalLM


def main():
    """Save short CPU float32 cached-token logits, greedy text, and tokenizer cases.

    Local checkpoint files are required; no model downloads are performed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    started = time.monotonic()
    print("Loading Clio in float32 for reference inference...", flush=True)
    model = StableLmForCausalLM.from_pretrained(
        str(args.model_dir), torch_dtype=torch.float32, attn_implementation="eager",
        local_files_only=True).eval()
    assert model.device.type == "cpu" and all(p.dtype == torch.float32 for p in model.parameters())
    print(f"Loaded in {time.monotonic()-started:.1f}s", flush=True)
    prompt = "Once upon a time"
    inputs = tokenizer.encode(prompt).ids
    started = time.monotonic()
    with torch.inference_mode():
        cache, logits = None, []
        for token in inputs:
            out = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logits.append(out.logits[0, 0].tolist())
        generated = []
        for step in range(16):
            token = out.logits[0, 0].argmax().item()
            generated.append(token)
            if token == 3 or step == 15:
                break
            out = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
    print(f"Inference in {time.monotonic()-started:.1f}s: {tokenizer.decode(generated)!r}", flush=True)
    texts = ["", prompt, "Hello 123!\n\t世界 🦊", "  leading   spaces", "a    b",
             "<|startoftext|>Hello<|endoftext|>", "☃️ café", "12", "a▁▁b", "\u2003\r\n",
             '[ Author: Jane; Title: Sky ]\n***\nShe said, "Hello!"',
             "<|spmspace|>" * 12, "\u0378\U0010ffff", " ", " "*33]
    rng = random.Random(42)
    alphabet = 'abcde ABCDE0123456789.!?\n\t▁日界☃🦊\u0378'
    texts += [''.join(rng.choice(alphabet) for _ in range(rng.randrange(1, 160))) for _ in range(200)]
    result = {
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "execution_mode": "cpu-float32-cached-token",
        "input_ids": inputs, "logits": logits,
        "generation": {"prompt": prompt, "ids": generated, "text": tokenizer.decode(generated)},
        "tokenization": [{"text": text, "ids": tokenizer.encode(text).ids,
                          "decoded": tokenizer.decode(tokenizer.encode(text).ids)} for text in texts],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
