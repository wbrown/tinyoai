"""Export local Clio numerical probes; no network or model code execution.

python scripts/clio_numerics.py /path/to/clio /path/to/probes
Requires the same reference dependencies as clio_reference.py. Records both
full-prompt and cached token-at-a-time float32 evaluation, plus isolated layer
operations. The probes are local test data, not distributed model assets.
"""
import argparse
import json
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import StableLmForCausalLM


def main():
    """Save full-prompt, cached-token, and isolated-operation probes for Go diagnostics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    model = StableLmForCausalLM.from_pretrained(
        str(args.model_dir), torch_dtype=torch.float32,
        attn_implementation="eager", local_files_only=True).eval()
    probes = {}
    handles = []
    for name, module in model.named_modules():
        if name.startswith(tuple(f"model.layers.{i}." for i in (0, 14, 27))) and isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
            def hook(module, inputs, output, name=name):
                """Copy one selected layer operation's input and output before buffers are reused."""
                probes[name + ".input"] = inputs[0][0].detach().clone()
                probes[name + ".output"] = output[0].detach().clone()
            handles.append(module.register_forward_hook(hook))
    with torch.inference_mode():
        model(torch.tensor([tokenizer.encode("Once upon a time").ids]), use_cache=False)
    for handle in handles:
        handle.remove()
    for i in (0, 14, 27):
        name = f"model.layers.{i}.input_layernorm"
        norm = model.model.layers[i].input_layernorm
        _, mean, rstd = torch.native_layer_norm(probes[name + ".input"],
                                               [model.config.hidden_size],
                                               norm.weight, norm.bias, norm.eps)
        probes[name + ".mean"] = mean.detach()
        probes[name + ".rstd"] = rstd.detach()
    save_file(probes, str(args.output_dir / "operations.safetensors"))
    cases = []
    texts = ["Once upon a time", "The rain tapped against the window as she opened the letter.",
             '[ Author: Jane; Title: The Observatory ]\n***\n"There is something moving among the stars," he said.']
    for text in texts:
        ids = tokenizer.encode(text).ids
        with torch.inference_mode():
            full = model(torch.tensor([ids]), use_cache=False).logits[0]
            cache = None
            rows = []
            for token in ids:
                out = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                rows.append(out.logits[0, 0])
            cached = torch.stack(rows)
        delta = (full - cached).double()
        print(f"{len(ids)} tokens: torch full vs cached max={delta.abs().max().item():.9g}, rms={delta.square().mean().sqrt().item():.9g}", flush=True)
        index = len(cases)
        save_file({"full": full.contiguous(), "cached": cached.contiguous()}, str(args.output_dir / f"case-{index}.safetensors"))
        cases.append({"text": text, "ids": ids, "full_cached_max": delta.abs().max().item(),
                      "full_cached_rms": delta.square().mean().sqrt().item()})
    metadata = {"torch": torch.__version__, "transformers": transformers.__version__,
                "torch_build": torch.__config__.show(), "cases": cases}
    (args.output_dir / "cases.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
