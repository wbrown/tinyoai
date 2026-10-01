"""Export a cached CPU float32 reference for long-prompt greedy parity.

python scripts/clio_greedy_reference.py /path/to/clio /path/to/greedy-probe
Then run TestClioGreedyParity with TINYOAI_CLIO_DIR and
TINYOAI_GREEDY_REFERENCE pointing to those directories. This reads local files
only. The full reference logits are saved as F32 safetensors for stable KL
comparisons without lossy decimal formatting.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import StableLmForCausalLM


def load_gguf_float32(model_dir, path):
    """Validation only: use llama.cpp's independent GGUF decoder, then PyTorch."""
    import gguf
    import numpy as np
    from transformers import StableLmConfig
    from transformers.models.stablelm.modeling_stablelm import StableLmRotaryEmbedding

    reader = gguf.GGUFReader(str(path))
    config = StableLmConfig.from_pretrained(str(model_dir), local_files_only=True)
    config._attn_implementation = "eager"
    config.layer_norm_eps = float(reader.fields["stablelm.attention.layer_norm_epsilon"].contents())
    with torch.device("meta"):
        model = StableLmForCausalLM(config)
    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.STABLELM, config.num_hidden_layers)
    tensors = {t.name: t for t in reader.tensors}
    for name, parameter in list(model.named_parameters()):
        tensor = tensors.pop(names.get_name(name, try_suffixes=(".weight", ".bias")))
        values = gguf.dequantize(tensor.data, tensor.tensor_type)
        values = np.array(values, dtype=np.float32, copy=True).reshape(tuple(parameter.shape))
        parent, leaf = name.rsplit(".", 1)
        model.get_submodule(parent).register_parameter(leaf, torch.nn.Parameter(torch.from_numpy(values), requires_grad=False))
    if tensors:
        raise ValueError(f"unused GGUF tensors: {list(tensors)}")
    # Meta construction also creates nonpersistent rotary buffers; rebuild them.
    for name, module in list(model.named_modules()):
        if isinstance(module, StableLmRotaryEmbedding):
            parent, leaf = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), leaf, StableLmRotaryEmbedding(config, device="cpu"))
    assert all(b.device.type == "cpu" for b in model.buffers())
    return model.eval()


# Original prose used only as a reproducible, ordinary long-context prompt.
PROMPT = """At midnight, Mara returned to the observatory with the brass key in her coat pocket. The building stood above the harbor, where the last fishing boats had already put out their lamps. Rain slipped down the green copper dome and gathered in the cracks of the stone steps. She had not been here since her father disappeared, and she had promised herself that she would never come back.

The letter had changed her mind. It arrived that morning in an envelope without a stamp, folded around a photograph of the northern sky. On the reverse, in her father's careful handwriting, were six words: Do not trust the seventh star. Beneath them was a date three days in the future.

Mara knew every instrument in the old tower. As a child she had carried notebooks up its spiral stair and fallen asleep beside the ticking clock while her father measured the passage of distant lights. He used to say that an astronomer needed patience before intelligence, and honesty before either. It had taken her twenty years to understand how difficult the second requirement could be.

The key turned easily. Inside, the air smelled of wet wool, dust, and the mineral oil used to protect the telescope's bearings. Someone had swept the entrance hall. A pair of muddy boots stood beside the umbrella rack, their toes pointing toward the stairs. Mara touched one with the edge of her shoe. The leather was still warm.

Above her, a chair scraped across the floor. She stopped breathing. The sound was followed by two slow knocks, then three quick ones: the little signal her father had used when he wanted her to bring him another cup of tea. She closed the outer door without letting its latch fall and began to climb.

On the landing she found a notebook open beneath a glass paperweight. Every page was filled with the same seven numbers. Some were written in pencil, some in red ink, and one row had been scratched through so hard that the paper was torn. A thin line connected the final number to a small drawing of a door. The door had no handle.

She took the notebook with her. The stair narrowed as it approached the dome, and the familiar ticking grew louder. Between the ticks she could hear another sound, a faint regular pulse like a second clock running a little too slowly. She tried to count the difference and lost her place when the floor above her creaked again.

At the top of the stairs, the telescope pointed through an opening in the roof. Its great iron frame had been polished until the old maker's name shone in the lamplight. A man stood beside the eyepiece with his back to her. He wore her father's gray coat, but he was much too young.

Mara held the notebook against her chest. She had prepared a dozen questions on the journey across town, and now none of them seemed useful. The stranger raised one hand without turning around. In the small circle of light beneath the telescope, she saw that he was holding another brass key.

"You came sooner than I expected," he said. "We still have time to decide which door to open."

"Where is my father?" Mara asked.

The stranger lowered his hand. For the first time, the steady ticking faltered. He looked toward the window, where the clouds were beginning to separate above the harbor, and then back at the notebook in her arms. "That depends," he said, "on what you remember about the night he left."
"""


def prompt_text():
    # Vary the records throughout the long context so incorrect cache reads
    # cannot simply retrieve an identical copy of a repeated paragraph.
    """Build deterministic varied prose so stale cache reads cannot hide behind repeats."""
    people = ["Mara", "Ellis", "Jun", "Orin", "Leah", "Tomas", "Ivo", "Nadia"]
    places = ["the harbor", "the west tower", "the old library", "the market",
              "the lower bridge", "the signal house", "the north quarry", "the station"]
    clues = ["a torn map", "a silver compass", "a blue glass bead", "an unsigned letter",
             "a broken watch", "a brass handle", "a folded photograph", "a copper disk",
             "a sealed bottle", "a red notebook", "a wooden whistle"]
    observations = [
        "The water was rising although the tide table promised that it would fall.",
        "The window reflected a light that nobody could see from the street.",
        "A messenger had arrived before dawn and refused to give his name.",
        "The clock had stopped, but the pendulum was still moving behind the glass.",
        "Someone had erased the final line of the record and replaced it with a question.",
        "A train could be heard beyond the hills, where the tracks had been removed years ago.",
        "The old caretaker insisted that he remembered the visitor from his childhood.",
        "There were fresh footprints in the dust on both sides of the locked door.",
        "The telescope had shifted toward the horizon without anyone touching its controls.",
        "A second set of figures appeared when the page was held against the lamp.",
        "The bell rang once at noon, and every bird in the square fell silent.",
        "The stars on the photograph formed a pattern different from the one in the atlas.",
        "An unfamiliar signature appeared beneath the name of the missing astronomer.",
    ]
    records = [PROMPT, "\nThe notebook contained the following records, each written on a separate dated page.\n"]
    for i in range(140):
        person, place, clue = people[i % len(people)], places[(i*3+1) % len(places)], clues[(i*7+2) % len(clues)]
        records.append(f"\nRecord {i+1}. At {i%24:02d}:{(i*13)%60:02d}, {person} returned from {place} carrying {clue}. "
                       f"{observations[(i*5+3)%len(observations)]} "
                       f"The measurement in the margin was {17+i*31}. "
                       f'"Keep this until I return," {person} told the assistant, who placed it beside the lamp.\n')
    return "".join(records)


def main():
    """Save a long CPU float32 greedy trajectory with cached one-token execution.

    An optional GGUF is decoded independently before inference. Full logits,
    token histories, runtime metadata, and timings remain available for audits."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--gguf", type=Path, help="independently dequantize this GGUF instead of loading SafeTensors; requires gguf==0.17.1")
    args = parser.parse_args()
    if args.prompt_tokens < 1 or args.new_tokens < 1:
        parser.error("token counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    ids = tokenizer.encode(prompt_text()).ids[:args.prompt_tokens]
    if len(ids) < args.prompt_tokens:
        parser.error(f"prompt contains only {len(ids)} tokens")
    print(f"Loading CPU float32 Clio; {len(ids)} prompt tokens, {args.new_tokens} requested output tokens", flush=True)
    model = load_gguf_float32(args.model_dir, args.gguf) if args.gguf else StableLmForCausalLM.from_pretrained(
        str(args.model_dir), torch_dtype=torch.float32,
        attn_implementation="eager", local_files_only=True).eval()
    assert model.device.type == "cpu" and all(p.dtype == torch.float32 for p in model.parameters())
    if len(ids) + args.new_tokens - 1 > model.config.max_position_embeddings:
        parser.error("requested evaluation exceeds model context")
    started = time.monotonic()
    cache, generated, rows = None, [], []
    with torch.inference_mode():
        for pos, token in enumerate(ids):
            out = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            if (pos + 1) % 64 == 0:
                print(f"Prefill {pos+1}/{len(ids)}, {time.monotonic()-started:.1f}s", flush=True)
        prefill_seconds = time.monotonic() - started
        for step in range(args.new_tokens):
            logits = out.logits[0, 0]
            token = logits.argmax().item()
            generated.append(token)
            rows.append(logits.clone())
            if (step + 1) % 32 == 0:
                print(f"Greedy {step+1}/{args.new_tokens}, {time.monotonic()-started:.1f}s", flush=True)
            if token == model.config.eos_token_id or step + 1 == args.new_tokens:
                break
            out = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
    decode_seconds = time.monotonic() - started - prefill_seconds
    save_file({"logits": torch.stack(rows)}, str(args.output_dir / "logits.safetensors"))
    metadata = {
        "execution_mode": "cpu-float32-cached-token", "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "torch_build": torch.__config__.show(),
        "prompt_ids": ids, "generated_ids": generated, "text": tokenizer.decode(generated),
        "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
    }
    if args.gguf:
        import hashlib
        from importlib.metadata import version
        digest = hashlib.sha256()
        with args.gguf.open("rb") as source:
            for block in iter(lambda: source.read(8 << 20), b""):
                digest.update(block)
        metadata.update(gguf_file=str(args.gguf), gguf_sha256=digest.hexdigest(), gguf_package=version("gguf"))
    (args.output_dir / "reference.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(generated)} greedy tokens to {args.output_dir}", flush=True)
    print(metadata["text"], flush=True)


if __name__ == "__main__":
    main()
