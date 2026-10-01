"""Regenerate small, deterministic test fixtures using Hugging Face.

Requires torch, transformers==4.57.6, tokenizers==0.22.2, safetensors.
Run from the repository root: python scripts/stablelm_reference.py
No pretrained weights or NovelAI tokenizer assets are redistributed here.
"""
import json
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, processors
from transformers import StableLmConfig, StableLmForCausalLM


def main():
    """Regenerate independent tiny StableLM, tokenizer, and rotary-boundary fixtures.

    The fixed seed and mixed BF16/F32 tensors exercise the loader and arithmetic
    without copying a pretrained checkpoint into the repository."""
    output = Path("testdata/stablelm")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    vocab = {}

    def add(piece):
        """Assign a stable vocabulary ID on the first appearance of a token piece."""
        if piece not in vocab:
            vocab[piece] = len(vocab)

    special = ["<|pad|>", "<|unknown|>", "<|startoftext|>", "<|endoftext|>"]
    for piece in special + list("0123456789"):
        add(piece)
    for i in range(256):
        add(f"<0x{i:02X}>")
    for piece in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?\n\t▁":
        add(piece)
    merges = [("a", "b"), ("b", "c"), ("ab", "c"), ("▁", "a"), ("▁a", "b"),
              ("▁", "▁"), ("▁▁", "▁▁"), ("1", "2"), ("H", "e"),
              ("He", "l"), ("Hel", "l"), ("Hell", "o")]
    for left, right in merges:
        add(left + right)
    tokenizer = Tokenizer(models.BPE(vocab, merges, unk_token="<|unknown|>", byte_fallback=True, fuse_unk=True))
    tokenizer.normalizer = normalizers.Sequence([normalizers.Replace(" ", "▁")])
    tokenizer.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()])
    tokenizer.add_special_tokens([AddedToken(s, normalized=False, special=True) for s in special])
    tokenizer.add_tokens([AddedToken(s, normalized=True) for s in list("0123456789") + ["▁▁", "▁▁▁▁"]])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|startoftext|> $A", pair="<|startoftext|> $A <|startoftext|>:1 $B:1",
        special_tokens=[("<|startoftext|>", 2)])
    tokenizer.save(str(output / "tokenizer.json"))

    config = StableLmConfig(
        vocab_size=len(vocab) + 1, hidden_size=16, intermediate_size=40,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, partial_rotary_factor=0.5, rope_theta=10000,
        use_parallel_residual=True, use_qkv_bias=False, qk_layernorm=False,
        tie_word_embeddings=False, hidden_act="silu", layer_norm_eps=1e-5,
        bos_token_id=2, eos_token_id=3, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    model = StableLmForCausalLM(config).eval()
    # Nonzero LayerNorm bias and BF16 matrices exercise all Clio tensor types.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim == 2:
                param.copy_((torch.randn_like(param) * 0.18).bfloat16().float())
            elif name.endswith("weight"):
                param.copy_(1 + torch.randn_like(param) * 0.1)
            else:
                param.copy_(torch.randn_like(param) * 0.05)
    config.save_pretrained(output)
    state = {name: value.bfloat16() if value.ndim == 2 else value for name, value in model.state_dict().items()}
    shards = [{}, {}]
    weight_map = {}
    for i, (name, tensor) in enumerate(state.items()):
        shard = i % 2
        shards[shard][name] = tensor.contiguous()
        weight_map[name] = f"model-{shard+1:05d}-of-00002.safetensors"
    for i, shard in enumerate(shards):
        save_file(shard, str(output / f"model-{i+1:05d}-of-00002.safetensors"))
    (output / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}, indent=2) + "\n")
    inputs = [2, vocab["a"], vocab["b"], vocab["c"], vocab["▁"], vocab["Hello"], vocab["!"]]
    with torch.no_grad():
        logits = model(torch.tensor([inputs]), use_cache=False).logits[0].tolist()
        prompt = "Hello abc"
        prompt_ids = tokenizer.encode(prompt).ids
        generated = model.generate(torch.tensor([prompt_ids]), max_new_tokens=8, do_sample=False,
                                   pad_token_id=0, eos_token_id=3)[0].tolist()[len(prompt_ids):]
    texts = ["", "abc ab bc", "Hello 123!\n\t世界 🦊", "  leading   spaces", "a    b",
             "<|startoftext|>Hello<|endoftext|>", "☃️ café", "12", "a▁▁b", "\u2003\r\n"]
    expected = {
        "transformers_version": transformers.__version__, "input_ids": inputs, "logits": logits,
        "generation": {"prompt": prompt, "ids": generated, "text": tokenizer.decode(generated)},
        "tokenization": [{"text": text, "ids": tokenizer.encode(text).ids,
                          "decoded": tokenizer.decode(tokenizer.encode(text).ids)} for text in texts],
    }
    (output / "expected.json").write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n")
    torch.manual_seed(123)
    vector = torch.randn(128)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
    rotary = []
    for position in (0, 1, 511, 767, 8128, 8191):
        angle = inv_freq * position
        cos, sin = torch.cat([angle.cos()] * 2), torch.cat([angle.sin()] * 2)
        rotated = vector.clone()
        rotated[:32] = vector[:32] * cos + torch.cat([-vector[16:32], vector[:16]]) * sin
        rotary.append({"position": position, "output": rotated.tolist()})
    (output / "rope.json").write_text(json.dumps({"input": vector.tolist(), "cases": rotary}, indent=2) + "\n")
    print(f"Wrote {output} (transformers {transformers.__version__})")


if __name__ == "__main__":
    main()
