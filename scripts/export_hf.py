"""Export a Dassie checkpoint to a Hugging Face LlamaForCausalLM folder.

    uv run python scripts/export_hf.py --ckpt checkpoints/50m/best.pt --out out/dassie-50m
"""
import argparse
import json
import shutil
from pathlib import Path

import torch

from dassie import Dassie, ModelConfig
from dassie.tokenizer import DEFAULT_PATH, EOT, EOT_ID


def to_hf_state_dict(sd: dict) -> dict:
    """Our module names already mirror HF Llama; only prefixes differ."""
    out = {}
    for k, v in sd.items():
        if k == "lm_head.weight":
            continue  # tied to embed_tokens
        out[f"model.{k}"] = v
    return out


def llama_config(cfg: ModelConfig) -> dict:
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.d_model,
        "intermediate_size": cfg.ffn_hidden,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_heads,
        "head_dim": cfg.head_dim,
        "max_position_embeddings": cfg.max_seq_len,
        "rms_norm_eps": cfg.norm_eps,
        "rope_theta": cfg.rope_theta,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": True,
        "bos_token_id": EOT_ID,
        "eos_token_id": EOT_ID,
        "pad_token_id": EOT_ID,
        "torch_dtype": "float32",
    }


def export(ckpt_path: str, out_dir: str, tokenizer_path: str = str(DEFAULT_PATH)) -> Path:
    from safetensors.torch import save_file

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sd = {k: v.float().contiguous() for k, v in to_hf_state_dict(ck["model"]).items()}
    save_file(sd, out / "model.safetensors", metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(llama_config(cfg), indent=2))
    (out / "generation_config.json").write_text(json.dumps({
        "bos_token_id": EOT_ID, "eos_token_id": EOT_ID, "pad_token_id": EOT_ID,
        "do_sample": True, "temperature": 0.8, "top_k": 50, "top_p": 0.95,
        "max_new_tokens": 300,
    }, indent=2))
    shutil.copy(tokenizer_path, out / "tokenizer.json")
    (out / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": EOT, "eos_token": EOT, "pad_token": EOT, "unk_token": EOT,
        "model_max_length": cfg.max_seq_len,
        "clean_up_tokenization_spaces": False,
    }, indent=2))
    (out / "special_tokens_map.json").write_text(json.dumps({
        "bos_token": EOT, "eos_token": EOT, "pad_token": EOT, "unk_token": EOT,
    }, indent=2))
    return out


def check_parity(ckpt_path: str, out_dir: str, atol: float = 1e-4) -> float:
    """Load both models and compare logits on a random batch."""
    from transformers import LlamaForCausalLM

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    ours = Dassie(cfg)
    ours.load_state_dict(ck["model"])
    ours.eval()
    hf = LlamaForCausalLM.from_pretrained(out_dir, torch_dtype=torch.float32).eval()
    x = torch.randint(0, cfg.vocab_size, (2, min(64, cfg.max_seq_len)))
    with torch.no_grad():
        a, _ = ours(x)
        b = hf(x).logits
    diff = (a - b).abs().max().item()
    assert diff < atol, f"logit mismatch: max abs diff {diff}"
    return diff


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="out/dassie-50m")
    p.add_argument("--tokenizer", default=str(DEFAULT_PATH))
    args = p.parse_args()
    out = export(args.ckpt, args.out, args.tokenizer)
    print(f"exported to {out}")
    print(f"parity check passed (max logit diff {check_parity(args.ckpt, args.out):.2e})")
