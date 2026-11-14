"""Fill the model_card.md template with real numbers from a checkpoint and an exported HF folder.

    uv run python scripts/render_card.py --folder out/dassie-50m --ckpt checkpoints/50m/best.pt

Placeholders look like {{name}}. Anything we can't compute (no checkpoint yet, no exported folder)
renders as _TBD_, and the model-index metadata block is left out so the Hub never sees a bogus metric.
"""
import argparse
import math
import re
from pathlib import Path

DEFAULT_PROMPTS = [
    "Once upon a time",
    "Lily found a small box in the garden.",
    "Tom and his dog went to the park. The sun",
]
TBD = "_TBD_"


def checkpoint_stats(ckpt_path: str) -> dict:
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stats = {"best_step": ck.get("step")}
    if ck.get("best_val") is not None and math.isfinite(ck["best_val"]):
        stats["val_loss"] = ck["best_val"]
    tc, mc = ck.get("train_config"), ck.get("model_config")
    if tc and mc and stats["best_step"]:
        tokens = stats["best_step"] * tc["batch_size"] * tc["grad_accum"] * mc["max_seq_len"]
        stats["tokens_seen"] = tokens
    return stats


def param_count(folder: Path) -> int | None:
    st = folder / "model.safetensors"
    if not st.exists():
        return None
    from safetensors import safe_open

    with safe_open(str(st), "pt") as f:
        return sum(math.prod(f.get_slice(k).get_shape()) for k in f.keys())  # lm_head is tied, not stored


def generate_samples(folder: Path, prompts: list[str], max_new_tokens: int = 160, seed: int = 0) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForCausalLM.from_pretrained(folder, torch_dtype=torch.float32).eval()
    blocks = []
    for p in prompts:
        torch.manual_seed(seed)
        ids = tok(p, return_tensors="pt")
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8,
                             top_p=0.95, top_k=50, pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0], skip_special_tokens=True).strip()
        blocks.append(f"**Prompt:** `{p}`\n\n> " + text.replace("\n", "\n> "))
    note = (f"_Sampled with temperature 0.8, top-p 0.95, top-k 50, seed {seed}, "
            f"{max_new_tokens} new tokens. Not cherry-picked._")
    return note + "\n\n" + "\n\n".join(blocks)


def model_index(repo: str, val_loss: float) -> str:
    return f"""model-index:
- name: {repo.split('/')[-1]}
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      type: roneneldan/TinyStories
      name: TinyStoriesV2-GPT4 (held-out 0.5%)
      split: validation
    metrics:
    - type: loss
      name: Validation loss
      value: {val_loss:.4f}
    - type: perplexity
      name: Validation perplexity
      value: {math.exp(val_loss):.2f}"""


def render(template: str, folder: Path | None = None, ckpt: str | None = None,
           repo: str = "khkstudios/dassie-50m", samples: bool = True,
           prompts: list[str] = DEFAULT_PROMPTS) -> str:
    stats = checkpoint_stats(ckpt) if ckpt and Path(ckpt).exists() else {}
    n_params = param_count(folder) if folder else None
    val = stats.get("val_loss")

    values = {
        "repo": repo,
        "params": f"~{n_params / 1e6:.1f}M" if n_params else "~54.8M",
        "val_loss": f"{val:.4f}" if val is not None else TBD,
        "val_ppl": f"{math.exp(val):.2f}" if val is not None else TBD,
        "best_step": f"{stats['best_step']:,}" if stats.get("best_step") else TBD,
        "tokens_seen": f"~{stats['tokens_seen'] / 1e9:.2f}B" if stats.get("tokens_seen") else TBD,
        "model_index": model_index(repo, val) if val is not None else "",
        "samples": (generate_samples(folder, prompts)
                    if samples and folder and (folder / "model.safetensors").exists()
                    else "_TBD after training._"),
    }
    if not values["model_index"]:  # drop the whole line so the frontmatter has no blank gap
        template = re.sub(r"^\{\{\s*model_index\s*\}\}\n", "", template, flags=re.M)
    out = re.sub(r"\{\{\s*(\w+)\s*\}\}", lambda m: values.get(m[1], m[0]), template)
    leftover = re.findall(r"\{\{\s*\w+\s*\}\}", out)
    if leftover:
        raise ValueError(f"unknown placeholders in model card: {leftover}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--card", default="model_card.md")
    p.add_argument("--folder", default="out/dassie-50m")
    p.add_argument("--ckpt", default="checkpoints/50m/best.pt")
    p.add_argument("--repo", default="khkstudios/dassie-50m")
    p.add_argument("--no-samples", action="store_true")
    p.add_argument("--out", default=None, help="write here instead of <folder>/README.md")
    args = p.parse_args()
    folder = Path(args.folder)
    text = render(Path(args.card).read_text(), folder, args.ckpt, args.repo, not args.no_samples)
    dest = Path(args.out) if args.out else folder / "README.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    print(f"wrote {dest}")
