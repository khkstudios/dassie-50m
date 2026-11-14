"""Generate stories from a training checkpoint.

    uv run python scripts/sample.py --ckpt checkpoints/debug/best.pt --prompt "Once upon a time" -n 3
"""
import argparse

import torch

from dassie import Dassie, ModelConfig
from dassie.tokenizer import EOT_ID, load_tokenizer
from dassie.utils import pick_device

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--prompt", default="Once upon a time")
p.add_argument("-n", "--num-samples", type=int, default=1)
p.add_argument("--max-new-tokens", type=int, default=300)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top-k", type=int, default=50)
p.add_argument("--top-p", type=float, default=0.95)
p.add_argument("--device", default="auto")
p.add_argument("--seed", type=int, default=None)
args = p.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)
device = pick_device(args.device)
ck = torch.load(args.ckpt, map_location=device, weights_only=False)
model = Dassie(ModelConfig(**ck["model_config"])).to(device).eval()
model.load_state_dict(ck["model"])
tok = load_tokenizer()

idx = torch.tensor([tok.encode(args.prompt).ids] * args.num_samples, device=device)
out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k, args.top_p, eos_id=EOT_ID)
for i, row in enumerate(out.tolist()):
    if EOT_ID in row[1:]:
        row = row[: row.index(EOT_ID, 1)]
    print(f"--- sample {i + 1} ---\n{tok.decode(row)}\n")
