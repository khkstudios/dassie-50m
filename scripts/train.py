"""Train Dassie.

    uv run python scripts/train.py --preset debug
    uv run python scripts/train.py --preset 50m --resume      # continue from checkpoints/50m/latest.pt
"""
import argparse
import csv
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from dassie import Dassie, PRESETS
from dassie.data import TokenData
from dassie.tokenizer import EOT_ID, load_tokenizer
from dassie.utils import is_apple_m1, lr_at, pick_device, seed_everything

p = argparse.ArgumentParser()
p.add_argument("--preset", default="debug", choices=list(PRESETS))
p.add_argument("--data-dir", default="data")
p.add_argument("--out-dir", default=None, help="default: checkpoints/<preset>")
p.add_argument("--device", default="auto")
p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"])
p.add_argument("--resume", action="store_true")
p.add_argument("--max-steps", type=int, default=None, help="override preset")
p.add_argument("--batch-size", type=int, default=None, help="override preset")
p.add_argument("--grad-accum", type=int, default=None, help="override preset")
args = p.parse_args()
sys.stdout.reconfigure(line_buffering=True)  # so `tail -f` on a redirected log works

preset = PRESETS[args.preset]
mcfg = preset.model
overrides = {k: v for k, v in dict(max_steps=args.max_steps, batch_size=args.batch_size,
                                   grad_accum=args.grad_accum).items() if v is not None}
tcfg = replace(preset.train, **overrides)
out_dir = Path(args.out_dir or f"checkpoints/{args.preset}")
out_dir.mkdir(parents=True, exist_ok=True)

device = pick_device(args.device)
if args.dtype == "auto":
    # M1 GPUs lack native bf16 (it's emulated and slower than fp32); M2+ and modern NVIDIA have it
    use_bf16 = (device == "cuda" and torch.cuda.is_bf16_supported()) or (device == "mps" and not is_apple_m1())
else:
    use_bf16 = args.dtype == "bfloat16"
dtype = torch.bfloat16 if use_bf16 else torch.float32
seed_everything(tcfg.seed)

train_data = TokenData(f"{args.data_dir}/train.bin", mcfg.max_seq_len)
val_data = TokenData(f"{args.data_dir}/val.bin", mcfg.max_seq_len)
tok = load_tokenizer()
assert tok.get_vocab_size() == mcfg.vocab_size

model = Dassie(mcfg).to(device)

# weight decay only on matrices (not norms); embedding is tied with lm_head
decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": tcfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
    lr=tcfg.lr, betas=(tcfg.beta1, tcfg.beta2), fused=device == "cuda",
)

step, best_val = 0, float("inf")
rng = np.random.default_rng(tcfg.seed)
if args.resume and (out_dir / "latest.pt").exists():
    ck = torch.load(out_dir / "latest.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    step, best_val = ck["step"], ck["best_val"]
    rng.bit_generator.state = ck["rng"]
    print(f"resumed from step {step} (best val {best_val:.4f})")

tokens_per_step = tcfg.batch_size * tcfg.grad_accum * mcfg.max_seq_len
print(f"device={device} dtype={dtype} params={model.num_params()/1e6:.2f}M "
      f"tokens/step={tokens_per_step:,} train_tokens={len(train_data):,} "
      f"epochs={tcfg.max_steps * tokens_per_step / len(train_data):.2f}")


def autocast():
    return torch.autocast(device_type=device, dtype=dtype, enabled=use_bf16)


@torch.no_grad()
def evaluate():
    model.eval()
    g = np.random.default_rng(0)  # same val batches every time -> comparable numbers
    losses = []
    for _ in range(tcfg.eval_batches):
        x, y = val_data.batch(tcfg.batch_size, device, g)
        with autocast():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def sample(prompt="Once upon a time", n=120):
    model.eval()
    idx = torch.tensor([tok.encode(prompt).ids], device=device)
    with autocast():
        out = model.generate(idx, n, eos_id=EOT_ID)
    model.train()
    return tok.decode(out[0].tolist())


def save(name, retries=3):
    """Atomic save: write a temp file then rename, so a crash never leaves a corrupt checkpoint."""
    state = {
        "model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
        "best_val": best_val, "rng": rng.bit_generator.state,
        "model_config": asdict(mcfg), "train_config": asdict(tcfg),
    }
    tmp = out_dir / f".{name}.tmp"
    for attempt in range(retries):
        try:
            torch.save(state, tmp)
            tmp.replace(out_dir / name)
            return
        except (RuntimeError, OSError) as e:
            print(f"save {name} failed ({e}); retry {attempt + 1}/{retries}")
            time.sleep(1)
    print(f"WARNING: could not save {name}; training continues")


log_path = out_dir / "log.csv"
new_log = not log_path.exists() or not args.resume
log_f = open(log_path, "w" if new_log else "a", newline="")
log = csv.writer(log_f)
if new_log:
    log.writerow(["step", "train_loss", "val_loss", "lr", "tok_per_s"])

model.train()
t0 = time.time()
while step < tcfg.max_steps:
    lr = lr_at(step, tcfg.warmup_steps, tcfg.max_steps, tcfg.lr, tcfg.min_lr)
    for g in opt.param_groups:
        g["lr"] = lr

    loss_acc = 0.0
    for _ in range(tcfg.grad_accum):
        x, y = train_data.batch(tcfg.batch_size, device, rng)
        with autocast():
            _, loss = model(x, y)
        (loss / tcfg.grad_accum).backward()
        loss_acc += loss.item() / tcfg.grad_accum
    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
    opt.step()
    opt.zero_grad(set_to_none=True)
    step += 1

    val_loss = ""
    if step % tcfg.eval_every == 0 or step == tcfg.max_steps:
        val_loss = evaluate()
        print(f"step {step:6d} | val {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            save("best.pt")
    if step % tcfg.log_every == 0:
        dt = time.time() - t0
        tps = tokens_per_step * tcfg.log_every / dt
        eta = (tcfg.max_steps - step) * dt / tcfg.log_every / 3600
        print(f"step {step:6d} | loss {loss_acc:.4f} | lr {lr:.2e} | {tps:,.0f} tok/s | eta {eta:.2f}h")
        log.writerow([step, f"{loss_acc:.4f}", val_loss if val_loss == "" else f"{val_loss:.4f}", f"{lr:.3e}", f"{tps:.0f}"])
        log_f.flush()
        t0 = time.time()
    if step % tcfg.sample_every == 0:
        print("--- sample ---\n" + sample() + "\n--------------")
    if step % tcfg.save_every == 0:
        save("latest.pt")

save("latest.pt")
print(f"done. best val {best_val:.4f}\n" + sample())
