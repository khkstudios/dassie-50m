import math
import random

import numpy as np
import torch


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_at(step: int, warmup: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Linear warmup, then cosine decay to min_lr."""
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (lr - min_lr)


def is_apple_m1() -> bool:
    import platform
    import subprocess
    if platform.system() != "Darwin":
        return False
    try:
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True).stdout
    except OSError:
        return False
    return "M1" in brand
