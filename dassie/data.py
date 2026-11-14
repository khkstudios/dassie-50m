"""Random-window batches from a flat memmapped token file."""
import numpy as np
import torch


class TokenData:
    def __init__(self, path: str, seq_len: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        assert len(self.data) > seq_len + 1, f"{path} too small"

    def __len__(self):
        return len(self.data)

    def batch(self, batch_size: int, device: str, gen: np.random.Generator):
        ix = gen.integers(0, len(self.data) - self.seq_len - 1, size=batch_size)
        chunk = np.stack([self.data[i: i + self.seq_len + 1] for i in ix]).astype(np.int64)
        x = torch.from_numpy(chunk[:, :-1])
        y = torch.from_numpy(chunk[:, 1:])
        if device == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)
