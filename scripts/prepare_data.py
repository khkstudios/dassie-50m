"""Tokenize the corpus into flat uint16 token files: data/train.bin, data/val.bin.

Each story is followed by <|endoftext|> (id 0). A deterministic ~0.5% of stories
are held out for validation.

    uv run python scripts/prepare_data.py --max-stories 20000   # debug subset
    uv run python scripts/prepare_data.py                       # full corpus
"""
import argparse
import zlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dassie.tokenizer import DEFAULT_PATH, EOT_ID, iter_stories, load_tokenizer

p = argparse.ArgumentParser()
p.add_argument("--input", default="data/TinyStoriesV2-GPT4-train.txt")
p.add_argument("--tokenizer", default=str(DEFAULT_PATH))
p.add_argument("--out-dir", default="data")
p.add_argument("--max-stories", type=int, default=None)
p.add_argument("--val-frac", type=float, default=0.005)
p.add_argument("--batch", type=int, default=4096)
args = p.parse_args()

tok = load_tokenizer(args.tokenizer)
assert tok.get_vocab_size() < 2**16, "uint16 storage needs vocab < 65536"
out = Path(args.out_dir)
files = {"train": open(out / "train.bin", "wb"), "val": open(out / "val.bin", "wb")}
counts = {"train": 0, "val": 0}
val_cut = int(args.val_frac * 2**32)


def flush(batch):
    for story, enc in zip(batch, tok.encode_batch(batch)):
        # hash-based split: stable across runs regardless of --max-stories
        split = "val" if zlib.crc32(story.encode()) < val_cut else "train"
        ids = np.array(enc.ids + [EOT_ID], dtype=np.uint16)
        files[split].write(ids.tobytes())
        counts[split] += len(ids)


batch = []
for story in tqdm(iter_stories(args.input, args.max_stories), unit=" stories"):
    batch.append(story)
    if len(batch) >= args.batch:
        flush(batch)
        batch = []
if batch:
    flush(batch)
for f in files.values():
    f.close()
print({k: f"{v:,} tokens" for k, v in counts.items()})
