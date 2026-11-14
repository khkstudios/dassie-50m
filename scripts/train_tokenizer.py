"""Train the BPE tokenizer on a slice of TinyStoriesV2.

    uv run python scripts/train_tokenizer.py --max-stories 500000
"""
import argparse
from pathlib import Path

from dassie.tokenizer import DEFAULT_PATH, iter_stories, train_tokenizer

p = argparse.ArgumentParser()
p.add_argument("--input", default="data/TinyStoriesV2-GPT4-train.txt")
p.add_argument("--out", default=str(DEFAULT_PATH))
p.add_argument("--vocab-size", type=int, default=8192)
p.add_argument("--max-stories", type=int, default=500_000)
args = p.parse_args()

tok = train_tokenizer(iter_stories(args.input, args.max_stories), args.vocab_size)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
tok.save(args.out)
sample = "Once upon a time, there was a little dog named Max. He liked to play in the park."
enc = tok.encode(sample)
print(f"saved {args.out}  vocab={tok.get_vocab_size()}")
print(f"{len(sample.split())} words -> {len(enc.ids)} tokens: {enc.tokens}")
assert tok.decode(enc.ids) == sample
