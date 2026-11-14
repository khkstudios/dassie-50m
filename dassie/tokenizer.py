"""Byte-level BPE tokenizer (HF `tokenizers`), saved as a standard tokenizer.json."""
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"
EOT_ID = 0
DEFAULT_PATH = Path("tokenizer/tokenizer.json")

_REPLACE = {
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "–": "-", "—": " - ", "…": "...", " ": " ",
    "​": "", "﻿": "",
}
_TABLE = str.maketrans(_REPLACE)


def clean_text(s: str) -> str:
    """Normalize typographic punctuation to ASCII and trim whitespace."""
    return s.translate(_TABLE).strip()


def iter_stories(path: str | Path, max_stories: int | None = None,
                 chunk_size: int = 1 << 20) -> Iterator[str]:
    """Stream stories split on <|endoftext|> without loading the whole file."""
    n, buf = 0, ""
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buf += chunk
            *stories, buf = buf.split(EOT)
            for s in stories:
                s = clean_text(s)
                if s:
                    yield s
                    n += 1
                    if max_stories and n >= max_stories:
                        return
    s = clean_text(buf)
    if s:
        yield s


def train_tokenizer(texts: Iterator[str], vocab_size: int = 8192) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],  # -> id 0
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    assert tok.token_to_id(EOT) == EOT_ID
    return tok


def load_tokenizer(path: str | Path = DEFAULT_PATH) -> Tokenizer:
    return Tokenizer.from_file(str(path))
