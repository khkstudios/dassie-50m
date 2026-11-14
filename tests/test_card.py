import math
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
pytest.importorskip("transformers")
yaml = pytest.importorskip("yaml")

from dassie import Dassie, ModelConfig  # noqa: E402
from dassie.config import TrainConfig  # noqa: E402
from export_hf import export  # noqa: E402
from render_card import TBD, render  # noqa: E402

TEMPLATE = (ROOT / "model_card.md").read_text()


def frontmatter(card: str) -> dict:
    assert card.startswith("---\n")
    return yaml.safe_load(card.split("---\n")[1])


def test_render_without_checkpoint_is_valid_and_tbd():
    card = render(TEMPLATE, folder=None, ckpt=None)
    meta = frontmatter(card)
    assert "model-index" not in meta
    assert meta["widget"][0]["text"] == "Once upon a time"
    assert TBD in card and "{{" not in card


def test_render_with_checkpoint_fills_metrics(tmp_path):
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=128, n_layers=2, n_heads=4, ffn_hidden=344, max_seq_len=128)
    ckpt = tmp_path / "best.pt"
    torch.save({"model": Dassie(cfg).state_dict(), "model_config": asdict(cfg),
                "train_config": asdict(TrainConfig()), "step": 4000, "best_val": 1.2345}, ckpt)
    folder = export(str(ckpt), str(tmp_path / "hf"))

    card = render(TEMPLATE, folder, str(ckpt), prompts=["Once upon a time"])
    metrics = frontmatter(card)["model-index"][0]["results"][0]["metrics"]
    assert metrics[0]["value"] == pytest.approx(1.2345)
    assert metrics[1]["value"] == pytest.approx(math.exp(1.2345), abs=0.01)
    assert "step 4,000" in card and "**Prompt:** `Once upon a time`" in card
    assert "{{" not in card
