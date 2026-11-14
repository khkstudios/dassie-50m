import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
pytest.importorskip("transformers")

from dassie import Dassie, ModelConfig  # noqa: E402
from export_hf import check_parity, export  # noqa: E402


def test_export_matches_hf_llama(tmp_path):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=8192, d_model=128, n_layers=2, n_heads=4, ffn_hidden=344, max_seq_len=128)
    m = Dassie(cfg)
    ckpt = tmp_path / "ck.pt"
    torch.save({"model": m.state_dict(), "model_config": asdict(cfg)}, ckpt)
    out = export(str(ckpt), str(tmp_path / "hf"))
    assert check_parity(str(ckpt), str(out)) < 1e-4
