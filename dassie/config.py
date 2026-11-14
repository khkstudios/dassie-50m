"""Model and training configuration, with named presets."""
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 640
    n_layers: int = 10
    n_heads: int = 10
    ffn_hidden: int = 1728          # SwiGLU hidden size (~8/3 * d_model, rounded)
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class TrainConfig:
    batch_size: int = 16           # sequences per micro-batch
    grad_accum: int = 8             # micro-batches per optimizer step
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 500
    max_steps: int = 8000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 50
    sample_every: int = 500
    log_every: int = 10
    save_every: int = 500
    seed: int = 1337


@dataclass
class Preset:
    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)


PRESETS = {
    # tiny config for debugging on the M1
    "debug": Preset(
        ModelConfig(d_model=256, n_layers=4, n_heads=4, ffn_hidden=688, max_seq_len=256),
        TrainConfig(batch_size=32, grad_accum=1, lr=2e-3, min_lr=2e-4, warmup_steps=100,
                    max_steps=1500, eval_every=100, eval_batches=20, sample_every=250, save_every=250),
    ),
    # the real ~54M parameter model (tokens/step = 32*8*512 = 131k)
    "50m": Preset(ModelConfig(), TrainConfig()),
}


def to_dict(cfg) -> dict:
    return asdict(cfg)
