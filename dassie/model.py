"""A Llama-style decoder-only transformer, written from scratch.

The layout deliberately mirrors HF's LlamaForCausalLM so weights can be
exported 1:1 (see scripts/export_hf.py).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """Normalize by root-mean-square (no mean subtraction, no bias), then scale."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def precompute_rope(head_dim: int, max_seq_len: int, theta: float):
    """cos/sin tables for rotary position embeddings, shape (T, head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)               # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)        # (T, head_dim)  HF "rotate_half" layout
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D); cos/sin: (T, D)
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class MLP(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """Pre-norm transformer block with residual connections."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Dassie(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # weight tying

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scale residual-output projections so activations don't grow with depth
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weights counted once

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"sequence length {T} > max {self.cfg.max_seq_len}"
        x = self.embed_tokens(idx)
        cos, sin = self.rope_cos[:T].to(x.dtype), self.rope_sin[:T].to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int | None = 50, top_p: float | None = 0.95, eos_id: int | None = None):
        """Autoregressive sampling (no KV cache — simple and fine at this size)."""
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                if top_p and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum = torch.softmax(sorted_logits, -1).cumsum(-1)
                    remove = cum - torch.softmax(sorted_logits, -1) > top_p
                    sorted_logits[remove] = -float("inf")
                    logits = torch.full_like(logits, -float("inf")).scatter(1, sorted_idx, sorted_logits)
                nxt = torch.multinomial(torch.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx
