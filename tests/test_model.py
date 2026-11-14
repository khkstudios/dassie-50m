import torch

from dassie import Dassie, ModelConfig, PRESETS


def tiny():
    return ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=4, ffn_hidden=128, max_seq_len=32)


def test_shapes_and_loss():
    m = Dassie(tiny())
    x = torch.randint(0, 128, (2, 16))
    logits, loss = m(x, torch.randint(0, 128, (2, 16)))
    assert logits.shape == (2, 16, 128)
    assert abs(loss.item() - torch.log(torch.tensor(128.0)).item()) < 0.5  # ~uniform at init


def test_causal():
    torch.manual_seed(0)
    m = Dassie(tiny()).eval()
    x = torch.randint(0, 128, (1, 16))
    y = x.clone()
    y[0, 10:] = torch.randint(0, 128, (6,))
    a, _ = m(x)
    b, _ = m(y)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-5)


def test_overfit_one_batch():
    torch.manual_seed(0)
    m = Dassie(tiny())
    x = torch.randint(0, 128, (4, 32))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(200):
        _, loss = m(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_50m_param_count():
    n = Dassie(PRESETS["50m"].model).num_params()
    assert 45e6 < n < 60e6, n
