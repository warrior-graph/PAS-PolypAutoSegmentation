import pytest
import torch

from pas.losses import deep_supervision_loss, structure_loss
from pas.models import build_model


@pytest.fixture(scope="module")
def model():
    return build_model({"name": "emcad", "encoder": "pvt_v2_b0"}, pretrained=False)


def test_output_shapes(model):
    model.eval()
    with torch.no_grad():
        outs = model(torch.randn(2, 3, 128, 128))
    assert len(outs) == 4
    assert all(o.shape == (2, 1, 128, 128) for o in outs)


def test_parameter_count_matches_paper(model):
    decoder = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("encoder"))
    assert 0.45e6 < decoder < 0.55e6  # EMCAD-B0 decoder: 0.506M in the paper


@pytest.mark.parametrize("mode", ["mutation", "sum", "last"])
def test_loss_modes(mode):
    outs = [torch.randn(2, 1, 32, 32, requires_grad=True) for _ in range(4)]
    mask = (torch.rand(2, 1, 32, 32) > 0.5).float()
    loss = deep_supervision_loss(outs, mask, mode)
    loss.backward()
    assert torch.isfinite(loss)


def test_structure_loss_prefers_correct_logits():
    mask = torch.zeros(1, 1, 32, 32)
    mask[..., 8:24, 8:24] = 1
    good = (mask * 2 - 1) * 10
    assert structure_loss(good, mask) < structure_loss(-good, mask)


def test_overfits_single_batch():
    torch.manual_seed(0)
    model = build_model({"name": "emcad", "encoder": "pvt_v2_b0"}, pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    mask[..., 16:48, 16:48] = 1
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(15):
        loss = deep_supervision_loss(model(x), mask, "sum")
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.7
