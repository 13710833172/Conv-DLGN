import torch

from src.config_profiles import apply_paper_table6_profile
from src.models.model_factory import build_soft_model_from_config
from src.models.logic_layer import DiffLogicLayer
from src.utils.output_grouping import compute_class_scores


def test_groupsum_uses_sum_not_mean_for_mnist_s():
    batch = 2
    num_classes = 10
    outputs_per_class = 1024
    tau = 6.5
    outputs = torch.ones(batch, num_classes * outputs_per_class)

    logits = compute_class_scores(outputs, num_classes=num_classes, tau=tau, beta=0.0)

    expected = torch.full((batch, num_classes), outputs_per_class / tau)
    torch.testing.assert_close(logits, expected)


def test_residual_a_init_prefers_a_gate_and_approximates_a():
    layer = DiffLogicLayer(
        input_dim=2,
        num_neurons=1,
        connectivity=torch.tensor([[0, 1]], dtype=torch.long),
        gate_init="residual_a",
    )
    probs = layer.gate_probs().detach()
    assert int(probs.argmax(dim=-1).item()) == 3
    assert float(probs[..., 3].item()) > 0.9

    x = torch.rand(128, 2)
    out = layer(x).squeeze(-1)
    a = x[:, 0]
    mae = float((out - a).abs().mean().item())
    assert mae < 0.1


def test_mnist_s_forward_shapes_match_paper_target():
    cfg = apply_paper_table6_profile(
        {
            "model": {
                "family": "logictreenet",
                "dataset": "mnist",
                "size": "S",
                "profile_source": "paper_table6",
                "num_classes": 10,
            },
            "data": {"dataset": "mnist"},
            "training": {},
            "output": {},
        }
    )
    model = build_soft_model_from_config(cfg["model"], connectivity_seed=0)
    x = torch.rand(4, 1, 28, 28)

    x1 = model.pool_layers[0](model.conv_layers[0](x))
    assert x1.shape == (4, 16, 12, 12)
    x2 = model.pool_layers[1](model.conv_layers[1](x1))
    assert x2.shape == (4, 48, 6, 6)
    x3 = model.pool_layers[2](model.conv_layers[2](x2))
    assert x3.shape == (4, 144, 3, 3)

    flat = x3.view(x3.size(0), -1)
    assert flat.shape == (4, 1296)

    h1 = model.head_layers[0](flat)
    assert h1.shape == (4, 40960)
    h2 = model.head_layers[1](h1)
    assert h2.shape == (4, 20480)
    h3 = model.head_layers[2](h2)
    assert h3.shape == (4, 10240)

    logits = compute_class_scores(h3, num_classes=10, tau=6.5, beta=0.0)
    assert logits.shape == (4, 10)
