import torch
import torch.nn.functional as F

from src.models.conv_logic_tree_layer import ConvLogicTreeLayer
from src.models.hard_conv_logic_tree_layer import HardConvLogicTreeLayer
from src.models.hard_logic_tree_net import HardLogicTreeNet
from src.models.logic_tree_net import LogicTreeNet
from src.models.or_pool import OrPool2d


def test_conv_logic_tree_shape():
    layer = ConvLogicTreeLayer(
        in_channels=9,
        out_channels=16,
        receptive_field_size=3,
        connectivity_seed=3,
        gate_init="normal",
    )
    x = torch.rand(4, 9, 32, 32)
    y = layer(x)
    assert y.shape == (4, 16, 32, 32)
    assert layer.gate_logits.shape == (16, 7, 16)
    assert layer.leaf_indices.shape == (16, 8)


def test_or_pool_matches_max_pool():
    x = torch.rand(2, 5, 8, 8)
    pool = OrPool2d(kernel_size=2, stride=2)
    y = pool(x)
    ref = F.max_pool2d(x, kernel_size=2, stride=2)
    assert torch.allclose(y, ref)


@torch.no_grad()
def test_residual_a_init_probability():
    layer = ConvLogicTreeLayer(
        in_channels=3,
        out_channels=4,
        receptive_field_size=3,
        connectivity_seed=0,
        gate_init="residual_a",
    )
    probs = layer.gate_probs()
    # exp(5)/(exp(5)+15*exp(0)) ~= 0.908
    p_a = probs[..., 3]
    assert torch.all(p_a > 0.85)


@torch.no_grad()
def test_soft_hard_consistency_for_conv_tree():
    soft = ConvLogicTreeLayer(
        in_channels=3,
        out_channels=6,
        receptive_field_size=3,
        connectivity_seed=123,
        gate_init="zeros",
    )
    soft.gate_logits.zero_()
    soft.gate_logits[..., 3] = 80.0  # force A gate

    hard = HardConvLogicTreeLayer(
        in_channels=3,
        out_channels=6,
        receptive_field_size=3,
        leaf_indices=soft.leaf_indices,
        gate_ids=soft.discretized_gate_ids(),
    )
    x = torch.randint(0, 2, (3, 3, 10, 10)).float()
    y_soft = soft(x)
    y_hard = hard(x)
    assert torch.allclose(y_soft, y_hard, atol=0.05)


@torch.no_grad()
def test_logictreenet_soft_hard_forward_shape():
    soft = LogicTreeNet(
        dataset="mnist",
        input_channels=1,
        input_image_size=28,
        k=16,
        output_gate_factor=2,
        num_classes=10,
        connectivity_seed=9,
    )
    hard = HardLogicTreeNet.from_soft_model(soft)
    x = torch.randint(0, 2, (2, 1, 28, 28)).float()
    y_soft = soft(x)
    y_hard = hard(x)
    assert y_soft.shape == y_hard.shape
    assert y_soft.shape[0] == 2
