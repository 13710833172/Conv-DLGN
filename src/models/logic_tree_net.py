"""
可微分版本的 Convolutional LogicTreeNet 主模型：
输入图像
→ 多层 ConvLogicTreeLayer + OrPool2d
→ flatten
→ 3 层 DiffLogicLayer head
→ 输出一组 class-score 前的逻辑神经元
"""

from __future__ import annotations

import math

import torch.nn as nn
from torch import Tensor

from src.models.conv_logic_tree_layer import ConvLogicTreeLayer
from src.models.logic_layer import DiffLogicLayer
from src.models.or_pool import OrPool2d


class LogicTreeNet(nn.Module):
    """Pure PyTorch approximation of the convolutional DLGN architecture."""

    def __init__(
        self,
        dataset: str,
        input_channels: int,
        input_image_size: int,
        k: int,
        output_gate_factor: int,
        num_classes: int,
        connectivity_seed: int,
        receptive_field_size: int = 3,
        gate_init: str = "residual_a",
        num_head_layers: int = 3,
        tree_stride: int = 1,
        output_neuron_factor: int = 320,
    ):
        super().__init__()
        ds = str(dataset).lower().replace("-", "").replace("_", "")
        if ds not in ("mnist", "cifar10"):
            raise ValueError(f"Unsupported dataset {dataset!r}, expected mnist or cifar10.")
        self.dataset = ds
        self.input_channels = int(input_channels)
        self.input_image_size = int(input_image_size)
        self.k = int(k)
        self.output_gate_factor = int(output_gate_factor)
        self.num_classes = int(num_classes)
        self.connectivity_seed = int(connectivity_seed)
        self.receptive_field_size = int(receptive_field_size)
        self.gate_init = str(gate_init)
        self.num_head_layers = int(num_head_layers)
        self.tree_stride = int(tree_stride)
        self.output_neuron_factor = int(output_neuron_factor)
        self.num_conv_blocks = 4 if self.dataset == "cifar10" else 3

        convs: list[ConvLogicTreeLayer] = []
        pools: list[OrPool2d] = []
        in_ch = self.input_channels
        spatial = self.input_image_size
        if self.dataset == "mnist":
            conv_out_channels = [self.k, 3 * self.k, 9 * self.k]
            conv_rf_schedule = [5, 3, 3] # 卷积核大小
            conv_padding_schedule = [0, 1, 1] # 填充大小
        else:
            conv_out_channels = [self.k, 4 * self.k, 16 * self.k, 32 * self.k]
            conv_rf_schedule = [self.receptive_field_size for _ in range(self.num_conv_blocks)]
            conv_padding_schedule = [self.receptive_field_size // 2 for _ in range(self.num_conv_blocks)]

        for li in range(self.num_conv_blocks):
            seed = self.connectivity_seed + li * 1_000_003
            rf = int(conv_rf_schedule[li])
            padding = int(conv_padding_schedule[li])
            conv = ConvLogicTreeLayer(
                in_channels=in_ch,
                out_channels=int(conv_out_channels[li]),
                receptive_field_size=rf,
                connectivity_seed=seed,
                gate_init=self.gate_init,
                stride=self.tree_stride,
                padding=padding,
            )
            convs.append(conv)
            pools.append(OrPool2d(kernel_size=2, stride=2))
            in_ch = int(conv_out_channels[li])
            conv_out_spatial = (spatial + 2 * padding - rf) // self.tree_stride + 1 # 卷积层输出的空间尺寸
            spatial = math.floor((conv_out_spatial - 2) / 2 + 1) # 池化层输出的空间尺寸
        self.conv_layers = nn.ModuleList(convs)
        self.pool_layers = nn.ModuleList(pools)
        self.backbone_spatial_size = int(spatial)
        self.backbone_feature_dim = int(in_ch * self.backbone_spatial_size * self.backbone_spatial_size)

        if self.output_neuron_factor <= 0:
            raise ValueError("output_neuron_factor must be > 0")
        if self.num_head_layers != 3:
            raise ValueError("LogicTreeNet paper reproduction expects exactly 3 head layers.")
        head1 = int(4 * self.output_neuron_factor * self.k * self.output_gate_factor)   # 1280*k*ox
        head2 = int(2 * self.output_neuron_factor * self.k * self.output_gate_factor)   # 640*k*ox
        head3 = int(1 * self.output_neuron_factor * self.k * self.output_gate_factor)   # 320*k*ox
        head_dims = [head1, head2, head3]
        self.head_width = head3
        if self.head_width % self.num_classes != 0:
            raise ValueError("head_width must be divisible by num_classes")

        dense_layers: list[DiffLogicLayer] = []
        dense_in = self.backbone_feature_dim
        for li, out_dim in enumerate(head_dims):
            seed = self.connectivity_seed + 100_000_000 + li * 1_000_003
            dense_layer = DiffLogicLayer(
                input_dim=dense_in,
                num_neurons=out_dim,
                connectivity_seed=seed,
                gate_init=self.gate_init,
            )
            dense_layers.append(
                dense_layer
            )
            dense_in = out_dim
        self.head_layers = nn.ModuleList(dense_layers)

    def forward(self, x: Tensor) -> Tensor:
        for conv, pool in zip(self.conv_layers, self.pool_layers, strict=True):
            x = pool(conv(x))
        x = x.view(x.size(0), -1) # 将特征图展平为一维向量，用于全连接层
        for dense in self.head_layers:
            x = dense(x)
        return x

    def forward_with_activations(self, x: Tensor) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """返回模型输出、每层特征图和每层门控概率。用于分析中间状态"""
        acts: list[Tensor] = []
        probs: list[Tensor] = []
        for conv, pool in zip(self.conv_layers, self.pool_layers, strict=True):
            x = conv(x)
            acts.append(x) # 存储卷积层的输出特征图
            probs.append(conv.gate_probs().view(-1, 16)) # 存储卷积层的门控概率
            x = pool(x)
        x = x.view(x.size(0), -1)
        for dense in self.head_layers:
            x = dense(x)
            acts.append(x)
            probs.append(dense.gate_probs())
        return x, acts, probs

    def conv_leaf_indices(self) -> list[Tensor]:
        return [layer.leaf_indices for layer in self.conv_layers]

    def conv_gate_ids(self) -> list[Tensor]:
        return [layer.discretized_gate_ids() for layer in self.conv_layers]

    def head_connectivity_tensors(self) -> list[Tensor]:
        return [layer.connectivity for layer in self.head_layers]

    def head_gate_ids(self) -> list[Tensor]:
        return [layer.gate_logits.argmax(dim=-1) for layer in self.head_layers]
