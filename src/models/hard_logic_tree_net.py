"""Hard (discretized) LogicTreeNet."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.hard_conv_logic_tree_layer import HardConvLogicTreeLayer
from src.models.hard_logic_layer import HardLogicLayer
from src.models.logic_tree_net import LogicTreeNet
from src.models.or_pool import OrPool2d


class HardLogicTreeNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        head_width: int,
        conv_layers: list[HardConvLogicTreeLayer],
        pool_layers: list[OrPool2d],
        head_layers: list[HardLogicLayer],
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.head_width = int(head_width)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.pool_layers = nn.ModuleList(pool_layers)
        self.head_layers = nn.ModuleList(head_layers)

    def forward(self, x: Tensor) -> Tensor:
        for conv, pool in zip(self.conv_layers, self.pool_layers, strict=True):
            x = pool(conv(x))
        x = x.view(x.size(0), -1) # flatten
        for dense in self.head_layers:
            x = dense(x)
        return x

    @torch.no_grad()
    def forward_layerwise(self, x: Tensor) -> list[Tensor]:
        acts: list[Tensor] = []
        for conv, pool in zip(self.conv_layers, self.pool_layers, strict=True):
            x = conv(x)
            acts.append(x)
            x = pool(x)
        x = x.view(x.size(0), -1)
        for dense in self.head_layers:
            x = dense(x)
            acts.append(x)
        return acts

    @classmethod
    # 从一个“软”（可微/可学习）LogicTreeNet模型，提取出其结构和已离散化（硬化）的参数，生成一个等价的“硬”HardLogicTreeNet模型。
    # 这样可以把训练好的软模型转化为只执行硬逻辑门、不再可训练的推理模型。
    def from_soft_model(cls, soft: LogicTreeNet) -> "HardLogicTreeNet":
        conv_layers: list[HardConvLogicTreeLayer] = []
        in_ch = soft.input_channels
        for conv in soft.conv_layers:
            conv_layers.append(
                HardConvLogicTreeLayer(
                    in_channels=in_ch,
                    out_channels=conv.out_channels,
                    receptive_field_size=conv.receptive_field_size,
                    leaf_indices=conv.leaf_indices.detach().clone(),
                    gate_ids=conv.discretized_gate_ids().detach().clone(),
                    stride=conv.stride,
                    padding=conv.padding,
                )
            )
            in_ch = conv.out_channels

        pool_layers = [OrPool2d(kernel_size=2, stride=2) for _ in soft.pool_layers]

        head_layers: list[HardLogicLayer] = []
        in_dim = soft.backbone_feature_dim
        for layer in soft.head_layers:
            head_layers.append(
                HardLogicLayer(
                    input_dim=in_dim,
                    num_neurons=layer.num_neurons,
                    connectivity=layer.connectivity.detach().clone(),
                    gate_ids=layer.gate_logits.argmax(dim=-1).detach().clone(),
                )
            )
            in_dim = layer.num_neurons

        return cls(
            input_channels=soft.input_channels,
            num_classes=soft.num_classes,
            head_width=soft.head_width,
            conv_layers=conv_layers,
            pool_layers=pool_layers,
            head_layers=head_layers,
        )
