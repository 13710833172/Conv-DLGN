"""
Convolutional differentiable logic-tree layer (depth=3, 8 leaves, 7 gates).

每个卷积核是一棵 depth=3 的二叉逻辑树：
8 个叶子输入
→ 4 个第一层逻辑门
→ 2 个第二层逻辑门
→ 1 个根节点逻辑门
→ 输出一个特征值
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.ops.logic_ops import weighted_sum_16_ops
from src.utils.connectivity import build_local_tree_leaf_connectivity


class ConvLogicTreeLayer(nn.Module):
    """Logic-tree convolution layer with fixed random leaf connectivity."""

    NUM_LEAVES = 8
    NUM_GATES = 7
    CHANNELS_PER_TREE = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int, # 每一层卷积树的个数=kernal树
        receptive_field_size: int = 3,
        connectivity_seed: int = 0,
        gate_init: str = "residual_a",
        stride: int = 1,
        padding: int | None = None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.receptive_field_size = int(receptive_field_size)
        self.stride = int(stride)
        self.padding = int(padding) if padding is not None else self.receptive_field_size // 2
        self.gate_init = str(gate_init)

        leaf_idx = build_local_tree_leaf_connectivity(
            in_channels=self.in_channels,
            receptive_field_size=self.receptive_field_size,
            out_channels=self.out_channels,
            seed=int(connectivity_seed),
            num_leaves=self.NUM_LEAVES,
            unique=True,
            channels_per_tree=self.CHANNELS_PER_TREE,
        )
        self.register_buffer("leaf_indices", leaf_idx.long(), persistent=True) # leaf_indices的形状是[out_channels, num_leaves]

        # 初始化 7 个逻辑门的参数，形状是[out_channels, 7, 16]
        if self.gate_init == "normal":
            logits = torch.randn(self.out_channels, self.NUM_GATES, 16)
        elif self.gate_init == "residual_a":
            logits = torch.zeros(self.out_channels, self.NUM_GATES, 16)
            # Operator id 3 = identity A.
            logits[..., 3] = 5.0
        else:
            raise ValueError(f"Unknown gate_init: {gate_init}")
        self.gate_logits = nn.Parameter(logits)

    def gate_probs(self) -> Tensor:
        return F.softmax(self.gate_logits, dim=-1)

    def discretized_gate_ids(self) -> Tensor:
        return self.gate_logits.argmax(dim=-1)

    def _mix_gate(self, a: Tensor, b: Tensor, probs: Tensor) -> Tensor:
        p = probs.view(1, self.out_channels, 1, 16)
        return weighted_sum_16_ops(a, b, p)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4: #假设[1, 2, 32, 32]
            raise ValueError(f"Expected BCHW input, got shape {tuple(x.shape)}")
        bsz, _, h, w = x.shape # batch size, channels, height, width
        # 用unfold先把每个空间位置对应的局部窗口都整理出来，方便每棵逻辑树在每个窗口里选叶子。
        patches = F.unfold(
            x,
            kernel_size=self.receptive_field_size,
            padding=self.padding,
            stride=self.stride,
        )  
        # F.unfold 的作用是：把图像张量里的局部滑动窗口提取出来，并展开成列。
        # 由[B, C, H, w] 展平成 [B, C*rf*rf, L]，L是kernel滑过的窗口的数量，L=H_out*W_out，假设为[1, 2*3*3, 32*32]=[1,18,1024]
        # L = 每张图上多少个 receptive field / 空间位置
        _, _, n_loc = patches.shape # batch size, channels, number of locations
        h_out = (h + 2 * self.padding - self.receptive_field_size) // self.stride + 1 # output height
        w_out = (w + 2 * self.padding - self.receptive_field_size) // self.stride + 1 # output width
        if h_out * w_out != n_loc:
            raise RuntimeError("Unfold output shape mismatch")

        leaves: list[Tensor] = [] # leaves 是一个 list，长度为 8，每个元素是一个形状为[B, out_channels, L]的Tensor，即每棵树在每个输出空间位置，都拿到了8个叶子输入。
        for leaf_idx in range(self.NUM_LEAVES):
            idx = self.leaf_indices[:, leaf_idx]  # [out_channels]，leaf_indices形状为[4, 8]
            # 假设leaf_indices[:, 0] = [0, 1, 0, 2]，则：
            # tree 0 的第 0 个 leaf 取 patches 的第 0 个候选位置
            # tree 1 的第 0 个 leaf 取 patches 的第 1 个候选位置
            # tree 2 的第 0 个 leaf 取 patches 的第 0 个候选位置
            # tree 3 的第 0 个 leaf 取 patches 的第 2 个候选位置
            leaves.append(patches[:, idx, :])   # [B, out_channels, L]，假设为[1,4,1024]，即每一层有4棵树
            # unfold后，每个窗口里的18个位置：
            #channel 0 的 3×3:
            # 0, 1, 2,
            # 3, 4, 5,
            # 6, 7, 8
            # channel 1 的 3×3:
            # 9, 10, 11,
            # 12, 13, 14,
            # 15, 16, 17
            # leaf_idx=0 时，patches[:, 0, :] 表示取所有窗口里 channel 0 的第 0 个 feature 值

        # 每一步都是在做所有 batch_size 的所有 kernel 的所有receptive field上做并行计算。
        probs = self.gate_probs()  # [out_channels, 7, 16]，4棵树，每棵树7个逻辑门，每个逻辑门有16种可选逻辑操作的概率
        n0 = self._mix_gate(leaves[0], leaves[1], probs[:, 0, :]) # leaves[0]形状为[1,4,1024]，所有树的第 0 个叶子，在所有空间位置上的取值。假如不考虑batch_size，则每一行代表一个kernel，每一列代表这个kernel在每个receptive field上的第0个叶子的取值。
        n1 = self._mix_gate(leaves[2], leaves[3], probs[:, 1, :])
        n2 = self._mix_gate(leaves[4], leaves[5], probs[:, 2, :])
        n3 = self._mix_gate(leaves[6], leaves[7], probs[:, 3, :])
        n4 = self._mix_gate(n0, n1, probs[:, 4, :])
        n5 = self._mix_gate(n2, n3, probs[:, 5, :])
        root = self._mix_gate(n4, n5, probs[:, 6, :])  # [B, out_channels, L]
        return root.view(bsz, self.out_channels, h_out, w_out)
