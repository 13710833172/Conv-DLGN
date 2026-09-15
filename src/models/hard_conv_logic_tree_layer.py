"""Hard (discretized) convolutional logic-tree layer."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.ops.logic_ops import build_truth_table


class HardConvLogicTreeLayer(nn.Module):
    NUM_LEAVES = 8
    NUM_GATES = 7

    def __init__(
        self,
        in_channels: int,
        out_channels: int, # 每一层卷积树的个数=kernal树
        receptive_field_size: int,
        leaf_indices: Tensor,
        gate_ids: Tensor,
        stride: int = 1,
        padding: int | None = None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.receptive_field_size = int(receptive_field_size)
        self.stride = int(stride)
        self.padding = int(padding) if padding is not None else self.receptive_field_size // 2

        if leaf_indices.shape != (self.out_channels, self.NUM_LEAVES):
            raise ValueError(
                f"leaf_indices must have shape ({self.out_channels}, {self.NUM_LEAVES}), "
                f"got {tuple(leaf_indices.shape)}"
            )
        if gate_ids.shape != (self.out_channels, self.NUM_GATES):
            raise ValueError(
                f"gate_ids must have shape ({self.out_channels}, {self.NUM_GATES}), "
                f"got {tuple(gate_ids.shape)}"
            )
        self.register_buffer("leaf_indices", leaf_indices.long(), persistent=True)
        self.register_buffer("gate_ids", gate_ids.long(), persistent=True)
        self.register_buffer("truth_table", build_truth_table(device=leaf_indices.device), persistent=True)

    def _hard_gate(self, a: Tensor, b: Tensor, gate_ids: Tensor) -> Tensor:
        ai = (a > 0.5).long() # 将a转换为0/1，0表示false，1表示true
        bi = (b > 0.5).long()
        gids = gate_ids.view(1, self.out_channels, 1).expand_as(ai)
        return self.truth_table[gids, ai, bi].float()

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:  # 假设 [1, 2, 32, 32]
            raise ValueError(f"Expected BCHW input, got shape {tuple(x.shape)}")
        bsz, _, h, w = x.shape
        # 用unfold先把每个空间位置对应的局部窗口都整理出来，方便每棵逻辑树在每个窗口里选叶子。
        patches = F.unfold(
            x,
            kernel_size=self.receptive_field_size,
            padding=self.padding,
            stride=self.stride,
        )  
        _, _, n_loc = patches.shape # batch size, channels, number of locations
        h_out = (h + 2 * self.padding - self.receptive_field_size) // self.stride + 1 # output height
        w_out = (w + 2 * self.padding - self.receptive_field_size) // self.stride + 1 # output width
        if h_out * w_out != n_loc:
            raise RuntimeError("Unfold output shape mismatch")

        leaves: list[Tensor] = []
        for leaf_idx in range(self.NUM_LEAVES):
            idx = self.leaf_indices[:, leaf_idx] # [out_channels]
            leaves.append(patches[:, idx, :]) 

        n0 = self._hard_gate(leaves[0], leaves[1], self.gate_ids[:, 0])
        n1 = self._hard_gate(leaves[2], leaves[3], self.gate_ids[:, 1])
        n2 = self._hard_gate(leaves[4], leaves[5], self.gate_ids[:, 2])
        n3 = self._hard_gate(leaves[6], leaves[7], self.gate_ids[:, 3])
        n4 = self._hard_gate(n0, n1, self.gate_ids[:, 4])
        n5 = self._hard_gate(n2, n3, self.gate_ids[:, 5])
        root = self._hard_gate(n4, n5, self.gate_ids[:, 6])
        return root.view(bsz, self.out_channels, h_out, w_out)
