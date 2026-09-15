"""OR pooling for differentiable logic activations.

The paper uses max t-conorm for OR pooling, which is equivalent to max-pooling.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OrPool2d(nn.Module):
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)

    def forward(self, x: Tensor) -> Tensor:
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
    # OR pooling 使用 max t-conorm，而 max t-conorm 就是 max operation。