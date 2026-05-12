"""
前馈网络模块 (FFN)

集成拉马努金初始化。
"""

import torch
import torch.nn as nn
from typing import Optional

from .ramanujan_initializer import RamanujanInitializer


class RamanujanFFN(nn.Module):
    """
    Transformer 前馈网络

    FFN(x) = W_2 · σ(W_1 · x + b_1) + b_2

    其中 σ 为激活函数 (ReLU, GELU, SiLU 等)

    拉马努金初始化应用于 W_1 和 W_2。
    """

    def __init__(self, d_model: int, dim_feedforward: int,
                 dropout: float = 0.0, activation: str = 'gelu',
                 layer_idx: int = 0,
                 initializer: Optional[RamanujanInitializer] = None):
        super().__init__()

        self.d_model = d_model
        self.dim_feedforward = dim_feedforward

        # 第一层：扩展
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        # 第二层：压缩
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu' or activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # 拉马努金初始化
        if initializer is None:
            initializer = RamanujanInitializer(max_depth=1000)

        # linear1 后面接激活函数，所以用对应的 nonlinearity
        act_name = activation if activation in ('relu', 'gelu', 'silu') else 'linear'
        initializer.init_linear(self.linear1, layer_idx, nonlinearity=act_name)
        initializer.init_linear(self.linear2, layer_idx, nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
