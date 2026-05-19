"""
Transformer Block (v1.6)

v1.6: FlashAttention 支持、滑动窗口注意力
"""

import logging
import torch
import torch.nn as nn
from typing import Optional

from .attention import RamanujanMultiHeadAttention
from .feedforward import RamanujanFFN
from .ramanujan_initializer import RamanujanInitializer

logger = logging.getLogger('acx_ramanujan')


class RamanujanTransformerBlock(nn.Module):
    """
    Pre-Norm Transformer Block (v1.6)

    v1.6: 支持 FlashAttention-3 和滑动窗口注意力
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.0, activation: str = 'gelu',
                 layer_idx: int = 0,
                 initializer: Optional[RamanujanInitializer] = None,
                 alpha: float = 0.3, lambda_decay: float = 0.5,
                 use_flash_attention: bool = True,
                 sliding_window_size: Optional[int] = None):
        super().__init__()

        if initializer is None:
            initializer = RamanujanInitializer(max_depth=1000)

        self.self_attn = RamanujanMultiHeadAttention(
            d_model, nhead, dropout, layer_idx, initializer,
            alpha=alpha, lambda_decay=lambda_decay,
            use_flash_attention=use_flash_attention,
            sliding_window_size=sliding_window_size,
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = RamanujanFFN(
            d_model, dim_feedforward, dropout, activation, layer_idx, initializer
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        initializer.init_layer_norm(self.norm1)
        initializer.init_layer_norm(self.norm2)

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, is_causal=is_causal)
        x = residual + self.dropout(attn_out)

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))

        return x
