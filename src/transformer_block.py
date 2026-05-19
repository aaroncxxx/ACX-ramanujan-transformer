"""
Transformer Block

集成拉马努金初始化的完整 Transformer 编码器/解码器块。
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import RamanujanMultiHeadAttention
from .feedforward import RamanujanFFN
from .ramanujan_initializer import RamanujanInitializer


class RamanujanTransformerBlock(nn.Module):
    """
    Pre-Norm Transformer Block

    结构:
        x → LayerNorm → MultiHeadAttention → + Residual
        x → LayerNorm → FFN → + Residual

    拉马努金初始化应用于所有线性层。
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.0, activation: str = 'gelu',
                 layer_idx: int = 0,
                 initializer: Optional[RamanujanInitializer] = None,
                 alpha: float = 0.3, lambda_decay: float = 0.5):
        super().__init__()

        if initializer is None:
            initializer = RamanujanInitializer(max_depth=1000)

        # 自注意力（自适应缩放）
        self.self_attn = RamanujanMultiHeadAttention(
            d_model, nhead, dropout, layer_idx, initializer,
            alpha=alpha, lambda_decay=lambda_decay
        )
        self.norm1 = nn.LayerNorm(d_model)

        # 前馈网络
        self.ffn = RamanujanFFN(
            d_model, dim_feedforward, dropout, activation, layer_idx, initializer
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # LayerNorm 保持标准初始化
        initializer.init_layer_norm(self.norm1)
        initializer.init_layer_norm(self.norm2)

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            is_causal: 是否使用因果掩码（解码器用）
        Returns:
            (B, T, D)
        """
        # 自注意力 + 残差
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, is_causal=is_causal)
        x = residual + self.dropout(attn_out)

        # FFN + 残差
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))

        return x
