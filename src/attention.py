"""
多头自注意力模块

集成拉马努金初始化 + 自适应缩放，确保注意力层在极深网络中保持梯度稳定。

自适应缩放公式:
    scale(l, d_k) = sqrt(d_k) * (1 + alpha * exp(-lambda_decay * l))

    - 浅层: 缩放更大 → logits 更小 → softmax 更平滑 → 注意力更分散
    - 深层: 收敛到标准 sqrt(d_k) → softmax 更尖锐
    - 与拉马努金系数的指数衰减行为一致
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .ramanujan_initializer import RamanujanInitializer


class RamanujanMultiHeadAttention(nn.Module):
    """
    多头自注意力，支持拉马努金初始化 + 自适应缩放

    标准 MHA:
        Q, K, V = xW_q, xW_k, xW_v
        attn = softmax(QK^T / sqrt(d_k)) V

    自适应缩放:
        attn = softmax(QK^T / scale(l, d_k)) V
        scale(l, d_k) = sqrt(d_k) * (1 + alpha * exp(-lambda_decay * l))
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0,
                 layer_idx: int = 0, initializer: Optional[RamanujanInitializer] = None,
                 alpha: float = 0.3, lambda_decay: float = 0.5):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.layer_idx = layer_idx

        # 自适应缩放: scale(l) = sqrt(d_k) * (1 + alpha * exp(-lambda * l))
        base_scale = math.sqrt(self.d_k)
        adaptive_factor = 1.0 + alpha * math.exp(-lambda_decay * layer_idx)
        self.scale = base_scale * adaptive_factor

        # 投影层
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 拉马努金初始化
        if initializer is None:
            initializer = RamanujanInitializer(max_depth=1000)

        for linear in [self.W_q, self.W_k, self.W_v, self.W_o]:
            initializer.init_linear(linear, layer_idx, nonlinearity='linear')

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (B, T, D)
            key: (B, S, D) or None (self-attention)
            value: (B, S, D) or None (self-attention)
            attn_mask: (T, S) or (B, T, S) or None
            is_causal: 是否使用因果掩码

        Returns:
            output: (B, T, D)
            attn_weights: (B, nhead, T, S) or None
        """
        if key is None:
            key = query
        if value is None:
            value = query

        B, T, _ = query.shape
        S = key.shape[1]

        # 线性投影 + 多头reshape
        Q = self.W_q(query).reshape(B, T, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).reshape(B, S, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).reshape(B, S, self.nhead, self.d_k).transpose(1, 2)

        # 缩放点积注意力
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 因果掩码
        if is_causal:
            causal_mask = torch.triu(
                torch.full((T, S), float('-inf'), device=query.device),
                diagonal=1
            )
            attn = attn + causal_mask.unsqueeze(0).unsqueeze(0)

        # 注意力掩码
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn = attn + attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn = attn + attn_mask.unsqueeze(1)

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        output = torch.matmul(attn_weights, V)

        # 合并多头
        output = output.transpose(1, 2).reshape(B, T, self.d_model)
        output = self.W_o(output)

        return output, attn_weights
