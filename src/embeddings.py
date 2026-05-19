"""
嵌入模块

Token 嵌入 + 可学习/正弦位置编码。
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from .ramanujan_initializer import RamanujanInitializer


class RamanujanPositionalEncoding(nn.Module):
    """
    正弦位置编码（标准版）

    PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RamanujanEmbeddings(nn.Module):
    """
    Token 嵌入 + 位置编码

    - Token 嵌入使用标准正态初始化（N(0, 0.02)）
    - 位置编码使用正弦版本
    """

    def __init__(self, vocab_size: int, d_model: int,
                 max_len: int = 5000, dropout: float = 0.1,
                 initializer: Optional[RamanujanInitializer] = None):
        super().__init__()

        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = RamanujanPositionalEncoding(d_model, max_len, dropout)

        if initializer is None:
            initializer = RamanujanInitializer(max_depth=1000)
        initializer.init_embedding(self.token_embedding, scale=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T) token indices
        Returns:
            (B, T, D) embedded representations
        """
        # sqrt(d_model) 缩放，同原始 Transformer
        embeddings = self.token_embedding(x) * math.sqrt(self.d_model)
        return self.positional_encoding(embeddings)
