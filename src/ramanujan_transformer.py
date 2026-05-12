"""
完整 Transformer 模型

支持编码器-only（BERT风格）和编码器-解码器（GPT风格）架构。
"""

import torch
import torch.nn as nn
from typing import Optional

from .embeddings import RamanujanEmbeddings
from .transformer_block import RamanujanTransformerBlock
from .ramanujan_initializer import RamanujanInitializer


class RamanujanTransformerEncoder(nn.Module):
    """
    Transformer 编码器（BERT / RoBERTa 风格）

    - Token Embedding + Positional Encoding
    - N × Transformer Block (self-attention, no causal mask)
    - Final LayerNorm
    """

    def __init__(self, vocab_size: int, d_model: int = 768,
                 nhead: int = 12, num_layers: int = 12,
                 dim_feedforward: int = 3072, dropout: float = 0.1,
                 activation: str = 'gelu',
                 max_len: int = 512,
                 max_depth: int = 1000):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        initializer = RamanujanInitializer(max_depth=max_depth)

        # 嵌入
        self.embeddings = RamanujanEmbeddings(
            vocab_size, d_model, max_len, dropout, initializer
        )

        # Transformer 块堆叠
        self.layers = nn.ModuleList([
            RamanujanTransformerBlock(
                d_model, nhead, dim_feedforward, dropout, activation,
                layer_idx=i, initializer=initializer
            )
            for i in range(num_layers)
        ])

        # 最终 LayerNorm
        self.final_norm = nn.LayerNorm(d_model)
        initializer.init_layer_norm(self.final_norm)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) token indices
            attention_mask: (B, T) padding mask (1=valid, 0=pad)

        Returns:
            (B, T, D) hidden states
        """
        x = self.embeddings(input_ids)

        for layer in self.layers:
            x = layer(x, is_causal=False)

        return self.final_norm(x)


class RamanujanTransformerDecoder(nn.Module):
    """
    Transformer 解码器（GPT 风格）

    - Token Embedding + Positional Encoding
    - N × Transformer Block (causal self-attention)
    - LM Head
    """

    def __init__(self, vocab_size: int, d_model: int = 768,
                 nhead: int = 12, num_layers: int = 12,
                 dim_feedforward: int = 3072, dropout: float = 0.1,
                 activation: str = 'gelu',
                 max_len: int = 2048,
                 max_depth: int = 1000):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        initializer = RamanujanInitializer(max_depth=max_depth)

        # 嵌入
        self.embeddings = RamanujanEmbeddings(
            vocab_size, d_model, max_len, dropout, initializer
        )

        # Transformer 块堆叠
        self.layers = nn.ModuleList([
            RamanujanTransformerBlock(
                d_model, nhead, dim_feedforward, dropout, activation,
                layer_idx=i, initializer=initializer
            )
            for i in range(num_layers)
        ])

        # 最终 LayerNorm
        self.final_norm = nn.LayerNorm(d_model)
        initializer.init_layer_norm(self.final_norm)

        # LM Head（与 token embedding 权重共享）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) token indices

        Returns:
            (B, T, vocab_size) logits
        """
        x = self.embeddings(input_ids)

        for layer in self.layers:
            x = layer(x, is_causal=True)

        x = self.final_norm(x)
        return self.lm_head(x)

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """
        自回归生成

        Args:
            input_ids: (B, T) prompt
            max_new_tokens: 最大生成长度
            temperature: 采样温度
            top_k: top-k 采样

        Returns:
            (B, T + max_new_tokens) 完整序列
        """
        for _ in range(max_new_tokens):
            # 截断到最大长度
            idx_cond = input_ids if input_ids.size(1) <= 2048 else input_ids[:, -2048:]

            # 前向传播
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k 过滤
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, idx_next], dim=1)

        return input_ids


# ─── 便捷构建函数 ──────────────────────────────────────────────────

def build_ramanujan_transformer(
    vocab_size: int,
    d_model: int = 768,
    nhead: int = 12,
    num_layers: int = 12,
    dim_feedforward: int = 3072,
    dropout: float = 0.1,
    activation: str = 'gelu',
    max_len: int = 512,
    max_depth: int = 1000,
    decoder_only: bool = True,
) -> nn.Module:
    """
    快速构建拉马努金 Transformer

    Args:
        vocab_size: 词表大小
        d_model: 模型维度
        nhead: 注意力头数
        num_layers: 层数
        dim_feedforward: FFN 中间维度
        dropout: dropout 率
        activation: 激活函数 ('gelu', 'relu', 'silu')
        max_len: 最大序列长度
        max_depth: 拉马努金系数最大深度
        decoder_only: True=GPT风格, False=BERT风格

    Returns:
        nn.Module: 完整的 Transformer 模型
    """
    if decoder_only:
        return RamanujanTransformerDecoder(
            vocab_size, d_model, nhead, num_layers,
            dim_feedforward, dropout, activation,
            max_len, max_depth
        )
    else:
        return RamanujanTransformerEncoder(
            vocab_size, d_model, nhead, num_layers,
            dim_feedforward, dropout, activation,
            max_len, max_depth
        )
