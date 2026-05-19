"""
完整 Transformer 模型 (v1.5)

支持编码器-only（BERT风格）和编码器-解码器（GPT风格）架构。
v1.5: 量化友好初始化、长上下文支持、自适应权重衰减
"""

import logging
import torch
import torch.nn as nn
from typing import Optional

from .embeddings import RamanujanEmbeddings
from .transformer_block import RamanujanTransformerBlock
from .ramanujan_initializer import RamanujanInitializer, compute_optimal_depth

logger = logging.getLogger('acx_ramanujan')


class RamanujanTransformerEncoder(nn.Module):
    """
    Transformer 编码器（BERT / RoBERTa 风格）

    - Token Embedding + Positional Encoding
    - N × Transformer Block (self-attention, no causal mask)
    - Final LayerNorm
    - v1.4: 梯度检查点支持
    """

    def __init__(self, vocab_size: int, d_model: int = 768,
                 nhead: int = 12, num_layers: int = 12,
                 dim_feedforward: int = 3072, dropout: float = 0.1,
                 activation: str = 'gelu',
                 max_len: int = 512,
                 max_depth: int = 1000,
                 alpha: float = 0.3, lambda_decay: float = 0.5,
                 gradient_checkpointing: bool = False,
                 quantization: str = 'none',
                 long_context_seq_len: int = 512):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.gradient_checkpointing = gradient_checkpointing

        initializer = RamanujanInitializer(
            max_depth=max_depth, num_layers=num_layers,
            quantization=quantization,
            long_context_seq_len=long_context_seq_len
        )

        # 嵌入
        self.embeddings = RamanujanEmbeddings(
            vocab_size, d_model, max_len, dropout, initializer
        )

        # Transformer 块堆叠
        self.layers = nn.ModuleList([
            RamanujanTransformerBlock(
                d_model, nhead, dim_feedforward, dropout, activation,
                layer_idx=i, initializer=initializer,
                alpha=alpha, lambda_decay=lambda_decay
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
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, use_reentrant=False
                )
            else:
                x = layer(x, is_causal=False)

        return self.final_norm(x)


class RamanujanTransformerDecoder(nn.Module):
    """
    Transformer 解码器（GPT 风格）

    - Token Embedding + Positional Encoding
    - N × Transformer Block (causal self-attention)
    - LM Head
    - v1.4: 梯度检查点支持
    """

    def __init__(self, vocab_size: int, d_model: int = 768,
                 nhead: int = 12, num_layers: int = 12,
                 dim_feedforward: int = 3072, dropout: float = 0.1,
                 activation: str = 'gelu',
                 max_len: int = 2048,
                 max_depth: int = 1000,
                 alpha: float = 0.3, lambda_decay: float = 0.5,
                 gradient_checkpointing: bool = False,
                 quantization: str = 'none',
                 long_context_seq_len: int = 2048):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.gradient_checkpointing = gradient_checkpointing

        initializer = RamanujanInitializer(
            max_depth=max_depth, num_layers=num_layers,
            quantization=quantization,
            long_context_seq_len=long_context_seq_len
        )

        # 嵌入
        self.embeddings = RamanujanEmbeddings(
            vocab_size, d_model, max_len, dropout, initializer
        )

        # Transformer 块堆叠
        self.layers = nn.ModuleList([
            RamanujanTransformerBlock(
                d_model, nhead, dim_feedforward, dropout, activation,
                layer_idx=i, initializer=initializer,
                alpha=alpha, lambda_decay=lambda_decay
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
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, True, use_reentrant=False
                )
            else:
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
    alpha: float = 0.3,
    lambda_decay: float = 0.5,
    gradient_checkpointing: bool = False,
    quantization: str = 'none',
    long_context_seq_len: int = 512,
) -> nn.Module:
    """
    快速构建拉马努金 Transformer (v1.5)

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
        alpha: 自适应缩放修正幅度 (默认 0.3)
        lambda_decay: 自适应缩放衰减速率 (默认 0.5)
        gradient_checkpointing: 启用梯度检查点以节省显存 (默认 False)
        quantization: 量化精度 ('none', 'int8', 'fp8', 'int4')
        long_context_seq_len: 长上下文序列长度 (用于 RoPE 方差修正)

    Returns:
        nn.Module: 完整的 Transformer 模型
    """
    logger.info(f"构建 Transformer: vocab={vocab_size}, d={d_model}, "
                f"layers={num_layers}, heads={nhead}, quantization={quantization}")

    if decoder_only:
        return RamanujanTransformerDecoder(
            vocab_size, d_model, nhead, num_layers,
            dim_feedforward, dropout, activation,
            max_len, max_depth, alpha, lambda_decay,
            gradient_checkpointing, quantization, long_context_seq_len
        )
    else:
        return RamanujanTransformerEncoder(
            vocab_size, d_model, nhead, num_layers,
            dim_feedforward, dropout, activation,
            max_len, max_depth, alpha, lambda_decay,
            gradient_checkpointing, quantization, long_context_seq_len
        )
