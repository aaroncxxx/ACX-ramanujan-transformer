"""
完整 Transformer 模型 (v1.6)

v1.6 新增:
    - FlashAttention-3 支持
    - 滑动窗口注意力
    - 混合精度训练支持
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
    """Transformer 编码器（BERT 风格）(v1.6)"""

    def __init__(self, vocab_size: int, d_model: int = 768,
                 nhead: int = 12, num_layers: int = 12,
                 dim_feedforward: int = 3072, dropout: float = 0.1,
                 activation: str = 'gelu',
                 max_len: int = 512,
                 max_depth: int = 1000,
                 alpha: float = 0.3, lambda_decay: float = 0.5,
                 gradient_checkpointing: bool = False,
                 quantization: str = 'none',
                 long_context_seq_len: int = 512,
                 use_flash_attention: bool = True,
                 sliding_window_size: Optional[int] = None):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.gradient_checkpointing = gradient_checkpointing

        initializer = RamanujanInitializer(
            max_depth=max_depth, num_layers=num_layers,
            quantization=quantization,
            long_context_seq_len=long_context_seq_len
        )

        self.embeddings = RamanujanEmbeddings(
            vocab_size, d_model, max_len, dropout, initializer
        )

        self.layers = nn.ModuleList([
            RamanujanTransformerBlock(
                d_model, nhead, dim_feedforward, dropout, activation,
                layer_idx=i, initializer=initializer,
                alpha=alpha, lambda_decay=lambda_decay,
                use_flash_attention=use_flash_attention,
                sliding_window_size=sliding_window_size,
            )
            for i in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        initializer.init_layer_norm(self.final_norm)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    """Transformer 解码器（GPT 风格）(v1.6)"""

    def __init__(self, vocab_size: int, d_model: int = 768,
                 nhead: int = 12, num_layers: int = 12,
                 dim_feedforward: int = 3072, dropout: float = 0.1,
                 activation: str = 'gelu',
                 max_len: int = 2048,
                 max_depth: int = 1000,
                 alpha: float = 0.3, lambda_decay: float = 0.5,
                 gradient_checkpointing: bool = False,
                 quantization: str = 'none',
                 long_context_seq_len: int = 2048,
                 use_flash_attention: bool = True,
                 sliding_window_size: Optional[int] = None):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.gradient_checkpointing = gradient_checkpointing

        initializer = RamanujanInitializer(
            max_depth=max_depth, num_layers=num_layers,
            quantization=quantization,
            long_context_seq_len=long_context_seq_len
        )

        self.embeddings = RamanujanEmbeddings(
            vocab_size, d_model, max_len, dropout, initializer
        )

        self.layers = nn.ModuleList([
            RamanujanTransformerBlock(
                d_model, nhead, dim_feedforward, dropout, activation,
                layer_idx=i, initializer=initializer,
                alpha=alpha, lambda_decay=lambda_decay,
                use_flash_attention=use_flash_attention,
                sliding_window_size=sliding_window_size,
            )
            for i in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        initializer.init_layer_norm(self.final_norm)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
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
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= 2048 else input_ids[:, -2048:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, idx_next], dim=1)

        return input_ids


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
    use_flash_attention: bool = True,
    sliding_window_size: Optional[int] = None,
) -> nn.Module:
    """
    快速构建拉马努金 Transformer (v1.6)

    v1.6 新增:
        use_flash_attention: 是否启用 FlashAttention-3 (默认 True)
        sliding_window_size: 滑动窗口注意力大小 (None = 全局)
    """
    logger.info(f"构建 Transformer: vocab={vocab_size}, d={d_model}, "
                f"layers={num_layers}, heads={nhead}, flash={use_flash_attention}")

    if decoder_only:
        return RamanujanTransformerDecoder(
            vocab_size, d_model, nhead, num_layers,
            dim_feedforward, dropout, activation,
            max_len, max_depth, alpha, lambda_decay,
            gradient_checkpointing, quantization, long_context_seq_len,
            use_flash_attention, sliding_window_size,
        )
    else:
        return RamanujanTransformerEncoder(
            vocab_size, d_model, nhead, num_layers,
            dim_feedforward, dropout, activation,
            max_len, max_depth, alpha, lambda_decay,
            gradient_checkpointing, quantization, long_context_seq_len,
            use_flash_attention, sliding_window_size,
        )
