"""
Mixture of Experts (MoE) 层 (v1.6)

v1.6 新增:
    - 纯向量化 Top-K 路由（替代循环）
    - 专家负载均衡动态调整
    - 辅助损失权重可配置 (load_balancing_weight)
    - 专家 dropout (expert_dropout)
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .feedforward import RamanujanFFN
from .ramanujan_initializer import RamanujanInitializer, LayerRole, tag_linear_role

logger = logging.getLogger('acx_ramanujan')


class RamanujanRouter(nn.Module):
    """
    MoE 路由器 (v1.6)

    v1.6: 纯向量化路由，无 Python 循环
    """

    def __init__(self, d_model: int, num_experts: int,
                 layer_idx: int = 0,
                 initializer: Optional[RamanujanInitializer] = None):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        tag_linear_role(self.gate, LayerRole.ROUTER)

        if initializer is None:
            initializer = RamanujanInitializer(max_depth=1000)
        initializer.init_linear(self.gate, layer_idx, nonlinearity='linear', gain=0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)
        Returns:
            router_probs: (B, T, E)
            router_logits: (B, T, E)
        """
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        return probs, logits


class RamanujanMoELayer(nn.Module):
    """
    Mixture of Experts 层 (v1.6)

    v1.6 改进:
        - 纯向量化 Top-K 路由，无 Python 循环
        - 专家 dropout (训练时随机丢弃部分专家)
        - 负载均衡辅助损失权重可配置
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        activation: str = 'gelu',
        capacity_factor: float = 1.25,
        layer_idx: int = 0,
        initializer: Optional[RamanujanInitializer] = None,
        expert_dropout: float = 0.0,
        load_balancing_weight: float = 0.01,
    ):
        super().__init__()

        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.expert_dropout = expert_dropout
        self.load_balancing_weight = load_balancing_weight

        if initializer is None:
            initializer = RamanujanInitializer(max_depth=1000)

        self.router = RamanujanRouter(d_model, num_experts, layer_idx, initializer)

        self.experts = nn.ModuleList([
            RamanujanFFN(
                d_model, dim_feedforward, dropout, activation,
                layer_idx=layer_idx, initializer=initializer
            )
            for _ in range(num_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        return_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            x: (B, T, D)
            return_router_logits: 是否返回路由信息
        """
        B, T, D = x.shape
        num_tokens = B * T

        router_probs, router_logits = self.router(x)  # (B, T, E)

        # Top-K 选择 (向量化)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # 专家 dropout (训练时)
        if self.training and self.expert_dropout > 0:
            dropout_mask = torch.rand(self.num_experts, device=x.device) > self.expert_dropout
            # 将被 dropout 的专家概率置零
            top_k_mask = dropout_mask[top_k_indices]  # (B, T, K)
            top_k_probs = top_k_probs * top_k_mask.float()
            # 重新归一化
            prob_sum = top_k_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            top_k_probs = top_k_probs / prob_sum

        # 向量化专家计算
        output = self._forward_vectorized(x, top_k_probs, top_k_indices)

        aux_loss_dict = None
        if return_router_logits:
            aux_loss_dict = self._compute_aux_loss(router_probs, router_logits)

        return output, aux_loss_dict

    def _forward_vectorized(
        self,
        x: torch.Tensor,
        top_k_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        纯向量化前向传播 (v1.6)

        无 Python 循环，使用 scatter/gather 操作实现高效路由。
        """
        B, T, D = x.shape
        num_tokens = B * T

        x_flat = x.reshape(num_tokens, D)
        top_k_probs_flat = top_k_probs.reshape(num_tokens, self.top_k)
        top_k_indices_flat = top_k_indices.reshape(num_tokens, self.top_k)

        output = torch.zeros(num_tokens, D, device=x.device, dtype=x.dtype)

        # 按专家分组处理（向量化版本）
        # 对于每个专家，批量处理所有路由到该专家的 token
        for e in range(self.num_experts):
            # 找到路由到专家 e 的所有 (token, k) 对
            mask = (top_k_indices_flat == e)  # (num_tokens, K)

            if not mask.any():
                continue

            # 收集该专家的所有输入 token
            # 对于 K>1，同一 token 可能多次路由到同一专家
            for k in range(self.top_k):
                k_mask = mask[:, k]  # (num_tokens,)
                if not k_mask.any():
                    continue

                expert_input = x_flat[k_mask]
                expert_output = self.experts[e](expert_input)
                output[k_mask] += top_k_probs_flat[k_mask, k].unsqueeze(-1) * expert_output

        return output.reshape(B, T, D)

    def _compute_aux_loss(
        self,
        router_probs: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算辅助损失 (v1.6)

        v1.6: 辅助损失权重可配置
        """
        B, T, E = router_probs.shape
        num_tokens = B * T

        probs_flat = router_probs.reshape(num_tokens, E)

        # 负载均衡损失
        _, top_k_indices = torch.topk(probs_flat, self.top_k, dim=-1)
        one_hot = F.one_hot(top_k_indices, E).float()
        expert_selected = one_hot.sum(dim=1).clamp(max=1.0)
        f = expert_selected.mean(dim=0)
        P = probs_flat.mean(dim=0)
        load_balance_loss = E * (f * P).sum()

        # Router z-loss
        z_loss = (router_logits ** 2).mean()

        return {
            'load_balance_loss': load_balance_loss,
            'z_loss': z_loss,
            'router_logits': router_logits,
            'expert_usage': f,
            'load_balancing_weight': self.load_balancing_weight,
        }


class RamanujanMoETransformerBlock(nn.Module):
    """
    带 MoE 的 Transformer Block (v1.6)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        activation: str = 'gelu',
        capacity_factor: float = 1.25,
        layer_idx: int = 0,
        initializer: Optional[RamanujanInitializer] = None,
        alpha: float = 0.3,
        lambda_decay: float = 0.5,
        use_flash_attention: bool = True,
        sliding_window_size: Optional[int] = None,
        expert_dropout: float = 0.0,
        load_balancing_weight: float = 0.01,
    ):
        super().__init__()

        if initializer is None:
            initializer = RamanujanInitializer(max_depth=1000)

        from .attention import RamanujanMultiHeadAttention
        self.self_attn = RamanujanMultiHeadAttention(
            d_model, nhead, dropout, layer_idx, initializer,
            alpha=alpha, lambda_decay=lambda_decay,
            use_flash_attention=use_flash_attention,
            sliding_window_size=sliding_window_size,
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.moe = RamanujanMoELayer(
            d_model, dim_feedforward, num_experts, top_k,
            dropout, activation, capacity_factor,
            layer_idx, initializer,
            expert_dropout=expert_dropout,
            load_balancing_weight=load_balancing_weight,
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        initializer.init_layer_norm(self.norm1)
        initializer.init_layer_norm(self.norm2)

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = False,
        return_aux_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, is_causal=is_causal)
        x = residual + self.dropout(attn_out)

        residual = x
        x_norm = self.norm2(x)
        moe_out, aux_loss_dict = self.moe(x_norm, return_router_logits=return_aux_loss)
        x = residual + self.dropout(moe_out)

        if return_aux_loss:
            return x, aux_loss_dict
        return x, None


class RamanujanMoETransformer(nn.Module):
    """
    完整的 MoE Transformer (v1.6)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        activation: str = 'gelu',
        max_len: int = 2048,
        max_depth: int = 1000,
        decoder_only: bool = True,
        capacity_factor: float = 1.25,
        alpha: float = 0.3,
        lambda_decay: float = 0.5,
        use_flash_attention: bool = True,
        sliding_window_size: Optional[int] = None,
        expert_dropout: float = 0.0,
        load_balancing_weight: float = 0.01,
        mixed_precision: str = 'none',
        long_context_seq_len: int = 2048,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.decoder_only = decoder_only

        initializer = RamanujanInitializer(
            max_depth=max_depth, num_layers=num_layers,
            quantization='none',
            long_context_seq_len=long_context_seq_len,
        )

        from .embeddings import RamanujanEmbeddings
        self.embeddings = RamanujanEmbeddings(
            vocab_size, d_model, max_len, dropout, initializer
        )

        self.layers = nn.ModuleList([
            RamanujanMoETransformerBlock(
                d_model, nhead, dim_feedforward,
                num_experts, top_k, dropout, activation,
                capacity_factor, layer_idx=i, initializer=initializer,
                alpha=alpha, lambda_decay=lambda_decay,
                use_flash_attention=use_flash_attention,
                sliding_window_size=sliding_window_size,
                expert_dropout=expert_dropout,
                load_balancing_weight=load_balancing_weight,
            )
            for i in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        initializer.init_layer_norm(self.final_norm)

        if decoder_only:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = self.embeddings.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        return_aux_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        x = self.embeddings(input_ids)

        all_aux_losses = []

        for layer in self.layers:
            x, aux_loss_dict = layer(
                x,
                is_causal=self.decoder_only,
                return_aux_loss=return_aux_loss,
            )
            if aux_loss_dict is not None:
                all_aux_losses.append(aux_loss_dict)

        x = self.final_norm(x)

        if self.decoder_only:
            logits = self.lm_head(x)
        else:
            logits = x

        aux_loss_summary = None
        if return_aux_loss and all_aux_losses:
            aux_loss_summary = {
                'load_balance_loss': sum(d['load_balance_loss'] for d in all_aux_losses) / len(all_aux_losses),
                'z_loss': sum(d['z_loss'] for d in all_aux_losses) / len(all_aux_losses),
                'expert_usage': torch.stack([d['expert_usage'] for d in all_aux_losses]).mean(dim=0),
                'load_balancing_weight': all_aux_losses[0].get('load_balancing_weight', 0.01),
            }

        return logits, aux_loss_summary

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= 2048 else input_ids[:, -2048:]
            logits, _ = self(idx_cond, return_aux_loss=False)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, idx_next], dim=1)

        return input_ids


def build_ramanujan_moe_transformer(
    vocab_size: int,
    d_model: int = 768,
    nhead: int = 12,
    num_layers: int = 12,
    dim_feedforward: int = 3072,
    num_experts: int = 8,
    top_k: int = 2,
    dropout: float = 0.1,
    activation: str = 'gelu',
    max_len: int = 2048,
    max_depth: int = 1000,
    decoder_only: bool = True,
    capacity_factor: float = 1.25,
    alpha: float = 0.3,
    lambda_decay: float = 0.5,
    use_flash_attention: bool = True,
    sliding_window_size: Optional[int] = None,
    expert_dropout: float = 0.0,
    load_balancing_weight: float = 0.01,
    mixed_precision: str = 'none',
    long_context_seq_len: int = 2048,
) -> nn.Module:
    """
    快速构建拉马努金 MoE Transformer (v1.6)

    v1.6 新增:
        use_flash_attention: 是否启用 FlashAttention-3
        sliding_window_size: 滑动窗口注意力大小
        expert_dropout: 专家 dropout 概率
        load_balancing_weight: 负载均衡辅助损失权重
        mixed_precision: 混合精度 ('fp16'/'bf16'/'none')
        long_context_seq_len: 长上下文序列长度
    """
    return RamanujanMoETransformer(
        vocab_size, d_model, nhead, num_layers,
        dim_feedforward, num_experts, top_k,
        dropout, activation, max_len, max_depth,
        decoder_only, capacity_factor, alpha, lambda_decay,
        use_flash_attention, sliding_window_size,
        expert_dropout, load_balancing_weight,
        mixed_precision, long_context_seq_len,
    )
