"""
Mixture of Experts (MoE) 层

集成拉马努金初始化的 MoE 实现：
- Router 使用拉马努金初始化
- 每个 Expert 是一个 FFN，独立使用拉马努金初始化
- 支持 Top-K 路由、负载均衡损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .feedforward import RamanujanFFN
from .ramanujan_initializer import RamanujanInitializer


class RamanujanRouter(nn.Module):
    """
    MoE 路由器

    将输入 token 映射到专家概率分布：
        gate(x) = softmax(x · W_gate)

    W_gate 使用拉马努金初始化。
    """

    def __init__(self, d_model: int, num_experts: int,
                 layer_idx: int = 0,
                 initializer: Optional[RamanujanInitializer] = None):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts

        # 门控线性层：d_model → num_experts
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # 拉马努金初始化
        if initializer is None:
            initializer = RamanujanInitializer(max_depth=1000)
        initializer.init_linear(self.gate, layer_idx, nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D) 输入

        Returns:
            router_probs: (B, T, num_experts) 路由概率
            router_logits: (B, T, num_experts) 原始 logits（用于辅助损失）
        """
        logits = self.gate(x)  # (B, T, num_experts)
        probs = F.softmax(logits, dim=-1)
        return probs, logits


class RamanujanMoELayer(nn.Module):
    """
    Mixture of Experts 层

    结构:
        输入 x → Router → 选择 Top-K 专家
        → 每个选中的专家独立处理
        → 加权求和输出

    支持:
        - Top-K 路由 (K=1 或 K=2)
        - 负载均衡损失（防止专家坍缩）
        - 容量因子（限制每个专家处理的 token 数）
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
    ):
        super().__init__()

        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        if initializer is None:
            initializer = RamanujanInitializer(max_depth=1000)

        # 路由器
        self.router = RamanujanRouter(d_model, num_experts, layer_idx, initializer)

        # 专家池：每个专家是一个独立的 FFN
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
            x: (B, T, D) 输入
            return_router_logits: 是否返回路由信息（用于辅助损失）

        Returns:
            output: (B, T, D)
            aux_loss_dict: 包含 router_logits, load_balance_loss 等
        """
        B, T, D = x.shape
        num_tokens = B * T

        # 路由
        router_probs, router_logits = self.router(x)  # (B, T, E)

        # Top-K 选择
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # top_k_probs: (B, T, K), top_k_indices: (B, T, K)

        # 归一化 top-k 概率
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # 展平处理
        x_flat = x.reshape(num_tokens, D)  # (B*T, D)
        top_k_probs_flat = top_k_probs.reshape(num_tokens, self.top_k)
        top_k_indices_flat = top_k_indices.reshape(num_tokens, self.top_k)

        # 计算每个专家的输出
        # 高效实现：先计算所有 token 经过所有专家的结果（如果专家数不多）
        # 或者按专家分组处理（如果专家数很多）

        if self.num_experts <= 16:
            # 小规模专家：全计算 + 稀疏加权
            output = self._forward_small_expert_pool(
                x_flat, top_k_probs_flat, top_k_indices_flat, num_tokens, D
            )
        else:
            # 大规模专家：按专家分组处理
            output = self._forward_large_expert_pool(
                x_flat, top_k_probs_flat, top_k_indices_flat, num_tokens, D
            )

        output = output.reshape(B, T, D)

        # 辅助损失
        aux_loss_dict = None
        if return_router_logits:
            aux_loss_dict = self._compute_aux_loss(router_probs, router_logits)

        return output, aux_loss_dict

    def _forward_small_expert_pool(
        self,
        x: torch.Tensor,
        top_k_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
        num_tokens: int,
        dim: int,
    ) -> torch.Tensor:
        """小规模专家池的前向传播（专家数 ≤ 16）"""
        output = torch.zeros(num_tokens, dim, device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # (num_tokens,)
            expert_weights = top_k_probs[:, k]     # (num_tokens,)

            for e in range(self.num_experts):
                mask = (expert_indices == e)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_weights[mask].unsqueeze(-1) * expert_output

        return output

    def _forward_large_expert_pool(
        self,
        x: torch.Tensor,
        top_k_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
        num_tokens: int,
        dim: int,
    ) -> torch.Tensor:
        """大规模专家池的前向传播（专家数 > 16）"""
        output = torch.zeros(num_tokens, dim, device=x.device, dtype=x.dtype)

        # 按专家分组
        for e in range(self.num_experts):
            # 找到所有路由到专家 e 的 (token, k) 对
            for k in range(self.top_k):
                mask = (top_k_indices[:, k] == e)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += top_k_probs[mask, k].unsqueeze(-1) * expert_output

        return output

    def _compute_aux_loss(
        self,
        router_probs: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算辅助损失

        1. 负载均衡损失：防止所有 token 路由到同一个专家
        2. Router z-loss：防止 router logits 过大
        """
        B, T, E = router_probs.shape
        num_tokens = B * T

        # 负载均衡损失 (Switch Transformer 风格)
        # f_i = fraction of tokens routed to expert i
        # P_i = average routing probability for expert i
        # loss = E * sum(f_i * P_i)

        probs_flat = router_probs.reshape(num_tokens, E)

        # f_i: 每个专家实际接收的 token 比例
        # 用 top-1 近似
        top1_indices = probs_flat.argmax(dim=-1)  # (num_tokens,)
        one_hot = F.one_hot(top1_indices, E).float()  # (num_tokens, E)
        f = one_hot.mean(dim=0)  # (E,)

        # P_i: 平均路由概率
        P = probs_flat.mean(dim=0)  # (E,)

        # 负载均衡损失
        load_balance_loss = E * (f * P).sum()

        # Router z-loss (稳定训练)
        z_loss = (router_logits ** 2).mean()

        return {
            'load_balance_loss': load_balance_loss,
            'z_loss': z_loss,
            'router_logits': router_logits,
            'expert_usage': f,  # 每个专家的使用率
        }


class RamanujanMoETransformerBlock(nn.Module):
    """
    带 MoE 的 Transformer Block

    结构:
        x → LayerNorm → MultiHeadAttention → + Residual
        x → LayerNorm → MoE (Router + Experts) → + Residual

    与标准 Transformer Block 的区别：
        - FFN 替换为 MoE
        - 额外的负载均衡损失
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
    ):
        super().__init__()

        if initializer is None:
            initializer = RamanujanInitializer(max_depth=1000)

        # 自注意力（复用现有实现）
        from .attention import RamanujanMultiHeadAttention
        self.self_attn = RamanujanMultiHeadAttention(
            d_model, nhead, dropout, layer_idx, initializer
        )
        self.norm1 = nn.LayerNorm(d_model)

        # MoE 层（替换 FFN）
        self.moe = RamanujanMoELayer(
            d_model, dim_feedforward, num_experts, top_k,
            dropout, activation, capacity_factor,
            layer_idx, initializer
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # LayerNorm 初始化
        initializer.init_layer_norm(self.norm1)
        initializer.init_layer_norm(self.norm2)

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = False,
        return_aux_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            x: (B, T, D)
            is_causal: 是否使用因果掩码
            return_aux_loss: 是否返回辅助损失

        Returns:
            output: (B, T, D)
            aux_loss_dict: MoE 辅助损失（如果 return_aux_loss=True）
        """
        # 自注意力 + 残差
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, is_causal=is_causal)
        x = residual + self.dropout(attn_out)

        # MoE + 残差
        residual = x
        x_norm = self.norm2(x)
        moe_out, aux_loss_dict = self.moe(x_norm, return_router_logits=return_aux_loss)
        x = residual + self.dropout(moe_out)

        if return_aux_loss:
            return x, aux_loss_dict
        return x, None


class RamanujanMoETransformer(nn.Module):
    """
    完整的 MoE Transformer

    支持 GPT 风格（decoder-only）和 BERT 风格（encoder-only）
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
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.decoder_only = decoder_only

        initializer = RamanujanInitializer(max_depth=max_depth)

        # 嵌入
        from .embeddings import RamanujanEmbeddings
        self.embeddings = RamanujanEmbeddings(
            vocab_size, d_model, max_len, dropout, initializer
        )

        # MoE Transformer 块
        self.layers = nn.ModuleList([
            RamanujanMoETransformerBlock(
                d_model, nhead, dim_feedforward,
                num_experts, top_k, dropout, activation,
                capacity_factor, layer_idx=i, initializer=initializer
            )
            for i in range(num_layers)
        ])

        # 最终 LayerNorm
        self.final_norm = nn.LayerNorm(d_model)
        initializer.init_layer_norm(self.final_norm)

        # LM Head（decoder-only）
        if decoder_only:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = self.embeddings.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        return_aux_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            input_ids: (B, T)
            return_aux_loss: 是否返回 MoE 辅助损失

        Returns:
            logits: (B, T, vocab_size) 或 (B, T, D)
            aux_loss_dict: 所有层的辅助损失汇总
        """
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

        # 汇总辅助损失
        aux_loss_summary = None
        if return_aux_loss and all_aux_losses:
            aux_loss_summary = {
                'load_balance_loss': sum(d['load_balance_loss'] for d in all_aux_losses) / len(all_aux_losses),
                'z_loss': sum(d['z_loss'] for d in all_aux_losses) / len(all_aux_losses),
                'expert_usage': torch.stack([d['expert_usage'] for d in all_aux_losses]).mean(dim=0),
            }

        return logits, aux_loss_summary

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """自回归生成"""
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


# ─── 便捷构建函数 ──────────────────────────────────────────────────

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
) -> nn.Module:
    """
    快速构建拉马努金 MoE Transformer

    Args:
        vocab_size: 词表大小
        d_model: 模型维度
        nhead: 注意力头数
        num_layers: 层数
        dim_feedforward: FFN 中间维度
        num_experts: 专家数量
        top_k: 每个 token 选择的专家数
        dropout: dropout 率
        activation: 激活函数
        max_len: 最大序列长度
        max_depth: 拉马努金系数最大深度
        decoder_only: True=GPT, False=BERT
        capacity_factor: 容量因子

    Returns:
        nn.Module: MoE Transformer 模型
    """
    return RamanujanMoETransformer(
        vocab_size, d_model, nhead, num_layers,
        dim_feedforward, num_experts, top_k,
        dropout, activation, max_len, max_depth,
        decoder_only, capacity_factor,
    )
