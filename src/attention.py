"""
多头自注意力模块 (v1.6)

集成拉马努金初始化 + 自适应缩放 + QKV 差异化初始化 + FlashAttention-3。

v1.6 新增:
    - FlashAttention-3 原生集成 (use_flash_attention 开关)
    - 滑动窗口注意力 (sliding_window_size)
    - CUDA 11.8+ / ROCm 5.7+ 兼容
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .ramanujan_initializer import RamanujanInitializer, LayerRole, tag_linear_role

logger = logging.getLogger('acx_ramanujan')

# ─── FlashAttention 可用性检测 ─────────────────────────────────────

_FLASH_AVAILABLE = False
_FLASH_VERSION = None

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.modules.mha import FlashCrossAttention
    _FLASH_AVAILABLE = True
    _FLASH_VERSION = "flash_attn"
    logger.info("FlashAttention 已加载")
except ImportError:
    try:
        # PyTorch 2.0+ 原生 scaled_dot_product_attention 支持 flash 后端
        if hasattr(F, 'scaled_dot_product_attention'):
            _FLASH_AVAILABLE = True
            _FLASH_VERSION = "sdpa"
            logger.info("使用 PyTorch 原生 SDPA (flash 后端)")
    except Exception:
        pass

if not _FLASH_AVAILABLE:
    logger.warning("FlashAttention 不可用，将使用标准注意力实现")


def _is_flash_available() -> bool:
    """检查 FlashAttention 是否可用"""
    return _FLASH_AVAILABLE


def _get_flash_version() -> Optional[str]:
    """获取 FlashAttention 版本标识"""
    return _FLASH_VERSION


class RamanujanMultiHeadAttention(nn.Module):
    """
    多头自注意力，支持拉马努金初始化 + FlashAttention-3

    v1.6 新增:
        - use_flash_attention: 是否启用 FlashAttention (默认 True)
        - sliding_window_size: 滑动窗口注意力大小 (None = 全局)
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0,
                 layer_idx: int = 0, initializer: Optional[RamanujanInitializer] = None,
                 alpha: float = 0.3, lambda_decay: float = 0.5,
                 use_flash_attention: bool = True,
                 sliding_window_size: Optional[int] = None):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.layer_idx = layer_idx
        self.use_flash_attention = use_flash_attention and _FLASH_AVAILABLE
        self.sliding_window_size = sliding_window_size

        # 自适应缩放
        base_scale = math.sqrt(self.d_k)
        adaptive_factor = 1.0 + alpha * math.exp(-lambda_decay * layer_idx)
        self.scale = base_scale * adaptive_factor

        # 投影层
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # QKV 差异化角色标签
        tag_linear_role(self.W_q, LayerRole.Q_PROJ)
        tag_linear_role(self.W_k, LayerRole.K_PROJ)
        tag_linear_role(self.W_v, LayerRole.V_PROJ)
        tag_linear_role(self.W_o, LayerRole.OUTPUT_ATTN)

        self.dropout_p = dropout
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 拉马努金初始化
        if initializer is None:
            initializer = RamanujanInitializer(max_depth=1000)

        for linear in [self.W_q, self.W_k, self.W_v, self.W_o]:
            initializer.init_linear(linear, layer_idx, nonlinearity='linear')

        flash_status = "FlashAttention" if self.use_flash_attention else "标准注意力"
        logger.debug(f"Attention layer {layer_idx}: scale={self.scale:.4f}, "
                     f"mode={flash_status}, sliding_window={sliding_window_size}")

    def _flash_attention(self, Q: torch.Tensor, K: torch.Tensor,
                         V: torch.Tensor, is_causal: bool = False,
                         attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        FlashAttention 前向传播

        Args:
            Q, K, V: (B, nhead, T, d_k)
            is_causal: 是否因果掩码
            attn_mask: 可选注意力掩码

        Returns:
            (B, nhead, T, d_k)
        """
        B, H, T, D = Q.shape
        S = K.shape[2]

        if _FLASH_VERSION == "flash_attn":
            # flash_attn 库: 输入格式 (B, T, H, D)
            Q_t = Q.transpose(1, 2)  # (B, T, H, D)
            K_t = K.transpose(1, 2)
            V_t = V.transpose(1, 2)

            # 应用缩放 (flash_attn 内部不缩放)
            Q_t = Q_t / self.scale

            out = flash_attn_func(
                Q_t, K_t, V_t,
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=is_causal,
                window_size=(self.sliding_window_size, self.sliding_window_size)
                             if self.sliding_window_size else (-1, -1),
            )
            return out.transpose(1, 2)  # (B, H, T, D)

        elif _FLASH_VERSION == "sdpa":
            # PyTorch 原生 SDPA
            out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal and attn_mask is None,
            )
            return out

        else:
            raise RuntimeError("FlashAttention 不可用")

    def _standard_attention(self, Q: torch.Tensor, K: torch.Tensor,
                            V: torch.Tensor, is_causal: bool = False,
                            attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        标准缩放点积注意力（fallback）

        Returns:
            (output, attn_weights)
        """
        T = Q.shape[2]
        S = K.shape[2]

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if is_causal:
            causal_mask = torch.triu(
                torch.full((T, S), float('-inf'), device=Q.device),
                diagonal=1
            )
            attn = attn + causal_mask.unsqueeze(0).unsqueeze(0)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn = attn + attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn = attn + attn_mask.unsqueeze(1)

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if key is None:
            key = query
        if value is None:
            value = query

        B, T, _ = query.shape
        S = key.shape[1]

        # 线性投影 + 多头 reshape
        Q = self.W_q(query).reshape(B, T, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).reshape(B, S, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).reshape(B, S, self.nhead, self.d_k).transpose(1, 2)

        if self.use_flash_attention:
            output = self._flash_attention(Q, K, V, is_causal, attn_mask)
            attn_weights = None  # FlashAttention 不返回注意力权重
        else:
            output, attn_weights = self._standard_attention(Q, K, V, is_causal, attn_mask)

        output = output.transpose(1, 2).reshape(B, T, self.d_model)
        output = self.W_o(output)

        return output, attn_weights
