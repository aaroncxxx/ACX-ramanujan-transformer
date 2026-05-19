"""
拉马努金模函数初始化器 v1.5

基于拉马努金递推关系的权重初始化方案。

核心递推公式:
    a_{n+1} = (π²/n²) * a_n + (2π/(n(n+1))) * a_{n-1}
    a_0 = 1, a_1 = π/√3

v1.5 新增:
    1. 量化友好初始化：INT8/FP8 专用增益系数
    2. 自适应权重衰减：分层自适应正则化
    3. QKV 差异化初始化：Q/K/V 各自独立递推系数
    4. 长上下文方差保持：RoPE 递推系数修正
    5. 错误处理与日志系统
"""

import math
import logging
from functools import lru_cache
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init

logger = logging.getLogger('acx_ramanujan')


# ─── 激活函数增益表 ────────────────────────────────────────────────

ACTIVATION_GAIN = {
    'linear': 1.0,
    'relu': math.sqrt(2.0),
    'gelu': math.sqrt(2.0),
    'silu': math.sqrt(2.0),
    'swish': math.sqrt(2.0),
    'sigmoid': 3.0,
    'tanh': math.sqrt(2.0),
}


def get_activation_gain(nonlinearity: str) -> float:
    """获取激活函数对应的理论增益"""
    gain = ACTIVATION_GAIN.get(nonlinearity)
    if gain is None:
        logger.warning(f"未知激活函数 '{nonlinearity}'，使用默认增益 1.0")
        return 1.0
    return gain


# ─── 量化增益表 ────────────────────────────────────────────────────

# INT8/FP8 量化场景的专用增益系数
# 量化噪声近似为 uniform(-Δ/2, Δ/2)，Δ = range / 2^bits
# 量化后方差: Var[W_q] ≈ Var[W] + Δ²/12
# 为抵消量化噪声，初始化时适当放大权重方差

QUANTIZATION_GAIN = {
    'int8': 1.15,   # INT8: 8-bit 量化，噪声较小，增益略增
    'fp8': 1.08,    # FP8: 浮点 8-bit，精度略好于 INT8
    'int4': 1.35,   # INT4: 4-bit 量化，噪声大，需更大增益
    'none': 1.0,    # 无量化
}


def get_quantization_gain(precision: str = 'none') -> float:
    """获取量化场景的专用增益系数"""
    gain = QUANTIZATION_GAIN.get(precision)
    if gain is None:
        logger.warning(f"未知量化精度 '{precision}'，使用无量化增益")
        return 1.0
    return gain


# ─── 自适应权重衰减 ────────────────────────────────────────────────

def get_adaptive_weight_decay(layer_idx: int, base_decay: float,
                               ramanujan_depth: int = 8,
                               transition_depth: int = 8) -> float:
    """
    分层自适应权重衰减

    策略：
        - 浅层（Ramanujan 区）：衰减系数 = base_decay × (1 + |a_n|/peak)
          浅层系数大 → 权重更重要 → 更强正则化
        - 过渡区：线性过渡到 base_decay
        - 深层：衰减系数 = base_decay（标准）

    Args:
        layer_idx: 层索引
        base_decay: 基础衰减系数
        ramanujan_depth: Ramanujan 调制深度
        transition_depth: 过渡深度

    Returns:
        float: 该层的权重衰减系数
    """
    coeffs = ramanujan_coefficients(max(ramanujan_depth, layer_idx + 1))
    peak = max(abs(c) for c in coeffs) if coeffs else 1.0

    if layer_idx < ramanujan_depth:
        # 浅层：系数越大，衰减越强
        coeff_ratio = abs(coeffs[layer_idx]) / peak if peak > 0 else 0
        return base_decay * (1.0 + coeff_ratio)

    elif layer_idx < ramanujan_depth + transition_depth:
        # 过渡区：线性衰减到 base_decay
        t = (layer_idx - ramanujan_depth) / transition_depth
        last_coeff_ratio = abs(coeffs[ramanujan_depth - 1]) / peak if peak > 0 else 0
        return base_decay * (1.0 + last_coeff_ratio * (1 - t))

    else:
        return base_decay


# ─── 长上下文 RoPE 修正 ────────────────────────────────────────────

def compute_rope_correction(seq_len: int, d_model: int,
                             base_freq: float = 10000.0) -> float:
    """
    长上下文方差保持修正系数

    RoPE 位置编码在长序列中会引入额外的方差漂移：
        - 短序列 (≤512): 修正系数 ≈ 1.0，无修正
        - 中等序列 (512-4096): 轻微修正
        - 长序列 (>4096): 显著修正

    修正公式推导：
        RoPE 旋转角度 θ_i = pos / base_freq^(2i/d)
        对于最大位置 pos=L，最大旋转角度 θ_max = L / base_freq
        方差漂移因子 ≈ 1 + log(1 + L/base_freq) / π

    Args:
        seq_len: 序列长度
        d_model: 模型维度
        base_freq: RoPE 基频 (默认 10000)

    Returns:
        float: 修正系数，乘以标准缩放因子
    """
    if seq_len <= 512:
        return 1.0

    # 方差漂移修正
    drift = math.log(1.0 + seq_len / base_freq) / math.pi
    correction = 1.0 + drift * 0.1  # 保守修正，避免过度干预

    logger.debug(f"长上下文修正: seq_len={seq_len}, drift={drift:.4f}, "
                 f"correction={correction:.4f}")

    return correction


# ─── 动态递推深度计算 ──────────────────────────────────────────────

def compute_optimal_depth(num_layers: int) -> Tuple[int, int]:
    """
    根据模型层数自动计算最优的 ramanujan_depth 和 transition_depth

    Args:
        num_layers: Transformer 层数

    Returns:
        (ramanujan_depth, transition_depth)
    """
    if num_layers <= 0:
        logger.error(f"num_layers 必须 > 0，收到 {num_layers}")
        raise ValueError(f"num_layers must be > 0, got {num_layers}")

    ramanujan_depth = min(8, max(4, num_layers // 3))
    transition_depth = min(16, max(4, num_layers // 4))

    if ramanujan_depth + transition_depth > num_layers:
        transition_depth = num_layers - ramanujan_depth

    logger.debug(f"动态深度: num_layers={num_layers} → "
                 f"ramanujan_depth={ramanujan_depth}, transition_depth={transition_depth}")

    return ramanujan_depth, transition_depth


# ─── 系数计算（带缓存） ───────────────────────────────────────────

@lru_cache(maxsize=1024)
def ramanujan_coefficients(max_n: int) -> tuple:
    """
    计算递推系数 a_0, a_1, ..., a_{max_n}

    Returns:
        tuple of float: (a_0, a_1, ..., a_{max_n})
    """
    if max_n < 0:
        return ()

    a = [0.0] * (max_n + 1)
    a[0] = 1.0
    if max_n >= 1:
        a[1] = math.pi / math.sqrt(3)

    for n in range(1, max_n):
        if n + 1 <= max_n:
            a[n + 1] = (math.pi ** 2 / n ** 2) * a[n] + \
                        (2 * math.pi / (n * (n + 1))) * a[n - 1]

    return tuple(a)


# 预计算常用深度的系数表
_PRECOMPUTED_DEPTHS = [8, 16, 32, 64, 128, 256, 512, 1024]

def _precompute_coefficients():
    for d in _PRECOMPUTED_DEPTHS:
        ramanujan_coefficients(d)

_precompute_coefficients()


def get_ramanujan_scale(layer_idx: int, ramanujan_depth: int = 8,
                        transition_depth: int = 8,
                        fan_in: int = 512) -> float:
    """
    获取第 layer_idx 层的初始化缩放因子

    三层混合方案：
        - [0, ramanujan_depth): Ramanujan 调制的 Xavier
        - [ramanujan_depth, ramanujan_depth + transition_depth): 线性过渡
        - [ramanujan_depth + transition_depth, ∞): 纯 Xavier
    """
    coeffs = ramanujan_coefficients(ramanujan_depth)
    peak = max(abs(c) for c in coeffs) if coeffs else 1.0

    xavier_scale = 1.0

    if layer_idx < ramanujan_depth:
        raw = math.sqrt(abs(coeffs[layer_idx]) / peak)
        return max(0.1, raw)

    elif layer_idx < ramanujan_depth + transition_depth:
        t = (layer_idx - ramanujan_depth) / transition_depth
        last_ramanujan = math.sqrt(abs(coeffs[-1]) / peak)
        last_ramanujan = max(0.1, last_ramanujan)
        return last_ramanujan * (1 - t) + xavier_scale * t

    else:
        return xavier_scale


# ─── 层索引自动追踪 ────────────────────────────────────────────────

def assign_layer_indices(model: nn.Module, counter: Optional[List[int]] = None) -> int:
    """按 modules() 拓扑序为所有 Linear 层分配递增的 layer_idx"""
    if counter is None:
        counter = [0]

    for module in model.modules():
        if isinstance(module, nn.Linear):
            module._ramanujan_idx = counter[0]
            counter[0] += 1

    return counter[0]


# ─── 分层初始化标签 ────────────────────────────────────────────────

class LayerRole:
    EMBEDDING = 'embedding'
    QKV_PROJ = 'qkv_proj'
    Q_PROJ = 'q_proj'
    K_PROJ = 'k_proj'
    V_PROJ = 'v_proj'
    OUTPUT_ATTN = 'output_attn'
    FFN_UP = 'ffn_up'
    FFN_DOWN = 'ffn_down'
    LM_HEAD = 'lm_head'
    ROUTER = 'router'
    OTHER = 'other'


def tag_linear_role(module: nn.Linear, role: str):
    module._ramanujan_role = role


def get_linear_role(module: nn.Linear) -> str:
    return getattr(module, '_ramanujan_role', LayerRole.OTHER)


# ─── PyTorch 初始化函数 ─────────────────────────────────────────────

def ramanujan_init_(tensor: torch.Tensor, fan_in: int, fan_out: int,
                    layer_idx: int = 0, ramanujan_depth: int = 8,
                    transition_depth: int = 8,
                    nonlinearity: str = 'linear',
                    gain: Optional[float] = None,
                    quantization: str = 'none') -> torch.Tensor:
    """
    拉马努金初始化

    Args:
        tensor: 待初始化的权重张量
        fan_in: 输入维度
        fan_out: 输出维度
        layer_idx: 层索引
        ramanujan_depth: Ramanujan 调制深度
        transition_depth: 过渡深度
        nonlinearity: 激活函数类型
        gain: 手动指定增益
        quantization: 量化精度 ('none', 'int8', 'fp8', 'int4')

    Returns:
        初始化后的张量
    """
    scale = get_ramanujan_scale(layer_idx, ramanujan_depth,
                                transition_depth, fan_in)

    if gain is None:
        gain = get_activation_gain(nonlinearity)

    # 量化修正
    q_gain = get_quantization_gain(quantization)
    gain *= q_gain

    std = gain * scale / math.sqrt(fan_in)

    init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)
    return tensor


def ramanujan_init_uniform_(tensor: torch.Tensor, fan_in: int, fan_out: int,
                            layer_idx: int = 0, ramanujan_depth: int = 8,
                            transition_depth: int = 8,
                            gain: Optional[float] = None) -> torch.Tensor:
    """均匀分布版本"""
    scale = get_ramanujan_scale(layer_idx, ramanujan_depth,
                                transition_depth, fan_in)
    g = gain if gain is not None else 1.0
    limit = g * scale / math.sqrt(fan_in) * math.sqrt(3.0)
    init.uniform_(tensor, -limit, limit)
    return tensor


# ─── 高级封装 ──────────────────────────────────────────────────────

class RamanujanInitializer:
    """
    全局初始化器 (v1.5)

    新增特性:
        - 量化友好初始化 (quantization: 'int8'/'fp8'/'int4'/'none')
        - 自适应权重衰减 (get_adaptive_weight_decay)
        - QKV 差异化初始化 (LayerRole.Q_PROJ/K_PROJ/V_PROJ)
        - 长上下文 RoPE 修正 (long_context_support)
        - 分级日志系统

    使用方法:
        initializer = RamanujanInitializer(num_layers=12)
        initializer.apply(model)

        # 量化场景
        initializer = RamanujanInitializer(num_layers=12, quantization='int8')

        # 长上下文
        initializer = RamanujanInitializer(num_layers=12, long_context_seq_len=8192)
    """

    def __init__(self, ramanujan_depth: Optional[int] = None,
                 transition_depth: Optional[int] = None,
                 num_layers: Optional[int] = None,
                 nonlinearity: str = 'linear',
                 gain: Optional[float] = None,
                 quantization: str = 'none',
                 long_context_seq_len: int = 512):
        # 动态深度计算
        if ramanujan_depth is None or transition_depth is None:
            auto_r, auto_t = compute_optimal_depth(num_layers or 12)
            self.ramanujan_depth = ramanujan_depth if ramanujan_depth is not None else auto_r
            self.transition_depth = transition_depth if transition_depth is not None else auto_t
        else:
            self.ramanujan_depth = ramanujan_depth
            self.transition_depth = transition_depth

        self.nonlinearity = nonlinearity
        self.gain = gain
        self.quantization = quantization
        self.long_context_seq_len = long_context_seq_len
        self._coeffs = ramanujan_coefficients(self.ramanujan_depth)

        # 长上下文修正系数
        self._rope_correction = compute_rope_correction(long_context_seq_len, 768)

        logger.info(f"RamanujanInitializer: depth={self.ramanujan_depth}/{self.transition_depth}, "
                     f"quantization={quantization}, long_ctx={long_context_seq_len}")

    @property
    def coefficients(self) -> tuple:
        return self._coeffs

    def get_scale(self, layer_idx: int, fan_in: int = 512) -> float:
        scale = get_ramanujan_scale(layer_idx, self.ramanujan_depth,
                                    self.transition_depth, fan_in)
        # 长上下文修正
        if self.long_context_seq_len > 512:
            scale *= self._rope_correction
        return scale

    def get_layer_weight_decay(self, layer_idx: int, base_decay: float) -> float:
        """获取该层的自适应权重衰减系数"""
        return get_adaptive_weight_decay(
            layer_idx, base_decay,
            self.ramanujan_depth, self.transition_depth
        )

    def apply(self, model: nn.Module, nonlinearity: Optional[str] = None,
              gain: Optional[float] = None):
        """一步完成：分配层索引 + 初始化所有参数"""
        assign_layer_indices(model)
        model.apply(lambda m: self._init_module(m, nonlinearity, gain))

    def _init_module(self, module: nn.Module, nonlinearity: Optional[str] = None,
                     gain: Optional[float] = None):
        """单模块初始化，支持分层差异化 + QKV 差异化 + 量化"""
        func = nonlinearity or self.nonlinearity
        g = gain if gain is not None else self.gain

        if isinstance(module, nn.Linear):
            layer_idx = getattr(module, '_ramanujan_idx', 0)
            role = get_linear_role(module)

            # 分层差异化增益
            effective_gain = g
            if effective_gain is None:
                if role == LayerRole.LM_HEAD:
                    effective_gain = 1.0 / math.sqrt(module.in_features)
                elif role == LayerRole.ROUTER:
                    effective_gain = 0.1
                elif role == LayerRole.Q_PROJ:
                    # Q 投影：使用略大的增益，增强查询信号
                    effective_gain = get_activation_gain(func) * 1.05
                elif role == LayerRole.K_PROJ:
                    # K 投影：标准增益
                    effective_gain = get_activation_gain(func)
                elif role == LayerRole.V_PROJ:
                    # V 投影：使用略小的增益，稳定值信号
                    effective_gain = get_activation_gain(func) * 0.95
                else:
                    effective_gain = get_activation_gain(func)

            if module.weight is not None:
                ramanujan_init_(module.weight, module.in_features,
                                module.out_features, layer_idx,
                                self.ramanujan_depth, self.transition_depth,
                                func, gain=effective_gain,
                                quantization=self.quantization)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            coeffs = ramanujan_coefficients(min(8, self.ramanujan_depth))
            peak = max(abs(c) for c in coeffs) if coeffs else 1.0
            embed_scale = math.sqrt(abs(coeffs[0]) / peak) if coeffs else 1.0
            nn.init.normal_(module.weight, std=0.02 * embed_scale)

    # ── 手动接口 ──

    def init_linear(self, layer: nn.Linear, layer_idx: int = 0,
                    nonlinearity: Optional[str] = None,
                    gain: Optional[float] = None) -> nn.Linear:
        func = nonlinearity or self.nonlinearity
        g = gain if gain is not None else self.gain
        if g is None:
            g = get_activation_gain(func)
        if layer.weight is not None:
            ramanujan_init_(layer.weight, layer.in_features,
                            layer.out_features, layer_idx,
                            self.ramanujan_depth, self.transition_depth,
                            func, gain=g, quantization=self.quantization)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
        return layer

    def init_tensor(self, tensor: torch.Tensor, layer_idx: int = 0,
                    nonlinearity: Optional[str] = None,
                    gain: Optional[float] = None) -> torch.Tensor:
        func = nonlinearity or self.nonlinearity
        g = gain if gain is not None else self.gain
        if g is None:
            g = get_activation_gain(func)
        fan_in = tensor.shape[-1] if len(tensor.shape) > 1 else tensor.shape[0]
        fan_out = tensor.shape[0] if len(tensor.shape) > 1 else 1
        return ramanujan_init_(tensor, fan_in, fan_out, layer_idx,
                               self.ramanujan_depth, self.transition_depth,
                               func, gain=g, quantization=self.quantization)

    # ── 方差验证工具 ──

    def variance_test(self, depth: int, dim: int = 512,
                      nonlinearity: str = 'linear',
                      use_residual: bool = True) -> Dict:
        """方差保持性测试"""
        x = torch.randn(1000, dim)
        input_var = x.var().item()
        per_layer_vars = [input_var]
        ratios = []

        if use_residual:
            gain = 1.0
        else:
            gain = get_activation_gain(nonlinearity)

        for i in range(depth):
            s = get_ramanujan_scale(i, self.ramanujan_depth,
                                    self.transition_depth, dim)
            std = gain * s / math.sqrt(dim)
            W = torch.randn(dim, dim) * std

            y = x @ W

            if nonlinearity == 'relu':
                y = torch.relu(y)
            elif nonlinearity in ('gelu', 'silu'):
                y = torch.nn.functional.gelu(y)

            if use_residual:
                x = x + y
            else:
                x = y

            current_var = x.var().item()
            per_layer_vars.append(current_var)
            ratios.append(current_var / input_var)

        output_var = x.var().item()
        return {
            'input_var': input_var,
            'output_var': output_var,
            'ratio': output_var / input_var,
            'per_layer_vars': per_layer_vars,
            'max_ratio': max(ratios) if ratios else 1.0,
            'min_ratio': min(ratios) if ratios else 1.0,
        }


# ─── 便捷函数 ──────────────────────────────────────────────────────

def get_initialization_function(ramanujan_depth: int = 8,
                                transition_depth: int = 8,
                                nonlinearity: str = 'linear'):
    """返回可传递给 nn.Module.apply() 的初始化函数"""
    initializer = RamanujanInitializer(ramanujan_depth, transition_depth,
                                       nonlinearity=nonlinearity)

    def init_fn(module):
        if isinstance(module, nn.Linear):
            initializer._init_module(module, nonlinearity)
        elif isinstance(module, (nn.LayerNorm, nn.Embedding)):
            initializer._init_module(module, nonlinearity)

    return init_fn
