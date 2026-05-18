"""
拉马努金模函数初始化器 v2（修复版）

基于拉马努金递推关系的权重初始化方案。

核心递推公式:
    a_{n+1} = (π²/n²) * a_n + (2π/(n(n+1))) * a_{n-1}
    a_0 = 1, a_1 = π/√3

数学性质:
    - 系数在 n≈4 处达到峰值 (~70)，之后指数衰减至零
    - 衰减由 (π²/n²) < 1 (n≥4) 的收缩性导致
    - 本质类似 Bessel 函数递推，非标准模形式递推

设计策略（三层混合方案）:
    1. 前 Ramanujan 层 (默认8层): 使用峰值归一化的系数调制 Xavier 初始化
       → 提供独特的"拉马努金轮廓"初始化模式
    2. 过渡层 (8-16层): 线性混合 Ramanujan 调制与纯 Xavier
       → 平滑过渡，避免突变
    3. 深层 (>16层): 标准 Xavier/Kaiming 初始化
       → 系数已衰减至零，无额外信息，回归经典方案

方差保持保证:
    每一层的标准差始终以 1/sqrt(fan_in) 为基准，Ramanujan 系数仅提供
    ±调制，不改变数量级。残差连接下的方差增长由 Transformer 原有的
    LayerNorm 处理，本初始化器不重复造轮子。

修复内容 (vs v1):
    1. nn.Tensor → torch.Tensor
    2. 缩放因子不再爆炸/坍缩：混合方案保证所有层 scale ∈ [floor, ceiling]
    3. layer_idx 自动传播：assign_layer_indices() 按拓扑序分配
    4. 增益参数：残差网络用 gain=1.0，纯前馈用 gain=sqrt(2)
    5. 残差方差测试
    6. 数学文档：明确系数衰减性质和方案选择依据
"""

import math
from functools import lru_cache
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.init as init


# ─── 拉马努金系数计算 ──────────────────────────────────────────────

@lru_cache(maxsize=512)
def ramanujan_coefficients(max_n: int) -> tuple:
    """
    计算递推系数 a_0, a_1, ..., a_{max_n}

    注意：系数在 n≈4 处达到峰值后指数衰减。
    对于 n > ~30，系数数值上接近零（< 1e-15）。
    这是递推公式中 (π²/n²) 项收缩性的自然结果，
    并非数值精度问题。

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


def get_ramanujan_scale(layer_idx: int, ramanujan_depth: int = 8,
                        transition_depth: int = 8,
                        fan_in: int = 512) -> float:
    """
    获取第 layer_idx 层的初始化缩放因子

    三层混合方案：
        - [0, ramanujan_depth): Ramanujan 调制的 Xavier
        - [ramanujan_depth, ramanujan_depth + transition_depth): 线性过渡
        - [ramanujan_depth + transition_depth, ∞): 纯 Xavier

    Args:
        layer_idx: 当前层索引
        ramanujan_depth: 使用 Ramanujan 调制的层数
        transition_depth: 过渡层数
        fan_in: 输入维度（用于 Xavier 基准）

    Returns:
        float: 缩放因子，乘以 sqrt(1/fan_in) 得到权重 std
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
    """
    按 modules() 拓扑序为所有 Linear 层分配递增的 layer_idx。

    结果存储在 module._ramanujan_idx 属性中。

    Args:
        model: 要初始化的模型
        counter: 内部计数器 [current_idx]，首次调用传 None

    Returns:
        int: 分配的 Linear 层总数
    """
    if counter is None:
        counter = [0]

    for module in model.modules():
        if isinstance(module, nn.Linear):
            module._ramanujan_idx = counter[0]
            counter[0] += 1

    return counter[0]


# ─── PyTorch 初始化函数 ─────────────────────────────────────────────

def ramanujan_init_(tensor: torch.Tensor, fan_in: int, fan_out: int,
                    layer_idx: int = 0, ramanujan_depth: int = 8,
                    transition_depth: int = 8,
                    nonlinearity: str = 'linear',
                    gain: Optional[float] = None) -> torch.Tensor:
    """
    拉马努金初始化

    W ~ N(0, (scale * gain / sqrt(fan_in))²)

    其中 scale 由 get_ramanujan_scale() 的三层混合方案决定。

    Args:
        tensor: 待初始化的权重张量
        fan_in: 输入维度
        fan_out: 输出维度
        layer_idx: 层索引
        ramanujan_depth: Ramanujan 调制深度
        transition_depth: 过渡深度
        nonlinearity: 'linear', 'relu', 'gelu', 'silu'（当 gain 未指定时用于推断）
        gain: 手动指定增益。None 时自动推断：
              - relu/gelu/silu → sqrt(2), linear → 1.0
              对于标准 Transformer（残差+LayerNorm），建议显式传 gain=1.0

    Returns:
        初始化后的张量
    """
    scale = get_ramanujan_scale(layer_idx, ramanujan_depth,
                                transition_depth, fan_in)

    if gain is None:
        if nonlinearity in ('relu', 'gelu', 'silu', 'swish'):
            gain = math.sqrt(2.0)
        else:
            gain = 1.0

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
    全局初始化器

    使用方法（推荐）:
        initializer = RamanujanInitializer()
        initializer.apply(model)

    或手动:
        assign_layer_indices(model)
        model.apply(initializer._init_fn)

    参数说明:
        ramanujan_depth: 使用 Ramanujan 系数调制的层数（默认8，覆盖系数峰值区域）
        transition_depth: 从 Ramanujan 过渡到 Xavier 的层数（默认8）
        nonlinearity: 默认非线性类型
        gain: 手动指定增益。None 时按 nonlinearity 自动推断。
              对于标准 Transformer（残差+LayerNorm），建议显式传 gain=1.0
    """

    def __init__(self, ramanujan_depth: int = 8, transition_depth: int = 8,
                 nonlinearity: str = 'linear', gain: Optional[float] = None):
        self.ramanujan_depth = ramanujan_depth
        self.transition_depth = transition_depth
        self.nonlinearity = nonlinearity
        self.gain = gain
        self._coeffs = ramanujan_coefficients(ramanujan_depth)

    @property
    def coefficients(self) -> tuple:
        return self._coeffs

    def get_scale(self, layer_idx: int, fan_in: int = 512) -> float:
        return get_ramanujan_scale(layer_idx, self.ramanujan_depth,
                                   self.transition_depth, fan_in)

    def apply(self, model: nn.Module, nonlinearity: Optional[str] = None,
              gain: Optional[float] = None):
        """
        一步完成：分配层索引 + 初始化所有参数

        Args:
            model: 要初始化的 PyTorch 模型
            nonlinearity: 覆盖默认非线性类型
            gain: 覆盖默认增益。对于标准 Transformer（残差+LayerNorm），
                  建议显式传 gain=1.0
        """
        assign_layer_indices(model)
        model.apply(lambda m: self._init_module(m, nonlinearity, gain))

    def _init_module(self, module: nn.Module, nonlinearity: Optional[str] = None,
                     gain: Optional[float] = None):
        """单模块初始化（内部使用）"""
        func = nonlinearity or self.nonlinearity
        g = gain if gain is not None else self.gain

        if isinstance(module, nn.Linear):
            layer_idx = getattr(module, '_ramanujan_idx', 0)
            if module.weight is not None:
                ramanujan_init_(module.weight, module.in_features,
                                module.out_features, layer_idx,
                                self.ramanujan_depth, self.transition_depth,
                                func, gain=g)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    # ── 手动接口（向后兼容）──

    def init_linear(self, layer: nn.Linear, layer_idx: int = 0,
                    nonlinearity: Optional[str] = None,
                    gain: Optional[float] = None) -> nn.Linear:
        func = nonlinearity or self.nonlinearity
        g = gain if gain is not None else self.gain
        if layer.weight is not None:
            ramanujan_init_(layer.weight, layer.in_features,
                            layer.out_features, layer_idx,
                            self.ramanujan_depth, self.transition_depth,
                            func, gain=g)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
        return layer

    def init_tensor(self, tensor: torch.Tensor, layer_idx: int = 0,
                    nonlinearity: Optional[str] = None,
                    gain: Optional[float] = None) -> torch.Tensor:
        func = nonlinearity or self.nonlinearity
        g = gain if gain is not None else self.gain
        fan_in = tensor.shape[-1] if len(tensor.shape) > 1 else tensor.shape[0]
        fan_out = tensor.shape[0] if len(tensor.shape) > 1 else 1
        return ramanujan_init_(tensor, fan_in, fan_out, layer_idx,
                               self.ramanujan_depth, self.transition_depth,
                               func, gain=g)

    # ── 方差验证工具 ──

    def variance_test(self, depth: int, dim: int = 512,
                      nonlinearity: str = 'linear',
                      use_residual: bool = True) -> Dict:
        """
        方差保持性测试

        模拟 depth 层全连接网络的信号传播，验证输出方差。

        注意：
        - 带残差时使用 gain=1.0（匹配 Transformer 的残差+LN 结构）
        - 无残差时使用 gain=sqrt(2)（匹配纯前馈+ReLU 结构）

        Returns:
            dict: {
                'input_var', 'output_var', 'ratio',
                'per_layer_vars', 'max_ratio', 'min_ratio'
            }
        """
        x = torch.randn(1000, dim)
        input_var = x.var().item()
        per_layer_vars = [input_var]
        ratios = []

        if use_residual:
            gain = 1.0
        elif nonlinearity in ('relu', 'gelu', 'silu'):
            gain = math.sqrt(2.0)
        else:
            gain = 1.0

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
    """
    返回可传递给 nn.Module.apply() 的初始化函数

    注意：使用前必须先调用 assign_layer_indices(model)。

    推荐改用 RamanujanInitializer.apply(model) 一步完成。
    """
    initializer = RamanujanInitializer(ramanujan_depth, transition_depth,
                                       nonlinearity)

    def init_fn(module):
        if isinstance(module, nn.Linear):
            initializer._init_module(module, nonlinearity)
        elif isinstance(module, (nn.LayerNorm, nn.Embedding)):
            initializer._init_module(module, nonlinearity)

    return init_fn
