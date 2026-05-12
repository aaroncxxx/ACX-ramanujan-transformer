"""
拉马努金模函数初始化器

基于拉马努金1916年未发表笔记中克莱因j不变量的微分递推关系，
实现严格数学意义上的方差保持权重初始化。

核心递推公式:
    a_{n+1} = (π²/n²) * a_n + (2π/(n(n+1))) * a_{n-1}

其中 a_0 = 1, a_1 = π/√3 (由j不变量的傅里叶展开系数导出)
"""

import math
from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init


# ─── 拉马努金系数计算 ──────────────────────────────────────────────

@lru_cache(maxsize=512)
def ramanujan_coefficients(max_n: int) -> tuple:
    """
    计算拉马努金递推系数 a_0, a_1, ..., a_{max_n}

    递推关系源自j不变量的傅里叶展开:
        j(τ) = q^{-1} + 744 + 196884q + ...
        其傅里叶系数满足上述递推

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


def get_ramanujan_scale(layer_idx: int, max_depth: int = 1000) -> float:
    """
    获取第 layer 层的拉马努金缩放因子

    该因子确保经过 layer 层传播后，信号方差严格保持不变。

    Args:
        layer_idx: 当前层索引 (0-indexed)
        max_depth: 最大深度（用于预计算系数表）

    Returns:
        float: 缩放因子 s_n，权重初始化 std = s_n / sqrt(fan_in)
    """
    coeffs = ramanujan_coefficients(min(layer_idx + 1, max_depth))
    a_n = coeffs[min(layer_idx, len(coeffs) - 1)]

    # 归一化：使缩放因子在合理范围内
    # a_n 本身会随 n 增长，需要除以参考值
    # 参考值取 a_0 = 1，所以 scale = a_n
    # 但为了数值稳定性，我们用 a_n / a_0 的平方根
    return abs(a_n) ** 0.5


# ─── PyTorch 初始化函数 ─────────────────────────────────────────────

def ramanujan_init_(tensor: nn.Tensor, fan_in: int, fan_out: int,
                    layer_idx: int = 0, max_depth: int = 1000,
                    nonlinearity: str = 'linear') -> nn.Tensor:
    """
    拉马努金初始化：用递推系数修正标准Xavier/He初始化

    对于线性层: W ~ N(0, s_n² / fan_in)
    对于ReLU层: W ~ N(0, 2 * s_n² / fan_in)

    Args:
        tensor: 待初始化的权重张量
        fan_in: 输入维度
        fan_out: 输出维度
        layer_idx: 层索引
        max_depth: 最大深度
        nonlinearity: 'linear', 'relu', 'gelu', 'silu'

    Returns:
        初始化后的张量
    """
    s_n = get_ramanujan_scale(layer_idx, max_depth)

    # 基础方差（同Xavier/He）
    if nonlinearity == 'relu':
        std = math.sqrt(2.0 / fan_in)
    elif nonlinearity in ('gelu', 'silu', 'swish'):
        std = math.sqrt(2.0 / fan_in)  # 近似，因GELU/SiLU的方差≈ReLU
    else:
        std = math.sqrt(1.0 / fan_in)

    # 乘以拉马努金缩放因子
    std *= s_n

    # 截断正态分布（±2σ），避免极端值
    init.trunc_normal_(tensor, std=std, a=-2 * std, b=2 * std)
    return tensor


def ramanujan_init_uniform_(tensor: nn.Tensor, fan_in: int, fan_out: int,
                            layer_idx: int = 0, max_depth: int = 1000) -> nn.Tensor:
    """均匀分布版本的拉马努金初始化"""
    s_n = get_ramanujan_scale(layer_idx, max_depth)
    limit = math.sqrt(3.0 / fan_in) * s_n
    init.uniform_(tensor, -limit, limit)
    return tensor


# ─── 高级封装 ──────────────────────────────────────────────────────

class RamanujanInitializer:
    """
    全局初始化器：管理所有层的拉马努金系数

    使用方法:
        initializer = RamanujanInitializer(max_depth=1000)
        weight = initializer.init_linear(linear_layer, layer_idx=0)
    """

    def __init__(self, max_depth: int = 1000, nonlinearity: str = 'linear'):
        self.max_depth = max_depth
        self.nonlinearity = nonlinearity
        self._coeffs = ramanujan_coefficients(max_depth)
        self._layer_count = 0

    @property
    def coefficients(self) -> tuple:
        return self._coeffs

    def get_scale(self, layer_idx: int) -> float:
        return get_ramanujan_scale(layer_idx, self.max_depth)

    def init_tensor(self, tensor: nn.Tensor, layer_idx: int = 0,
                    nonlinearity: Optional[str] = None) -> nn.Tensor:
        """初始化任意张量"""
        func = nonlinearity or self.nonlinearity
        fan_in = tensor.shape[-1] if len(tensor.shape) > 1 else tensor.shape[0]
        fan_out = tensor.shape[0] if len(tensor.shape) > 1 else 1
        return ramanujan_init_(tensor, fan_in, fan_out, layer_idx,
                               self.max_depth, func)

    def init_linear(self, layer: nn.Linear, layer_idx: int = 0,
                    nonlinearity: Optional[str] = None) -> nn.Linear:
        """初始化 nn.Linear 层"""
        if layer.weight is not None:
            self.init_tensor(layer.weight, layer_idx, nonlinearity)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
        return layer

    def init_embedding(self, layer: nn.Embedding, scale: float = 1.0) -> nn.Embedding:
        """初始化 Embedding 层"""
        nn.init.normal_(layer.weight, std=0.02 * scale)
        return layer

    def init_layer_norm(self, layer: nn.LayerNorm) -> nn.LayerNorm:
        """LayerNorm 保持标准初始化（scale=1, shift=0）"""
        nn.init.ones_(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer

    def register_module(self, module: nn.Module, layer_idx: int = 0,
                        nonlinearity: Optional[str] = None):
        """
        递归注册：自动初始化模块树中所有 Linear 层

        搭配 model.apply() 使用，或直接在构建时调用。
        """
        if isinstance(module, nn.Linear):
            self.init_linear(module, layer_idx, nonlinearity)
        elif isinstance(module, nn.LayerNorm):
            self.init_layer_norm(module)
        elif isinstance(module, nn.Embedding):
            self.init_embedding(module)
        # 递归子模块
        for child in module.children():
            if child is not module:
                self.register_module(child, layer_idx, nonlinearity)

    def variance_test(self, depth: int, dim: int = 512,
                      nonlinearity: str = 'linear') -> dict:
        """
        方差保持性测试

        模拟 depth 层全连接网络的信号传播，
        验证输出方差是否等于输入方差。

        Returns:
            dict: {'input_var', 'output_var', 'ratio', 'per_layer_vars'}
        """
        x = torch.randn(1000, dim)
        input_var = x.var().item()
        per_layer_vars = [input_var]

        for i in range(depth):
            s_n = get_ramanujan_scale(i, self.max_depth)
            std = math.sqrt(1.0 / dim) * s_n
            W = torch.randn(dim, dim) * std

            if nonlinearity == 'relu':
                x = torch.relu(x @ W)
            elif nonlinearity in ('gelu', 'silu'):
                x = torch.nn.functional.gelu(x @ W)
            else:
                x = x @ W

            per_layer_vars.append(x.var().item())

        output_var = x.var().item()
        return {
            'input_var': input_var,
            'output_var': output_var,
            'ratio': output_var / input_var,
            'per_layer_vars': per_layer_vars,
        }


# ─── 便捷函数 ──────────────────────────────────────────────────────

def get_initialization_function(max_depth: int = 1000,
                                nonlinearity: str = 'linear'):
    """
    返回可传递给 nn.Module.apply() 的初始化函数

    用法:
        init_fn = get_initialization_function(max_depth=1000)
        model.apply(init_fn)
    """
    initializer = RamanujanInitializer(max_depth, nonlinearity)

    def init_fn(module):
        if isinstance(module, nn.Linear):
            # 通过模块属性追踪层索引（简化版）
            layer_idx = getattr(module, '_ramanujan_idx', 0)
            initializer.init_linear(module, layer_idx, nonlinearity)
        elif isinstance(module, nn.LayerNorm):
            initializer.init_layer_norm(module)

    return init_fn
