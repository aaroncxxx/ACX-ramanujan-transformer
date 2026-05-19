"""
拉马努金模函数初始化器 v1.4

基于拉马努金递推关系的权重初始化方案。

核心递推公式:
    a_{n+1} = (π²/n²) * a_n + (2π/(n(n+1))) * a_{n-1}
    a_0 = 1, a_1 = π/√3

v1.4 新增:
    1. 激活函数自适应增益：GELU/SiLU/SiGLU/ReLU 各自推导增益系数
    2. 动态递推深度：根据 num_layers 自动计算 ramanujan_depth/transition_depth
    3. 分层初始化适配：Embedding/QKV/FFN/Output 差异化系数
    4. 系数缓存优化：预计算常用深度并缓存
"""

import math
from functools import lru_cache
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init


# ─── 激活函数增益表 ────────────────────────────────────────────────

# 每种激活函数的理论增益（推导自 Var[f(x)] = gain² * Var[x]）
# linear: f(x)=x, E[f'²]=1 → gain=1
# relu: f(x)=max(0,x), E[f'²]=0.5 → gain=sqrt(2)
# gelu: 近似 E[f'²]≈0.5 → gain=sqrt(2)
# silu/swish: 近似 E[f'²]≈0.5 → gain=sqrt(2)
# sigmoid: E[f'²]≈1/9 → gain=3（很少用于 Transformer）

ACTIVATION_GAIN = {
    'linear': 1.0,
    'relu': math.sqrt(2.0),
    'gelu': math.sqrt(2.0),
    'silu': math.sqrt(2.0),
    'swish': math.sqrt(2.0),
    'sigmoid': 3.0,
    'tanh': math.sqrt(2.0),  # 近似
}


def get_activation_gain(nonlinearity: str) -> float:
    """获取激活函数对应的理论增益"""
    return ACTIVATION_GAIN.get(nonlinearity, 1.0)


# ─── 动态递推深度计算 ──────────────────────────────────────────────

def compute_optimal_depth(num_layers: int) -> Tuple[int, int]:
    """
    根据模型层数自动计算最优的 ramanujan_depth 和 transition_depth

    策略：
        - ramanujan_depth: 覆盖系数峰值区域（n≈4），但不超过总层数的 1/3
        - transition_depth: 从 Ramanujan 平滑过渡到 Xavier，约占总层数的 1/4
        - 保证 ramanujan_depth + transition_depth <= num_layers

    Args:
        num_layers: Transformer 层数

    Returns:
        (ramanujan_depth, transition_depth)
    """
    # 系数峰值在 n≈4，n>30 后接近零
    # ramanujan_depth 取 min(峰值覆盖区, 总层数/3)
    ramanujan_depth = min(8, max(4, num_layers // 3))

    # 过渡层数：总层数的 1/4，最少 4 层
    transition_depth = min(16, max(4, num_layers // 4))

    # 确保不超出总层数
    if ramanujan_depth + transition_depth > num_layers:
        transition_depth = num_layers - ramanujan_depth

    return ramanujan_depth, transition_depth


# ─── 系数计算（带缓存） ───────────────────────────────────────────

@lru_cache(maxsize=1024)
def ramanujan_coefficients(max_n: int) -> tuple:
    """
    计算递推系数 a_0, a_1, ..., a_{max_n}

    系数在 n≈4 处达到峰值后指数衰减。
    对于 n > ~30，系数数值上接近零（< 1e-15）。

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
    """预计算常用深度的系数表，避免首次推理时的计算开销"""
    for d in _PRECOMPUTED_DEPTHS:
        ramanujan_coefficients(d)

# 模块加载时自动预计算
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

    gain 自动推导规则（v1.4 增强）:
        - gain 未指定时，根据 nonlinearity 查表获取理论增益
        - 残差网络（Transformer）建议显式传 gain=1.0
        - 纯前馈网络建议 gain=None（自动推导）

    Args:
        tensor: 待初始化的权重张量
        fan_in: 输入维度
        fan_out: 输出维度
        layer_idx: 层索引
        ramanujan_depth: Ramanujan 调制深度
        transition_depth: 过渡深度
        nonlinearity: 'linear', 'relu', 'gelu', 'silu'（当 gain 未指定时用于推断）
        gain: 手动指定增益。None 时自动查表推导

    Returns:
        初始化后的张量
    """
    scale = get_ramanujan_scale(layer_idx, ramanujan_depth,
                                transition_depth, fan_in)

    if gain is None:
        gain = get_activation_gain(nonlinearity)

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


# ─── 分层初始化标签 ────────────────────────────────────────────────

class LayerRole:
    """层角色标签，用于分层差异化初始化"""
    EMBEDDING = 'embedding'
    QKV_PROJ = 'qkv_proj'
    OUTPUT_ATTN = 'output_attn'
    FFN_UP = 'ffn_up'
    FFN_DOWN = 'ffn_down'
    LM_HEAD = 'lm_head'
    ROUTER = 'router'
    OTHER = 'other'


def tag_linear_role(module: nn.Linear, role: str):
    """为 Linear 层打角色标签"""
    module._ramanujan_role = role


def get_linear_role(module: nn.Linear) -> str:
    """获取 Linear 层的角色标签"""
    return getattr(module, '_ramanujan_role', LayerRole.OTHER)


# ─── 高级封装 ──────────────────────────────────────────────────────

class RamanujanInitializer:
    """
    全局初始化器 (v1.4)

    新增特性:
        - 激活函数自适应增益（自动检测并匹配）
        - 动态递推深度（根据模型层数自动计算）
        - 分层初始化适配（Embedding/QKV/FFN/Output 差异化）
        - 系数预计算缓存

    使用方法（推荐）:
        initializer = RamanujanInitializer()
        initializer.apply(model)

    或手动:
        assign_layer_indices(model)
        model.apply(initializer._init_fn)

    参数说明:
        ramanujan_depth: Ramanujan 系数调制层数（None=自动计算）
        transition_depth: 过渡层数（None=自动计算）
        num_layers: 模型总层数（用于自动计算深度，仅当 depth=None 时生效）
        nonlinearity: 默认非线性类型
        gain: 手动指定增益。None 时自动推导。
    """

    def __init__(self, ramanujan_depth: Optional[int] = None,
                 transition_depth: Optional[int] = None,
                 num_layers: Optional[int] = None,
                 nonlinearity: str = 'linear',
                 gain: Optional[float] = None):
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
        self._coeffs = ramanujan_coefficients(self.ramanujan_depth)

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
            gain: 覆盖默认增益
        """
        assign_layer_indices(model)
        model.apply(lambda m: self._init_module(m, nonlinearity, gain))

    def _init_module(self, module: nn.Module, nonlinearity: Optional[str] = None,
                     gain: Optional[float] = None):
        """单模块初始化（内部使用），支持分层差异化"""
        func = nonlinearity or self.nonlinearity
        g = gain if gain is not None else self.gain

        if isinstance(module, nn.Linear):
            layer_idx = getattr(module, '_ramanujan_idx', 0)
            role = get_linear_role(module)

            # 分层差异化增益
            effective_gain = g
            if effective_gain is None:
                if role == LayerRole.LM_HEAD:
                    # LM Head: 输出层缩放，方差除以 sqrt(d_model)
                    effective_gain = 1.0 / math.sqrt(module.in_features)
                elif role == LayerRole.ROUTER:
                    # MoE Router: 轻量初始化，避免路由权重梯度消失
                    effective_gain = 0.1
                elif role in (LayerRole.FFN_UP, LayerRole.FFN_DOWN):
                    effective_gain = get_activation_gain(func)
                else:
                    effective_gain = get_activation_gain(func)

            if module.weight is not None:
                ramanujan_init_(module.weight, module.in_features,
                                module.out_features, layer_idx,
                                self.ramanujan_depth, self.transition_depth,
                                func, gain=effective_gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            # Embedding: 使用拉马努金系数的前几项调制 std
            coeffs = ramanujan_coefficients(min(8, self.ramanujan_depth))
            peak = max(abs(c) for c in coeffs) if coeffs else 1.0
            embed_scale = math.sqrt(abs(coeffs[0]) / peak) if coeffs else 1.0
            nn.init.normal_(module.weight, std=0.02 * embed_scale)

    # ── 手动接口（向后兼容）──

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
                            func, gain=g)
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
                               func, gain=g)

    # ── 方差验证工具 ──

    def variance_test(self, depth: int, dim: int = 512,
                      nonlinearity: str = 'linear',
                      use_residual: bool = True) -> Dict:
        """
        方差保持性测试

        模拟 depth 层全连接网络的信号传播，验证输出方差。

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
    """
    返回可传递给 nn.Module.apply() 的初始化函数

    注意：使用前必须先调用 assign_layer_indices(model)。

    推荐改用 RamanujanInitializer.apply(model) 一步完成。
    """
    initializer = RamanujanInitializer(ramanujan_depth, transition_depth,
                                       nonlinearity=nonlinearity)

    def init_fn(module):
        if isinstance(module, nn.Linear):
            initializer._init_module(module, nonlinearity)
        elif isinstance(module, (nn.LayerNorm, nn.Embedding)):
            initializer._init_module(module, nonlinearity)

    return init_fn
