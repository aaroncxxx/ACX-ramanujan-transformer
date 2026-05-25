"""
ACX-Ramanujan-Transformer

基于拉马努金模函数递推关系的神经网络权重初始化方法。

快速使用:
    from acx_ramanujan import patch_model, RamanujanInitializer

    # 一行代码初始化任意 PyTorch 模型
    model = patch_model(model, quantization='int8')

    # 或者精细控制
    initializer = RamanujanInitializer(num_layers=24, quantization='int8')
    initializer.apply(model)
"""

__version__ = '2.0.0'
__author__ = 'aaroncxxx'

from .src.ramanujan_initializer import (
    RamanujanInitializer,
    ramanujan_init_,
    ramanujan_coefficients,
    get_ramanujan_scale,
    get_adaptive_weight_decay,
    compute_rope_correction,
    LayerRole,
    ACTIVATION_GAIN,
    QUANTIZATION_GAIN,
)

from .src.patch import patch_model, quick_init


__all__ = [
    # 核心
    'RamanujanInitializer',
    'patch_model',
    'quick_init',
    # 底层
    'ramanujan_init_',
    'ramanujan_coefficients',
    'get_ramanujan_scale',
    'get_adaptive_weight_decay',
    'compute_rope_correction',
    # 常量
    'LayerRole',
    'ACTIVATION_GAIN',
    'QUANTIZATION_GAIN',
    # 元信息
    '__version__',
]
