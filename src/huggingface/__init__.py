"""
HuggingFace Transformers 兼容层 (v1.6)

提供 RamanujanPreTrainedModel 基类和预定义模型实现。
"""

from .modeling_ramanujan import (
    RamanujanPreTrainedModel,
    RamanujanGPT2,
    RamanujanLlama,
    RamanujanMistral,
    RamanujanConfig,
)

__all__ = [
    'RamanujanPreTrainedModel',
    'RamanujanGPT2',
    'RamanujanLlama',
    'RamanujanMistral',
    'RamanujanConfig',
]
