"""
patch_model — 一行代码初始化任意 PyTorch 模型

核心 API:
    from acx_ramanujan import patch_model

    # 最简单用法
    model = patch_model(model)

    # 完整参数
    model = patch_model(model, quantization='int8', long_context_seq_len=8192)

    # 快捷函数
    model = quick_init(model, num_layers=24)
"""

import math
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from .ramanujan_initializer import (
    RamanujanInitializer,
    assign_layer_indices,
    tag_linear_role,
    LayerRole,
)

logger = logging.getLogger('acx_ramanujan')


def _detect_num_layers(model: nn.Module) -> int:
    """自动检测模型层数（通过 ModuleList / ModuleDict 模式）"""
    max_depth = 0

    # 方法 1: 查找名为 'layers', 'h', 'blocks', 'encoder.layer', 'decoder.layers' 的 ModuleList
    layer_names = ['layers', 'h', 'blocks', 'layer', 'encoder.layer',
                   'decoder.layers', 'transformer.h', 'model.layers']

    for name, module in model.named_modules():
        for target in layer_names:
            if name == target or name.endswith(f'.{target}'):
                if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
                    max_depth = max(max_depth, len(module))

    # 方法 2: 统计 Transformer Block 数量
    if max_depth == 0:
        block_count = 0
        for name, module in model.named_modules():
            cls_name = module.__class__.__name__.lower()
            if any(kw in cls_name for kw in ['transformerblock', 'block', 'layer']):
                if any(isinstance(m, nn.MultiheadAttention) or
                       hasattr(m, 'self_attn') or
                       any('attention' in n for n, _ in m.named_children())
                       for m in module.modules()):
                    block_count += 1
        max_depth = block_count

    # 方法 3: fallback 到 Linear 层数 / 6（粗估）
    if max_depth == 0:
        linear_count = sum(1 for _ in model.modules() if isinstance(_, nn.Linear))
        max_depth = max(1, linear_count // 6)

    return max_depth


def _auto_tag_roles(model: nn.Module):
    """自动检测并标记 QKV / FFN / Output 层角色"""
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        name_lower = name.lower()

        # Attention 相关
        if any(kw in name_lower for kw in ['q_proj', 'query', 'c_attn']):
            if 'q_proj' in name_lower or 'query' in name_lower:
                tag_linear_role(module, LayerRole.Q_PROJ)
            else:
                tag_linear_role(module, LayerRole.QKV_PROJ)
        elif any(kw in name_lower for kw in ['k_proj', 'key']):
            tag_linear_role(module, LayerRole.K_PROJ)
        elif any(kw in name_lower for kw in ['v_proj', 'value']):
            tag_linear_role(module, LayerRole.V_PROJ)
        elif any(kw in name_lower for kw in ['out_proj', 'c_proj', 'o_proj']):
            tag_linear_role(module, LayerRole.OUTPUT_ATTN)
        # FFN 相关
        elif any(kw in name_lower for kw in ['fc1', 'up_proj', 'gate_proj', 'w1', 'c_fc']):
            tag_linear_role(module, LayerRole.FFN_UP)
        elif any(kw in name_lower for kw in ['fc2', 'down_proj', 'w2', 'c_proj']):
            # 避免和 attention 的 c_proj 冲突
            if 'attn' not in name_lower and 'attention' not in name_lower:
                tag_linear_role(module, LayerRole.FFN_DOWN)
        # LM Head
        elif any(kw in name_lower for kw in ['lm_head', 'output_projection', 'cls']):
            tag_linear_role(module, LayerRole.LM_HEAD)
        # Router (MoE)
        elif any(kw in name_lower for kw in ['router', 'gate']):
            tag_linear_role(module, LayerRole.ROUTER)
        # Embedding（非 Linear，单独处理）


def patch_model(model: nn.Module,
                num_layers: Optional[int] = None,
                ramanujan_depth: Optional[int] = None,
                transition_depth: Optional[int] = None,
                quantization: str = 'none',
                long_context_seq_len: int = 512,
                nonlinearity: str = 'linear',
                gain: Optional[float] = None,
                auto_tag: bool = True,
                verbose: bool = False) -> nn.Module:
    """
    一行代码为任意 PyTorch 模型应用 Ramanujan 初始化。

    Args:
        model: 任意 nn.Module
        num_layers: 模型层数（None = 自动检测）
        ramanujan_depth: Ramanujan 调制深度（None = 自动计算）
        transition_depth: 过渡深度（None = 自动计算）
        quantization: 量化精度 ('none', 'int8', 'fp8', 'int4')
        long_context_seq_len: 长上下文序列长度
        nonlinearity: 激活函数类型
        gain: 手动增益
        auto_tag: 自动标记 QKV/FFN 角色
        verbose: 打印详细信息

    Returns:
        初始化后的模型（原地修改，同时返回引用）

    Example:
        import torch.nn as nn
        from acx_ramanujan import patch_model

        model = nn.TransformerEncoder(...)
        model = patch_model(model, quantization='int8')

        # HuggingFace 模型也可以
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model = patch_model(model)
    """
    # 自动检测层数
    if num_layers is None:
        num_layers = _detect_num_layers(model)
        if verbose:
            print(f"[acx] 自动检测层数: {num_layers}")

    if num_layers <= 0:
        logger.warning("无法检测模型层数，使用默认值 12")
        num_layers = 12

    # 自动标记角色
    if auto_tag:
        _auto_tag_roles(model)

    # 创建初始化器
    initializer = RamanujanInitializer(
        ramanujan_depth=ramanujan_depth,
        transition_depth=transition_depth,
        num_layers=num_layers,
        nonlinearity=nonlinearity,
        gain=gain,
        quantization=quantization,
        long_context_seq_len=long_context_seq_len,
    )

    # 应用初始化
    initializer.apply(model)

    if verbose:
        print(f"[acx] Ramanujan 初始化完成")
        print(f"  深度: {initializer.ramanujan_depth}/{initializer.transition_depth}")
        print(f"  量化: {quantization}")
        print(f"  长上下文: {long_context_seq_len}")

    return model


def quick_init(model: nn.Module, num_layers: int = 12,
               quantization: str = 'none') -> nn.Module:
    """
    快捷初始化：最少参数，最简用法。

    Args:
        model: PyTorch 模型
        num_layers: 层数
        quantization: 量化精度

    Returns:
        初始化后的模型

    Example:
        model = quick_init(model, num_layers=24)
    """
    return patch_model(model, num_layers=num_layers,
                       quantization=quantization, auto_tag=True)


# ─── HuggingFace 集成辅助 ────────────────────────────────────────

def patch_hf_model(model, quantization: str = 'none',
                   long_context_seq_len: int = 512):
    """
    专门为 HuggingFace Transformers 模型设计的初始化。

    自动处理 HuggingFace 特有的模块命名和结构。

    Args:
        model: HuggingFace PreTrainedModel
        quantization: 量化精度
        long_context_seq_len: 长上下文长度

    Returns:
        初始化后的模型

    Example:
        from transformers import AutoModelForCausalLM
        from acx_ramanujan import patch_hf_model

        model = AutoModelForCausalLM.from_config(config)
        model = patch_hf_model(model, quantization='int8')
    """
    # HuggingFace 模型通常有 config 属性
    num_layers = None
    if hasattr(model, 'config'):
        config = model.config
        for attr in ['num_hidden_layers', 'n_layer', 'num_layers',
                     'encoder_layers', 'decoder_layers']:
            if hasattr(config, attr):
                num_layers = getattr(config, attr)
                break

    return patch_model(model, num_layers=num_layers,
                       quantization=quantization,
                       long_context_seq_len=long_context_seq_len,
                       auto_tag=True, verbose=True)
