"""
模型导出工具 (v1.6)

支持:
    - ONNX 格式导出
    - TorchScript 导出
    - 精度一致性验证
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger('acx_ramanujan')


def export_onnx(
    model: nn.Module,
    output_path: str,
    vocab_size: int = 50257,
    seq_len: int = 128,
    opset_version: int = 17,
    dynamic_axes: bool = True,
) -> str:
    """
    导出模型为 ONNX 格式

    Args:
        model: 训练好的模型
        output_path: 输出路径
        vocab_size: 词表大小（用于 dummy input）
        seq_len: 序列长度
        opset_version: ONNX opset 版本
        dynamic_axes: 是否使用动态轴

    Returns:
        导出文件路径
    """
    try:
        import onnx
    except ImportError:
        raise ImportError("需要安装 onnx: pip install onnx")

    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dummy input
    dummy_input = torch.randint(0, vocab_size, (1, seq_len))

    # 动态轴配置
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'},
        }

    # 导出
    logger.info(f"导出 ONNX 模型至 {output_path}")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes=dynamic_axes_dict,
        do_constant_folding=True,
    )

    # 验证
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    file_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"ONNX 导出成功: {output_path} ({file_size:.1f} MB)")

    return str(output_path)


def export_torchscript(
    model: nn.Module,
    output_path: str,
    vocab_size: int = 50257,
    seq_len: int = 128,
) -> str:
    """
    导出模型为 TorchScript 格式

    Args:
        model: 训练好的模型
        output_path: 输出路径
        vocab_size: 词表大小
        seq_len: 序列长度

    Returns:
        导出文件路径
    """
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randint(0, vocab_size, (1, seq_len))

    logger.info(f"导出 TorchScript 模型至 {output_path}")

    try:
        scripted = torch.jit.trace(model, dummy_input)
        scripted.save(str(output_path))
    except Exception as e:
        logger.warning(f"TorchScript trace 失败: {e}，尝试 script")
        scripted = torch.jit.script(model)
        scripted.save(str(output_path))

    file_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"TorchScript 导出成功: {output_path} ({file_size:.1f} MB)")

    return str(output_path)


def validate_export(
    original_model: nn.Module,
    exported_path: str,
    vocab_size: int = 50257,
    seq_len: int = 128,
    tolerance: float = 1e-4,
    format: str = 'onnx',
) -> Dict[str, Any]:
    """
    验证导出模型的精度一致性

    Args:
        original_model: 原始 PyTorch 模型
        exported_path: 导出模型路径
        vocab_size: 词表大小
        seq_len: 序列长度
        tolerance: 允许的精度误差
        format: 导出格式 ('onnx' / 'torchscript')

    Returns:
        验证结果字典
    """
    original_model.eval()

    # 测试输入
    test_input = torch.randint(0, vocab_size, (1, seq_len))

    # 原始模型输出
    with torch.no_grad():
        original_output = original_model(test_input)
        if isinstance(original_output, dict):
            original_output = original_output['logits']

    max_diff = float('inf')

    if format == 'onnx':
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(exported_path)
            ort_output = session.run(None, {'input_ids': test_input.numpy()})[0]
            original_np = original_output.numpy()
            max_diff = float(abs(original_np - ort_output).max())
        except ImportError:
            logger.warning("onnxruntime 未安装，跳过精度验证")
            return {'status': 'skipped', 'reason': 'onnxruntime not installed'}

    elif format == 'torchscript':
        loaded = torch.jit.load(exported_path)
        with torch.no_grad():
            loaded_output = loaded(test_input)
        max_diff = float((original_output - loaded_output).abs().max().item())

    passed = max_diff < tolerance
    result = {
        'status': 'passed' if passed else 'failed',
        'max_diff': max_diff,
        'tolerance': tolerance,
        'format': format,
    }

    if passed:
        logger.info(f"精度验证通过: max_diff={max_diff:.6f} < {tolerance}")
    else:
        logger.warning(f"精度验证失败: max_diff={max_diff:.6f} >= {tolerance}")

    return result
