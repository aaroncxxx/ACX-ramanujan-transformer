"""
方差可视化工具 (v1.6)

支持:
    - 层输出方差变化曲线
    - 梯度分布直方图
    - 注意力权重热力图

用法:
    python experiments/visualize_variance.py --checkpoint model.pt --type variance
    python experiments/visualize_variance.py --checkpoint model.pt --type gradient
    python experiments/visualize_variance.py --checkpoint model.pt --type attention
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ramanujan_transformer import build_ramanujan_transformer
from src.ramanujan_initializer import RamanujanInitializer, get_ramanujan_scale

logger = logging.getLogger('acx_ramanujan')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib 未安装，可视化功能不可用")


def visualize_variance(
    model: Optional[nn.Module] = None,
    output_dir: str = 'figures',
    num_layers: int = 12,
    d_model: int = 768,
    seq_len: int = 128,
    batch_size: int = 32,
):
    """
    绘制层输出方差变化曲线

    验证拉马努金初始化的方差保持效果
    """
    if not HAS_MATPLOTLIB:
        logger.error("需要 matplotlib: pip install matplotlib")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 初始化缩放因子曲线
    ax = axes[0]
    depths = list(range(min(num_layers, 64)))
    scales = [get_ramanujan_scale(i, min(8, num_layers // 3), min(16, num_layers // 4), d_model)
              for i in depths]
    ax.plot(depths, scales, 'r-', linewidth=2, label='Ramanujan Scale')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Xavier (baseline)')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Init Scale Factor')
    ax.set_title('Ramanujan Initialization Scale')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 实际方差传播（如果有模型）
    ax = axes[1]
    if model is not None:
        model.eval()
        device = next(model.parameters()).device

        # 注册 hook 收集每层输出方差
        layer_vars = []
        hooks = []

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, torch.Tensor):
                layer_vars.append(output.var().item())

        for layer in model.layers if hasattr(model, 'layers') else []:
            hooks.append(layer.register_forward_hook(hook_fn))

        with torch.no_grad():
            x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
            _ = model(x)

        for h in hooks:
            h.remove()

        if layer_vars:
            ax.plot(range(len(layer_vars)), layer_vars, 'b-o', markersize=3)
            ax.axhline(y=layer_vars[0], color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Layer Index')
            ax.set_ylabel('Output Variance')
            ax.set_title('Layer Output Variance (Actual)')
            ax.grid(True, alpha=0.3)
    else:
        # 理论方差传播模拟
        for method, color in [('ramanujan', 'red'), ('xavier', 'blue'), ('he', 'green')]:
            vars_list = []
            x = torch.randn(batch_size, d_model)
            vars_list.append(x.var().item())

            for i in range(min(num_layers, 64)):
                if method == 'ramanujan':
                    s = get_ramanujan_scale(i, min(8, num_layers // 3), min(16, num_layers // 4), d_model)
                    std = s / np.sqrt(d_model)
                elif method == 'xavier':
                    std = 1.0 / np.sqrt(d_model)
                else:
                    std = np.sqrt(2.0 / d_model)

                W = torch.randn(d_model, d_model) * std
                x = x + x @ W  # residual connection
                vars_list.append(x.var().item())

            ax.plot(range(len(vars_list)), vars_list, color=color, label=method)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Variance')
        ax.set_title('Theoretical Variance Propagation')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / 'variance_curve.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"方差曲线已保存: {save_path}")


def visualize_gradients(
    model: nn.Module,
    output_dir: str = 'figures',
    vocab_size: int = 1000,
    seq_len: int = 128,
    batch_size: int = 16,
):
    """
    绘制梯度分布直方图
    """
    if not HAS_MATPLOTLIB:
        logger.error("需要 matplotlib")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    device = next(model.parameters()).device

    # 前向 + 反向
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    if hasattr(model, 'forward') and 'return_aux_loss' in model.forward.__code__.co_varnames:
        logits, aux = model(x, return_aux_loss=False)
    else:
        logits = model(x)

    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()

    # 收集梯度
    grad_data = []
    layer_names = []
    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            grad_data.append(param.grad.cpu().numpy().flatten())
            layer_names.append(name.split('.')[-2] + '.' + name.split('.')[-1])

    # 绘制直方图
    fig, ax = plt.subplots(figsize=(12, 6))

    n_layers = min(len(grad_data), 20)  # 最多显示 20 层
    for i in range(n_layers):
        data = grad_data[i]
        ax.hist(data, bins=50, alpha=0.5, label=layer_names[i])

    ax.set_xlabel('Gradient Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Gradient Distribution by Layer')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / 'gradient_distribution.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"梯度分布已保存: {save_path}")

    model.zero_grad()


def visualize_attention(
    model: nn.Module,
    output_dir: str = 'figures',
    vocab_size: int = 1000,
    seq_len: int = 64,
    layer_idx: int = 0,
):
    """
    绘制注意力权重热力图
    """
    if not HAS_MATPLOTLIB:
        logger.error("需要 matplotlib")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    # 注册 hook 获取注意力权重
    attn_weights = None

    def attn_hook(module, input, output):
        nonlocal attn_weights
        if isinstance(output, tuple) and len(output) > 1:
            attn_weights = output[1]

    # 找到目标注意力层
    target_layer = None
    if hasattr(model, 'layers'):
        if layer_idx < len(model.layers):
            target_layer = model.layers[layer_idx]
            if hasattr(target_layer, 'self_attn'):
                target_layer = target_layer.self_attn

    if target_layer is None:
        logger.warning("未找到目标注意力层")
        return

    hook = target_layer.register_forward_hook(attn_hook)

    with torch.no_grad():
        x = torch.randint(0, vocab_size, (1, seq_len), device=device)
        _ = model(x)

    hook.remove()

    if attn_weights is None:
        logger.warning("未捕获到注意力权重（可能使用了 FlashAttention）")
        return

    # 绘制热力图
    weights = attn_weights[0].cpu().numpy()  # (nhead, T, T)
    n_heads = weights.shape[0]

    fig, axes = plt.subplots(1, min(n_heads, 4), figsize=(4 * min(n_heads, 4), 4))
    if n_heads == 1:
        axes = [axes]

    for i, ax in enumerate(axes[:min(n_heads, 4)]):
        im = ax.imshow(weights[i], cmap='viridis', aspect='auto')
        ax.set_title(f'Head {i}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax)

    plt.suptitle(f'Attention Weights (Layer {layer_idx})')
    plt.tight_layout()
    save_path = output_dir / f'attention_layer{layer_idx}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"注意力热力图已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='ACX-Ramanujan 可视化工具')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--type', type=str, default='variance',
                        choices=['variance', 'gradient', 'attention', 'all'],
                        help='可视化类型')
    parser.add_argument('--output-dir', type=str, default='figures', help='输出目录')
    parser.add_argument('--num-layers', type=int, default=12, help='层数')
    parser.add_argument('--d-model', type=int, default=768, help='模型维度')
    parser.add_argument('--vocab-size', type=int, default=1000, help='词表大小')
    parser.add_argument('--seq-len', type=int, default=128, help='序列长度')
    parser.add_argument('--layer-idx', type=int, default=0, help='注意力层索引')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model = None
    if args.checkpoint:
        model = build_ramanujan_transformer(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            decoder_only=True,
        )
        state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"已加载模型: {args.checkpoint}")

    if args.type in ('variance', 'all'):
        visualize_variance(model, args.output_dir, args.num_layers, args.d_model,
                           args.seq_len)

    if args.type in ('gradient', 'all') and model is not None:
        visualize_gradients(model, args.output_dir, args.vocab_size, args.seq_len)

    if args.type in ('attention', 'all') and model is not None:
        visualize_attention(model, args.output_dir, args.vocab_size,
                           args.seq_len // 2, args.layer_idx)

    logger.info("可视化完成")


if __name__ == '__main__':
    main()
