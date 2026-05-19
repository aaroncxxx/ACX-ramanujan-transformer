#!/usr/bin/env python3
"""
ACX-Ramanujan-Transformer CLI

用法:
    acx-rt train [--config configs/default.yaml] [--moe]
    acx-rt verify [--depth 200] [--dim 512]
    acx-rt benchmark [--layers 6,12,24]
    acx-rt info
"""

import argparse
import sys
import math
from pathlib import Path

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def cmd_info(args):
    """显示项目信息和当前配置"""
    from src.ramanujan_initializer import (
        ramanujan_coefficients, compute_optimal_depth, ACTIVATION_GAIN
    )

    print("=" * 50)
    print("ACX-Ramanujan-Transformer v1.4.0")
    print("=" * 50)

    print("\n📐 核心递推公式:")
    print("   a_{n+1} = (π²/n²) · a_n + (2π/(n(n+1))) · a_{n-1}")
    print("   a_0 = 1, a_1 = π/√3")

    print("\n📊 系数表 (前 10 项):")
    coeffs = ramanujan_coefficients(9)
    for i, c in enumerate(coeffs):
        print(f"   a_{i} = {c:.6f}")

    print(f"\n🔧 激活函数增益表:")
    for name, gain in ACTIVATION_GAIN.items():
        print(f"   {name:>10s}: {gain:.4f}")

    print(f"\n🏗️ 动态深度计算:")
    for n in [6, 12, 24, 48, 96]:
        r, t = compute_optimal_depth(n)
        print(f"   {n:3d} 层 → ramanujan_depth={r}, transition_depth={t}")


def cmd_verify(args):
    """方差保持性验证"""
    from src.ramanujan_initializer import RamanujanInitializer

    print(f"方差保持性验证 (depth={args.depth}, dim={args.dim})")
    print("=" * 50)

    init = RamanujanInitializer(num_layers=args.depth)

    for use_res in [False, True]:
        label = "残差模式" if use_res else "前馈模式"
        result = init.variance_test(
            depth=args.depth, dim=args.dim,
            nonlinearity=args.activation,
            use_residual=use_res
        )
        print(f"\n{label}:")
        print(f"  输入方差: {result['input_var']:.4f}")
        print(f"  输出方差: {result['output_var']:.4f}")
        print(f"  方差比:   {result['ratio']:.6f}")
        print(f"  最大比:   {result['max_ratio']:.6f}")
        print(f"  最小比:   {result['min_ratio']:.6f}")


def cmd_benchmark(args):
    """基准对比"""
    import torch
    import torch.nn as nn
    from src.ramanujan_transformer import build_ramanujan_transformer

    layers_list = [int(x) for x in args.layers.split(',')]
    print(f"基准对比 (层数: {layers_list})")
    print("=" * 50)

    for num_layers in layers_list:
        model = build_ramanujan_transformer(
            vocab_size=1000, d_model=256, nhead=4,
            num_layers=num_layers, dim_feedforward=1024,
            decoder_only=True
        )
        params = sum(p.numel() for p in model.parameters())
        print(f"\n  {num_layers:3d} 层: {params:>12,} 参数 ({params/1e6:.1f}M)")


def cmd_train(args):
    """训练"""
    import torch
    from experiments.train import generate_synthetic_data, train
    from src.ramanujan_transformer import build_ramanujan_transformer

    print("拉马努金 Transformer 训练")
    print("=" * 50)

    train_data = generate_synthetic_data(args.vocab_size, args.train_samples, args.seq_len)
    val_data = generate_synthetic_data(args.vocab_size, args.val_samples, args.seq_len)

    model = build_ramanujan_transformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        decoder_only=True,
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {params:,}")

    best_loss = train(
        model, train_data, val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
    )
    print(f"\n训练完成! Best val loss: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(
        prog='acx-rt',
        description='ACX-Ramanujan-Transformer CLI'
    )
    sub = parser.add_subparsers(dest='command')

    # info
    sub.add_parser('info', help='显示项目信息')

    # verify
    p_verify = sub.add_parser('verify', help='方差保持性验证')
    p_verify.add_argument('--depth', type=int, default=200)
    p_verify.add_argument('--dim', type=int, default=512)
    p_verify.add_argument('--activation', default='linear')

    # benchmark
    p_bench = sub.add_parser('benchmark', help='基准对比')
    p_bench.add_argument('--layers', default='6,12,24')

    # train
    p_train = sub.add_parser('train', help='训练')
    p_train.add_argument('--vocab-size', type=int, default=1000)
    p_train.add_argument('--d-model', type=int, default=256)
    p_train.add_argument('--nhead', type=int, default=4)
    p_train.add_argument('--num-layers', type=int, default=6)
    p_train.add_argument('--dim-feedforward', type=int, default=1024)
    p_train.add_argument('--epochs', type=int, default=20)
    p_train.add_argument('--batch-size', type=int, default=32)
    p_train.add_argument('--lr', type=float, default=3e-4)
    p_train.add_argument('--warmup-ratio', type=float, default=0.1)
    p_train.add_argument('--train-samples', type=int, default=800)
    p_train.add_argument('--val-samples', type=int, default=200)
    p_train.add_argument('--seq-len', type=int, default=128)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        'info': cmd_info,
        'verify': cmd_verify,
        'benchmark': cmd_benchmark,
        'train': cmd_train,
    }
    commands[args.command](args)


if __name__ == '__main__':
    main()
