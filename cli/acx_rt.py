#!/usr/bin/env python3
"""
ACX-Ramanujan-Transformer CLI (v1.6)

用法:
    acx-rt info
    acx-rt verify [--depth 200] [--dim 512]
    acx-rt benchmark [--layers 6,12,24]
    acx-rt train [--config ...] [--resume ...] [--nproc_per_node N] [--logger wandb] [--mixed-precision fp16]
    acx-rt visualize [--checkpoint ...] [--type variance]
    acx-rt export [--checkpoint ...] [--format onnx]
"""

import argparse
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def cmd_info(args):
    """显示项目信息"""
    from src.ramanujan_initializer import (
        ramanujan_coefficients, compute_optimal_depth, ACTIVATION_GAIN, QUANTIZATION_GAIN
    )

    print("=" * 50)
    print("ACX-Ramanujan-Transformer v1.6.0")
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

    print(f"\n🔢 量化增益表:")
    for name, gain in QUANTIZATION_GAIN.items():
        print(f"   {name:>10s}: {gain:.4f}")

    print(f"\n🏗️ 动态深度计算:")
    for n in [6, 12, 24, 48, 96]:
        r, t = compute_optimal_depth(n)
        print(f"   {n:3d} 层 → ramanujan_depth={r}, transition_depth={t}")

    # FlashAttention 状态
    from src.attention import _is_flash_available, _get_flash_version
    flash_ok = _is_flash_available()
    flash_ver = _get_flash_version()
    print(f"\n⚡ FlashAttention: {'✅ 可用' if flash_ok else '❌ 不可用'} ({flash_ver})")


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
    from src.ramanujan_transformer import build_ramanujan_transformer

    layers_list = [int(x) for x in args.layers.split(',')]
    print(f"基准对比 (层数: {layers_list})")
    print("=" * 50)

    for num_layers in layers_list:
        model = build_ramanujan_transformer(
            vocab_size=1000, d_model=256, nhead=4,
            num_layers=num_layers, dim_feedforward=1024,
            decoder_only=True,
        )
        params = sum(p.numel() for p in model.parameters())
        print(f"\n  {num_layers:3d} 层: {params:>12,} 参数 ({params/1e6:.1f}M)")


def cmd_train(args):
    """训练 (v1.6)"""
    import torch
    import logging
    from experiments.train import generate_synthetic_data, train
    from src.ramanujan_transformer import build_ramanujan_transformer
    from src.checkpoint import CheckpointManager
    from src.logging_utils import TrainingLogger

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

    print("拉马努金 Transformer 训练 (v1.6)")
    print("=" * 50)

    # DDP 支持
    if args.nproc_per_node and args.nproc_per_node > 1:
        print(f"DDP 训练: {args.nproc_per_node} GPU")
        import torch.multiprocessing as mp
        mp.spawn(
            _ddp_train_worker,
            args=(args,),
            nprocs=args.nproc_per_node,
            join=True,
        )
        return

    # 单卡训练
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}, 混合精度: {args.mixed_precision}")

    train_data = generate_synthetic_data(args.vocab_size, args.train_samples, args.seq_len)
    val_data = generate_synthetic_data(args.vocab_size, args.val_samples, args.seq_len)

    model = build_ramanujan_transformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        decoder_only=True,
        use_flash_attention=args.use_flash_attention,
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {params:,}")

    checkpoint_manager = CheckpointManager(args.checkpoint_dir)
    training_logger = TrainingLogger(logger_type=args.logger, config=vars(args))

    best_loss = train(
        model, train_data, val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        device=device,
        mixed_precision=args.mixed_precision,
        checkpoint_manager=checkpoint_manager,
        training_logger=training_logger,
        resume_path=args.resume,
    )
    print(f"\n训练完成! Best val loss: {best_loss:.4f}")


def _ddp_train_worker(rank, args):
    """DDP 训练 worker"""
    import torch
    import torch.distributed as dist
    import logging
    from torch.nn.parallel import DistributedDataParallel as DDP
    from experiments.train import (
        generate_synthetic_data, train, setup_distributed, cleanup_distributed
    )
    from src.ramanujan_transformer import build_ramanujan_transformer
    from src.checkpoint import CheckpointManager
    from src.logging_utils import TrainingLogger

    logging.basicConfig(level=logging.INFO)

    setup_distributed(rank, args.nproc_per_node)
    device = f'cuda:{rank}'

    train_data = generate_synthetic_data(args.vocab_size, args.train_samples, args.seq_len)
    val_data = generate_synthetic_data(args.vocab_size, args.val_samples, args.seq_len)

    model = build_ramanujan_transformer(
        vocab_size=args.vocab_size, d_model=args.d_model,
        nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, decoder_only=True,
        use_flash_attention=args.use_flash_attention,
    ).to(device)

    model = DDP(model, device_ids=[rank])

    checkpoint_manager = CheckpointManager(args.checkpoint_dir) if rank == 0 else None
    training_logger = TrainingLogger(logger_type=args.logger, config=vars(args)) if rank == 0 else None

    train(
        model, train_data, val_data,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, warmup_ratio=args.warmup_ratio,
        device=device, mixed_precision=args.mixed_precision,
        checkpoint_manager=checkpoint_manager,
        training_logger=training_logger,
        resume_path=args.resume,
    )

    cleanup_distributed()


def cmd_visualize(args):
    """可视化"""
    from experiments.visualize_variance import (
        visualize_variance, visualize_gradients, visualize_attention
    )
    from src.ramanujan_transformer import build_ramanujan_transformer
    import torch

    model = None
    if args.checkpoint:
        model = build_ramanujan_transformer(
            vocab_size=args.vocab_size, d_model=args.d_model,
            num_layers=args.num_layers, decoder_only=True,
        )
        state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        print(f"已加载模型: {args.checkpoint}")

    if args.type in ('variance', 'all'):
        visualize_variance(model, args.output_dir, args.num_layers, args.d_model)

    if args.type in ('gradient', 'all') and model is not None:
        visualize_gradients(model, args.output_dir, args.vocab_size)

    if args.type in ('attention', 'all') and model is not None:
        visualize_attention(model, args.output_dir, args.vocab_size,
                           layer_idx=args.layer_idx)

    print(f"可视化完成，输出目录: {args.output_dir}")


def cmd_export(args):
    """模型导出"""
    import torch
    from src.ramanujan_transformer import build_ramanujan_transformer
    from src.export import export_onnx, export_torchscript, validate_export

    model = build_ramanujan_transformer(
        vocab_size=args.vocab_size, d_model=args.d_model,
        num_layers=args.num_layers, decoder_only=True,
    )

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        print(f"已加载模型: {args.checkpoint}")

    model.eval()

    if args.format == 'onnx':
        output_path = args.output or 'model.onnx'
        export_onnx(model, output_path, vocab_size=args.vocab_size, seq_len=args.seq_len)
        result = validate_export(model, output_path, vocab_size=args.vocab_size,
                                seq_len=args.seq_len, format='onnx')
        print(f"精度验证: {result['status']} (max_diff={result['max_diff']:.6f})")

    elif args.format == 'torchscript':
        output_path = args.output or 'model.pt'
        export_torchscript(model, output_path, vocab_size=args.vocab_size, seq_len=args.seq_len)

    print(f"导出完成: {args.format}")


def main():
    parser = argparse.ArgumentParser(
        prog='acx-rt',
        description='ACX-Ramanujan-Transformer CLI v1.6'
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

    # train (v1.6 增强)
    p_train = sub.add_parser('train', help='训练')
    p_train.add_argument('--config', type=str, default=None)
    p_train.add_argument('--resume', type=str, default=None, help='恢复训练检查点路径')
    p_train.add_argument('--nproc-per-node', type=int, default=None, help='DDP GPU 数量')
    p_train.add_argument('--logger', type=str, default='none',
                         choices=['wandb', 'tensorboard', 'none'])
    p_train.add_argument('--mixed-precision', type=str, default='none',
                         choices=['fp16', 'bf16', 'none'])
    p_train.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    p_train.add_argument('--use-flash-attention', action='store_true', default=True)
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

    # visualize (v1.6 新增)
    p_vis = sub.add_parser('visualize', help='可视化')
    p_vis.add_argument('--checkpoint', type=str, default=None)
    p_vis.add_argument('--type', type=str, default='variance',
                       choices=['variance', 'gradient', 'attention', 'all'])
    p_vis.add_argument('--output-dir', type=str, default='figures')
    p_vis.add_argument('--num-layers', type=int, default=12)
    p_vis.add_argument('--d-model', type=int, default=768)
    p_vis.add_argument('--vocab-size', type=int, default=1000)
    p_vis.add_argument('--seq-len', type=int, default=128)
    p_vis.add_argument('--layer-idx', type=int, default=0)

    # export (v1.6 新增)
    p_export = sub.add_parser('export', help='模型导出')
    p_export.add_argument('--checkpoint', type=str, default=None)
    p_export.add_argument('--format', type=str, default='onnx',
                          choices=['onnx', 'torchscript'])
    p_export.add_argument('--output', type=str, default=None)
    p_export.add_argument('--vocab-size', type=int, default=50257)
    p_export.add_argument('--d-model', type=int, default=768)
    p_export.add_argument('--num-layers', type=int, default=12)
    p_export.add_argument('--seq-len', type=int, default=128)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        'info': cmd_info,
        'verify': cmd_verify,
        'benchmark': cmd_benchmark,
        'train': cmd_train,
        'visualize': cmd_visualize,
        'export': cmd_export,
    }
    commands[args.command](args)


if __name__ == '__main__':
    main()
