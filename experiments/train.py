"""
训练脚本 (v1.6)

v1.6 新增:
    - 原生混合精度训练 (torch.cuda.amp)
    - 分布式 DDP 训练支持
    - 完善断点续训 (--resume)
    - WandB/TensorBoard 日志集成
"""

import sys
import os
import math
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from pathlib import Path
from typing import Optional, Dict, Tuple
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ramanujan_transformer import build_ramanujan_transformer
from src.checkpoint import CheckpointManager
from src.logging_utils import TrainingLogger

logger = logging.getLogger('acx_ramanujan')


def get_piecewise_exp_schedule_with_warmup(optimizer, warmup_steps: int,
                                            total_steps: int,
                                            alpha: float = 3.0,
                                            beta: float = 2.0,
                                            phase1_ratio: float = 0.3):
    """三段式分段指数调度"""
    phase1_steps = int((total_steps - warmup_steps) * phase1_ratio)
    phase2_steps = total_steps - warmup_steps - phase1_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        s = current_step - warmup_steps

        if s < phase1_steps:
            return math.exp(-alpha * s / phase1_steps)

        s2 = s - phase1_steps
        phase1_end_lr = math.exp(-alpha)
        return phase1_end_lr * math.exp(-beta * s2 / phase2_steps)

    return LambdaLR(optimizer, lr_lambda)


def generate_synthetic_data(vocab_size: int = 1000, num_samples: int = 1000,
                            seq_len: int = 128):
    """生成合成数据"""
    data = torch.randint(0, vocab_size, (num_samples, seq_len))
    return data[:, :-1], data[:, 1:]


def setup_distributed(rank: int, world_size: int):
    """初始化分布式训练"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """清理分布式训练"""
    dist.destroy_process_group()


def train(
    model: nn.Module,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    val_data: Tuple[torch.Tensor, torch.Tensor],
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
    warmup_ratio: float = 0.1,
    device: str = 'cpu',
    mixed_precision: str = 'none',
    checkpoint_manager: Optional[CheckpointManager] = None,
    training_logger: Optional[TrainingLogger] = None,
    resume_path: Optional[str] = None,
    grad_clip: float = 1.0,
    weight_decay: float = 0.01,
    seed: int = 42,
) -> float:
    """
    训练循环 (v1.6)

    v1.6 新增:
        - mixed_precision: 'fp16'/'bf16'/'none'
        - checkpoint_manager: 断点续训
        - training_logger: 日志集成
        - resume_path: 恢复训练路径
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_x, train_y = train_data
    val_x, val_y = val_data

    total_steps = (len(train_x) // batch_size) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_piecewise_exp_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 混合精度 Scaler
    scaler = None
    use_amp = mixed_precision in ('fp16', 'bf16')
    amp_dtype = torch.float16 if mixed_precision == 'fp16' else torch.bfloat16

    if use_amp and device != 'cpu':
        scaler = torch.cuda.amp.GradScaler(enabled=(mixed_precision == 'fp16'))

    # 断点续训
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    if checkpoint_manager is not None and resume_path is not None:
        resume_info = checkpoint_manager.load(
            model, optimizer, scheduler, scaler, path=resume_path, device=device
        )
        start_epoch = resume_info['epoch']
        global_step = resume_info['global_step']
        best_val_loss = resume_info['best_val_loss']
        logger.info(f"从 epoch {start_epoch} 恢复训练 (step={global_step})")

    # 日志
    if training_logger is not None:
        training_logger.log_model_info(model, global_step)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for i in range(0, len(train_x), batch_size):
            batch_x = train_x[i:i+batch_size].to(device)
            batch_y = train_y[i:i+batch_size].to(device)

            optimizer.zero_grad()

            if use_amp and device != 'cpu':
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    logits = model(batch_x)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
            else:
                logits = model(batch_x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            scheduler.step()
            global_step += 1

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches

        # 验证
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for i in range(0, len(val_x), batch_size):
                batch_x = val_x[i:i+batch_size].to(device)
                batch_y = val_y[i:i+batch_size].to(device)

                if use_amp and device != 'cpu':
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        logits = model(batch_x)
                        loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
                else:
                    logits = model(batch_x)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        # 日志
        if training_logger is not None:
            training_logger.log({
                'train/loss': avg_train_loss,
                'val/loss': avg_val_loss,
                'train/lr': scheduler.get_last_lr()[0],
                'train/epoch': epoch + 1,
            }, step=global_step)

        # 检查点
        if checkpoint_manager is not None and (epoch + 1) % 5 == 0:
            checkpoint_manager.save(
                model, optimizer, scheduler, scaler,
                epoch=epoch + 1, global_step=global_step,
                best_val_loss=best_val_loss,
            )

        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1:3d}: train_loss={avg_train_loss:.4f}, "
                        f"val_loss={avg_val_loss:.4f}, best={best_val_loss:.4f}")

    # 最终保存
    if checkpoint_manager is not None:
        checkpoint_manager.save(
            model, optimizer, scheduler, scaler,
            epoch=epochs, global_step=global_step,
            best_val_loss=best_val_loss,
        )

    if training_logger is not None:
        training_logger.close()

    return best_val_loss


def main():
    import argparse

    parser = argparse.ArgumentParser(description='ACX-Ramanujan 训练')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练检查点路径')
    parser.add_argument('--mixed-precision', type=str, default='none',
                        choices=['fp16', 'bf16', 'none'], help='混合精度')
    parser.add_argument('--logger', type=str, default='none',
                        choices=['wandb', 'tensorboard', 'none'], help='日志后端')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='检查点目录')
    parser.add_argument('--device', type=str, default='auto', help='设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # 模型参数
    parser.add_argument('--vocab-size', type=int, default=1000)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--dim-feedforward', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup-ratio', type=float, default=0.1)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--train-samples', type=int, default=800)
    parser.add_argument('--val-samples', type=int, default=200)
    parser.add_argument('--use-flash-attention', action='store_true', default=True)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

    # 设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    logger.info(f"设备: {device}, 混合精度: {args.mixed_precision}")

    # 数据
    train_data = generate_synthetic_data(args.vocab_size, args.train_samples, args.seq_len)
    val_data = generate_synthetic_data(args.vocab_size, args.val_samples, args.seq_len)

    # 模型
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
    logger.info(f"模型参数量: {params:,}")

    # 检查点
    checkpoint_manager = CheckpointManager(args.checkpoint_dir)

    # 日志
    training_logger = TrainingLogger(
        logger_type=args.logger,
        config=vars(args),
    )

    # 训练
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
        seed=args.seed,
    )

    logger.info(f"训练完成! Best val loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
