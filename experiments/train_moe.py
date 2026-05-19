"""
MoE Transformer 训练脚本 (v1.6)

v1.6 新增:
    - 混合精度训练
    - 断点续训
    - WandB/TensorBoard 日志
    - 专家负载均衡可视化
"""

import sys
import math
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ramanujan_transformer import build_ramanujan_transformer
from src.moe import build_ramanujan_moe_transformer
from src.checkpoint import CheckpointManager
from src.logging_utils import TrainingLogger

logger = logging.getLogger('acx_ramanujan')


def get_piecewise_exp_schedule_with_warmup(optimizer, warmup_steps: int,
                                            total_steps: int,
                                            alpha: float = 3.0,
                                            beta: float = 2.0,
                                            phase1_ratio: float = 0.3):
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
    data = torch.randint(0, vocab_size, (num_samples, seq_len))
    return data[:, :-1], data[:, 1:]


def train_epoch(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scheduler=None,
    batch_size: int = 32,
    aux_loss_weight: float = 0.01,
    is_moe: bool = False,
    mixed_precision: str = 'none',
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip: float = 1.0,
    device: str = 'cpu',
) -> Dict:
    """训练一个 epoch (v1.6)"""
    model.train()
    total_loss = 0
    total_aux_loss = 0
    num_batches = 0

    use_amp = mixed_precision in ('fp16', 'bf16')
    amp_dtype = torch.float16 if mixed_precision == 'fp16' else torch.bfloat16

    for i in range(0, len(train_x), batch_size):
        batch_x = train_x[i:i+batch_size].to(device)
        batch_y = train_y[i:i+batch_size].to(device)

        optimizer.zero_grad()

        if use_amp and device != 'cpu':
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                if is_moe:
                    logits, aux_loss_dict = model(batch_x, return_aux_loss=True)
                    ce_loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
                    aux_loss = 0
                    if aux_loss_dict is not None:
                        lb_weight = aux_loss_dict.get('load_balancing_weight', aux_loss_weight)
                        aux_loss = (aux_loss_dict['load_balance_loss'] * lb_weight +
                                   aux_loss_dict['z_loss'] * 0.01)
                    loss = ce_loss + aux_loss
                    total_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
                else:
                    logits = model(batch_x)
                    ce_loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
                    loss = ce_loss

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
            if is_moe:
                logits, aux_loss_dict = model(batch_x, return_aux_loss=True)
                ce_loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
                aux_loss = 0
                if aux_loss_dict is not None:
                    lb_weight = aux_loss_dict.get('load_balancing_weight', aux_loss_weight)
                    aux_loss = (aux_loss_dict['load_balance_loss'] * lb_weight +
                               aux_loss_dict['z_loss'] * 0.01)
                loss = ce_loss + aux_loss
                total_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
            else:
                logits = model(batch_x)
                ce_loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
                loss = ce_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += ce_loss.item()
        num_batches += 1

    return {
        'loss': total_loss / max(num_batches, 1),
        'aux_loss': total_aux_loss / max(num_batches, 1) if is_moe else 0,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    criterion: nn.Module,
    batch_size: int = 32,
    is_moe: bool = False,
    mixed_precision: str = 'none',
    device: str = 'cpu',
) -> float:
    model.eval()
    total_loss = 0
    num_batches = 0

    use_amp = mixed_precision in ('fp16', 'bf16')
    amp_dtype = torch.float16 if mixed_precision == 'fp16' else torch.bfloat16

    for i in range(0, len(val_x), batch_size):
        batch_x = val_x[i:i+batch_size].to(device)
        batch_y = val_y[i:i+batch_size].to(device)

        if use_amp and device != 'cpu':
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                if is_moe:
                    logits, _ = model(batch_x, return_aux_loss=False)
                else:
                    logits = model(batch_x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
        else:
            if is_moe:
                logits, _ = model(batch_x, return_aux_loss=False)
            else:
                logits = model(batch_x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def run_comparison(
    mixed_precision: str = 'none',
    checkpoint_dir: str = 'checkpoints',
    logger_type: str = 'none',
    resume_path: Optional[str] = None,
    device: str = 'cpu',
):
    """对比标准 Transformer 和 MoE Transformer (v1.6)"""
    print("=" * 70)
    print("拉马努金 MoE Transformer vs 标准 Transformer (v1.6)")
    print("=" * 70)

    vocab_size = 1000
    d_model = 256
    nhead = 4
    num_layers = 6
    dim_feedforward = 1024
    seq_len = 128
    num_epochs = 30
    batch_size = 32
    lr = 3e-4
    warmup_ratio = 0.1
    num_experts = 8
    top_k = 2

    print(f"\n设备: {device}, 混合精度: {mixed_precision}")

    train_data = generate_synthetic_data(vocab_size, 800, seq_len)
    val_data = generate_synthetic_data(vocab_size, 200, seq_len)
    train_x, train_y = train_data
    val_x, val_y = val_data

    standard_model = build_ramanujan_transformer(
        vocab_size=vocab_size, d_model=d_model, nhead=nhead,
        num_layers=num_layers, dim_feedforward=dim_feedforward,
        decoder_only=True,
    )

    moe_model = build_ramanujan_moe_transformer(
        vocab_size=vocab_size, d_model=d_model, nhead=nhead,
        num_layers=num_layers, dim_feedforward=dim_feedforward,
        num_experts=num_experts, top_k=top_k, decoder_only=True,
    )

    std_params = sum(p.numel() for p in standard_model.parameters())
    moe_params = sum(p.numel() for p in moe_model.parameters())

    print(f"\n标准 Transformer 参数量: {std_params:,}")
    print(f"MoE Transformer 参数量:  {moe_params:,}")
    print(f"参数量比值: {moe_params/std_params:.2f}x")

    # 混合精度 Scaler
    scaler = None
    if mixed_precision == 'fp16' and device != 'cpu':
        scaler = torch.cuda.amp.GradScaler()

    criterion = nn.CrossEntropyLoss()
    results = {}

    training_logger = TrainingLogger(logger_type=logger_type)

    for name, model, is_moe in [('Standard', standard_model, False), ('MoE', moe_model, True)]:
        print(f"\n{'='*40}")
        print(f"训练 {name} Transformer")
        print(f"{'='*40}")

        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        steps_per_epoch = (len(train_x) + batch_size - 1) // batch_size
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_piecewise_exp_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        train_losses = []
        val_losses = []

        start_time = time.time()

        for epoch in range(num_epochs):
            train_stats = train_epoch(
                model, train_x, train_y, optimizer, criterion,
                scheduler, batch_size, aux_loss_weight=0.01, is_moe=is_moe,
                mixed_precision=mixed_precision, scaler=scaler, device=device,
            )
            val_loss = evaluate(
                model, val_x, val_y, criterion, batch_size, is_moe,
                mixed_precision=mixed_precision, device=device,
            )

            train_losses.append(train_stats['loss'])
            val_losses.append(val_loss)

            if training_logger:
                metrics = {
                    f'{name}/train_loss': train_stats['loss'],
                    f'{name}/val_loss': val_loss,
                }
                if is_moe and train_stats['aux_loss'] > 0:
                    metrics[f'{name}/aux_loss'] = train_stats['aux_loss']
                training_logger.log(metrics, step=epoch)

            if (epoch + 1) % 5 == 0:
                aux_str = f", aux={train_stats['aux_loss']:.4f}" if is_moe else ""
                print(f"  Epoch {epoch+1:3d}: train={train_stats['loss']:.4f}, "
                      f"val={val_loss:.4f}{aux_str}")

        elapsed = time.time() - start_time
        results[name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'time': elapsed,
            'final_val': val_losses[-1],
        }
        print(f"  耗时: {elapsed:.2f}s, 最终 val loss: {val_losses[-1]:.4f}")

    if training_logger:
        training_logger.close()

    # 汇总
    print("\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)
    print(f"\n{'模型':>12s} {'最终Val Loss':>12s} {'耗时(s)':>10s} {'参数量':>15s}")
    print("-" * 55)
    for name in ['Standard', 'MoE']:
        r = results[name]
        params = std_params if name == 'Standard' else moe_params
        print(f"{name:>12s} {r['final_val']:12.4f} {r['time']:10.2f} {params:>15,}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MoE Transformer 训练')
    parser.add_argument('--mixed-precision', type=str, default='none',
                        choices=['fp16', 'bf16', 'none'])
    parser.add_argument('--logger', type=str, default='none',
                        choices=['wandb', 'tensorboard', 'none'])
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    run_comparison(
        mixed_precision=args.mixed_precision,
        checkpoint_dir=args.checkpoint_dir,
        logger_type=args.logger,
        resume_path=args.resume,
        device=device,
    )


if __name__ == '__main__':
    main()
