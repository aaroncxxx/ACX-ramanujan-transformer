"""
MoE Transformer 训练脚本

对比：
1. 标准 Transformer（所有 token 共享同一个 FFN）
2. MoE Transformer（每个 token 路由到不同的专家）

验证拉马努金初始化在 MoE 架构中的有效性。
"""

import sys
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ramanujan_transformer import build_ramanujan_transformer
from src.moe import build_ramanujan_moe_transformer


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """线性 warmup + 余弦衰减"""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def generate_synthetic_data(vocab_size: int = 1000, num_samples: int = 1000,
                            seq_len: int = 128):
    """生成合成数据"""
    data = torch.randint(0, vocab_size, (num_samples, seq_len))
    return data[:, :-1], data[:, 1:]


def train_epoch(model: nn.Module, train_x: torch.Tensor, train_y: torch.Tensor,
                optimizer: optim.Optimizer, criterion: nn.Module,
                scheduler=None,
                batch_size: int = 32, aux_loss_weight: float = 0.01,
                is_moe: bool = False) -> Dict:
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_aux_loss = 0
    num_batches = 0

    for i in range(0, len(train_x), batch_size):
        batch_x = train_x[i:i+batch_size]
        batch_y = train_y[i:i+batch_size]

        optimizer.zero_grad()

        if is_moe:
            logits, aux_loss_dict = model(batch_x, return_aux_loss=True)
            ce_loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))

            # MoE 辅助损失
            aux_loss = 0
            if aux_loss_dict is not None:
                aux_loss = (aux_loss_dict['load_balance_loss'] +
                           aux_loss_dict['z_loss'] * 0.01)

            loss = ce_loss + aux_loss_weight * aux_loss
            total_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
        else:
            logits = model(batch_x)
            ce_loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
            loss = ce_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
def evaluate(model: nn.Module, val_x: torch.Tensor, val_y: torch.Tensor,
             criterion: nn.Module, batch_size: int = 32,
             is_moe: bool = False) -> float:
    """验证"""
    model.eval()
    total_loss = 0
    num_batches = 0

    for i in range(0, len(val_x), batch_size):
        batch_x = val_x[i:i+batch_size]
        batch_y = val_y[i:i+batch_size]

        if is_moe:
            logits, _ = model(batch_x, return_aux_loss=False)
        else:
            logits = model(batch_x)

        loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def get_expert_usage(model: nn.Module) -> Dict[int, float]:
    """获取每层的专家使用率"""
    usage = {}
    for idx, layer in enumerate(model.layers):
        if hasattr(layer, 'moe'):
            # 计算 router 的平均概率分布
            probs = layer.moe.router.gate.weight.data
            avg_probs = probs.mean(dim=0)  # (num_experts,)
            usage[idx] = avg_probs.tolist()
    return usage


def run_comparison():
    """对比标准 Transformer 和 MoE Transformer"""
    print("=" * 70)
    "拉马努金 MoE Transformer vs 标准 Transformer"
    print("=" * 70)

    # 超参数
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

    # 生成数据
    print("\n生成训练数据...")
    train_data = generate_synthetic_data(vocab_size, 800, seq_len)
    val_data = generate_synthetic_data(vocab_size, 200, seq_len)
    train_x, train_y = train_data
    val_x, val_y = val_data

    # 构建模型
    print("构建模型...")

    # 1. 标准 Transformer
    standard_model = build_ramanujan_transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        decoder_only=True,
    )

    # 2. MoE Transformer
    moe_model = build_ramanujan_moe_transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_experts=num_experts,
        top_k=top_k,
        decoder_only=True,
    )

    # 参数量对比
    std_params = sum(p.numel() for p in standard_model.parameters())
    moe_params = sum(p.numel() for p in moe_model.parameters())

    print(f"\n标准 Transformer 参数量: {std_params:,}")
    print(f"MoE Transformer 参数量:  {moe_params:,}")
    print(f"参数量比值: {moe_params/std_params:.2f}x")
    print(f"专家数: {num_experts}, Top-K: {top_k}")

    # 训练
    criterion = nn.CrossEntropyLoss()
    results = {}

    for name, model, is_moe in [
        ('Standard', standard_model, False),
        ('MoE', moe_model, True),
    ]:
        print(f"\n{'='*40}")
        print(f"训练 {name} Transformer")
        print(f"{'='*40}")

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        steps_per_epoch = (len(train_x) + batch_size - 1) // batch_size
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        train_losses = []
        val_losses = []

        start_time = time.time()

        for epoch in range(num_epochs):
            train_stats = train_epoch(
                model, train_x, train_y, optimizer, criterion,
                scheduler, batch_size, aux_loss_weight=0.01, is_moe=is_moe,
            )
            val_loss = evaluate(model, val_x, val_y, criterion, batch_size, is_moe)

            train_losses.append(train_stats['loss'])
            val_losses.append(val_loss)

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

    # MoE 专家使用分析
    print(f"\nMoE 专家使用率:")
    usage = get_expert_usage(moe_model)
    for layer_idx, probs in usage.items():
        probs_str = ", ".join([f"{p:.3f}" for p in probs])
        print(f"  Layer {layer_idx}: [{probs_str}]")

    return results


if __name__ == '__main__':
    run_comparison()
