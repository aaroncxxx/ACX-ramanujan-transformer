"""
基准测试

对比不同初始化方法在实际训练中的表现：
- 收敛速度
- 梯度范数
- 最终性能
"""

import sys
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ramanujan_transformer import build_ramanujan_transformer
from src.ramanujan_initializer import RamanujanInitializer


def build_baseline_transformer(vocab_size: int = 1000, d_model: int = 256,
                                nhead: int = 4, num_layers: int = 8,
                                dim_feedforward: int = 1024,
                                init_method: str = 'xavier') -> nn.Module:
    """构建标准 Transformer（非拉马努金初始化）用于对比"""
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation='gelu',
        ),
        num_layers=num_layers,
    )

    # 应用指定初始化
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if init_method == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_method == 'he':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_method == 'normal':
                nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    model.apply(init_weights)
    return model


def train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
               optimizer: optim.Optimizer, criterion: nn.Module) -> Dict:
    """单步训练"""
    model.train()
    optimizer.zero_grad()

    output = model(x)
    loss = criterion(output, y)

    # 记录梯度范数
    grad_norms = []
    loss.backward()

    for p in model.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())

    optimizer.step()

    avg_grad = sum(grad_norms) / max(len(grad_norms), 1)
    return {
        'loss': loss.item(),
        'avg_grad_norm': avg_grad,
        'max_grad_norm': max(grad_norms) if grad_norms else 0,
    }


def run_benchmark(num_epochs: int = 50, batch_size: int = 32,
                  seq_len: int = 64, d_model: int = 256,
                  num_layers: int = 8, num_runs: int = 3):
    """运行基准测试"""
    print("=" * 70)
    print("基准测试：Xavier vs He vs 拉马努金 初始化")
    print("=" * 70)

    vocab_size = 1000
    nhead = 4

    results = {
        'xavier': {'losses': [], 'grad_norms': [], 'times': []},
        'he': {'losses': [], 'grad_norms': [], 'times': []},
        'ramanujan': {'losses': [], 'grad_norms': [], 'times': []},
    }

    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")

        for method in ['xavier', 'he', 'ramanujan']:
            print(f"\n  方法: {method}")

            if method == 'ramanujan':
                model = build_ramanujan_transformer(
                    vocab_size=vocab_size, d_model=d_model,
                    nhead=nhead, num_layers=num_layers,
                    dim_feedforward=d_model * 4,
                    decoder_only=False,
                )
            else:
                model = build_baseline_transformer(
                    vocab_size, d_model, nhead, num_layers,
                    d_model * 4, method
                )

            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            criterion = nn.CrossEntropyLoss()

            epoch_losses = []
            epoch_grads = []

            start_time = time.time()

            for epoch in range(num_epochs):
                # 生成随机序列
                x = torch.randint(0, vocab_size, (batch_size, seq_len))
                y = torch.randint(0, vocab_size, (batch_size, seq_len))

                stats = train_step(model, x, y, optimizer, criterion)
                epoch_losses.append(stats['loss'])
                epoch_grads.append(stats['avg_grad_norm'])

                if (epoch + 1) % 10 == 0:
                    print(f"    Epoch {epoch+1:3d}: loss={stats['loss']:.4f}, "
                          f"grad_norm={stats['avg_grad_norm']:.4f}")

            elapsed = time.time() - start_time

            results[method]['losses'].append(epoch_losses)
            results[method]['grad_norms'].append(epoch_grads)
            results[method]['times'].append(elapsed)

            print(f"    耗时: {elapsed:.2f}s, 最终loss: {epoch_losses[-1]:.4f}")

    # 汇总
    print("\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)
    print(f"\n{'方法':>12s} {'最终Loss':>10s} {'梯度范数':>10s} {'耗时(s)':>10s}")
    print("-" * 45)
    for method in ['xavier', 'he', 'ramanujan']:
        avg_loss = sum(r[-1] for r in results[method]['losses']) / num_runs
        avg_grad = sum(r[-1] for r in results[method]['grad_norms']) / num_runs
        avg_time = sum(results[method]['times']) / num_runs
        print(f"{method:>12s} {avg_loss:10.4f} {avg_grad:10.4f} {avg_time:10.2f}")

    return results


if __name__ == '__main__':
    run_benchmark()
