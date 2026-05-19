"""
训练脚本

在 WikiText-2 数据集上训练语言模型，对比不同初始化方法。
"""

import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ramanujan_transformer import build_ramanujan_transformer


def get_piecewise_exp_schedule_with_warmup(optimizer, warmup_steps: int,
                                            total_steps: int,
                                            alpha: float = 3.0,
                                            beta: float = 2.0,
                                            phase1_ratio: float = 0.3):
    """
    三段式分段指数调度：线性 warmup → 快速指数衰减 → 缓慢指数衰减

    Args:
        optimizer: 优化器
        warmup_steps: warmup 步数
        total_steps: 总训练步数
        alpha: phase1 衰减强度（结束时 lr ≈ lr_max * e^(-alpha)）
        beta: phase2 衰减强度
        phase1_ratio: phase1 占总步数的比例
    """
    phase1_steps = int((total_steps - warmup_steps) * phase1_ratio)
    phase2_steps = total_steps - warmup_steps - phase1_steps

    def lr_lambda(current_step: int) -> float:
        # Phase 0: 线性 warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        s = current_step - warmup_steps

        # Phase 1: 快速指数衰减
        if s < phase1_steps:
            return math.exp(-alpha * s / phase1_steps)

        # Phase 2: 缓慢指数衰减
        s2 = s - phase1_steps
        phase1_end_lr = math.exp(-alpha)  # 连续性：phase1 结束时的 lr
        return phase1_end_lr * math.exp(-beta * s2 / phase2_steps)

    return LambdaLR(optimizer, lr_lambda)


def generate_synthetic_data(vocab_size: int = 1000, num_samples: int = 1000,
                            seq_len: int = 128):
    """生成合成数据（实际使用时替换为真实数据集）"""
    data = torch.randint(0, vocab_size, (num_samples, seq_len))
    return data[:, :-1], data[:, 1:]  # input, target (shifted by 1)


def train(model: nn.Module, train_data, val_data, epochs: int = 20,
          batch_size: int = 32, lr: float = 3e-4, warmup_ratio: float = 0.1,
          device: str = 'cpu'):
    """训练循环"""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    train_x, train_y = train_data
    val_x, val_y = val_data

    total_steps = (len(train_x) // batch_size) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_piecewise_exp_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for i in range(0, len(train_x), batch_size):
            batch_x = train_x[i:i+batch_size].to(device)
            batch_y = train_y[i:i+batch_size].to(device)

            logits = model(batch_x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                logits = model(batch_x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: train_loss={avg_train_loss:.4f}, "
                  f"val_loss={avg_val_loss:.4f}, best={best_val_loss:.4f}")

    return best_val_loss


def main():
    print("=" * 50)
    print("拉马努金 Transformer 训练实验")
    print("=" * 50)

    vocab_size = 1000
    d_model = 256
    nhead = 4
    num_layers = 6
    seq_len = 128

    # 生成数据
    print("\n生成训练数据...")
    train_data = generate_synthetic_data(vocab_size, 800, seq_len)
    val_data = generate_synthetic_data(vocab_size, 200, seq_len)

    # 构建模型
    print("构建拉马努金 Transformer...")
    model = build_ramanujan_transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=d_model * 4,
        max_len=seq_len,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count:,}")

    # 训练
    print("\n开始训练...")
    best_loss = train(model, train_data, val_data, epochs=20)

    print(f"\n训练完成! Best val loss: {best_loss:.4f}")

    # 保存模型
    save_path = Path(__file__).resolve().parent.parent / 'experiments' / 'model.pt'
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至 {save_path}")


if __name__ == '__main__':
    main()
