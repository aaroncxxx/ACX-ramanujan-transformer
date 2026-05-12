"""
训练脚本

在 WikiText-2 数据集上训练语言模型，对比不同初始化方法。
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ramanujan_transformer import build_ramanujan_transformer


def generate_synthetic_data(vocab_size: int = 1000, num_samples: int = 1000,
                            seq_len: int = 128):
    """生成合成数据（实际使用时替换为真实数据集）"""
    data = torch.randint(0, vocab_size, (num_samples, seq_len))
    return data[:, :-1], data[:, 1:]  # input, target (shifted by 1)


def train(model: nn.Module, train_data, val_data, epochs: int = 20,
          batch_size: int = 32, lr: float = 3e-4, device: str = 'cpu'):
    """训练循环"""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_x, train_y = train_data
    val_x, val_y = val_data

    best_val_loss = float('inf')

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

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

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
