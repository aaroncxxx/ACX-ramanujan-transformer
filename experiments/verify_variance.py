"""
方差保持性验证

模拟深度网络信号传播，验证拉马努金初始化的严格方差保持性质。
对比 Xavier、He、拉马努金三种初始化方法。
"""

import sys
import math
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ramanujan_initializer import get_ramanujan_scale, ramanujan_coefficients


def simulate_variance(depth: int, dim: int = 512, num_trials: int = 100,
                      method: str = 'ramanujan', nonlinearity: str = 'linear'):
    """
    模拟指定深度网络的方差传播

    Returns:
        list of float: 每层的输出方差
    """
    variances = []

    for _ in range(num_trials):
        x = torch.randn(256, dim)
        input_var = x.var().item()

        for i in range(depth):
            if method == 'ramanujan':
                s = get_ramanujan_scale(i, max(depth, 1000))
                std = math.sqrt(1.0 / dim) * s
            elif method == 'xavier':
                std = math.sqrt(1.0 / dim)
            elif method == 'he':
                std = math.sqrt(2.0 / dim)
            else:
                raise ValueError(f"Unknown method: {method}")

            W = torch.randn(dim, dim) * std

            if nonlinearity == 'relu':
                x = torch.relu(x @ W)
            else:
                x = x @ W

        variances.append(x.var().item())

    # 计算每层的平均方差
    return variances


def run_experiment(max_depth: int = 200, dim: int = 512):
    """运行完整方差实验"""
    print("=" * 60)
    print("方差保持性验证实验")
    print("=" * 60)

    methods = ['xavier', 'he', 'ramanujan']
    colors = {'xavier': 'blue', 'he': 'orange', 'ramanujan': 'red'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 实验1：线性网络
    print("\n[实验1] 线性网络 (无激活函数)")
    depths_to_test = [50, 100, 200]
    for depth in depths_to_test:
        print(f"\n  深度 = {depth}:")
        for method in methods:
            vars_list = simulate_variance(depth, dim, 50, method, 'linear')
            ratio = vars_list[-1] / vars_list[0] if vars_list[0] > 0 else float('inf')
            print(f"    {method:12s}: 最终方差比 = {ratio:.4f}")

    # 实验2：ReLU网络
    print("\n[实验2] ReLU网络")
    for depth in depths_to_test:
        print(f"\n  深度 = {depth}:")
        for method in methods:
            std_factor = 2.0 if method != 'ramanujan' else 1.0
            vars_list = simulate_variance(depth, dim, 50, method, 'relu')
            ratio = vars_list[-1] / vars_list[0] if vars_list[0] > 0 else float('inf')
            print(f"    {method:12s}: 最终方差比 = {ratio:.4f}")

    # 绘制方差传播曲线
    print("\n[绘图] 生成方差传播曲线...")

    for ax, nonlin, title in [(axes[0], 'linear', 'Linear Network'),
                               (axes[1], 'relu', 'ReLU Network')]:
        depth = 200
        for method in methods:
            # 多次运行取平均
            all_vars = []
            for _ in range(20):
                x = torch.randn(256, dim)
                layer_vars = [x.var().item()]

                for i in range(depth):
                    if method == 'ramanujan':
                        s = get_ramanujan_scale(i, depth)
                        std = math.sqrt(1.0 / dim) * s
                    elif method == 'xavier':
                        std = math.sqrt(1.0 / dim)
                    else:
                        std = math.sqrt(2.0 / dim)

                    W = torch.randn(dim, dim) * std

                    if nonlin == 'relu':
                        x = torch.relu(x @ W)
                    else:
                        x = x @ W

                    layer_vars.append(x.var().item())
                all_vars.append(layer_vars)

            avg_vars = [sum(v) / len(v) for v in zip(*all_vars)]
            ax.plot(range(depth + 1), avg_vars, label=method, color=colors[method])

        ax.axhline(y=avg_vars[0], color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Variance')
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(0, avg_vars[0] * 3)

    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent.parent / 'figures' / 'variance_preservation.png',
                dpi=150)
    print("  → 保存至 figures/variance_preservation.png")
    plt.close()


def show_ramanujan_coefficients(max_n: int = 50):
    """展示拉马努金递推系数"""
    print("\n" + "=" * 60)
    print("拉马努金递推系数 (前50项)")
    print("=" * 60)

    coeffs = ramanujan_coefficients(max_n)
    print(f"\n{'n':>4s} {'a_n':>20s} {'√a_n':>12s} {'a_n/a_0':>12s}")
    print("-" * 50)
    for i, c in enumerate(coeffs[:20]):
        print(f"{i:4d} {c:20.10f} {abs(c)**0.5:12.6f} {c/coeffs[0]:12.6f}")


if __name__ == '__main__':
    show_ramanujan_coefficients()
    run_experiment()
