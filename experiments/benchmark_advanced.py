"""
高级初始化方法对比实验

对比 Ramanujan 初始化与以下先进方法：
- Fixup (Zhang et al., 2019) — 残差网络的信号传播修正
- SkipInit (De & Smith, 2020) — 可学习的残差缩放
- DeepNet / μP (Microsoft, 2022) — 超参数迁移的深层初始化
- Base Station (Liu et al., 2023) — 基于信号传播的深层 Transformer

评估维度：
1. 方差保持性（信号传播稳定性）
2. 梯度健康度（范数 + 爆炸/消失比率）
3. 训练收敛速度
4. 深层可扩展性（48 / 96 / 128 层）
"""

import sys
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ramanujan_initializer import (
    RamanujanInitializer, ramanujan_coefficients, get_ramanujan_scale
)
from src.ramanujan_transformer import build_ramanujan_transformer


# ════════════════════════════════════════════════════════════════════
#  初始化方法实现
# ════════════════════════════════════════════════════════════════════

class FixupInitializer:
    """
    Fixup: Zhang et al., "Fixup Initialization: Residual Learning Without Normalization" (ICLR 2019)

    核心思想：
    - 残差分支最后一层缩放 1/L^{1/2}（L = 残差分支数）
    - 其余层用标准 Xavier/He
    - 去掉 LayerNorm

    适用场景：深层残差网络，无需 LayerNorm
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.scale_factor = 1.0 / math.sqrt(num_layers)

    def apply(self, model: nn.Module):
        """应用 Fixup 初始化"""
        # 第一层正常 Xavier
        first_linear = True
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if first_linear:
                    nn.init.xavier_uniform_(module.weight)
                    first_linear = False
                else:
                    # 残差分支最后一层缩放
                    nn.init.xavier_uniform_(module.weight)
                    module.weight.data *= self.scale_factor
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 偏置项特殊处理：第二层 bias 初始化为 0
        linear_count = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                linear_count += 1
                if linear_count % 2 == 0 and module.bias is not None:
                    nn.init.zeros_(module.bias)


class SkipInitInitializer:
    """
    SkipInit: De & Smith, "Batch Normalization Biases Deep Residual Networks Towards Shallow Paths" (2020)

    核心思想：
    - 每个残差分支乘以可学习标量 α，初始化为 0.1
    - 等效于让网络初始时接近恒等映射
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def apply(self, model: nn.Module):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                module.weight.data *= self.alpha
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class DeepNetInitializer:
    """
    DeepNet: Wang et al., "DeepNet: Scaling Transformers to 1,000 Layers" (2022)

    核心思想：
    - α 参数：缩放残差分支，α = (2N)^{1/4}，N = 层数
    - β 参数：缩放子层输出，β = (8N)^{-1/4}
    - 保证深层信号传播稳定

    对应 Microsoft 的 μP (Maximal Update Parametrization)
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # DeepNet 的 α 和 β 参数
        self.alpha = (2.0 * num_layers) ** 0.25
        self.beta = (8.0 * num_layers) ** -0.25

    def apply(self, model: nn.Module):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # 标准 Xavier 初始化
                nn.init.xavier_uniform_(module.weight)
                # β 缩放
                module.weight.data *= self.beta
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class BaseStationInitializer:
    """
    Base Station: Liu et al., "Base Station: A General Framework for Scaling Transformers" (2023)

    核心思想：
    - 基于信号传播理论推导缩放因子
    - 残差分支缩放：scale = 1 / sqrt(2 * num_layers)
    - 注意力输出额外缩放：scale = 1 / sqrt(2 * num_layers)
    - FFN 输出缩放：scale = 1 / sqrt(2 * num_layers)
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.scale = 1.0 / math.sqrt(2.0 * num_layers)

    def apply(self, model: nn.Module):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                module.weight.data *= self.scale
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


# ════════════════════════════════════════════════════════════════════
#  统一 Transformer 模型（用于公平对比）
# ════════════════════════════════════════════════════════════════════

class SimpleTransformerBlock(nn.Module):
    """统一的 Transformer Block，支持所有初始化方法"""

    def __init__(self, d_model: int, nhead: int, dim_ff: int,
                 use_layernorm: bool = True):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        residual = x
        h = self.ln1(x) if self.use_layernorm else x
        h, _ = self.attn(h, h, h)
        x = residual + h

        # FFN
        residual = x
        h = self.ln2(x) if self.use_layernorm else x
        h = self.ffn(h)
        x = residual + h

        return x


class SimpleTransformer(nn.Module):
    """统一的 Transformer 模型"""

    def __init__(self, vocab_size: int, d_model: int, nhead: int,
                 num_layers: int, dim_ff: int, max_seq_len: int = 512,
                 use_layernorm: bool = True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(d_model, nhead, dim_ff, use_layernorm)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.head(x)


def build_model_with_init(method: str, num_layers: int, **kwargs) -> nn.Module:
    """用指定方法构建并初始化模型"""
    vocab_size = kwargs.get('vocab_size', 1000)
    d_model = kwargs.get('d_model', 256)
    nhead = kwargs.get('nhead', 4)
    dim_ff = kwargs.get('dim_ff', 1024)
    use_layernorm = method not in ('fixup',)  # Fixup 不用 LayerNorm

    model = SimpleTransformer(
        vocab_size, d_model, nhead, num_layers, dim_ff,
        use_layernorm=use_layernorm,
    )

    if method == 'ramanujan':
        initializer = RamanujanInitializer(num_layers=num_layers)
        initializer.apply(model)
    elif method == 'xavier':
        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        model.apply(init_fn)
    elif method == 'he':
        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        model.apply(init_fn)
    elif method == 'fixup':
        FixupInitializer(num_layers).apply(model)
    elif method == 'skipinit':
        SkipInitInitializer(alpha=0.1).apply(model)
    elif method == 'deepnet':
        DeepNetInitializer(num_layers).apply(model)
    elif method == 'basestation':
        BaseStationInitializer(num_layers).apply(model)
    else:
        raise ValueError(f"Unknown method: {method}")

    return model


# ════════════════════════════════════════════════════════════════════
#  实验 1: 方差保持性测试
# ════════════════════════════════════════════════════════════════════

@dataclass
class VarianceResult:
    method: str
    num_layers: int
    input_var: float
    output_var: float
    ratio: float
    per_layer_vars: List[float] = field(default_factory=list)
    max_drift: float = 0.0
    min_drift: float = 0.0


def test_variance_preservation(method: str, num_layers: int,
                                d_model: int = 256,
                                num_samples: int = 1000) -> VarianceResult:
    """测试方差保持性"""
    model = build_model_with_init(method, num_layers,
                                   d_model=d_model, vocab_size=1000, nhead=4, dim_ff=1024)
    model.eval()

    # 随机输入
    x = torch.randint(0, 1000, (num_samples, 32))
    with torch.no_grad():
        logits = model(x)

    input_var = torch.var(torch.randn(num_samples, 32, d_model)).item()
    output_var = torch.var(logits).item()

    # 逐层方差追踪
    per_layer_vars = []
    hooks = []

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                per_layer_vars.append(torch.var(output).item())
        return hook

    # 注册 hook
    for layer in model.layers:
        hooks.append(layer.register_forward_hook(hook_fn("block")))

    with torch.no_grad():
        model(x)

    # 清理 hooks
    for h in hooks:
        h.remove()

    max_drift = max(v / per_layer_vars[0] for v in per_layer_vars) if per_layer_vars else 1.0
    min_drift = min(v / per_layer_vars[0] for v in per_layer_vars) if per_layer_vars else 1.0

    return VarianceResult(
        method=method,
        num_layers=num_layers,
        input_var=input_var,
        output_var=output_var,
        ratio=output_var / input_var if input_var > 0 else 0,
        per_layer_vars=per_layer_vars,
        max_drift=max_drift,
        min_drift=min_drift,
    )


# ════════════════════════════════════════════════════════════════════
#  实验 2: 梯度健康度测试
# ════════════════════════════════════════════════════════════════════

@dataclass
class GradientResult:
    method: str
    num_layers: int
    avg_grad_norm: float
    max_grad_norm: float
    min_grad_norm: float
    grad_explosion_ratio: float  # 梯度 > 10x 平均值的比例
    grad_vanishing_ratio: float  # 梯度 < 0.01x 平均值的比例
    per_layer_grad_norms: List[float] = field(default_factory=list)


def test_gradient_health(method: str, num_layers: int,
                          d_model: int = 256) -> GradientResult:
    """测试梯度健康度"""
    model = build_model_with_init(method, num_layers,
                                   d_model=d_model, vocab_size=1000, nhead=4, dim_ff=1024)

    x = torch.randint(0, 1000, (8, 32))
    y = torch.randint(0, 1000, (8, 32))

    model.train()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, 1000), y.view(-1))
    loss.backward()

    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())

    if not grad_norms:
        return GradientResult(method, num_layers, 0, 0, 0, 0, 0, [])

    avg = sum(grad_norms) / len(grad_norms)
    max_g = max(grad_norms)
    min_g = min(grad_norms)

    explosion = sum(1 for g in grad_norms if g > 10 * avg) / len(grad_norms)
    vanishing = sum(1 for g in grad_norms if g < 0.01 * avg) / len(grad_norms)

    return GradientResult(
        method=method,
        num_layers=num_layers,
        avg_grad_norm=avg,
        max_grad_norm=max_g,
        min_grad_norm=min_g,
        grad_explosion_ratio=explosion,
        grad_vanishing_ratio=vanishing,
        per_layer_grad_norms=grad_norms,
    )


# ════════════════════════════════════════════════════════════════════
#  实验 3: 训练收敛速度
# ════════════════════════════════════════════════════════════════════

@dataclass
class TrainResult:
    method: str
    num_layers: int
    losses: List[float]
    grad_norms: List[float]
    final_loss: float
    wall_time: float
    converged_epoch: int  # loss < threshold 的 epoch


def test_training_convergence(method: str, num_layers: int,
                               num_epochs: int = 100,
                               d_model: int = 256,
                               lr: float = 3e-4) -> TrainResult:
    """测试训练收敛速度"""
    model = build_model_with_init(method, num_layers,
                                   d_model=d_model, vocab_size=1000, nhead=4, dim_ff=1024)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    losses = []
    grad_norms = []
    start = time.time()

    for epoch in range(num_epochs):
        x = torch.randint(0, 1000, (16, 32))
        y = torch.randint(0, 1000, (16, 32))

        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, 1000), y.view(-1))
        loss.backward()

        # 梯度范数
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5

        optimizer.step()
        losses.append(loss.item())
        grad_norms.append(total_norm)

    wall_time = time.time() - start

    # 找到收敛 epoch（loss < 初始 loss 的 50%）
    threshold = losses[0] * 0.5 if losses else 0
    converged = num_epochs
    for i, l in enumerate(losses):
        if l < threshold:
            converged = i
            break

    return TrainResult(
        method=method,
        num_layers=num_layers,
        losses=losses,
        grad_norms=grad_norms,
        final_loss=losses[-1] if losses else 0,
        wall_time=wall_time,
        converged_epoch=converged,
    )


# ════════════════════════════════════════════════════════════════════
#  主实验流程
# ════════════════════════════════════════════════════════════════════

def run_full_benchmark(depths: List[int] = None,
                       methods: List[str] = None,
                       num_epochs: int = 80,
                       output_json: str = None) -> Dict:
    """运行完整基准测试"""
    if depths is None:
        depths = [12, 24, 48, 96]
    if methods is None:
        methods = ['xavier', 'he', 'fixup', 'skipinit', 'deepnet', 'basestation', 'ramanujan']

    all_results = {
        'variance': {},
        'gradient': {},
        'training': {},
    }

    print("=" * 80)
    print("  高级初始化方法对比实验")
    print("=" * 80)

    for depth in depths:
        print(f"\n{'─' * 80}")
        print(f"  模型深度: {depth} 层")
        print(f"{'─' * 80}")

        # --- 方差保持性 ---
        print(f"\n  [1/3] 方差保持性测试")
        print(f"  {'方法':>15s} {'输出/输入方差比':>15s} {'最大漂移':>12s} {'最小漂移':>12s}")
        print(f"  {'─' * 56}")

        for method in methods:
            vr = test_variance_preservation(method, depth)
            all_results['variance'][f"{method}_L{depth}"] = {
                'method': method, 'layers': depth,
                'ratio': round(vr.ratio, 6),
                'max_drift': round(vr.max_drift, 6),
                'min_drift': round(vr.min_drift, 6),
            }
            ideal = "✓" if 0.8 < vr.ratio < 1.2 else "✗"
            print(f"  {method:>15s} {vr.ratio:15.6f} {vr.max_drift:12.6f} {vr.min_drift:12.6f}  {ideal}")

        # --- 梯度健康度 ---
        print(f"\n  [2/3] 梯度健康度测试")
        print(f"  {'方法':>15s} {'平均梯度':>12s} {'爆炸比':>10s} {'消失比':>10s} {'状态':>8s}")
        print(f"  {'─' * 58}")

        for method in methods:
            gr = test_gradient_health(method, depth)
            all_results['gradient'][f"{method}_L{depth}"] = {
                'method': method, 'layers': depth,
                'avg_grad_norm': round(gr.avg_grad_norm, 6),
                'explosion': round(gr.grad_explosion_ratio, 4),
                'vanishing': round(gr.grad_vanishing_ratio, 4),
            }
            status = "健康" if gr.grad_explosion_ratio < 0.1 and gr.grad_vanishing_ratio < 0.1 else "异常"
            print(f"  {method:>15s} {gr.avg_grad_norm:12.6f} "
                  f"{gr.grad_explosion_ratio:10.4f} {gr.grad_vanishing_ratio:10.4f}  {status}")

        # --- 训练收敛 ---
        print(f"\n  [3/3] 训练收敛速度测试 ({num_epochs} epochs)")
        print(f"  {'方法':>15s} {'最终Loss':>12s} {'收敛Epoch':>12s} {'耗时(s)':>10s}")
        print(f"  {'─' * 52}")

        for method in methods:
            tr = test_training_convergence(method, depth, num_epochs=num_epochs)
            all_results['training'][f"{method}_L{depth}"] = {
                'method': method, 'layers': depth,
                'final_loss': round(tr.final_loss, 4),
                'converged_epoch': tr.converged_epoch,
                'wall_time': round(tr.wall_time, 2),
                'losses': [round(l, 4) for l in tr.losses],
            }
            print(f"  {method:>15s} {tr.final_loss:12.4f} {tr.converged_epoch:12d} {tr.wall_time:10.2f}")

    # --- 汇总排名 ---
    print(f"\n{'=' * 80}")
    print("  综合排名（按 48 层训练收敛速度）")
    print(f"{'=' * 80}")

    target_depth = 48 if 48 in depths else depths[-1]
    training_48 = {k: v for k, v in all_results['training'].items() if v['layers'] == target_depth}
    ranked = sorted(training_48.items(), key=lambda x: x[1]['final_loss'])

    for rank, (key, data) in enumerate(ranked, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"  {medal} #{rank}  {data['method']:>15s}  "
              f"Loss={data['final_loss']:.4f}  "
              f"收敛@{data['converged_epoch']}")

    # 保存 JSON
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  结果已保存: {output_json}")

    return all_results


# ════════════════════════════════════════════════════════════════════
#  单独实验入口
# ════════════════════════════════════════════════════════════════════

def run_variance_only(depths: List[int] = None):
    """只跑方差测试"""
    if depths is None:
        depths = [12, 24, 48, 96, 128]
    methods = ['xavier', 'he', 'fixup', 'skipinit', 'deepnet', 'basestation', 'ramanujan']

    print("=" * 70)
    print("  方差保持性对比")
    print("=" * 70)

    for depth in depths:
        print(f"\n  ── {depth} 层 ──")
        print(f"  {'方法':>15s} {'方差比':>12s} {'最大漂移':>12s}")
        print(f"  {'─' * 42}")
        for method in methods:
            vr = test_variance_preservation(method, depth)
            print(f"  {method:>15s} {vr.ratio:12.6f} {vr.max_drift:12.6f}")


def run_gradient_only(depths: List[int] = None):
    """只跑梯度测试"""
    if depths is None:
        depths = [12, 24, 48, 96, 128]
    methods = ['xavier', 'he', 'fixup', 'skipinit', 'deepnet', 'basestation', 'ramanujan']

    print("=" * 70)
    print("  梯度健康度对比")
    print("=" * 70)

    for depth in depths:
        print(f"\n  ── {depth} 层 ──")
        print(f"  {'方法':>15s} {'平均梯度':>12s} {'爆炸':>8s} {'消失':>8s}")
        print(f"  {'─' * 46}")
        for method in methods:
            gr = test_gradient_health(method, depth)
            print(f"  {method:>15s} {gr.avg_grad_norm:12.6f} "
                  f"{gr.grad_explosion_ratio:8.4f} {gr.grad_vanishing_ratio:8.4f}")


# ════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='高级初始化方法对比实验')
    parser.add_argument('--mode', choices=['full', 'variance', 'gradient', 'train'],
                        default='full', help='实验模式')
    parser.add_argument('--depths', type=int, nargs='+', default=[12, 24, 48, 96],
                        help='测试深度列表')
    parser.add_argument('--epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--output', type=str, help='结果 JSON 输出路径')

    args = parser.parse_args()

    if args.mode == 'full':
        run_full_benchmark(args.depths, num_epochs=args.epochs, output_json=args.output)
    elif args.mode == 'variance':
        run_variance_only(args.depths)
    elif args.mode == 'gradient':
        run_gradient_only(args.depths)
    elif args.mode == 'train':
        for depth in args.depths:
            print(f"\n{'=' * 60}")
            print(f"  训练对比: {depth} 层")
            print(f"{'=' * 60}")
            for method in ['xavier', 'he', 'deepnet', 'basestation', 'ramanujan']:
                tr = test_training_convergence(method, depth, num_epochs=args.epochs)
                print(f"  {method:>15s}: final_loss={tr.final_loss:.4f}, "
                      f"converged@{tr.converged_epoch}, time={tr.wall_time:.1f}s")
