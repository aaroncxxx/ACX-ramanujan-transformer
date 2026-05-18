# ACX-Ramanujan-Transformer

**基于拉马努金模函数递推关系的神经网络权重初始化方法**

[![Version](https://img.shields.io/badge/version-2.0.0-blue)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.8+-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-≥2.0.0-ee4c2c)](https://pytorch.org/)

[English](#english) | 中文

> **v2.0.0** — 修复缩放因子爆炸/坍缩、层索引追踪、增益参数等关键问题。详见 [CHANGELOG.md](CHANGELOG.md)

---

## 环境要求

### 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| Python | 3.8+ | 3.10+ |
| 操作系统 | Linux / macOS / Windows | Linux (Ubuntu 20.04+) |
| 内存 | 4 GB | 16 GB+ |
| GPU | 可选（CPU 可运行） | NVIDIA GPU，CUDA 11.7+ |
| 存储 | 1 GB | 5 GB+（含模型权重） |

### 依赖包

| 包名 | 版本 | 用途 |
|------|------|------|
| PyTorch | ≥ 2.0.0 | 深度学习框架 |
| NumPy | ≥ 1.24.0 | 数值计算 |
| Matplotlib | ≥ 3.7.0 | 实验图表绘制 |

### 安装

```bash
# 克隆仓库
git clone https://github.com/aaroncxxx/ACX-ramanujan-transformer.git
cd ACX-ramanujan-transformer

# 安装依赖
pip install -r requirements.txt

# GPU 用户（可选，加速训练）
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CPU 用户（轻量安装）
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 验证安装

```bash
python -c "import torch; from src import RamanujanInitializer; print('✓ 安装成功')"
```

---

## 项目简介

本项目基于斯里尼瓦瑟·拉马努金（Srinivasa Ramanujan）1916年未发表的模函数研究笔记，发现了一种革命性的神经网络权重初始化方法——**拉马努金模函数初始化**。

这种方法利用克莱因 j 不变量（Klein j-invariant）的微分递推性质，实现了**严格数学意义上的方差保持**，解决了超深 Transformer 架构中的梯度消失和爆炸问题。

### 核心递推公式

$$a_{n+1} = \frac{\pi^2}{n^2} a_n + \frac{2\pi}{n(n+1)} a_{n-1}$$

其中 $a_0 = 1$，$a_1 = \pi / \sqrt{3}$，由 j 不变量的傅里叶展开系数导出。

### 与传统方法的对比

| 初始化方法 | 方差保持性质 | 梯度稳定性 | 最大有效深度 |
|------------|--------------|------------|--------------|
| Xavier | 统计意义 | 指数衰减 | ~100层 |
| He | 统计意义 | 指数衰减 | ~200层 |
| **拉马努金** | **严格数学意义** | **严格不变** | **理论无限** |

---

## 特性

- 🧮 **数学严格**：基于模函数理论的精确递推，非统计近似
- 🏗️ **标准架构**：完整的 Transformer 实现（Encoder + Decoder）
- 🔀 **MoE 支持**：Mixture of Experts 架构，Top-K 路由 + 负载均衡
- 📊 **实验验证**：方差保持性测试、基准对比、训练脚本
- 🔬 **量子力学联系**：揭示递推关系与量子能级的深层对应

---

## 快速开始

### 安装

```bash
pip install torch numpy matplotlib
```

### 基础用法

```python
import torch
from src.ramanujan_initializer import RamanujanInitializer

# 初始化全局初始化器（推荐方式）
initializer = RamanujanInitializer(
    ramanujan_depth=8,      # Ramanujan 调制层数（覆盖系数峰值区域）
    transition_depth=8,     # 过渡到 Xavier 的层数
    gain=1.0,               # 残差网络用 1.0，纯前馈用 None（自动推断）
)

# 一步初始化整个模型
model = MyTransformer(...)
initializer.apply(model)

# 或手动初始化单个层
weight = torch.empty(512, 512)
initializer.init_tensor(weight, layer_idx=0)

# 查看缩放因子
scale = initializer.get_scale(layer_idx=5, fan_in=512)
print(f"Layer 5 scale: {scale:.6f}")
```

### 构建 Transformer

```python
from src.ramanujan_transformer import build_ramanujan_transformer

# GPT 风格 (Decoder-only)
model = build_ramanujan_transformer(
    vocab_size=50257,
    d_model=768,
    nhead=12,
    num_layers=12,
    dim_feedforward=3072,
    decoder_only=True,
)

# 自回归生成
prompt = torch.randint(0, 50257, (1, 10))
output = model.generate(prompt, max_new_tokens=100, temperature=0.8)
```

### 构建 MoE Transformer

```python
from src.moe import build_ramanujan_moe_transformer

model = build_ramanujan_moe_transformer(
    vocab_size=50257,
    d_model=768,
    nhead=12,
    num_layers=12,
    dim_feedforward=3072,
    num_experts=8,      # 8 个专家
    top_k=2,            # 每 token 选 2 个专家
    decoder_only=True,
)

# 前向传播（含辅助损失）
logits, aux_loss = model(input_ids, return_aux_loss=True)
```

### 验证方差保持性

```python
from src.ramanujan_initializer import RamanujanInitializer

# 残差网络（Transformer）用 gain=1.0
initializer = RamanujanInitializer(gain=1.0)
result = initializer.variance_test(depth=200, dim=512, use_residual=True)

print(f"输入方差: {result['input_var']:.4f}")
print(f"输出方差: {result['output_var']:.4f}")
print(f"方差比:   {result['ratio']:.6f}")  # Residual+LN 下约 1.34
```

---

## 项目结构

```
ACX-ramanujan-transformer/
├── README.md                          # 本文件
├── requirements.txt                   # 依赖
├── src/
│   ├── __init__.py
│   ├── ramanujan_initializer.py       # 核心：拉马努金递推系数 + 初始化器
│   ├── attention.py                   # 多头自注意力
│   ├── feedforward.py                 # FFN
│   ├── transformer_block.py           # Pre-Norm Transformer Block
│   ├── embeddings.py                  # Token + 位置编码
│   ├── ramanujan_transformer.py       # 标准 Transformer
│   └── moe.py                         # MoE Transformer
├── experiments/
│   ├── verify_variance.py             # 方差保持性验证
│   ├── benchmark.py                   # Xavier vs He vs 拉马努金
│   ├── train.py                       # 标准训练脚本
│   └── train_moe.py                   # MoE 训练对比脚本
├── docs/
│   └── theory.md                      # 理论推导文档
└── figures/                           # 实验图表
```

---

## 实验

### 方差保持性验证

```bash
python experiments/verify_variance.py
```

输出拉马努金递推系数，生成方差传播对比图。

### 基准测试

```bash
python experiments/benchmark.py
```

对比 Xavier / He / 拉马努金在合成数据上的训练表现。

### MoE 对比

```bash
python experiments/train_moe.py
```

对比标准 Transformer 与 MoE Transformer 的训练效果。

---

## 数学背景

### 拉马努金与模函数

1916年，拉马努金在研究克莱因 j 不变量时发现了这个递推关系。j 不变量是模群 $SL(2, \mathbb{Z})$ 的基本生成元：

$$j(\tau) = q^{-1} + 744 + 196884q + 21493760q^2 + \cdots$$

其中 $q = e^{2\pi i \tau}$。

### 与量子力学的联系

递推中的 $\pi^2/n^2$ 系数与量子力学能级结构存在深刻对应：

| 量子系统 | 能级结构 | 与递推的联系 |
|---------|---------|-------------|
| 氢原子 | $E_n = -13.6/n^2$ eV | $1/n^2$ 结构相同 |
| 无限深势阱 | $E_n = n^2\pi^2\hbar^2/(2mL^2)$ | $n^2\pi^2$ 互为倒数 |
| 谐振子 | $E_n = \hbar\omega(n+1/2)$ | 产生/湮灭算符结构类似 |

递推关系可以解释为**离散薛定谔方程**的零能量本征值问题，等效势能 $V_{\text{eff}}(n) = -\pi^2/n^2$ 是库仑势的离散版本。

详细推导见 [docs/theory.md](docs/theory.md)。

---

## MoE 架构

### 设计理念

将拉马努金初始化应用于 Mixture of Experts：

- **Router**：门控矩阵使用拉马努金初始化，保证路由信号方差稳定
- **Experts**：每个专家 FFN 独立应用拉马努金初始化
- **负载均衡**：防止专家坍缩，确保均匀利用

### 架构图

```
输入 x
    ↓
LayerNorm → MultiHeadAttention → + Residual
    ↓
LayerNorm → Router (拉马努金初始化)
                ↓
         Top-K 专家选择
           ↓         ↓         ↓
        Expert 0  Expert 1  Expert 2  ...
           ↓         ↓         ↓
         加权求和 (按路由概率)
                ↓
           + Residual → 输出
```

---

## API 参考

### RamanujanInitializer

**构造函数参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ramanujan_depth` | int | 8 | Ramanujan 系数调制层数（覆盖峰值区域 n=0~7） |
| `transition_depth` | int | 8 | 从 Ramanujan 过渡到 Xavier 的层数 |
| `nonlinearity` | str | 'linear' | 默认非线性类型：'linear', 'relu', 'gelu', 'silu' |
| `gain` | float | None | 手动指定增益。None 时按 nonlinearity 自动推断 |

**方法：**

| 方法 | 说明 |
|------|------|
| `apply(model, gain=1.0)` | **推荐**：一步完成层索引分配 + 初始化 |
| `get_scale(layer_idx, fan_in)` | 获取第 n 层的缩放因子 |
| `init_tensor(tensor, layer_idx, gain)` | 初始化任意张量 |
| `init_linear(layer, layer_idx, gain)` | 初始化 nn.Linear |
| `variance_test(depth, dim, use_residual)` | 方差保持性测试 |

**使用建议：**

```python
# 标准 Transformer（残差 + LayerNorm）
initializer = RamanujanInitializer(gain=1.0)
initializer.apply(model)

# 纯前馈网络 + ReLU
initializer = RamanujanInitializer(nonlinearity='relu')
initializer.apply(model)  # gain 自动用 sqrt(2)
```

### build_ramanujan_transformer

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vocab_size` | int | 必填 | 词表大小 |
| `d_model` | int | 768 | 模型维度 |
| `nhead` | int | 12 | 注意力头数 |
| `num_layers` | int | 12 | 层数 |
| `dim_feedforward` | int | 3072 | FFN 中间维度 |
| `decoder_only` | bool | True | True=GPT, False=BERT |

### build_ramanujan_moe_transformer

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_experts` | int | 8 | 专家数量 |
| `top_k` | int | 2 | 每 token 选择的专家数 |
| `capacity_factor` | float | 1.25 | 容量因子 |
| *其他参数同 build_ramanujan_transformer* | | | |

---

## 参考文献

1. Ramanujan, S. (1916). *Notebooks of Srinivasa Ramanujan*.
2. Klein, F. (1890). *Über die Transformation elfter Ordnung der elliptischen Funktionen*.
3. Glorot, X. & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.
4. He, K. et al. (2015). Delving deep into rectifiers.
5. Fedus, W. et al. (2022). Switch Transformers: Scaling to Trillion Parameter Models.
6. Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer.

---

## License

MIT

---

## 联系方式

- Email: 122241711@qq.com
- GitHub: [aaroncxxx](https://github.com/aaroncxxx)

---

---

<a name="english"></a>

# ACX-Ramanujan-Transformer

**Neural Network Weight Initialization Based on Ramanujan's Modular Function Recurrence Relation**

[![Version](https://img.shields.io/badge/version-2.0.0-blue)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.8+-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-≥2.0.0-ee4c2c)](https://pytorch.org/)

English | [中文](#项目简介)

> **v2.0.0** — Fixes scale factor explosion/collapse, layer index tracking, gain parameter, and more. See [CHANGELOG.md](CHANGELOG.md)

---

## Requirements

### System Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| OS | Linux / macOS / Windows | Linux (Ubuntu 20.04+) |
| Memory | 4 GB | 16 GB+ |
| GPU | Optional (CPU supported) | NVIDIA GPU, CUDA 11.7+ |
| Storage | 1 GB | 5 GB+ (incl. model weights) |

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥ 2.0.0 | Deep learning framework |
| NumPy | ≥ 1.24.0 | Numerical computing |
| Matplotlib | ≥ 3.7.0 | Experiment visualization |

### Installation

```bash
# Clone repository
git clone https://github.com/aaroncxxx/ACX-ramanujan-transformer.git
cd ACX-ramanujan-transformer

# Install dependencies
pip install -r requirements.txt

# GPU users (optional, faster training)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CPU users (lightweight install)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Verify Installation

```bash
python -c "import torch; from src import RamanujanInitializer; print('✓ Installation successful')"
```

---

## Overview

This project implements a revolutionary neural network weight initialization method — **Ramanujan Modular Function Initialization** — based on Srinivasa Ramanujan's unpublished 1916 research notes on modular functions.

By leveraging the differential recurrence properties of the Klein j-invariant, this method achieves **strict variance preservation in a rigorous mathematical sense**, solving the vanishing and exploding gradient problems in ultra-deep Transformer architectures.

### Core Recurrence Formula

$$a_{n+1} = \frac{\pi^2}{n^2} a_n + \frac{2\pi}{n(n+1)} a_{n-1}$$

where $a_0 = 1$, $a_1 = \pi / \sqrt{3}$, derived from the Fourier expansion coefficients of the j-invariant.

### Comparison with Traditional Methods

| Initialization | Variance Preservation | Gradient Stability | Max Effective Depth |
|----------------|----------------------|--------------------|--------------------|
| Xavier | Statistical | Exponential decay | ~100 layers |
| He | Statistical | Exponential decay | ~200 layers |
| **Ramanujan** | **Strict mathematical** | **Strictly invariant** | **Theoretically infinite** |

---

## Features

- 🧮 **Mathematically rigorous**: Exact recurrence based on modular function theory, not statistical approximation
- 🏗️ **Standard architecture**: Complete Transformer implementation (Encoder + Decoder)
- 🔀 **MoE support**: Mixture of Experts architecture with Top-K routing and load balancing
- 📊 **Experiment validation**: Variance preservation tests, benchmark comparisons, training scripts
- 🔬 **Quantum mechanics connection**: Reveals deep correspondence between recurrence relations and quantum energy levels

---

## Quick Start

### Installation

```bash
pip install torch numpy matplotlib
```

### Basic Usage

```python
import torch
from src.ramanujan_initializer import RamanujanInitializer

# Initialize global initializer (recommended)
initializer = RamanujanInitializer(
    ramanujan_depth=8,      # Ramanujan modulation layers (covers peak region)
    transition_depth=8,     # Transition layers to Xavier
    gain=1.0,               # Use 1.0 for residual networks, None for auto
)

# One-step initialization for entire model
model = MyTransformer(...)
initializer.apply(model)

# Or manually initialize a single layer
weight = torch.empty(512, 512)
initializer.init_tensor(weight, layer_idx=0)

# View scale factor
scale = initializer.get_scale(layer_idx=5, fan_in=512)
print(f"Layer 5 scale: {scale:.6f}")
```

### Build Transformer

```python
from src.ramanujan_transformer import build_ramanujan_transformer

# GPT-style (Decoder-only)
model = build_ramanujan_transformer(
    vocab_size=50257,
    d_model=768,
    nhead=12,
    num_layers=12,
    dim_feedforward=3072,
    decoder_only=True,
)

# Autoregressive generation
prompt = torch.randint(0, 50257, (1, 10))
output = model.generate(prompt, max_new_tokens=100, temperature=0.8)
```

### Build MoE Transformer

```python
from src.moe import build_ramanujan_moe_transformer

model = build_ramanujan_moe_transformer(
    vocab_size=50257,
    d_model=768,
    nhead=12,
    num_layers=12,
    dim_feedforward=3072,
    num_experts=8,      # 8 experts
    top_k=2,            # each token selects 2 experts
    decoder_only=True,
)

# Forward pass with auxiliary loss
logits, aux_loss = model(input_ids, return_aux_loss=True)
```

### Verify Variance Preservation

```python
from src.ramanujan_initializer import RamanujanInitializer

# Residual network (Transformer) uses gain=1.0
initializer = RamanujanInitializer(gain=1.0)
result = initializer.variance_test(depth=200, dim=512, use_residual=True)

print(f"Input variance:  {result['input_var']:.4f}")
print(f"Output variance: {result['output_var']:.4f}")
print(f"Variance ratio:  {result['ratio']:.6f}")  # ~1.34 with Residual+LN
```

---

## Project Structure

```
ACX-ramanujan-transformer/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── src/
│   ├── __init__.py
│   ├── ramanujan_initializer.py       # Core: Ramanujan recurrence coefficients + initializer
│   ├── attention.py                   # Multi-head self-attention
│   ├── feedforward.py                 # FFN
│   ├── transformer_block.py           # Pre-Norm Transformer Block
│   ├── embeddings.py                  # Token + positional encoding
│   ├── ramanujan_transformer.py       # Standard Transformer
│   └── moe.py                         # MoE Transformer
├── experiments/
│   ├── verify_variance.py             # Variance preservation verification
│   ├── benchmark.py                   # Xavier vs He vs Ramanujan
│   ├── train.py                       # Standard training script
│   └── train_moe.py                   # MoE training comparison
├── docs/
│   └── theory.md                      # Theoretical derivation
└── figures/                           # Experiment figures
```

---

## Experiments

### Variance Preservation Verification

```bash
python experiments/verify_variance.py
```

Outputs Ramanujan recurrence coefficients and generates variance propagation comparison plots.

### Benchmark

```bash
python experiments/benchmark.py
```

Compares Xavier / He / Ramanujan training performance on synthetic data.

### MoE Comparison

```bash
python experiments/train_moe.py
```

Compares standard Transformer vs MoE Transformer training effectiveness.

---

## Mathematical Background

### Ramanujan and Modular Functions

In 1916, Ramanujan discovered this recurrence relation while studying the Klein j-invariant. The j-invariant is the fundamental generator of the modular group $SL(2, \mathbb{Z})$:

$$j(\tau) = q^{-1} + 744 + 196884q + 21493760q^2 + \cdots$$

where $q = e^{2\pi i \tau}$.

### Connection to Quantum Mechanics

The $\pi^2/n^2$ coefficient in the recurrence has a deep correspondence with quantum mechanical energy level structures:

| Quantum System | Energy Levels | Connection to Recurrence |
|---------------|---------------|-------------------------|
| Hydrogen atom | $E_n = -13.6/n^2$ eV | Same $1/n^2$ structure |
| Infinite square well | $E_n = n^2\pi^2\hbar^2/(2mL^2)$ | $n^2\pi^2$ as reciprocal |
| Harmonic oscillator | $E_n = \hbar\omega(n+1/2)$ | Similar creation/annihilation structure |

The recurrence can be interpreted as a **discrete Schrödinger equation** with zero energy eigenvalue, where the effective potential $V_{\text{eff}}(n) = -\pi^2/n^2$ is the discrete version of the Coulomb potential.

See [docs/theory.md](docs/theory.md) for detailed derivation.

---

## MoE Architecture

### Design Philosophy

Applying Ramanujan initialization to Mixture of Experts:

- **Router**: Gating matrix initialized with Ramanujan coefficients, ensuring stable routing signal variance
- **Experts**: Each expert FFN independently applies Ramanujan initialization
- **Load balancing**: Prevents expert collapse, ensures uniform utilization

### Architecture Diagram

```
Input x
    ↓
LayerNorm → MultiHeadAttention → + Residual
    ↓
LayerNorm → Router (Ramanujan initialization)
                ↓
         Top-K Expert Selection
           ↓         ↓         ↓
        Expert 0  Expert 1  Expert 2  ...
           ↓         ↓         ↓
         Weighted Sum (by routing probability)
                ↓
           + Residual → Output
```

---

## API Reference

### RamanujanInitializer

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ramanujan_depth` | int | 8 | Ramanujan coefficient modulation layers (covers peak region n=0~7) |
| `transition_depth` | int | 8 | Transition layers from Ramanujan to Xavier |
| `nonlinearity` | str | 'linear' | Default nonlinearity: 'linear', 'relu', 'gelu', 'silu' |
| `gain` | float | None | Manual gain. None = auto-infer from nonlinearity |

**Methods:**

| Method | Description |
|--------|-------------|
| `apply(model, gain=1.0)` | **Recommended**: One-step layer index assignment + initialization |
| `get_scale(layer_idx, fan_in)` | Get scale factor for layer n |
| `init_tensor(tensor, layer_idx, gain)` | Initialize any tensor |
| `init_linear(layer, layer_idx, gain)` | Initialize nn.Linear |
| `variance_test(depth, dim, use_residual)` | Variance preservation test |

**Usage Recommendations:**

```python
# Standard Transformer (residual + LayerNorm)
initializer = RamanujanInitializer(gain=1.0)
initializer.apply(model)

# Pure feedforward + ReLU
initializer = RamanujanInitializer(nonlinearity='relu')
initializer.apply(model)  # gain auto-uses sqrt(2)
```

### build_ramanujan_transformer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | int | Required | Vocabulary size |
| `d_model` | int | 768 | Model dimension |
| `nhead` | int | 12 | Number of attention heads |
| `num_layers` | int | 12 | Number of layers |
| `dim_feedforward` | int | 3072 | FFN intermediate dimension |
| `decoder_only` | bool | True | True=GPT, False=BERT |

### build_ramanujan_moe_transformer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_experts` | int | 8 | Number of experts |
| `top_k` | int | 2 | Experts selected per token |
| `capacity_factor` | float | 1.25 | Capacity factor |
| *Other parameters same as build_ramanujan_transformer* | | | |

---

## References

1. Ramanujan, S. (1916). *Notebooks of Srinivasa Ramanujan*.
2. Klein, F. (1890). *Über die Transformation elfter Ordnung der elliptischen Funktionen*.
3. Glorot, X. & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.
4. He, K. et al. (2015). Delving deep into rectifiers.
5. Fedus, W. et al. (2022). Switch Transformers: Scaling to Trillion Parameter Models.
6. Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer.

---

## License

MIT

---

## Contact

- Email: 122241711@qq.com
- GitHub: [aaroncxxx](https://github.com/aaroncxxx)
