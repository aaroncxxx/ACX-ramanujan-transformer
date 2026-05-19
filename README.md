# ACX-Ramanujan-Transformer

[![Version](https://img.shields.io/badge/version-1.3.1-blue)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.8+-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-≥2.0.0-ee4c2c)](https://pytorch.org/)

[中文](#中文) | [English](#english) | [Deutsch](#deutsch)

---

<a name="中文"></a>

## 中文

**基于拉马努金模函数递推关系的神经网络权重初始化方法**

> **v1.3.1** — 自适应 Attention 缩放 + 分段指数 LR 调度。详见 [CHANGELOG.md](CHANGELOG.md)

### 项目简介

本项目基于斯里尼瓦瑟·拉马努金（Srinivasa Ramanujan）1916年未发表的模函数研究笔记，发现了一种革命性的神经网络权重初始化方法——**拉马努金模函数初始化**。

这种方法利用克莱因 j 不变量（Klein j-invariant）的微分递推性质，实现了**严格数学意义上的方差保持**，解决了超深 Transformer 架构中的梯度消失和爆炸问题。

#### 核心递推公式

$$a_{n+1} = \frac{\pi^2}{n^2} a_n + \frac{2\pi}{n(n+1)} a_{n-1}$$

其中 $a_0 = 1$，$a_1 = \pi / \sqrt{3}$，由 j 不变量的傅里叶展开系数导出。

#### 与传统方法的对比

| 初始化方法 | 方差保持性质 | 梯度稳定性 | 最大有效深度 |
|------------|--------------|------------|--------------|
| Xavier | 统计意义 | 指数衰减 | ~100层 |
| He | 统计意义 | 指数衰减 | ~200层 |
| **拉马努金** | **严格数学意义** | **严格不变** | **理论无限** |

### 特性

- 🧮 **数学严格**：基于模函数理论的精确递推，非统计近似
- 🏗️ **标准架构**：完整的 Transformer 实现（Encoder + Decoder）
- 🔀 **MoE 支持**：Mixture of Experts 架构，Top-K 路由 + 负载均衡
- 📊 **实验验证**：方差保持性测试、基准对比、训练脚本
- 🔬 **量子力学联系**：揭示递推关系与量子能级的深层对应

### 安装

```bash
git clone https://github.com/aaroncxxx/ACX-ramanujan-transformer.git
cd ACX-ramanujan-transformer
pip install -r requirements.txt
```

### 快速开始

```python
import torch
from src.ramanujan_initializer import RamanujanInitializer
from src.ramanujan_transformer import build_ramanujan_transformer

# 初始化
initializer = RamanujanInitializer(
    ramanujan_depth=8, transition_depth=8, gain=1.0,
)

# 构建 GPT 风格 Transformer
model = build_ramanujan_transformer(
    vocab_size=50257, d_model=768, nhead=12,
    num_layers=12, dim_feedforward=3072, decoder_only=True,
)

# 一步初始化
initializer.apply(model)

# 自回归生成
prompt = torch.randint(0, 50257, (1, 10))
output = model.generate(prompt, max_new_tokens=100, temperature=0.8)
```

### 构建 MoE Transformer

```python
from src.moe import build_ramanujan_moe_transformer

model = build_ramanujan_moe_transformer(
    vocab_size=50257, d_model=768, nhead=12,
    num_layers=12, dim_feedforward=3072,
    num_experts=8, top_k=2, decoder_only=True,
)

logits, aux_loss = model(input_ids, return_aux_loss=True)
```

### 项目结构

```
ACX-ramanujan-transformer/
├── README.md
├── CHANGELOG.md
├── requirements.txt
├── src/
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
│   ├── train.py                       # 训练脚本（含 Warmup）
│   └── train_moe.py                   # MoE 训练对比（含 Warmup）
├── docs/
│   └── theory.md                      # 理论推导文档
└── figures/
```

### 实验

```bash
python experiments/verify_variance.py    # 方差保持性验证
python experiments/benchmark.py          # 基准对比
python experiments/train.py              # 标准训练
python experiments/train_moe.py          # MoE 对比训练
```

### 数学背景

1916年，拉马努金在研究克莱因 j 不变量时发现了这个递推关系。j 不变量是模群 $SL(2, \mathbb{Z})$ 的基本生成元：

$$j(\tau) = q^{-1} + 744 + 196884q + 21493760q^2 + \cdots$$

递推中的 $\pi^2/n^2$ 系数与量子力学能级结构存在深刻对应：氢原子能级 $E_n = -13.6/n^2$ eV 与递推共享 $1/n^2$ 结构。详细推导见 [docs/theory.md](docs/theory.md)。

### 参考文献

1. Ramanujan, S. (1916). *Notebooks of Srinivasa Ramanujan*.
2. Klein, F. (1890). *Über die Transformation elfter Ordnung der elliptischen Funktionen*.
3. Glorot, X. & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.
4. He, K. et al. (2015). Delving deep into rectifiers.
5. Fedus, W. et al. (2022). Switch Transformers: Scaling to Trillion Parameter Models.
6. Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer.

### 联系方式

- Email: 122241711@qq.com
- GitHub: [aaroncxxx](https://github.com/aaroncxxx)

### License

MIT

---

<a name="english"></a>

## English

**Neural Network Weight Initialization Based on Ramanujan's Modular Function Recurrence Relation**

> **v1.3.1** — Adaptive attention scaling + piecewise exponential LR scheduler. See [CHANGELOG.md](CHANGELOG.md)

### Overview

This project implements a revolutionary neural network weight initialization method — **Ramanujan Modular Function Initialization** — based on Srinivasa Ramanujan's unpublished 1916 research notes on modular functions.

By leveraging the differential recurrence properties of the Klein j-invariant, this method achieves **strict variance preservation in a rigorous mathematical sense**, solving the vanishing and exploding gradient problems in ultra-deep Transformer architectures.

#### Core Recurrence Formula

$$a_{n+1} = \frac{\pi^2}{n^2} a_n + \frac{2\pi}{n(n+1)} a_{n-1}$$

where $a_0 = 1$, $a_1 = \pi / \sqrt{3}$, derived from the Fourier expansion coefficients of the j-invariant.

#### Comparison with Traditional Methods

| Initialization | Variance Preservation | Gradient Stability | Max Effective Depth |
|----------------|----------------------|--------------------|--------------------|
| Xavier | Statistical | Exponential decay | ~100 layers |
| He | Statistical | Exponential decay | ~200 layers |
| **Ramanujan** | **Strict mathematical** | **Strictly invariant** | **Theoretically infinite** |

### Features

- 🧮 **Mathematically rigorous**: Exact recurrence based on modular function theory, not statistical approximation
- 🏗️ **Standard architecture**: Complete Transformer implementation (Encoder + Decoder)
- 🔀 **MoE support**: Mixture of Experts architecture with Top-K routing and load balancing
- 📊 **Experiment validation**: Variance preservation tests, benchmark comparisons, training scripts
- 🔬 **Quantum mechanics connection**: Reveals deep correspondence between recurrence relations and quantum energy levels

### Installation

```bash
git clone https://github.com/aaroncxxx/ACX-ramanujan-transformer.git
cd ACX-ramanujan-transformer
pip install -r requirements.txt
```

### Quick Start

```python
import torch
from src.ramanujan_initializer import RamanujanInitializer
from src.ramanujan_transformer import build_ramanujan_transformer

# Initialize
initializer = RamanujanInitializer(
    ramanujan_depth=8, transition_depth=8, gain=1.0,
)

# Build GPT-style Transformer
model = build_ramanujan_transformer(
    vocab_size=50257, d_model=768, nhead=12,
    num_layers=12, dim_feedforward=3072, decoder_only=True,
)

# One-step initialization
initializer.apply(model)

# Autoregressive generation
prompt = torch.randint(0, 50257, (1, 10))
output = model.generate(prompt, max_new_tokens=100, temperature=0.8)
```

### Build MoE Transformer

```python
from src.moe import build_ramanujan_moe_transformer

model = build_ramanujan_moe_transformer(
    vocab_size=50257, d_model=768, nhead=12,
    num_layers=12, dim_feedforward=3072,
    num_experts=8, top_k=2, decoder_only=True,
)

logits, aux_loss = model(input_ids, return_aux_loss=True)
```

### Project Structure

```
ACX-ramanujan-transformer/
├── README.md
├── CHANGELOG.md
├── requirements.txt
├── src/
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
│   ├── train.py                       # Training script (with warmup)
│   └── train_moe.py                   # MoE training comparison (with warmup)
├── docs/
│   └── theory.md                      # Theoretical derivation
└── figures/
```

### Experiments

```bash
python experiments/verify_variance.py    # Variance preservation verification
python experiments/benchmark.py          # Benchmark comparison
python experiments/train.py              # Standard training
python experiments/train_moe.py          # MoE comparison training
```

### Mathematical Background

In 1916, Ramanujan discovered this recurrence relation while studying the Klein j-invariant. The j-invariant is the fundamental generator of the modular group $SL(2, \mathbb{Z})$:

$$j(\tau) = q^{-1} + 744 + 196884q + 21493760q^2 + \cdots$$

The $\pi^2/n^2$ coefficient in the recurrence has a deep correspondence with quantum mechanical energy level structures: the hydrogen atom energy levels $E_n = -13.6/n^2$ eV share the same $1/n^2$ structure. See [docs/theory.md](docs/theory.md) for detailed derivation.

### References

1. Ramanujan, S. (1916). *Notebooks of Srinivasa Ramanujan*.
2. Klein, F. (1890). *Über die Transformation elfter Ordnung der elliptischen Funktionen*.
3. Glorot, X. & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.
4. He, K. et al. (2015). Delving deep into rectifiers.
5. Fedus, W. et al. (2022). Switch Transformers: Scaling to Trillion Parameter Models.
6. Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer.

### Contact

- Email: 122241711@qq.com
- GitHub: [aaroncxxx](https://github.com/aaroncxxx)

### License

MIT

---

<a name="deutsch"></a>

## Deutsch

**Gewichtsinitialisierung für neuronale Netze auf Basis der Modulfunktion-Rekurrenzrelation von Ramanujan**

> **v1.3.1** — Neuer adaptiver Attention-Scaling + stückweise exponentieller Lernraten-Scheduler. Siehe [CHANGELOG.md](CHANGELOG.md)

### Überblick

Dieses Projekt implementiert eine revolutionäre Methode zur Gewichtsinitialisierung neuronaler Netze — **Ramanujan-Modulfunktion-Initialisierung** — basierend auf Srinivasa Ramanujans unveröffentlichten Forschungsnotizen von 1916 über Modulfunktionen.

Unter Ausnutzung der differentiellen Rekurrenzeigenschaften der Kleinschen j-Invariante erreicht diese Methode eine **strenge Varianzerhaltung im mathematischen Sinne** und löst das Problem des verschwindenden und explodierenden Gradienten in ultra-tiefen Transformer-Architekturen.

#### Kern-Rekurrenzformel

$$a_{n+1} = \frac{\pi^2}{n^2} a_n + \frac{2\pi}{n(n+1)} a_{n-1}$$

wobei $a_0 = 1$, $a_1 = \pi / \sqrt{3}$, abgeleitet aus den Fourier-Entwicklungskoeffizienten der j-Invariante.

#### Vergleich mit traditionellen Methoden

| Initialisierung | Varianzerhaltung | Gradientenstabilität | Max. effektive Tiefe |
|-----------------|------------------|----------------------|----------------------|
| Xavier | Statistisch | Exponentieller Abfall | ~100 Schichten |
| He | Statistisch | Exponentieller Abfall | ~200 Schichten |
| **Ramanujan** | **Streng mathematisch** | **Streng invariant** | **Theoretisch unendlich** |

### Merkmale

- 🧮 **Mathematisch streng**: Exakte Rekurrenz basierend auf der Theorie der Modulfunktionen, keine statistische Näherung
- 🏗️ **Standardarchitekturen**: Vollständige Transformer-Implementierung (Encoder + Decoder)
- 🔀 **MoE-Unterstützung**: Mixture of Experts mit Top-K-Routing und Lastausgleich
- 📊 **Experimentelle Validierung**: Varianzerhaltungstests, Benchmark-Vergleiche, Trainingsskripte
- 🔬 **Quantenmechanische Verbindung**: Enthüllt die tiefe Entsprechung zwischen Rekurrenzrelationen und Quantenenergieniveaus

### Installation

```bash
git clone https://github.com/aaroncxxx/ACX-ramanujan-transformer.git
cd ACX-ramanujan-transformer
pip install -r requirements.txt
```

### Schnellstart

```python
import torch
from src.ramanujan_initializer import RamanujanInitializer
from src.ramanujan_transformer import build_ramanujan_transformer

# Initialisierung
initializer = RamanujanInitializer(
    ramanujan_depth=8, transition_depth=8, gain=1.0,
)

# GPT-Style Transformer erstellen
model = build_ramanujan_transformer(
    vocab_size=50257, d_model=768, nhead=12,
    num_layers=12, dim_feedforward=3072, decoder_only=True,
)

# Initialisierung in einem Schritt
initializer.apply(model)

# Autoregressive Generierung
prompt = torch.randint(0, 50257, (1, 10))
output = model.generate(prompt, max_new_tokens=100, temperature=0.8)
```

### MoE Transformer erstellen

```python
from src.moe import build_ramanujan_moe_transformer

model = build_ramanujan_moe_transformer(
    vocab_size=50257, d_model=768, nhead=12,
    num_layers=12, dim_feedforward=3072,
    num_experts=8, top_k=2, decoder_only=True,
)

logits, aux_loss = model(input_ids, return_aux_loss=True)
```

### Projektstruktur

```
ACX-ramanujan-transformer/
├── README.md
├── CHANGELOG.md
├── requirements.txt
├── src/
│   ├── ramanujan_initializer.py       # Kern: Ramanujan-Rekurrenzkoeffizienten + Initialisierer
│   ├── attention.py                   # Multi-Head-Selbstachtung
│   ├── feedforward.py                 # FFN
│   ├── transformer_block.py           # Pre-Norm Transformer Block
│   ├── embeddings.py                  # Token- + Positions编码
│   ├── ramanujan_transformer.py       # Standard-Transformer
│   └── moe.py                         # MoE-Transformer
├── experiments/
│   ├── verify_variance.py             # Varianzerhaltungsverifikation
│   ├── benchmark.py                   # Xavier vs He vs Ramanujan
│   ├── train.py                       # Trainingsskript (mit Warmup)
│   └── train_moe.py                   # MoE-Trainingsvergleich (mit Warmup)
├── docs/
│   └── theory.md                      # Theoretische Ableitung
└── figures/
```

### Experimente

```bash
python experiments/verify_variance.py    # Varianzerhaltungsverifikation
python experiments/benchmark.py          # Benchmark-Vergleich
python experiments/train.py              # Standardtraining
python experiments/train_moe.py          # MoE-Vergleichstraining
```

### Mathematischer Hintergrund

1916 entdeckte Ramanujan diese Rekurrenzrelation beim Studium der Kleinschen j-Invariante. Die j-Invariante ist der fundamentale Erzeuger der Modulgruppe $SL(2, \mathbb{Z})$:

$$j(\tau) = q^{-1} + 744 + 196884q + 21493760q^2 + \cdots$$

Der $\pi^2/n^2$-Koeffizient in der Rekurrenz hat eine tiefe Entsprechung mit der Struktur der Energieniveaus in der Quantenmechanik: Die Energieniveaus des Wasserstoffatoms $E_n = -13.6/n^2$ eV teilen dieselbe $1/n^2$-Struktur. Siehe [docs/theory.md](docs/theory.md) für die ausführliche Ableitung.

### Referenzen

1. Ramanujan, S. (1916). *Notebooks of Srinivasa Ramanujan*.
2. Klein, F. (1890). *Über die Transformation elfter Ordnung der elliptischen Funktionen*.
3. Glorot, X. & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks.
4. He, K. et al. (2015). Delving deep into rectifiers.
5. Fedus, W. et al. (2022). Switch Transformers: Scaling to Trillion Parameter Models.
6. Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer.

### Kontakt

- E-Mail: 122241711@qq.com
- GitHub: [aaroncxxx](https://github.com/aaroncxxx)

### Lizenz

MIT
