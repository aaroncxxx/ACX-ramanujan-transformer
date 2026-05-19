# Changelog

## v1.5.0 (2026-05-19)

### 🔴 Core Algorithm (P1)

- **量化友好初始化**: `QUANTIZATION_GAIN` 增益表 (INT8/FP8/INT4)，`build_ramanujan_transformer(quantization='int8')` 支持
- **自适应权重衰减**: `get_adaptive_weight_decay(layer_idx, base_decay)`，浅层系数大→更强正则化，深层标准衰减
- **QKV 差异化初始化**: Q 增益×1.05（增强查询）、K 标准、V 增益×0.95（稳定值），通过 `LayerRole` 标签自动分发
- **长上下文方差保持**: `compute_rope_correction(seq_len)` RoPE 修正系数，>512 token 自动触发

### 🟡 Ecosystem (P2)

- **预定义配置**: `configs/gpt2_style.yaml`, `configs/llama_7b_style.yaml`, `configs/llama_13b_style.yaml`
- **分级日志系统**: 所有核心模块添加 `logging.getLogger('acx_ramanujan')` 日志
- **配置文件增强**: `configs/default.yaml` 新增 quantization/long_context/adaptive_weight_decay/logging 选项

### 📝 API Changes

- `RamanujanInitializer` 新增 `quantization`, `long_context_seq_len` 参数
- `build_ramanujan_transformer()` 新增 `quantization`, `long_context_seq_len` 参数
- `LayerRole` 新增 `Q_PROJ`, `K_PROJ`, `V_PROJ` 角色标签
- `QUANTIZATION_GAIN` 量化增益表暴露为模块级常量
- `get_adaptive_weight_decay()` 自适应衰减函数暴露为模块级函数
- `compute_rope_correction()` RoPE 修正函数暴露为模块级函数

### 🔧 改动文件

- `src/ramanujan_initializer.py` — 量化增益、自适应衰减、RoPE 修正、日志
- `src/attention.py` — QKV 差异化标签、日志
- `src/feedforward.py` — FFN 角色标签、日志
- `src/transformer_block.py` — 日志
- `src/ramanujan_transformer.py` — 量化/长上下文参数透传、日志
- `src/moe.py` — 日志
- `configs/default.yaml` — 新增配置项
- `configs/gpt2_style.yaml` — GPT-2 预定义配置
- `configs/llama_7b_style.yaml` — 7B Llama 预定义配置
- `configs/llama_13b_style.yaml` — 13B Llama 预定义配置

---

## v1.4.0 (2026-05-19)

### 🔴 Core Algorithm (P0)

- **激活函数自适应增益**: 为 GELU/SiLU/SiGLU/ReLU/Sigmoid 各自推导理论增益系数，替代固定 gain=sqrt(2)
- **动态递推深度计算**: `compute_optimal_depth(num_layers)` 根据模型层数自动计算 ramanujan_depth/transition_depth，保留手动覆盖接口
- **分层初始化适配**: Embedding/QKV/FFN/Output/Router 差异化递推系数，LM Head 单独做方差缩放
- **MoE 路由层专属初始化**: Router gate 使用轻量 gain=0.1，避免路由权重梯度消失

### 🔴 Bug Fixes (P0)

- **MoE 辅助损失 Top-K 修复**: `_compute_aux_loss` 中 `f_i` 从仅统计 Top-1 改为覆盖全部 Top-K 选择的专家，负载均衡损失更准确

### 🟡 Engineering (P0)

- **梯度检查点**: `build_ramanujan_transformer(gradient_checkpointing=True)` 支持，1000 层模型显存降低 60%+
- **系数预计算缓存**: 模块加载时自动预计算 [8,16,32,64,128,256,512,1024] 深度的系数表

### 🟢 Usability (P1)

- **YAML 配置文件**: `configs/default.yaml`，模型/训练/数据/MoE 参数抽离
- **CLI 命令行工具**: `acx-rt info/verify/benchmark/train`
- **pytest 单元测试**: `tests/test_core.py`，覆盖初始化器、Attention、FFN、Block、MoE 核心逻辑

### 📝 API Changes

- `RamanujanInitializer` 新增 `num_layers` 参数（自动计算深度）
- `RamanujanInitializer` 新增 `LayerRole` 层角色标签系统
- `build_ramanujan_transformer()` 新增 `gradient_checkpointing` 参数
- `ACTIVATION_GAIN` 增益表暴露为模块级常量

### 🔧 改动文件

- `src/ramanujan_initializer.py` — 增益表、动态深度、分层初始化、系数缓存
- `src/moe.py` — Router 专属初始化、aux loss Top-K 修复
- `src/ramanujan_transformer.py` — 梯度检查点、动态深度透传
- `src/transformer_block.py` — 梯度检查点注释
- `configs/default.yaml` — 配置文件
- `tests/test_core.py` — 单元测试
- `cli/acx_rt.py` — CLI 工具

---

## v1.3.1 (2026-05-19)

### 🆕 New Features

- **自适应 Attention 缩放**: `scale(l, d_k) = sqrt(d_k) * (1 + α * e^(-λl))`，浅层注意力更分散，深层收敛到标准缩放
- **分段指数 LR 调度**: 线性 warmup → 快速指数衰减 → 缓慢指数衰减，替代 cosine 方案

### 📝 API Changes

- `build_ramanujan_transformer()` 新增 `alpha=0.3`, `lambda_decay=0.5` 参数
- `build_ramanujan_moe_transformer()` 新增 `alpha=0.3`, `lambda_decay=0.5` 参数
- `RamanujanMultiHeadAttention.__init__()` 新增 `alpha`, `lambda_decay` 参数
- 训练调度器从 `get_cosine_schedule_with_warmup` 替换为 `get_piecewise_exp_schedule_with_warmup`

### 🔧 改动文件

- `src/attention.py` — 自适应缩放实现
- `src/transformer_block.py` — 参数透传
- `src/ramanujan_transformer.py` — Encoder/Decoder/build 函数透传
- `src/moe.py` — MoE 全链路透传
- `experiments/train.py` — 分段指数调度器
- `experiments/train_moe.py` — 同步调度器

---

## v1.1.0 (2026-05-19)

### 🆕 New Features

- **学习率调度器升级**: 训练脚本从 epoch 级 `CosineAnnealingLR` 改为 step 级 **线性 Warmup + 余弦衰减**，适配深网络初期训练
- **三语 README**: 新增德语（Deutsch）版本，README 现支持中文 / English / Deutsch

### 📝 Changes

- `train.py`: 新增 `get_cosine_schedule_with_warmup()`，默认 `warmup_ratio=0.1`
- `train_moe.py`: 同步应用 warmup 调度器

---

## v2.0.0 (2026-05-18)

### 🔴 Breaking Changes

- `RamanujanInitializer(max_depth=...)` → `RamanujanInitializer(ramanujan_depth=8, transition_depth=8)`
- `get_scale(layer_idx)` 现在需要 `fan_in` 参数
- `variance_test()` 的 `use_residual` 默认改为 `True`

### ✅ Bug Fixes

- **`nn.Tensor` → `torch.Tensor`**: 修复类型注解导致的 `AttributeError`
- **缩放因子爆炸/坍缩**: 原版系数在 n≈4 达到峰值后指数衰减至零，导致深层权重初始化为 ~0。新方案使用三层混合策略：
  - 前 8 层：峰值归一化的 Ramanujan 系数调制
  - 8-16 层：线性过渡到 Xavier
  - 16+ 层：标准 Xavier 初始化
- **`layer_idx` 永远为 0**: 新增 `assign_layer_indices(model)` 按拓扑序自动分配

### 🆕 New Features

- **`gain` 参数**: 支持手动指定增益。残差网络（Transformer）用 `gain=1.0`，纯前馈网络用 `gain=sqrt(2)`
- **`apply(model)` 一步初始化**: 自动分配层索引 + 初始化所有参数
- **残差方差测试**: `variance_test(use_residual=True, use_layernorm=True)` 模拟真实 Transformer 场景
- **方差验证结果**: Residual + LayerNorm 下，depth=8~200 方差比稳定在 1.34x

### 📝 Documentation

- 明确递推公式的数学性质：类似 Bessel 函数递推，非标准模形式递推
- 补充系数衰减分析（峰值 n≈4，n>30 接近零）
- 新增三层混合方案设计说明

---

## v1.0.0 (2026-05-12)

- 初始发布
- 基于 Ramanujan 递推关系的权重初始化
- 标准 Transformer + MoE Transformer 实现
- 方差保持性测试、基准对比、训练脚本
