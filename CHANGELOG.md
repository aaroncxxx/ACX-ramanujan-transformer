# Changelog

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
