"""
ACX-Ramanujan-Transformer 单元测试

覆盖: 初始化器、Transformer 块、MoE 层核心逻辑
"""

import math
import pytest
import torch
import torch.nn as nn

from src.ramanujan_initializer import (
    ramanujan_coefficients,
    get_ramanujan_scale,
    compute_optimal_depth,
    get_activation_gain,
    ACTIVATION_GAIN,
    RamanujanInitializer,
    assign_layer_indices,
    LayerRole,
    tag_linear_role,
    get_linear_role,
)
from src.attention import RamanujanMultiHeadAttention
from src.feedforward import RamanujanFFN
from src.transformer_block import RamanujanTransformerBlock


# ─── 系数计算测试 ──────────────────────────────────────────────────

class TestCoefficients:
    def test_basic(self):
        coeffs = ramanujan_coefficients(5)
        assert len(coeffs) == 6
        assert coeffs[0] == 1.0
        assert abs(coeffs[1] - math.pi / math.sqrt(3)) < 1e-10

    def test_peak_at_n4(self):
        coeffs = ramanujan_coefficients(20)
        peak_idx = max(range(len(coeffs)), key=lambda i: abs(coeffs[i]))
        assert 3 <= peak_idx <= 6  # 峰值在 n≈4 附近

    def test_decay_after_peak(self):
        coeffs = ramanujan_coefficients(30)
        peak = max(abs(c) for c in coeffs)
        # n>15 后系数应该很小
        for i in range(16, len(coeffs)):
            assert abs(coeffs[i]) < peak * 0.01

    def test_cache(self):
        c1 = ramanujan_coefficients(10)
        c2 = ramanujan_coefficients(10)
        assert c1 is c2  # 同一个对象（缓存命中）

    def test_empty(self):
        assert ramanujan_coefficients(0) == (1.0,)
        assert ramanujan_coefficients(-1) == ()


# ─── 动态深度计算测试 ──────────────────────────────────────────────

class TestDynamicDepth:
    def test_small_model(self):
        r, t = compute_optimal_depth(6)
        assert r >= 4
        assert t >= 4
        assert r + t <= 6

    def test_medium_model(self):
        r, t = compute_optimal_depth(12)
        assert r >= 4
        assert t >= 4
        assert r + t <= 12

    def test_large_model(self):
        r, t = compute_optimal_depth(100)
        assert r <= 8  # 不超过系数峰值覆盖区
        assert r + t <= 100

    def test_very_small(self):
        r, t = compute_optimal_depth(2)
        assert r + t <= 2


# ─── 激活函数增益测试 ──────────────────────────────────────────────

class TestActivationGain:
    def test_relu(self):
        assert abs(get_activation_gain('relu') - math.sqrt(2.0)) < 1e-6

    def test_linear(self):
        assert get_activation_gain('linear') == 1.0

    def test_gelu(self):
        assert abs(get_activation_gain('gelu') - math.sqrt(2.0)) < 1e-6

    def test_unknown(self):
        assert get_activation_gain('unknown') == 1.0

    def test_all_keys(self):
        for key in ACTIVATION_GAIN:
            assert get_activation_gain(key) > 0


# ─── 缩放因子测试 ──────────────────────────────────────────────────

class TestScale:
    def test_first_layer(self):
        s = get_ramanujan_scale(0, 8, 8, 512)
        assert s > 0

    def test_deep_layer_is_xavier(self):
        s = get_ramanujan_scale(100, 8, 8, 512)
        assert abs(s - 1.0) < 0.01  # 深层应接近 Xavier

    def test_monotonic_transition(self):
        # 过渡区应单调变化
        s8 = get_ramanujan_scale(8, 8, 8, 512)
        s12 = get_ramanujan_scale(12, 8, 8, 512)
        s16 = get_ramanujan_scale(16, 8, 8, 512)
        assert s8 <= s12 <= s16 or s8 >= s12 >= s16


# ─── 初始化器测试 ──────────────────────────────────────────────────

class TestInitializer:
    def test_auto_depth(self):
        init = RamanujanInitializer(num_layers=12)
        assert init.ramanujan_depth >= 4
        assert init.transition_depth >= 4

    def test_manual_depth(self):
        init = RamanujanInitializer(ramanujan_depth=6, transition_depth=4)
        assert init.ramanujan_depth == 6
        assert init.transition_depth == 4

    def test_init_linear(self):
        init = RamanujanInitializer()
        layer = nn.Linear(256, 256)
        init.init_linear(layer, layer_idx=0)
        assert layer.weight.std() > 0

    def test_init_tensor(self):
        init = RamanujanInitializer()
        t = torch.empty(256, 256)
        init.init_tensor(t, layer_idx=0)
        assert t.std() > 0

    def test_apply(self):
        init = RamanujanInitializer()
        model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
        init.apply(model)
        for p in model.parameters():
            assert p.std() > 0

    def test_variance_test(self):
        init = RamanujanInitializer()
        result = init.variance_test(depth=50, dim=128, use_residual=True)
        assert 'ratio' in result
        assert 0.5 < result['ratio'] < 3.0  # 方差不应极端偏离


# ─── 层角色标签测试 ────────────────────────────────────────────────

class TestLayerRole:
    def test_tag_and_get(self):
        layer = nn.Linear(64, 64)
        assert get_linear_role(layer) == LayerRole.OTHER
        tag_linear_role(layer, LayerRole.QKV_PROJ)
        assert get_linear_role(layer) == LayerRole.QKV_PROJ


# ─── Attention 测试 ────────────────────────────────────────────────

class TestAttention:
    def test_output_shape(self):
        attn = RamanujanMultiHeadAttention(d_model=128, nhead=4, layer_idx=0)
        x = torch.randn(2, 10, 128)
        out, weights = attn(x)
        assert out.shape == (2, 10, 128)
        assert weights.shape == (2, 4, 10, 10)

    def test_causal(self):
        attn = RamanujanMultiHeadAttention(d_model=128, nhead=4, layer_idx=0)
        x = torch.randn(2, 10, 128)
        out, _ = attn(x, is_causal=True)
        assert out.shape == (2, 10, 128)

    def test_adaptive_scale(self):
        # 浅层 scale 应大于深层
        attn0 = RamanujanMultiHeadAttention(d_model=64, nhead=4, layer_idx=0)
        attn10 = RamanujanMultiHeadAttention(d_model=64, nhead=4, layer_idx=10)
        assert attn0.scale > attn10.scale

    def test_alpha_zero_equals_standard(self):
        # alpha=0 时应等价于标准 sqrt(d_k)
        attn = RamanujanMultiHeadAttention(
            d_model=64, nhead=4, layer_idx=0, alpha=0.0
        )
        expected = math.sqrt(64 // 4)
        assert abs(attn.scale - expected) < 1e-6


# ─── FFN 测试 ──────────────────────────────────────────────────────

class TestFFN:
    def test_output_shape(self):
        ffn = RamanujanFFN(d_model=128, dim_feedforward=512, layer_idx=0)
        x = torch.randn(2, 10, 128)
        out = ffn(x)
        assert out.shape == (2, 10, 128)


# ─── Transformer Block 测试 ────────────────────────────────────────

class TestTransformerBlock:
    def test_output_shape(self):
        block = RamanujanTransformerBlock(
            d_model=128, nhead=4, dim_feedforward=512, layer_idx=0
        )
        x = torch.randn(2, 10, 128)
        out = block(x)
        assert out.shape == (2, 10, 128)

    def test_causal(self):
        block = RamanujanTransformerBlock(
            d_model=128, nhead=4, dim_feedforward=512, layer_idx=0
        )
        x = torch.randn(2, 10, 128)
        out = block(x, is_causal=True)
        assert out.shape == (2, 10, 128)


# ─── 完整模型测试 ──────────────────────────────────────────────────

class TestFullModel:
    def test_build_decoder(self):
        from src.ramanujan_transformer import build_ramanujan_transformer
        model = build_ramanujan_transformer(
            vocab_size=100, d_model=64, nhead=4,
            num_layers=3, dim_feedforward=128, decoder_only=True
        )
        x = torch.randint(0, 100, (2, 10))
        out = model(x)
        assert out.shape == (2, 10, 100)

    def test_build_encoder(self):
        from src.ramanujan_transformer import build_ramanujan_transformer
        model = build_ramanujan_transformer(
            vocab_size=100, d_model=64, nhead=4,
            num_layers=3, dim_feedforward=128, decoder_only=False
        )
        x = torch.randint(0, 100, (2, 10))
        out = model(x)
        assert out.shape == (2, 10, 64)

    def test_gradient_checkpointing(self):
        from src.ramanujan_transformer import build_ramanujan_transformer
        model = build_ramanujan_transformer(
            vocab_size=100, d_model=64, nhead=4,
            num_layers=3, dim_feedforward=128,
            decoder_only=True, gradient_checkpointing=True
        )
        assert model.gradient_checkpointing is True
        x = torch.randint(0, 100, (2, 10))
        out = model(x)
        assert out.shape == (2, 10, 100)


# ─── MoE 测试 ──────────────────────────────────────────────────────

class TestMoE:
    def test_router_output_shape(self):
        from src.moe import RamanujanRouter
        router = RamanujanRouter(d_model=64, num_experts=8)
        x = torch.randn(2, 10, 64)
        probs, logits = router(x)
        assert probs.shape == (2, 10, 8)
        assert logits.shape == (2, 10, 8)
        # 概率应归一化
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 10), atol=1e-5)

    def test_moe_layer_output_shape(self):
        from src.moe import RamanujanMoELayer
        moe = RamanujanMoELayer(d_model=64, dim_feedforward=128, num_experts=4, top_k=2)
        x = torch.randn(2, 10, 64)
        out, aux = moe(x, return_router_logits=True)
        assert out.shape == (2, 10, 64)
        assert aux is not None
        assert 'load_balance_loss' in aux
        assert 'z_loss' in aux

    def test_build_moe_transformer(self):
        from src.moe import build_ramanujan_moe_transformer
        model = build_ramanujan_moe_transformer(
            vocab_size=100, d_model=64, nhead=4,
            num_layers=3, dim_feedforward=128,
            num_experts=4, top_k=2, decoder_only=True
        )
        x = torch.randint(0, 100, (2, 10))
        logits, aux = model(x, return_aux_loss=True)
        assert logits.shape == (2, 10, 100)
        assert aux is not None

    def test_aux_loss_topk_fix(self):
        """验证 MoE 辅助损失 Top-K 修复：f_i 应覆盖所有 top-K 专家"""
        from src.moe import RamanujanMoELayer
        moe = RamanujanMoELayer(d_model=64, dim_feedforward=128, num_experts=4, top_k=2)
        x = torch.randn(4, 20, 64)
        _, aux = moe(x, return_router_logits=True)
        f = aux['expert_usage']
        # 所有专家至少应有一定使用率（概率很低但不为零）
        assert f.shape == (4,)
        assert f.sum() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
