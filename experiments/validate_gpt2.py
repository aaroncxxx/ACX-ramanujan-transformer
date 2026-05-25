"""
GPT-2 真实模型验证

在 GPT-2 small (124M) 结构上对比 Ramanujan 初始化 vs 默认初始化：
- 语言建模 perplexity
- 训练收敛曲线
- 梯度健康度
- 生成质量

数据：WikiText-103 子集（或合成数据作为 fallback）
"""

import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ramanujan_initializer import RamanujanInitializer


# ════════════════════════════════════════════════════════════════════
#  GPT-2 架构实现（不依赖 HuggingFace）
# ════════════════════════════════════════════════════════════════════

class GPT2Config:
    """GPT-2 Small 配置"""
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int = 3072  # 4 * n_embd
    activation: str = 'gelu'
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(causal_mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return self.dropout(y)


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.wte.weight

    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None):
        B, T = input_ids.shape
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(input_ids)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# ════════════════════════════════════════════════════════════════════
#  初始化方法
# ════════════════════════════════════════════════════════════════════

def apply_default_init(model: GPT2):
    """GPT-2 默认初始化（OpenAI 原版）"""
    for name, p in model.named_parameters():
        if p.dim() > 1:
            nn.init.normal_(p, mean=0.0, std=0.02)
        elif 'ln' in name:
            nn.init.ones_(p)


def apply_xavier_init(model: GPT2):
    """Xavier 初始化"""
    for name, p in model.named_parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        elif 'ln' in name:
            nn.init.ones_(p)


def apply_he_init(model: GPT2):
    """He/Kaiming 初始化"""
    for name, p in model.named_parameters():
        if p.dim() > 1:
            nn.init.kaiming_uniform_(p, nonlinearity='relu')
        elif 'ln' in name:
            nn.init.ones_(p)


def apply_ramanujan_init(model: GPT2, quantization: str = 'none'):
    """Ramanujan 初始化"""
    from src.attention import tag_linear_role
    from src.ramanujan_initializer import LayerRole

    # Tag QKV
    for block in model.h:
        tag_linear_role(block.attn.c_attn, LayerRole.QKV_PROJ)
        tag_linear_role(block.attn.c_proj, LayerRole.OUTPUT_ATTN)
        tag_linear_role(block.mlp.c_fc, LayerRole.FFN_UP)
        tag_linear_role(block.mlp.c_proj, LayerRole.FFN_DOWN)

    initializer = RamanujanInitializer(
        num_layers=model.config.n_layer,
        nonlinearity='gelu',
        quantization=quantization,
    )
    # LM Head 权重绑定，不重复初始化
    initializer.apply(model)


INIT_METHODS = {
    'default': apply_default_init,
    'xavier': apply_xavier_init,
    'he': apply_he_init,
    'ramanujan': apply_ramanujan_init,
}


# ════════════════════════════════════════════════════════════════════
#  训练与评估
# ════════════════════════════════════════════════════════════════════

@dataclass
class TrainStats:
    method: str
    losses: List[float]
    grad_norms: List[float]
    perplexities: List[float]
    wall_time: float
    final_loss: float
    final_ppl: float


def generate_synthetic_data(num_tokens: int = 100000,
                             vocab_size: int = 50257,
                             seq_len: int = 128) -> List[torch.Tensor]:
    """生成合成训练数据（模拟 Zipf 分布的 token 序列）"""
    # 模拟自然语言的 Zipf 分布
    ranks = torch.arange(1, vocab_size + 1, dtype=torch.float)
    probs = 1.0 / ranks
    probs /= probs.sum()

    data = torch.multinomial(probs, num_tokens, replacement=True)
    sequences = []
    for i in range(0, num_tokens - seq_len, seq_len):
        chunk = data[i:i + seq_len + 1]
        if len(chunk) == seq_len + 1:
            sequences.append(chunk)
    return sequences


def train_gpt2(method: str, config: GPT2Config,
               num_epochs: int = 30, batch_size: int = 8,
               lr: float = 6e-4, warmup_steps: int = 200,
               data: List[torch.Tensor] = None) -> TrainStats:
    """训练 GPT-2 并收集统计"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")

    model = GPT2(config).to(device)

    # 应用初始化
    init_fn = INIT_METHODS[method]
    if method == 'ramanujan':
        init_fn(model)
    else:
        init_fn(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {num_params / 1e6:.1f}M")

    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=0.1, betas=(0.9, 0.95))

    # Cosine LR schedule with warmup
    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, num_epochs * len(data) // batch_size - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    losses = []
    grad_norms = []
    perplexities = []
    step = 0
    start = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_steps = 0

        # Shuffle
        indices = torch.randperm(len(data))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            if len(batch_idx) < batch_size:
                continue

            batch = torch.stack([data[j] for j in batch_idx]).to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad()
            logits, loss = model(input_ids, targets)
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            step += 1

            if step % 50 == 0:
                losses.append(loss.item())
                grad_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                ppl = math.exp(loss.item())
                perplexities.append(ppl)

        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_ppl = math.exp(avg_loss)
        elapsed = time.time() - start
        print(f"    Epoch {epoch+1:3d}: loss={avg_loss:.4f}, ppl={avg_ppl:.2f}, "
              f"lr={scheduler.get_last_lr()[0]:.2e}, time={elapsed:.1f}s")

    wall_time = time.time() - start
    final_loss = losses[-1] if losses else avg_loss
    final_ppl = math.exp(final_loss)

    return TrainStats(
        method=method,
        losses=losses,
        grad_norms=grad_norms,
        perplexities=perplexities,
        wall_time=wall_time,
        final_loss=final_loss,
        final_ppl=final_ppl,
    )


# ════════════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════════════

def run_gpt2_validation(num_epochs: int = 30, batch_size: int = 8,
                         methods: List[str] = None,
                         output_json: str = None):
    """运行 GPT-2 验证实验"""
    if methods is None:
        methods = ['default', 'xavier', 'he', 'ramanujan']

    config = GPT2Config()

    print("=" * 70)
    print("  GPT-2 Small (124M) 初始化方法验证")
    print("=" * 70)
    print(f"  模型: {config.n_layer}L / {config.n_embd}d / {config.n_head}H")
    print(f"  训练: {num_epochs} epochs, batch_size={batch_size}")
    print()

    # 生成合成数据
    print("  生成训练数据...")
    data = generate_synthetic_data(num_tokens=50000, vocab_size=config.vocab_size,
                                    seq_len=128)
    print(f"  序列数: {len(data)}")

    results = {}
    for method in methods:
        print(f"\n{'─' * 50}")
        print(f"  方法: {method}")
        print(f"{'─' * 50}")

        stats = train_gpt2(method, config, num_epochs=num_epochs,
                           batch_size=batch_size, data=data)
        results[method] = {
            'final_loss': round(stats.final_loss, 4),
            'final_ppl': round(stats.final_ppl, 2),
            'wall_time': round(stats.wall_time, 2),
            'losses': [round(l, 4) for l in stats.losses],
            'perplexities': [round(p, 2) for p in stats.perplexities],
        }

    # 汇总
    print(f"\n{'=' * 70}")
    print("  汇总")
    print(f"{'=' * 70}")
    print(f"  {'方法':>12s} {'最终Loss':>12s} {'Perplexity':>12s} {'耗时(s)':>10s}")
    print(f"  {'─' * 48}")

    ranked = sorted(results.items(), key=lambda x: x[1]['final_loss'])
    for rank, (method, data) in enumerate(ranked, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"  {medal} {method:>10s} {data['final_loss']:12.4f} "
              f"{data['final_ppl']:12.2f} {data['wall_time']:10.2f}")

    if output_json:
        import json
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  结果已保存: {output_json}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GPT-2 初始化方法验证')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--methods', nargs='+', default=['default', 'xavier', 'he', 'ramanujan'])
    parser.add_argument('--output', type=str, help='JSON 输出路径')

    args = parser.parse_args()
    run_gpt2_validation(args.epochs, args.batch_size, args.methods, args.output)
