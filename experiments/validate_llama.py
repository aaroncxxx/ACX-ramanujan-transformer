"""
Llama 结构验证

在 Llama-style Transformer 上验证 Ramanujan 初始化：
- Pre-RMSNorm（非 LayerNorm）
- SwiGLU FFN（非标准 FFN）
- RoPE 位置编码
- GQA (Grouped Query Attention) 支持

对比：默认初始化 vs Xavier vs Ramanujan
"""

import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ramanujan_initializer import RamanujanInitializer, LayerRole, tag_linear_role


# ════════════════════════════════════════════════════════════════════
#  Llama 架构实现
# ════════════════════════════════════════════════════════════════════

class LlamaRMSNorm(nn.Module):
    """RMSNorm (Root Mean Square Layer Normalization)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int = 2048,
                           base: float = 10000.0) -> torch.Tensor:
    """预计算 RoPE 频率"""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """应用 Rotary Position Embedding"""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs.unsqueeze(0).unsqueeze(0)
    x_rotated = torch.view_as_real(x_complex * freqs).reshape(*x.shape)
    return x_rotated.type_as(x)


class LlamaAttention(nn.Module):
    """Multi-Head Attention with RoPE"""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int = None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // self.n_kv_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        q = apply_rope(q, freqs[:T])
        k = apply_rope(k, freqs[:T])

        # GQA: repeat k/v if n_rep > 1
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(causal, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(y)


class LlamaSwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network"""
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        # 对齐到 256 的倍数（Llama 惯例）
        hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LlamaBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int = None):
        super().__init__()
        self.attention_norm = LlamaRMSNorm(dim)
        self.attention = LlamaAttention(dim, n_heads, n_kv_heads)
        self.ffn_norm = LlamaRMSNorm(dim)
        self.ffn = LlamaSwiGLU(dim)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), freqs)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LlamaModel(nn.Module):
    def __init__(self, vocab_size: int = 32000, dim: int = 768,
                 n_layers: int = 12, n_heads: int = 12,
                 n_kv_heads: int = None, max_seq_len: int = 2048):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            LlamaBlock(dim, n_heads, n_kv_heads) for _ in range(n_layers)
        ])
        self.norm = LlamaRMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.freqs = precompute_rope_freqs(dim // n_heads, max_seq_len)

    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None):
        x = self.tok_embeddings(input_ids)
        freqs = self.freqs.to(x.device)
        for layer in self.layers:
            x = layer(x, freqs)
        x = self.norm(x)
        logits = self.output(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ════════════════════════════════════════════════════════════════════
#  初始化方法
# ════════════════════════════════════════════════════════════════════

def apply_default_llama_init(model: LlamaModel):
    """Llama 默认初始化"""
    for name, p in model.named_parameters():
        if p.dim() > 1:
            nn.init.normal_(p, mean=0.0, std=0.02)


def apply_xavier_llama_init(model: LlamaModel):
    for name, p in model.named_parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def apply_ramanujan_llama_init(model: LlamaModel, quantization: str = 'none'):
    """Ramanujan 初始化（适配 Llama 结构）"""
    # Tag roles
    for block in model.layers:
        tag_linear_role(block.attention.wq, LayerRole.Q_PROJ)
        tag_linear_role(block.attention.wk, LayerRole.K_PROJ)
        tag_linear_role(block.attention.wv, LayerRole.V_PROJ)
        tag_linear_role(block.attention.wo, LayerRole.OUTPUT_ATTN)
        tag_linear_role(block.ffn.w1, LayerRole.FFN_UP)
        tag_linear_role(block.ffn.w2, LayerRole.FFN_DOWN)
        tag_linear_role(block.ffn.w3, LayerRole.FFN_UP)

    initializer = RamanujanInitializer(
        num_layers=len(model.layers),
        nonlinearity='silu',
        quantization=quantization,
    )
    initializer.apply(model)


# ════════════════════════════════════════════════════════════════════
#  训练与评估
# ════════════════════════════════════════════════════════════════════

def generate_data(num_tokens: int = 50000, vocab_size: int = 32000,
                  seq_len: int = 128) -> List[torch.Tensor]:
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


def train_llama(method: str, n_layers: int = 12, dim: int = 512,
                n_heads: int = 8, num_epochs: int = 20,
                batch_size: int = 4, lr: float = 3e-4,
                data: List[torch.Tensor] = None) -> Dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LlamaModel(
        vocab_size=32000, dim=dim, n_layers=n_layers, n_heads=n_heads
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    参数量: {num_params / 1e6:.1f}M")

    # 初始化
    if method == 'default':
        apply_default_llama_init(model)
    elif method == 'xavier':
        apply_xavier_llama_init(model)
    elif method == 'ramanujan':
        apply_ramanujan_llama_init(model)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    losses = []
    start = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        steps = 0

        indices = torch.randperm(len(data))
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            if len(batch_idx) < batch_size:
                continue

            batch = torch.stack([data[j] for j in batch_idx]).to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]

            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        avg = epoch_loss / max(steps, 1)
        losses.append(avg)
        ppl = math.exp(min(avg, 20))
        elapsed = time.time() - start
        print(f"    Epoch {epoch+1:2d}: loss={avg:.4f}, ppl={ppl:.2f}, "
              f"time={elapsed:.1f}s")

    return {
        'method': method,
        'losses': losses,
        'final_loss': losses[-1] if losses else 0,
        'final_ppl': math.exp(min(losses[-1], 20)) if losses else 0,
        'wall_time': time.time() - start,
    }


def run_llama_validation(n_layers_list: List[int] = None,
                          num_epochs: int = 20,
                          methods: List[str] = None):
    """运行 Llama 验证"""
    if n_layers_list is None:
        n_layers_list = [12, 24, 48]
    if methods is None:
        methods = ['default', 'xavier', 'ramanujan']

    print("=" * 70)
    print("  Llama-style Transformer 验证")
    print("  Pre-RMSNorm + SwiGLU + RoPE + GQA")
    print("=" * 70)

    data = generate_data(num_tokens=30000, vocab_size=32000, seq_len=128)
    print(f"  训练序列数: {len(data)}")

    all_results = {}

    for n_layers in n_layers_list:
        dim = 512 if n_layers <= 24 else 384  # 节省内存
        n_heads = 8

        print(f"\n{'─' * 60}")
        print(f"  Llama: {n_layers}L / {dim}d / {n_heads}H")
        print(f"{'─' * 60}")

        for method in methods:
            print(f"\n  方法: {method}")
            result = train_llama(method, n_layers=n_layers, dim=dim,
                                n_heads=n_heads, num_epochs=num_epochs,
                                batch_size=4, data=data)
            all_results[f"{method}_L{n_layers}"] = result

    # 汇总
    print(f"\n{'=' * 70}")
    print("  汇总")
    print(f"{'=' * 70}")
    print(f"  {'配置':>20s} {'方法':>12s} {'最终Loss':>12s} {'PPL':>10s}")
    print(f"  {'─' * 56}")

    for n_layers in n_layers_list:
        for method in methods:
            key = f"{method}_L{n_layers}"
            r = all_results[key]
            print(f"  {f'L{n_layers}':>20s} {method:>12s} "
                  f"{r['final_loss']:12.4f} {r['final_ppl']:10.2f}")

    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Llama 初始化验证')
    parser.add_argument('--layers', type=int, nargs='+', default=[12, 24, 48])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--methods', nargs='+', default=['default', 'xavier', 'ramanujan'])

    args = parser.parse_args()
    run_llama_validation(args.layers, args.epochs, args.methods)
