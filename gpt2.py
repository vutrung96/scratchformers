from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class GPT2Config:
    d_model: int
    n_layers: int
    n_heads: int
    seq_len: int
    vocab_size: int


class MLP(nn.Module):
    """
    Implementation of an MLP layer

    b: batch size
    s: sequence length
    d: embedding dimension
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.up_proj = nn.Linear(self.d_model, 4 * self.d_model)
        self.down_proj = nn.Linear(4 * self.d_model, self.d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.down_proj(self.gelu(self.up_proj(x)))


class MultiHeadAttention(nn.Module):
    """
    Implementation of Multi Head Attention.

    b: batch size
    s: sequence length
    d: embedding dimension
    d_h: single head embedding dimension
    n_h: number of heads
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads

        self.c_attn = nn.Linear(self.d_model, 3 * self.d_model)
        self.c_proj = nn.Linear(self.d_model, self.d_model)

        assert self.d_model % self.n_heads == 0, (
            "The embedding dimension must divide the number of attention heads"
        )

    def forward(self, x):
        # x: b x s x d
        b, s, d = x.shape

        # qkv: b x s x 3*d
        qkv = self.c_attn(x).reshape(b, s, 3, self.d_model)
        assert qkv.shape == (b, s, 3, d)

        # q, k, v: b x n_h x s x d_h
        d_head = self.d_model // self.n_heads
        q = qkv[:, :, 0, :].reshape(b, s, self.n_heads, d_head).transpose(1, 2)
        k = qkv[:, :, 1, :].reshape(b, s, self.n_heads, d_head).transpose(1, 2)
        v = qkv[:, :, 2, :].reshape(b, s, self.n_heads, d_head).transpose(1, 2)
        causal_mask = torch.tril(torch.ones(b, self.n_heads, s, s)) == 0

        # attn: b x n_h x s x s
        attn = torch.softmax(
            ((q @ torch.transpose(k, -1, -2)) / np.sqrt(d_head)).masked_fill(
                causal_mask, -float("inf")
            ),
            -1,
        )
        assert torch.allclose(torch.sum(attn, -1), torch.ones(b, self.n_heads, s))

        # o: b x s x d
        o = (attn @ v).reshape(b, s, d)
        o_proj = self.c_proj(o)
        assert o_proj.shape == (b, s, d)
        return o_proj


class Block(nn.Module):
    """
    Implementation of a GPT2 block.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.mha = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x += self.mha(self.ln_1(x))
        x += self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """
    Implementation of a GPT2 model.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.d_model)
        self.blocks = [Block(config) for _ in range(config.n_layers)]
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.seq_len, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

    def forward(self, x):
        """
        b: batch size
        s: sequence length
        d: model dim
        v: vocab size
        """
        # x: b x s
        b, s = x.shape
        assert s <= cfg.seq_len, f"Input sequence length cannot exceed cfg.seq_len: {cfg.seq_len}!"

        pe = self.wpe(torch.arange(s)).unsqueeze(0).expand(b, s, -1)
        
        # x: b x s x d
        x = self.wte(x) + pe
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # logits: b x s x v
        logits = self.lm_head(x)
        return logits


d_model = 768
n_heads = 12
n_layers = 12
seq_len = 1024
vocab_size = 50257

emb = torch.rand((2, 10, 768))
cfg = GPT2Config(d_model=d_model, n_heads=12, n_layers=12, seq_len=1024, vocab_size=50257)
gpt = GPT2(cfg)
output = gpt(torch.tensor([[0, 1, 2]]))
print(output.shape)
# breakpoint()
