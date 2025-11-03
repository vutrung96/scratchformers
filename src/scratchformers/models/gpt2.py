import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors import safe_open


@dataclass
class GPT2Config:
    d_model: int
    n_layers: int
    n_heads: int
    seq_len: int
    vocab_size: int


class NewGELUActivation(nn.Module):
    """
    Adapted from transformers to reduce numerical differences.

    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://huggingface.co/papers/1606.08415
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0)))
            )
        )


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
        self.gelu = NewGELUActivation()

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
        causal_mask = (torch.tril(torch.ones(b, self.n_heads, s, s)) == 0).to(0)

        # attn: b x n_h x s x s
        attn = torch.softmax(
            ((q @ torch.transpose(k, -1, -2)) / np.sqrt(d_head)).masked_fill(
                causal_mask, -float("inf")
            ),
            -1,
        )

        # attn: b x n_h x s x s
        # v: b x n_h x s x d_h
        # o: b x n_h x s x d_h -> b x s x n_h x d_h -> b x s d
        o = torch.transpose(attn @ v, 1, 2).reshape(b, s, d)
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
        x = x + self.mha(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """
    Implementation of a GPT2 model.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.seq_len, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        self.seq_len = config.seq_len

    def forward(self, x):
        """
        b: batch size
        s: sequence length
        d: model dim
        v: vocab size
        """
        # x: b x s
        b, s = x.shape
        assert s <= self.seq_len, (
            f"Input sequence length cannot exceed cfg.seq_len: {self.seq_len}!"
        )

        pe = self.wpe(torch.arange(s).to(0)).unsqueeze(0).expand(b, s, -1)

        # x: b x s x d
        x = self.wte(x) + pe
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # logits: b x s x v
        logits = self.lm_head(x)
        return logits


def create_gpt2_from_hf_weights():
    """
    Create gpt2 from HF weights.

    Maps HuggingFace GPT2 weight names to the custom model's naming convention:
    - h.{i} -> blocks.{i}
    - h.{i}.attn -> blocks.{i}.mha
    - h.{i}.mlp.c_fc -> blocks.{i}.mlp.up_proj
    - h.{i}.mlp.c_proj -> blocks.{i}.mlp.down_proj
    """
    cfg = GPT2Config(d_model=768, n_heads=12, n_layers=12, seq_len=1024, vocab_size=50257)
    gpt = GPT2(cfg)

    st_file = hf_hub_download("openai-community/gpt2", filename="model.safetensors")
    hf_tensors = {}
    with safe_open(st_file, framework="pt") as f:
        for k in f.keys():
            hf_tensors[k] = f.get_tensor(k)

    # Get model's state dict
    model_state_dict = gpt.state_dict()

    # Create mapping from HF keys to model keys
    mapped_weights = {}

    for hf_key, hf_tensor in hf_tensors.items():
        # Skip the attention bias (causal mask buffer, not a learnable parameter)
        # This is h.{i}.attn.bias but NOT h.{i}.attn.c_attn.bias or h.{i}.attn.c_proj.bias
        # We check that there's no 'c_' before 'attn.bias'
        if hf_key.endswith("attn.bias") and not hf_key.endswith("c_attn.bias"):
            continue

        # Map HF key to model key
        model_key = hf_key
        model_key = model_key.replace("h.", "blocks.")
        model_key = model_key.replace(".attn.", ".mha.")
        model_key = model_key.replace(".mlp.c_fc.", ".mlp.up_proj.")
        model_key = model_key.replace(".mlp.c_proj.", ".mlp.down_proj.")

        # Check if key exists in model
        if model_key not in model_state_dict:
            print(f"Warning: Key {model_key} (from HF key {hf_key}) not found in model")
            continue

        def should_transpose_weights(model_key):
            # We need to transpose these weights because HF GPT2 uses Conv1D whereas
            # we use Linear, and the weights are transposed.
            if not model_key.endswith(".weight"):
                return False
            transposed_weights_patterns = ["up_proj", "down_proj", "c_proj", "c_attn"]
            for pattern in transposed_weights_patterns:
                if pattern in model_key:
                    return True
            return False

        if should_transpose_weights(model_key):
            hf_tensor = torch.transpose(hf_tensor, 0, 1)

        # Check if shapes already match
        if hf_tensor.shape != model_state_dict[model_key].shape:
            raise ValueError(
                f"Shape mismatch for {model_key}: "
                f"HF shape {hf_tensor.shape} vs Model shape {model_state_dict[model_key].shape}"
            )
        mapped_weights[model_key] = hf_tensor

    # Load the mapped weights into the model
    gpt.load_state_dict(mapped_weights, strict=False)

    print(f"Successfully loaded {len(mapped_weights)} tensors into the model")
    print(f"Model has {len(model_state_dict)} parameters total")

    # Report any missing keys
    missing_keys = set(model_state_dict.keys()) - set(mapped_weights.keys())
    if missing_keys:
        print(f"Missing keys in HF weights (not loaded): {missing_keys}")

    return gpt
