from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # Q, K, V juntos
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)      # projeção de saída
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # máscara triangular: posição t só enxerga posições <= t
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # lote, nº de tokens, tamanho do vetor
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, tamanho_da_cabeça)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # notas de atenção: Q·Kᵀ / raiz(d)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v  # média ponderada dos valores
        # junta as cabeças: volta para (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # tokens trocam informação entre si
        x = x + self.mlp(self.ln_2(x))   # cada token é processado sozinho
        return x
@dataclass
class GPTConfig:
    block_size: int = 1024   # tamanho máximo do contexto (nº de tokens)
    vocab_size: int = 50257  # nº de tokens do vocabulário
    n_layer: int = 12        # nº de blocos Transformer
    n_head: int = 12         # nº de cabeças de atenção
    n_embd: int = 768        # tamanho do vetor de cada token

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),   # token -> vetor
            wpe = nn.Embedding(config.block_size, config.n_embd),   # posição -> vetor
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),                     # LayerNorm final
        ))
        # vetor -> nota (logit) para cada palavra do vocabulário
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

model = GPT(GPTConfig())
n_params = sum(p.numel() for p in model.parameters())
print(f"modelo criado: {n_params:,} parâmetros")