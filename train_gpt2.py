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

    def forward(self, idx):
        # idx tem formato (B, T): lote de sequências de índices de tokens
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # embeddings de posição e de token
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # formato (T)
        pos_emb = self.transformer.wpe(pos)  # embeddings de posição, formato (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # embeddings de token, formato (B, T, n_embd)
        x = tok_emb + pos_emb                # soma: cada token sabe "o quê" e "onde"
        # passa pelos blocos Transformer
        for block in self.transformer.h:
            x = block(x)
        # LayerNorm final e camada de saída
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # formato (B, T, vocab_size)
        return logits  

    @classmethod
    def from_pretrained(cls, model_type):
        """Carrega os pesos do GPT-2 treinado da Hugging Face"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head e n_embd dependem do tamanho do modelo
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M parâmetros
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M parâmetros
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M parâmetros
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M parâmetros
        }[model_type]
        config_args['vocab_size'] = 50257  # sempre 50257 nos modelos GPT-2
        config_args['block_size'] = 1024   # sempre 1024 nos modelos GPT-2

        # cria o modelo feito do zero (pesos aleatórios)
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # ignora a máscara (não é parâmetro)

        # carrega o modelo da Hugging Face
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copia os pesos conferindo nomes e formatos
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # máscara, ignorar
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]         # máscara, ignorar
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # a OpenAI usou "Conv1D" nessas camadas; aqui é nn.Linear,
        # então esses pesos precisam ser transpostos na cópia
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

# -----------------------------------------------------------------------------
# detecta automaticamente o dispositivo: usa a GPU (cuda) se existir, senão a cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"usando o dispositivo: {device}")

# pega um lote de dados
import tiktoken
enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
text = text[:1000]                  # só os primeiros 1000 caracteres
tokens = enc.encode(text)
B, T = 4, 32                        # 4 sequências de 32 tokens
buf = torch.tensor(tokens[:B*T + 1])
x = buf[:-1].view(B, T)             # entradas
y = buf[1:].view(B, T)              # respostas esperadas (próximo token)

# calcula os logits
model = GPT(GPTConfig())
model.to(device)
logits = model(x.to(device))

print(logits.shape)
import sys; sys.exit(0)             

# prefix tokens
model.eval()                        # modo de uso (não de treino)
num_return_sequences = 5   # quantas frases gerar
max_length = 30            # tamanho máximo de cada frase (em tokens)

# tokens do início da frase (prefixo)
import tiktoken
enc = tiktoken.get_encoding('gpt2')                       # tokenizador do GPT-2
tokens = enc.encode("Hello, I'm a language model,")       # texto -> lista de 8 números
tokens = torch.tensor(tokens, dtype=torch.long)           # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8): a mesma frase 5 vezes
x = tokens.to(device)

# geração de texto. neste momento x tem formato (B, T), com B = 5 e T = 8
# fixa a semente aleatória em 42 (resultados repetíveis)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # passa x pelo modelo para obter os logits
    with torch.no_grad():
        logits = model(x)  # (B, T, vocab_size)
        # pega só os logits da última posição
        logits = logits[:, -1, :]  # (B, vocab_size)
        # transforma em probabilidades
        probs = F.softmax(logits, dim=-1)
        # amostragem top-k com k = 50 (padrão do pipeline da Hugging Face)
        # topk_probs fica (5, 50), topk_indices fica (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # sorteia um token entre os 50 mais prováveis
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        # pega o índice real do token sorteado
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # acrescenta o token no fim da sequência
        x = torch.cat((x, xcol), dim=1)

# imprime o texto gerado
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()   # tokens da frase i
    decoded = enc.decode(tokens)          # números -> texto
    print(">", decoded)