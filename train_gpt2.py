from dataclasses import dataclass
import math
import os
# import sys
import time
import inspect
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
from hellaswag import render_example, iterate_examples

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # Q, K, V juntos
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)      # projeção de saída
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # máscara triangular: posição t só enxerga posições <= t
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     #.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # lote, nº de tokens, tamanho do vetor
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, tamanho_da_cabeça)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # notas de atenção: Q·Kᵀ / raiz(d) -  atenção "manual" (cria a matriz (T, T) inteira para todas as queries e keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # média ponderada dos valores
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        
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
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
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

        # compartilhamento de pesos: a entrada (wte) e a saída (lm_head) usam a mesma matriz
        self.transformer.wte.weight = self.lm_head.weight

        # inicializa os parâmetros
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # camadas marcadas (c_proj): desvio menor
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
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
        loss= None
        if targets is not None:
            # cross-entropy: compara as notas de cada posição com o token correto
            # (B, T, vocab_size) -> (B*T, vocab_size)   e   (B, T) -> (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss 
    
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

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # todos os parâmetros que precisam de gradiente
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # grupos do otimizador: parâmetros 2D (pesos das matrizes e embeddings) recebem weight decay;
        # os demais (bias e LayerNorm) não recebem
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # usa a versão "fused" do AdamW se estiver disponível (mais rápida na GPU)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type =='cuda'

        if master_process:
            print(f"using fused AdamW: {use_fused}")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)   
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

def get_most_likely_row(tokens, mask, logits):
    # loss em todas as posições
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # loss média só na região do final (onde mask == 1)
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # o final com a menor loss é o mais provável
    pred_norm = avg_loss.argmin().item()
    return pred_norm

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T

        self.process_rank = process_rank      # número deste processo (GPU)
        self.num_processes = num_processes    # total de processos (GPUs)
        assert split in {'train', 'val'}

        # ao iniciar, lê os tokens do disco e guarda na memória
        #with open('input.txt', 'r') as f:
            # text = f.read()
        # enc = tiktoken.get_encoding('gpt2')
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)
        # print(f"loaded {len(self.tokens)} tokens")
        # print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # estado: posição atual no texto
        # self.current_position = 0

        # estado: cada processo começa num ponto diferente do texto
        # self.current_position = self.B * self.T * self.process_rank

        # lista os arquivos de dados (shards)
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()


    def reset(self):
        # estado: começa no primeiro arquivo
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)  # entradas
        y = (buf[1:]).view(B, T)   # respostas esperadas

        # avança a posição no texto
        # self.current_position += B * T
        # se o próximo lote passar do fim do texto, volta ao começo
        # if self.current_position + (B * T + 1) > len(self.tokens):
            # self.current_position = 0
        # return x, y

        # avança a posição, pulando os pedaços dos outros processos
        # self.current_position += B * T * self.num_processes
        # se o próximo lote passar do fim do texto, volta ao começo
        # if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            # self.current_position = self.B * self.T * self.process_rank
        # return x, y

        # avança a posição no arquivo
        self.current_position += B * T * self.num_processes
        # se o próximo lote passar do fim do arquivo, vai para o próximo arquivo
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# execução simples:
# python train_gpt2.py
# execução DDP, por exemplo com 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py - USAR APENAS SE A MÁQUINA TIVER MAIS DE UMA GPU

# roda o treino
from torch.distributed import init_process_group, destroy_process_group

# configura o DDP (distributed data parallel)
# o comando torchrun define as variáveis de ambiente RANK, LOCAL_RANK e WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1  # é uma execução DDP?
if ddp:
    # o DDP, por enquanto, exige CUDA; o dispositivo é escolhido de acordo com o rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])               # número do processo (0, 1, 2...)
    ddp_local_rank = int(os.environ['LOCAL_RANK'])   # número da GPU nesta máquina
    ddp_world_size = int(os.environ['WORLD_SIZE'])   # total de processos (GPUs)
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0                   # este processo cuida de logs, checkpoints etc.
else:
    # execução normal, sem DDP (uma GPU só)
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # detecta o dispositivo automaticamente
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"    

# -----------------------------------------------------------------------------
# detecta automaticamente o dispositivo: usa a GPU (cuda) se existir, senão a cpu
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"usando o dispositivo: {device}")

# fixa a semente aleatória (resultados repetíveis)
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288   # 2**19, cerca de 0,5 milhão de tokens por atualização dos pesos
B = 4                       # micro lote (no vídeo: 16)
T = 1024                    # tamanho da sequência

# assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
# grad_accum_steps = total_batch_size // (B * T)

assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# train_loader = DataLoaderLite(B=B, T=T)
# train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

# train_loader = DataLoaderLite(B=4, T=32)     # lotes de 4 sequências de 32 tokens
# train_loader = DataLoaderLite(B=16, T=1024)  # lotes de 16 sequências de 1024 tokens
# train_loader = DataLoaderLite(B=2, T=1024)   # lotes de 2 sequências de 1024 tokens
# train_loader = DataLoaderLite(B=4, T=1024)   # lotes de 4 sequências de 1024 tokens
# train_loader = DataLoaderLite(B=8, T=1024)   # lotes de 8 sequências de 1024 tokens

# usa TF32 nas multiplicações de matrizes (mais rápido, com precisão um pouco menor)
torch.set_float32_matmul_precision('high')

# cria o modelo
# model = GPT(GPTConfig())
model = GPT(GPTConfig(vocab_size=50304))    # vocabulário arredondado para um "nice number"
model.to(device)
# model = torch.compile(model)              # compila o modelo: junta operações e reduz idas e voltas à memória da GPU

use_hellaswag = False                       # a fonte original do HellaSwag (github.com/rowanz/hellaswag) foi bloqueada por DMCA

# use_compile = False                       # torch.compile atrapalha a HellaSwag e a geração de texto (no código original: TODO fix)

use_compile = True                          # otimizar, já que não está usando o HellaSwag
if use_compile:
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # sempre guarda o modelo "puro", sem o DDP

# taxa de aprendizado variável: comneça pequena, sobe (aquecimento) e cai devagar (queda em cosseno)
# max_lr = 3e-4             # taxa de aprendizado máxima
max_lr = 6e-4               # taxa de aprendizado máxima
min_lr = max_lr * 0.1       # taxa mínima: 10% da máxima
warmup_steps = 10           # passos de aquecimento
max_steps = 50              # total de passos

# warmup_steps = 715         # passos de aquecimento
# max_steps = 19073          # total de passos - ≈ 10 bilhões de tokens ÷ 524.288 por passo

# warmup_steps = 64          # passos de aquecimento - mesma proporção do vídeo (715 de 19073 ≈ 3,75%)
# max_steps = 1716           # total de passos - ≈ 900 milhões de tokens ÷ 524.288 por passo

def get_lr(it):
    # 1) aquecimento linear: sobe de quase 0 até max_lr nos primeiros warmup_steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) depois de max_steps, fica na taxa mínima
    if it > max_steps:
        return min_lr
    # 3) no meio, desce em formato de cosseno até min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # vai de 1 até 0
    return min_lr + coeff * (max_lr - min_lr)

# otimização

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # otimizador AdamW, taxa de aprendizado 0,0003
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)  # otimizador AdamW, taxa de aprendizado 0,0003
# optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device_type)

# cria a pasta onde ficam os logs e os checkpoints
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:  # abre para escrita, para limpar o arquivo
    pass

# for i in range(50):
for step in range(max_steps):
    t0 = time.time()                                                    # marca o início do passo
    last_step = (step == max_steps - 1)                                 # True só no último passo

    # de tempos em tempos, avalia a loss de validação
    # if step % 100 == 0:
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # salva um checkpoint do modelo
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, checkpoint_path)

    # de tempos em tempos, avalia a HellaSwag
    if (step % 250 == 0 or last_step) and (not use_compile) and use_hellaswag:
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # cada processo avalia só os exemplos em que i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # transforma o exemplo em tokens e rótulo
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # calcula os logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # junta os resultados de todos os processos
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # de tempos em tempos, gera texto com o modelo (exceto no passo 0, que é só ruído)
    # if step > 0 and step % 100 == 0:
    # if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
    if ((step > 0 and step % 250 == 0) or last_step):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # passa pelo modelo para obter os logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)  # (B, T, vocab_size)
                # pega só os logits da última posição
                logits = logits[:, -1, :]   # (B, vocab_size)
                # transforma em probabilidades
                probs = F.softmax(logits, dim=-1)
                # amostragem top-k com k = 50
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # sorteia um token entre os 50 mais prováveis
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                # pega o índice real do token sorteado
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                # acrescenta o token no fim da sequência
                xgen = torch.cat((xgen, xcol), dim=1)
        # imprime o texto gerado
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # laço de treino
    model.train()
    optimizer.zero_grad() 
    loss_accum = 0.0                                                        # zera os gradientes do passo anterior

    # soma os gradientes de vários micro lotes antes de atualizar os pesos
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()                                    # pega o próximo lote do texto
        x, y = x.to(device), y.to(device)                                   # envia o lote para a GPU
    
        # bfloat16: parte das contas usa números de 16 bits (mais rápido e com menos memória)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)                                      # calcula a previsão e o erro (loss)
        # divide a loss para que a soma dos gradientes vire uma média
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()

        if ddp:
            # só sincroniza os gradientes entre as GPUs no último micro lote
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

        loss.backward()                                                     # calcula os gradientes (backpropagation)
    
    if ddp:
        # média da loss entre todas as GPUs
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # limita o tamanho total dos gradientes a 1,0 (evita passos grandes demais)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # define a taxa de aprendizado deste passo
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()                                           # atualiza os pesos
    torch.cuda.synchronize()                                   # espera a GPU terminar o trabalho
    t1 = time.time()                                           # marca o fim do passo
    dt = (t1 - t0)                                             # diferença de tempo em segundos

    # tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)      # tokens processados por segundo
    # tokens_processed = train_loader.B * train_loader.T * grad_accum_steps

    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed/ dt
    
    # if master_process:
        # print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()

# sys.exit(0)             

# # prefix tokens
# model.eval()               # modo de uso (não de treino)
# num_return_sequences = 5   # quantas frases gerar
# max_length = 30            # tamanho máximo de cada frase (em tokens)

# # tokens do início da frase (prefixo)
# enc = tiktoken.get_encoding('gpt2')                           # tokenizador do GPT-2
# tokens = enc.encode("Hello, I'm a language model,")           # texto -> lista de 8 números
# tokens = torch.tensor(tokens, dtype=torch.long)               # (8,)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8): a mesma frase 5 vezes
# x = tokens.to(device)

# # geração de texto. neste momento x tem formato (B, T), com B = 5 e T = 8
# # fixa a semente aleatória em 42 (resultados repetíveis)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     # passa x pelo modelo para obter os logits
#     with torch.no_grad():
#         logits = model(x)  # (B, T, vocab_size)
#         # pega só os logits da última posição
#         logits = logits[:, -1, :]  # (B, vocab_size)
#         # transforma em probabilidades
#         probs = F.softmax(logits, dim=-1)
#         # amostragem top-k com k = 50 (padrão do pipeline da Hugging Face)
#         # topk_probs fica (5, 50), topk_indices fica (5, 50)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         # sorteia um token entre os 50 mais prováveis
#         ix = torch.multinomial(topk_probs, 1)  # (B, 1)
#         # pega o índice real do token sorteado
#         xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
#         # acrescenta o token no fim da sequência
#         x = torch.cat((x, xcol), dim=1)

# # imprime o texto gerado
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()   # tokens da frase i
#     decoded = enc.decode(tokens)          # números -> texto
#     print(">", decoded)