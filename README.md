# GPT-2 do zero

🇺🇸 [English version](README.en.md)

Implementação, a partir do zero, do GPT-2 de 124 milhões de parâmetros, seguindo o vídeo [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU), de Andrej Karpathy. O código foi escrito etapa por etapa, com um commit a cada etapa, e conferido com o repositório de referência [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt).

Trabalho desenvolvido para a disciplina **Grandes Modelos de Linguagem** (CESAR School, 2026.2).

## Resultado do treino final

| Item | Valor |
|---|---|
| GPU | 1 × NVIDIA A100 (Google Colab) |
| Tokens de treino | ≈ 900 milhões (9 arquivos do FineWeb-Edu) |
| Passos | 1.716 (524.288 tokens por passo) |
| Tempo | ≈ 82 minutos (≈ 183 mil tokens/s) |
| Loss de validação | 10,95 → **3,74** |
| Referência: GPT-2 124M original (OpenAI) | 3,29 |

Exemplo de texto gerado no último passo, a partir de "Hello, I'm a language model,":

> Hello, I'm a language model, I use it to describe the word but I can put it into an input-based form for a new word. This

## Arquivos

| Arquivo | Função |
|---|---|
| `train_gpt2.py` | Arquitetura do modelo, carregamento de dados e laço de treino |
| `fineweb.py` | Download e tokenização do conjunto de dados FineWeb-Edu |
| `hellaswag.py` | Avaliação HellaSwag (presente no código, mas desligada) |
| `input.txt` | Tiny Shakespeare, usado nos primeiros testes |
| `rodar_colab.ipynb` | Notebook para rodar o treino na GPU A100 do Google Colab |
| `gpt2-do-zero.ipynb` | Cópia do notebook executado no treino final, com as saídas gravadas |
| `log/log.txt` | Losses de treino e validação do treino final |
| `requirements.txt` | Dependências do projeto |

## Como executar

### 1. Instalar as dependências

```bash
pip install -r requirements.txt
```

Para usar a GPU, o PyTorch precisa ser a versão com CUDA. Exemplo com CUDA 12.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

No Windows, o `torch.compile` também exige o Triton:

```bash
pip install triton-windows
```

### 2. Preparar os dados

```bash
python fineweb.py
```

O script lê a amostra `sample-10BT` do FineWeb-Edu em streaming e grava 10 arquivos de 100 milhões de tokens (cerca de 2 GB) na pasta `edu_fineweb10B/`. O primeiro arquivo é usado para validação e os outros nove para treino.

### 3. Treinar

Com uma GPU:

```bash
python train_gpt2.py
```

Com várias GPUs na mesma máquina (exemplo com 8):

```bash
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```

Os principais parâmetros ficam no início do laço de treino em `train_gpt2.py`:

| Parâmetro | Valor no repositório | Treino final (A100) |
|---|---|---|
| `B` (micro lote) | 4 | 16 |
| `T` (contexto) | 1024 | 1024 |
| `warmup_steps` | 10 | 64 |
| `max_steps` | 50 | 1716 |
| `use_compile` | True | True |
| `use_hellaswag` | False | False |

O valor de `B` deve caber na memória da GPU. Na RTX 4060 (8 GB), o maior valor que funcionou foi `B = 4`. O tamanho total do lote (524.288 tokens) é mantido pela acumulação de gradientes, qualquer que seja o `B`.

### 4. Treinar no Google Colab

O notebook `rodar_colab.ipynb` monta o Google Drive, clona este repositório, copia os dados já tokenizados do Drive, ajusta `B`, `warmup_steps` e `max_steps` para o treino final e salva a pasta `log/` de volta no Drive ao terminar.

## Diferenças em relação ao vídeo

- **Escala:** 1 GPU, ≈ 900 milhões de tokens e 1.716 passos, contra 8 GPUs, 10 bilhões de tokens e 19.073 passos no vídeo.
- **`torch.compile`:** ligado, com a geração de texto ativa. No código final do vídeo ele fica desligado.
- **HellaSwag:** desligado (`use_hellaswag = False`), porque o repositório de origem dos dados está bloqueado no GitHub. O código da avaliação foi mantido.
- **`fineweb.py`:** leitura em streaming, limitada a 10 arquivos, e bloco `if __name__ == "__main__":`, exigido pelo `multiprocessing` no Windows.

## Referências

- KARPATHY, A. *Let's reproduce GPT-2 (124M)*. 2024. Vídeo. https://www.youtube.com/watch?v=l8pRSuU81PU
- KARPATHY, A. *build-nanogpt*. GitHub, 2024. https://github.com/karpathy/build-nanogpt
- RADFORD, A. et al. *Language models are unsupervised multitask learners*. OpenAI, 2019.
- VASWANI, A. et al. *Attention is all you need*. 2017. https://arxiv.org/abs/1706.03762
- PENEDO, G. et al. *FineWeb-Edu*. Hugging Face, 2024. https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu