# GPT-2 from scratch

🇧🇷 [Versão em português](README.md)

From-scratch implementation of the 124-million-parameter GPT-2, following Andrej Karpathy's video [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU). The code was written step by step, with one commit per step, and checked against the reference repository [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt).

Developed for the **Large Language Models** course (CESAR School, 2026.2).

## Final training run

| Item | Value |
|---|---|
| GPU | 1 × NVIDIA A100 (Google Colab) |
| Training tokens | ≈ 900 million (9 FineWeb-Edu shards) |
| Steps | 1,716 (524,288 tokens per step) |
| Time | ≈ 82 minutes (≈ 183k tokens/s) |
| Validation loss | 10.95 → **3.74** |
| Reference: original GPT-2 124M (OpenAI) | 3.29 |

Sample generated at the last step, from the prompt "Hello, I'm a language model,":

> Hello, I'm a language model, I use it to describe the word but I can put it into an input-based form for a new word. This

## Files

| File | Purpose |
|---|---|
| `train_gpt2.py` | Model architecture, data loading and training loop |
| `fineweb.py` | Download and tokenization of the FineWeb-Edu dataset |
| `hellaswag.py` | HellaSwag evaluation (present in the code, but turned off) |
| `input.txt` | Tiny Shakespeare, used in the first tests |
| `rodar_colab.ipynb` | Notebook to run training on the Google Colab A100 GPU |
| `gpt2-do-zero.ipynb` | Copy of the notebook used for the final run, with cell outputs saved |
| `log/log.txt` | Training and validation losses from the final run |
| `requirements.txt` | Project dependencies |

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

To use the GPU, PyTorch must be the CUDA build. Example with CUDA 12.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

On Windows, `torch.compile` also requires Triton:

```bash
pip install triton-windows
```

### 2. Prepare the data

```bash
python fineweb.py
```

The script streams the `sample-10BT` subset of FineWeb-Edu and writes 10 shards of 100 million tokens each (about 2 GB) to `edu_fineweb10B/`. The first shard is used for validation and the other nine for training.

### 3. Train

On a single GPU:

```bash
python train_gpt2.py
```

On several GPUs in the same machine (example with 8):

```bash
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```

The main parameters are set right before the training loop in `train_gpt2.py`:

| Parameter | Value in the repository | Final run (A100) |
|---|---|---|
| `B` (micro-batch) | 4 | 16 |
| `T` (context) | 1024 | 1024 |
| `warmup_steps` | 10 | 64 |
| `max_steps` | 50 | 1716 |
| `use_compile` | True | True |
| `use_hellaswag` | False | False |

`B` must fit in GPU memory. On an RTX 4060 (8 GB), the largest value that worked was `B = 4`. The total batch size (524,288 tokens) is kept by gradient accumulation, whatever the value of `B`.

### 4. Train on Google Colab

The `rodar_colab.ipynb` notebook mounts Google Drive, clones this repository, copies the already tokenized data from Drive, sets `B`, `warmup_steps` and `max_steps` for the final run, and saves the `log/` folder back to Drive when training ends.

## Differences from the video

- **Scale:** 1 GPU, ≈ 900 million tokens and 1,716 steps, versus 8 GPUs, 10 billion tokens and 19,073 steps in the video.
- **`torch.compile`:** turned on, with text generation active. In the video's final code it is turned off.
- **HellaSwag:** turned off (`use_hellaswag = False`) because the source repository of the data is blocked on GitHub. The evaluation code was kept.
- **`fineweb.py`:** streaming download limited to 10 shards, plus an `if __name__ == "__main__":` block, required by `multiprocessing` on Windows.

## References

- KARPATHY, A. *Let's reproduce GPT-2 (124M)*. 2024. Video. https://www.youtube.com/watch?v=l8pRSuU81PU
- KARPATHY, A. *build-nanogpt*. GitHub, 2024. https://github.com/karpathy/build-nanogpt
- RADFORD, A. et al. *Language models are unsupervised multitask learners*. OpenAI, 2019.
- VASWANI, A. et al. *Attention is all you need*. 2017. https://arxiv.org/abs/1706.03762
- PENEDO, G. et al. *FineWeb-Edu*. Hugging Face, 2024. https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu