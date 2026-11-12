# GPT From Scratch

A compact, educational, decoder-only GPT language model implemented in PyTorch. It includes causal self-attention, positional embeddings, transformer blocks, a training loop, checkpoint saving, and text generation. The starter corpus is a tiny fictional sample so the project works without downloading a dataset.

This is a learning project, not a pretrained or production language model. It trains a character-level model from random initialization; useful language quality requires a much larger text corpus, more compute, and further engineering.

## Project files

- `model.py` — tokenizer, GPT configuration, causal attention, transformer blocks, and model generation.
- `train.py` — reads a UTF-8 text corpus, trains the model, and saves a checkpoint.
- `generate.py` — loads a checkpoint and samples text.
- `data/input.txt` — small example corpus for a quick first run.
- `requirements.txt` — Python dependency list.

## Requirements

- Python 3.10 or newer
- PyTorch 2.1 or newer

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For CUDA acceleration, install the PyTorch build that matches your CUDA version using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). The project automatically selects CUDA, Apple Silicon MPS, or CPU when available.

## Train

Run a short demo training job:

```bash
python train.py --steps 300 --block-size 64 --batch-size 16
```

The model reads `data/input.txt` by default and writes `checkpoints/gpt.pt`. To train on your own plain-text corpus:

```bash
python train.py --data path/to/your_corpus.txt --out checkpoints/my_model.pt
```

Useful options include `--steps`, `--block-size`, `--n-embd`, `--n-head`, `--n-layer`, `--learning-rate`, `--dropout`, `--eval-interval`, `--device`, and `--seed`. The embedding size must be divisible by the number of attention heads. The training corpus needs at least `2 * (block-size + 1)` characters so both training and validation can form a batch. Training on a tiny corpus is only a code demonstration and will overfit quickly.

## Generate text

After training:

```bash
python generate.py --checkpoint checkpoints/gpt.pt --prompt "Mira " --tokens 300
```

Control sampling with `--temperature` and `--top-k`; lower temperature usually makes output less varied. Generation prompts must use characters present in the training corpus vocabulary.

## How the model works

1. The character tokenizer maps each character in the training text to an integer token.
2. Token and position embeddings are added together.
3. Repeated transformer blocks apply masked self-attention and a feed-forward network, each with residual connections and layer normalization.
4. The model predicts the next character at every position. Cross-entropy loss trains those predictions against the text shifted by one character.
5. Generation repeatedly samples a next character while limiting attention to the most recent `block-size` characters.

The implementation uses PyTorch's scaled dot-product attention with a causal mask. It does not download weights, call a hosted model, or use the Hugging Face Transformers library.

## Limitations

- Character-level tokenization is simple to inspect but inefficient compared with modern subword tokenizers.
- The sample corpus is intentionally very small and is not suitable for evaluating language ability.
- The implementation omits distributed training, mixed precision, learning-rate schedules, checkpoint resume, and production safety or serving features.
