"""Generate text from a checkpoint produced by train.py."""

import argparse
from pathlib import Path

import torch

from model import CharTokenizer, GPT, GPTConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gpt.pt"))
    parser.add_argument("--prompt", default="", help="Prompt text; empty starts from a newline or first known character")
    parser.add_argument("--tokens", type=int, default=500, help="Number of new characters to sample")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}. Train a model first.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(args.checkpoint, map_location=device)
    tokenizer = CharTokenizer(checkpoint["vocabulary"])
    config = GPTConfig(**checkpoint["model_config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    prompt = args.prompt
    if not prompt:
        prompt = "\n" if "\n" in tokenizer.stoi else tokenizer.characters[0]
    try:
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    except ValueError as error:
        parser.error(str(error))
    result = model.generate(context, args.tokens, args.temperature, args.top_k)
    print(tokenizer.decode(result[0].tolist()), end="")


if __name__ == "__main__":
    main()
