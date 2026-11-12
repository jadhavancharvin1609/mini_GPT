"""Train the small character-level GPT on a plain-text file."""

import argparse
import random
from pathlib import Path

import torch

from model import CharTokenizer, GPT, GPTConfig, config_to_dict


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/input.txt"), help="UTF-8 training text")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/gpt.pt"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.block_size < 1:
        parser.error("steps, batch-size, and block-size must be positive")
    if not args.data.is_file():
        parser.error(f"Training file not found: {args.data}. Pass --data PATH to a UTF-8 text file.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    text = args.data.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    minimum_chars = 2 * (args.block_size + 1)
    if len(token_ids) < minimum_chars:
        parser.error(f"Training text must contain at least 2 * (block-size + 1) ({minimum_chars}) characters")
    split = max(1, int(0.9 * len(token_ids)))
    split = min(split, len(token_ids) - 1)
    if len(token_ids) - split <= args.block_size:
        split = len(token_ids) - args.block_size - 1
    train_ids, val_ids = token_ids[:split], token_ids[split:]

    def get_batch(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(len(source) - args.block_size, (args.batch_size,))
        x = torch.stack([source[i : i + args.block_size] for i in starts])
        y = torch.stack([source[i + 1 : i + args.block_size + 1] for i in starts])
        return x.to(device), y.to(device)

    config = GPTConfig(
        vocab_size=len(tokenizer.characters), block_size=args.block_size, n_embd=args.n_embd,
        n_head=args.n_head, n_layer=args.n_layer, dropout=args.dropout,
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    @torch.no_grad()
    def estimate_loss() -> dict[str, float]:
        model.eval()
        result = {}
        for name, source in (("train", train_ids), ("val", val_ids)):
            losses = torch.zeros(args.eval_batches)
            for i in range(args.eval_batches):
                x, y = get_batch(source)
                _, loss = model(x, y)
                losses[i] = loss.item()
            result[name] = losses.mean().item()
        model.train()
        return result

    model.train()
    for step in range(args.steps):
        if step % args.eval_interval == 0 or step == args.steps - 1:
            losses = estimate_loss()
            print(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
        x, y = get_batch(train_ids)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_config": config_to_dict(config), "model_state": model.state_dict(),
         "vocabulary": tokenizer.characters, "training_data": str(args.data)},
        args.out,
    )
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
