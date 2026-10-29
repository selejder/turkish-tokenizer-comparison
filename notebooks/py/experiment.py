"""Run the character-tokenizer and byte-level BPE comparison."""

from pathlib import Path

import torch
import torch.nn as nn

from generate import generate
from gpt import GPT
from parameters import (
    BATCH_SIZE,
    BLOCK_SIZE,
    EMBED_DIM,
    EVALUATION_STEPS,
    HEAD_SIZE,
    LEARNING_RATE,
    MAX_NEW_TOKENS,
    NUM_HEADS,
    NUM_MERGES,
    NUM_TRANSFORMER_BLOCKS,
    PROMPT,
    SAMPLE_CHARS,
    SEED,
    TRAINING_STEPS,
)
from tokenizer import BPETokenizer, CharacterTokenizer, dataset
from training import bits_per_byte, evaluation, training


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "Bianet_tr_minimal-cleaned.txt"


def build_model(vocab_size: int):
    """Build the same GPT architecture for either tokenizer."""
    return GPT(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        embed_dim=EMBED_DIM,
        head_size=HEAD_SIZE,
        num_head=NUM_HEADS,
        num_transformer=NUM_TRANSFORMER_BLOCKS,
    )


def print_loss_summary(name: str, losses: list[float], validation_loss: float):
    print(
        f"{name}: train loss {losses[0]:.4f} -> {losses[-1]:.4f}; "
        f"validation loss {validation_loss:.4f}"
    )


def make_control_characters_visible(text: str):
    """Keep generated control bytes from changing terminal behavior."""
    visible = []
    for character in text:
        if character.isprintable() or character in "\n\t":
            visible.append(character)
        else:
            visible.append(f"\\x{ord(character):02x}")
    return "".join(visible)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Clean corpus not found at {DATA_PATH}. "
            "Run corpus.py after placing the raw Bianet file in data/raw."
        )

    torch.manual_seed(SEED)
    data = DATA_PATH.read_text(encoding="utf-8")
    sample = data[:SAMPLE_CHARS]
    train_data, test_data, val_data = dataset(sample)

    # Fit both tokenizers on the same training text. Test data stays untouched.
    tokenizer_chr = CharacterTokenizer()
    tokenizer_chr.fit(train_data)
    train_ids_chr = tokenizer_chr.encode(train_data)
    val_ids_chr = tokenizer_chr.encode(val_data)

    tokenizer_bpe = BPETokenizer()
    tokenizer_bpe.fit(train_data, NUM_MERGES)
    train_ids_bpe = tokenizer_bpe.encode(train_data)
    val_ids_bpe = tokenizer_bpe.encode(val_data)

    print("=== Data and tokenizers ===")
    print(
        f"characters: train={len(train_data)}, validation={len(val_data)}, "
        f"held-out test={len(test_data)}"
    )
    print(
        f"character tokenizer: vocab={tokenizer_chr.vocab_size}, "
        f"train tokens={len(train_ids_chr)}"
    )
    print(
        f"byte-level BPE: vocab={tokenizer_bpe.vocab_size}, "
        f"train tokens={len(train_ids_bpe)}, "
        f"bytes/token={len(train_data.encode('utf-8')) / len(train_ids_bpe):.3f}"
    )

    model_chr = build_model(tokenizer_chr.vocab_size)
    model_bpe = build_model(tokenizer_bpe.vocab_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer_chr = torch.optim.Adam(model_chr.parameters(), lr=LEARNING_RATE)
    optimizer_bpe = torch.optim.Adam(model_bpe.parameters(), lr=LEARNING_RATE)

    train_generator_chr = torch.Generator().manual_seed(SEED + 1)
    train_generator_bpe = torch.Generator().manual_seed(SEED + 2)
    eval_generator_chr = torch.Generator().manual_seed(SEED + 3)
    eval_generator_bpe = torch.Generator().manual_seed(SEED + 4)

    loss_list_chr = training(
        encoded_data=train_ids_chr,
        vocab_size=tokenizer_chr.vocab_size,
        model=model_chr,
        optimizer=optimizer_chr,
        loss_fn=loss_fn,
        iteration=TRAINING_STEPS,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        generator=train_generator_chr,
    )
    loss_list_bpe = training(
        encoded_data=train_ids_bpe,
        vocab_size=tokenizer_bpe.vocab_size,
        model=model_bpe,
        optimizer=optimizer_bpe,
        loss_fn=loss_fn,
        iteration=TRAINING_STEPS,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        generator=train_generator_bpe,
    )

    avg_loss_chr = evaluation(
        model=model_chr,
        vocab_size=tokenizer_chr.vocab_size,
        data=val_ids_chr,
        loss_fn=loss_fn,
        iteration0=EVALUATION_STEPS,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        generator=eval_generator_chr,
    )
    avg_loss_bpe = evaluation(
        model=model_bpe,
        vocab_size=tokenizer_bpe.vocab_size,
        data=val_ids_bpe,
        loss_fn=loss_fn,
        iteration0=EVALUATION_STEPS,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        generator=eval_generator_bpe,
    )

    print("\n=== Training and validation ===")
    print_loss_summary("character", loss_list_chr, avg_loss_chr)
    print_loss_summary("byte-level BPE", loss_list_bpe, avg_loss_bpe)
    print(
        f"character bits-per-byte: "
        f"{bits_per_byte(model_chr, val_ids_chr, val_data, BLOCK_SIZE):.4f}"
    )
    print(
        f"byte-level BPE bits-per-byte: "
        f"{bits_per_byte(model_bpe, val_ids_bpe, val_data, BLOCK_SIZE):.4f}"
    )

    unknown_prompt_chars = sorted(set(PROMPT) - set(tokenizer_chr.stoi))
    print(f"character prompt unknowns: {unknown_prompt_chars}")

    prompt_ids_bpe = torch.tensor([tokenizer_bpe.encode(PROMPT)], dtype=torch.long)
    prompt_ids_chr = torch.tensor([tokenizer_chr.encode(PROMPT)], dtype=torch.long)
    generated_bpe = generate(
        MAX_NEW_TOKENS,
        prompt_ids_bpe,
        model_bpe,
        generator=torch.Generator().manual_seed(SEED + 5),
    )
    generated_chr = generate(
        MAX_NEW_TOKENS,
        prompt_ids_chr,
        model_chr,
        generator=torch.Generator().manual_seed(SEED + 6),
    )

    decoded_bpe = tokenizer_bpe.decode(generated_bpe[0].tolist(), errors="replace")
    decoded_chr = tokenizer_chr.decode(generated_chr[0].tolist())

    print("\n=== Generated text ===")
    print(f"byte-level BPE (replacement characters={decoded_bpe.count('�')}):")
    print(make_control_characters_visible(decoded_bpe))
    print("\ncharacter:")
    print(decoded_chr)


if __name__ == "__main__":
    main()
