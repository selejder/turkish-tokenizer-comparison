"""Training and evaluation helpers shared by both models."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt import GPT
from parameters import BATCH_SIZE, BLOCK_SIZE, EVALUATION_STEPS, TRAINING_STEPS
from tokenizer import batcher

def training(
    encoded_data,
    vocab_size: int,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    iteration: int = TRAINING_STEPS,
    block_size: int = BLOCK_SIZE,
    batch_size: int = BATCH_SIZE,
    generator: torch.Generator | None = None,
):
    """Train one model and return its per-step loss history."""
    loss_list = []

    model.train()
    for _ in range(iteration):

        X, target = batcher(
            encoded_data,
            block_size=block_size,
            batch_size=batch_size,
            generator=generator,
        )

        logits = model(X)

        logits = logits.reshape(-1, vocab_size)
        target = target.reshape(-1)

        loss = loss_fn(logits, target)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loss_list.append(loss.item())

    return loss_list


@torch.no_grad()
def evaluation(
    model: nn.Module,
    vocab_size: int,
    data,
    loss_fn,
    iteration0: int = EVALUATION_STEPS,
    block_size: int = BLOCK_SIZE,
    batch_size: int = BATCH_SIZE,
    generator: torch.Generator | None = None,
):
    """Estimate validation loss without updating model parameters."""

    losses = []
    was_training = model.training
    model.eval()

    for _ in range(iteration0):
        X, target = batcher(
            data,
            block_size=block_size,
            batch_size=batch_size,
            generator=generator,
        )
        logits = model(X)
        logits = logits.reshape(-1, vocab_size)
        target = target.reshape(-1)
        loss = loss_fn(logits, target)
        losses.append(loss.item())

    average_loss = sum(losses) / len(losses)
    model.train(was_training)
    return average_loss


@torch.no_grad()
def bits_per_byte(model: nn.Module, encoded_data, raw_text: str, block_size: int = BLOCK_SIZE):
    """Compute blockwise validation negative log-likelihood per source byte."""
    token_ids = list(encoded_data)
    if len(token_ids) < 2:
        raise ValueError("encoded_data must contain at least two tokens")

    byte_count = len(raw_text.encode("utf-8"))
    if byte_count == 0:
        raise ValueError("raw_text must contain at least one byte")

    was_training = model.training
    model.eval()
    total_nll = 0.0

    for start in range(0, len(token_ids) - 1, block_size):
        end = min(start + block_size, len(token_ids) - 1)
        X = torch.tensor([token_ids[start:end]], dtype=torch.long)
        target = torch.tensor([token_ids[start + 1 : end + 1]], dtype=torch.long)
        logits = model(X)
        total_nll += F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            reduction="sum",
        ).item()

    model.train(was_training)
    return total_nll / (byte_count * math.log(2))













