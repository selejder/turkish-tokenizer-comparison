"""Autoregressive token generation."""

import torch
import torch.nn as nn

@torch.no_grad()
def generate(
    max_new_tokens: int,
    context_ids: torch.Tensor,
    model: nn.Module,
    generator: torch.Generator | None = None,
):
    """Append sampled tokens while respecting the model context window."""
    if context_ids.ndim != 2:
        raise ValueError("context_ids must have shape [batch, time]")
    if context_ids.shape[1] == 0:
        raise ValueError("context_ids must contain at least one token")

    generated_text = context_ids.clone()
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        # Crop before every forward call, including the first one.
        model_input = generated_text[:, -model.block_size:]  # type: ignore
        logits = model(model_input)
        last_logit = logits[:, -1, :]
        possibilities = torch.softmax(last_logit, dim=-1)
        next_id = torch.multinomial(
            possibilities,
            num_samples=1,
            generator=generator,
        )
        generated_text = torch.cat((generated_text, next_id), dim=1)

    model.train(was_training)
    return generated_text
