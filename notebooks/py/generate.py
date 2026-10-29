from gpt import GPT
import torch
import torch.nn as nn

@torch.no_grad()
def generate(max_new_tokens, context_ids, model: nn.Module):
    model_input = context_ids
    generated_text = context_ids.clone()


    model.eval()
    for i in range(max_new_tokens):
        logits = model(model_input)
        last_logit = logits[:, -1, :]
        possibilities = torch.softmax(last_logit, dim=-1)
        next_id = torch.multinomial(possibilities, num_samples=1)
        generated_text = torch.cat((generated_text, next_id), dim=1)
        model_input = generated_text[:, -model.block_size:] #type: ignore

    return generated_text

