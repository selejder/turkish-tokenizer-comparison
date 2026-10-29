import torch
import torch.nn as nn
from pathlib import Path
import gpt
from gpt import GPT
from tokenizer import BPETokenizer, CharachterTokenizer, batcher, dataset

num_merge = 100 
block_size = 256
batch_size = 32
embed_dim = 256
head_size = 16
num_head = 16
num_transformer = 8
learning_rate = 1e-3
iteration = 6

def training(encoded_data, vocab_size, model: GPT, optimizer: torch.optim.Optimizer, loss_fn ,iteration: int = 20):
    loss_list = []

    model.train()
    for i in range(iteration):

        X, target = batcher( # type: ignore
        encoded_data,
        block_size=block_size,
        batch_size=batch_size)

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
def evaluation(model: nn.Module, vocab_size: int, data, loss_fn, iteration0: int = 20):

    losses = []

    model.eval()

    for i in range(iteration0):
        X, target = batcher(data, block_size=block_size, batch_size=batch_size)
        logits = model(X)
        logits = logits.reshape(-1, vocab_size)
        target = target.reshape(-1)
        loss = loss_fn(logits, target)
        losses.append(loss.item())

        average_loss = sum(losses) / len(losses)

    model.train()
    return average_loss













