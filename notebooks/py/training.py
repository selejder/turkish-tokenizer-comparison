
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

project_root = Path(__file__).resolve().parents[2]
doc_name = "Bianet_tr_minimal-cleaned.txt"
folder_path = project_root / "data" / "interim"
doc_path = folder_path / doc_name

with open(doc_path, mode='r', encoding="utf-8") as file:
    data = file.read()

len_data = len(data)
sample = data[:10000]

train_data, test_data, val_data = dataset(sample)


# Character Tokenizer
tokenizer_chr = CharachterTokenizer()
tokenizer_chr.fit(train_data)
vocab_size_chr = tokenizer_chr.vocab_size
encoded_data_chr = tokenizer_chr.encode(train_data)

model_chr = GPT(
    vocab_size=vocab_size_chr,
    block_size=block_size,
    batch_size=batch_size,
    embed_dim=embed_dim,
    head_size=head_size,
    num_head=num_head,
    num_transformer=num_transformer
)

#---------------------------------------------

# BPE Tokenizer
tokenizer_bpe = BPETokenizer()
tokenizer_bpe.fit(train_data, num_merge)
vocab_size_bpe = tokenizer_bpe.vocab_size
encoded_data_bpe = tokenizer_bpe.encode(train_data)

model_bpe = GPT(
    vocab_size=vocab_size_bpe,
    block_size=block_size,
    batch_size=batch_size,
    embed_dim=embed_dim,
    head_size=head_size,
    num_head=num_head,
    num_transformer=num_transformer
)



loss_fn = nn.CrossEntropyLoss()
optimizer_bpe = torch.optim.Adam(model_bpe.parameters(), lr=learning_rate)
optimizer_chr = torch.optim.Adam(model_chr.parameters(), lr=learning_rate);


def training(encoded_data, vocab_size, model: GPT, optimizer: torch.optim.Optimizer, iteration: int = 20):
    loss_list = []
    len_data = len(data)

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


loss_list_bpe = training(encoded_data_bpe, vocab_size_bpe, model_bpe, optimizer_bpe)
loss_list_chr = training(encoded_data_chr, vocab_size_chr, model_chr, optimizer_chr)

print(f"Loss list of bpe: {loss_list_bpe}\n\nLoss list of chr: {loss_list_chr}")



@torch.no_grad()
def evaluation(model: nn.Module, iteration0: int = 20):
    data_for_eval = {
        'bpe_val' : tokenizer_bpe.encode(val_data),
        'chr_val' : tokenizer_chr.encode(val_data)
    }

    average_loss = {}

    model.eval()
    for name, data0 in data_for_eval.items():
        losses = []

        for i in range(iteration0):
            X, target = batcher(data0, block_size=block_size, batch_size=batch_size)
            logits = model(X)
            logits = logits.reshape(-1, vocab_size_bpe)
            target = target.reshape(-1)
            loss = loss_fn(logits, target)
            losses.append(loss.item())

        average_loss[name] = sum(losses) / len(losses)

    model.train()
    return average_loss


average_loss = evaluation(model=model_bpe)
print(f"\n----Losses----\n{average_loss}")
        












