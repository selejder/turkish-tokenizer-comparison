from pathlib import Path
from tokenizer import batcher, dataset, BPETokenizer, CharachterTokenizer
from generate import generate
from gpt import GPT
from training import evaluation, training
import torch
import torch.nn as nn

num_merge = 100 #type: ignore
block_size = 256
batch_size = 32
embed_dim = 256
head_size = 16
num_head = 16
num_transformer = 8
learning_rate = 1e-3
iteration = 6

project_root = Path(__file__).resolve().parents[2]

data_path = (
    project_root
    / "data"
    / "interim"
    / "Bianet_tr_minimal-cleaned.txt"
)

data = data_path.read_text(encoding="utf-8")

len_data = len(data)
sample = data[:10000]

train_data, test_data, val_data = dataset(sample)


#------CharachterTokenizer-------
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


#------BPETokenizer-------
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

loss_list_bpe = training(
    encoded_data=encoded_data_bpe, 
    vocab_size=vocab_size_bpe, 
    model=model_bpe, 
    optimizer=optimizer_bpe, 
    loss_fn=loss_fn)

loss_list_chr = training(
    encoded_data=encoded_data_chr, 
    vocab_size=vocab_size_chr, 
    model=model_chr, 
    optimizer=optimizer_chr, 
    loss_fn=loss_fn)

avg_loss_bpe = evaluation(
    model=model_bpe,
    vocab_size=vocab_size_bpe,
    data=tokenizer_bpe.encode(val_data),
    loss_fn=loss_fn,
    iteration0=10
)

avg_loss_chr = evaluation(
    model=model_chr,
    vocab_size=vocab_size_chr,
    data=tokenizer_chr.encode(val_data),
    loss_fn=loss_fn,
    iteration0=10   
)

propmpt_encoded_bpe = tokenizer_bpe.encode("Yapay Zeka")
X = torch.tensor([propmpt_encoded_bpe], dtype=torch.long)
generated_text_bpe = generate(
    max_new_tokens=100,
    context_ids=X,
    model=model_bpe
)
X_decode = tokenizer_bpe.decode(generated_text_bpe[0].tolist())

propmpt_encoded_chr = tokenizer_chr.encode("Yapay Zeka")
Y = torch.tensor([propmpt_encoded_chr], dtype=torch.long)
generated_text_chr = generate(
    max_new_tokens=100,
    context_ids=Y,
    model=model_chr
)
Y_decode = tokenizer_chr.decode(generated_text_chr[0].tolist())

print(X_decode)
print(Y_decode)