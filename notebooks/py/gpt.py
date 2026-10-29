"""Small GPT model shared by both tokenizer experiments."""

import torch
import torch.nn as nn

from parameters import (
    BATCH_SIZE,
    BLOCK_SIZE,
    EMBED_DIM,
    HEAD_SIZE,
    NUM_HEADS,
    NUM_TRANSFORMER_BLOCKS,
)

# Aliases keep the original notebook imports and default arguments working.
block_size = BLOCK_SIZE
embed_dim = EMBED_DIM
num_head = NUM_HEADS
head_size = HEAD_SIZE
batch_size = BATCH_SIZE
num_transformer = NUM_TRANSFORMER_BLOCKS


#*****************************************************
#---------------EmbeddingOperation-------------------
#*****************************************************
class EmbeddingOperation(nn.Module):
    def __init__(self, vocab_size: int = 211, block_size: int = block_size, embed_dim: int = embed_dim):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)

    def forward(self, ix):
        _, T = ix.shape
        tokens = self.token_embedding(ix)
        position_ids = torch.arange(T, device=ix.device)
        positions = self.position_embedding(position_ids)

        output = tokens + positions

        return output



#*****************************************************
#---------------SingleHeadAttention-------------------
#*****************************************************
class SingleHeadAttention(nn.Module):
    def __init__(self, block_size: int = block_size, embed_dim: int = embed_dim, head_size: int = head_size):
        super().__init__()
        self.head_size = head_size

        self.q_proj = nn.Linear(in_features=embed_dim, out_features=head_size, bias=False)
        self.k_proj = nn.Linear(in_features=embed_dim, out_features=head_size, bias=False)
        self.v_proj = nn.Linear(in_features=embed_dim, out_features=head_size, bias=False)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer('mask', mask)

    def forward(self, x):
        _, T, _ = x.shape

        Q = self.q_proj(x) # Q -> [batch_size, block_size, head_size]
        K = self.k_proj(x) # K -> [batch_size, block_size, head_size]
        V = self.v_proj(x) # V -> [batch_size, block_size, head_size]
        K_t = K.transpose(-2, -1) # K_t -> [batch_size, head_size, block_size]
        wei = Q @ K_t # wei -> [batch_size, block_size, block_size]

        scaled_score = wei * (self.head_size ** -0.5)

        causal_mask = self.mask[:T, :T]  # type: ignore
        forbidden = causal_mask == 0
        masked_score = scaled_score.masked_fill(
            forbidden,
            float("-inf")
        )

        attention_weights = torch.softmax(masked_score, dim = -1)

        output = attention_weights @ V # output -> [batch_size, head_size, block_size]

        return output


#*****************************************************
#---------------MultiHeadAttention--------------------
#*****************************************************
class MultiHeadAttention(nn.Module):
    def __init__(self, block_size: int = block_size, embed_dim: int = embed_dim, head_size: int = head_size, num_head: int = num_head,):
        super().__init__()
        if num_head * head_size != embed_dim:
            raise ValueError("num_head * head_size must equal embed_dim")
        self.heads = nn.ModuleList([
            SingleHeadAttention(
                block_size=block_size,
                embed_dim=embed_dim,
                head_size=head_size
            )
            for _ in range(num_head)
        ]) # we are creating the multiheadattention which encompasse num_head single head attention 

        self.out_projection = nn.Linear(in_features=num_head*head_size, out_features=embed_dim) #

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]

        head_outputs = torch.cat(head_outputs, dim = -1)

        output = self.out_projection(head_outputs)

        return output



#*****************************************************
#---------------FeedForwardNetwork--------------------
#*****************************************************
class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim: int = embed_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=4*embed_dim),
            nn.ReLU(),
            # Preserve the extra hidden projection from the original project.
            nn.Linear(in_features=embed_dim*4, out_features=embed_dim*4),
            nn.ReLU(),
            nn.Linear(in_features=embed_dim*4, out_features=embed_dim)
        )

    def forward(self, x):
        output = self.layers(x)

        return output


#*****************************************************
#-----------------------Transformer-------------------
#*****************************************************
class MyTransformer(nn.Module):
    def __init__(self, 
                 block_size: int = block_size, 
                 batch_size: int = batch_size, 
                 embed_dim: int = embed_dim, 
                 head_size: int =  head_size,
                 num_head: int = num_head):
        super().__init__()

        self.ln1 = nn.LayerNorm(normalized_shape = embed_dim)
        self.ln2 = nn.LayerNorm(normalized_shape = embed_dim)
        self.MHA = MultiHeadAttention(block_size=block_size, embed_dim=embed_dim, head_size=head_size, num_head=num_head)
        self.ffn = FeedForwardNetwork(embed_dim=embed_dim)

    def forward(self, x):
        x = self.MHA(self.ln1(x)) + x
        x = self.ffn(self.ln2(x)) + x

        output = x

        return output
    

#*****************************************************
#-----------------------GPT---------------------------
#*****************************************************
class GPT(nn.Module):
    def __init__(self, 
                 block_size: int = block_size,
                 batch_size: int = batch_size,
                 num_head: int = num_head,
                 vocab_size: int = 211,
                 embed_dim: int = embed_dim,
                 head_size: int = head_size,
                 num_transformer: int = num_transformer
                ):
        super().__init__()
        if num_head * head_size != embed_dim:
            raise ValueError("num_head * head_size must equal embed_dim")

        self.embedding_layer = EmbeddingOperation(
            vocab_size=vocab_size,
            block_size=block_size,
            embed_dim=embed_dim
        )
        self.transformers = nn.ModuleList([MyTransformer(
                block_size = block_size,
                batch_size = batch_size,
                embed_dim = embed_dim,
                head_size = head_size,
                num_head = num_head
            )
            for _ in range(num_transformer)
                                        ])
        self.final_layer = nn.Linear(in_features=embed_dim, out_features=vocab_size)

        self.block_size = block_size
        self.batch_size = batch_size

    def forward(self, x,  target = None):
        if x.ndim != 2:
            raise ValueError("GPT input must have shape [batch, time]")
        if x.shape[1] > self.block_size:
            raise ValueError(
                f"sequence length {x.shape[1]} exceeds block_size {self.block_size}"
            )

        x = self.embedding_layer(x)
        for transformer in self.transformers:
            x = transformer(x)
        x = self.final_layer(x)

        output = x

        return output

#*****************************************************
#---------------------ANALIZING-----------------------
#*****************************************************










