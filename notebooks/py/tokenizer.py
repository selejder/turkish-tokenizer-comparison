"""Dataset helpers and the two tokenizers used in the comparison."""

from collections.abc import Sequence

import torch


def dataset(text, train: float = 0.8, test: float = 0.1, validation: float = 0.1):
    """Split raw text before fitting either tokenizer."""
    if not text:
        raise ValueError("text must not be empty")
    if abs((train + test + validation) - 1.0) > 1e-9:
        raise ValueError("train, test and validation ratios must sum to 1.0")

    train_split = int(len(text) * train)
    test_split = int(len(text) * (train + test))

    train_data = text[:train_split]
    test_data = text[train_split:test_split]
    val_data = text[test_split:]

    return train_data, test_data, val_data


def batcher(
    encoded_ids,
    block_size: int = 256,
    batch_size: int = 32,
    generator: torch.Generator | None = None,
):
    """Create random next-token training batches."""
    if len(encoded_ids) <= block_size:
        raise ValueError(
            f"encoded data must contain more than {block_size} tokens; "
            f"received {len(encoded_ids)}"
        )

    X = []
    targets = []

    for _ in range(batch_size):
        ix = torch.randint(
            0,
            len(encoded_ids) - block_size,
            (1,),
            generator=generator,
        ).item()

        X.append(encoded_ids[ix : ix + block_size])
        targets.append(encoded_ids[ix + 1 : ix + block_size + 1])

    X = torch.tensor(X, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return X, targets


def get_stats(ids: Sequence[int]):

    stats = {}

    for pair in zip(ids, ids[1:]):
        if pair in stats:
            stats[pair] += 1
        else:
            stats[pair] = 1

    return stats


def merge(ids: Sequence[int], pair: tuple[int, int], idx: int):

    new_ids = []

    ix = 0
    while ix < len(ids):

        if ix < len(ids) - 1 and pair[0] == ids[ix] and pair[1] == ids[ix + 1]:
            new_ids.append(idx)
            ix += 2

        else:
            new_ids.append(ids[ix])
            ix += 1
    return new_ids


def train_bpe(text, num_merge):
    ids = list(text.encode("utf-8"))

    merges = {}
    idx = 256
    for i in range(num_merge):
        stats = get_stats(ids)
        if not stats:
            break        

        # target_pair = max(stats, key=stats.get)
        target_pair = max(stats, key=lambda pair: stats[pair])
        ids = merge(ids, target_pair, idx)
        merges[target_pair] = idx
        idx += 1

    return merges



def encoder(text, merges):
    encoded_ids = list(text.encode("utf-8"))

    for pair, token_id in merges.items():
        encoded_ids = merge(encoded_ids, pair, token_id)

    return encoded_ids


def decoder(encoded_ids, vocab, errors: str = "strict"):
    text = []
    for idx in encoded_ids:
        ch = vocab[idx]
        text.append(ch)

    text = b''.join(text)
    text = text.decode("utf-8", errors=errors)

    return text


def create_vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for pair, token_id in merges.items():
        vocab[token_id] = vocab[pair[0]] + vocab[pair[1]]

    return vocab


class BPETokenizer:
    """A small educational byte-level BPE tokenizer."""

    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.vocab_size = len(self.vocab)

    def fit(self, text, num_merge):
        """Learn BPE merges from training text only."""
        if not text:
            raise ValueError("training text must not be empty")
        if num_merge < 0:
            raise ValueError("num_merge must be non-negative")

        self.merges = train_bpe(text, num_merge)
        self.vocab = create_vocab(self.merges)
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        encoded_ids = list(text.encode("utf-8"))

        for pair, token_id in self.merges.items():
            encoded_ids = merge(encoded_ids, pair, token_id)

        return encoded_ids

    def decode(self, encoded_ids, errors: str = "strict"):
        text = []

        for idx in encoded_ids:
            ch = self.vocab[idx]
            text.append(ch)

        text = b''.join(text)
        text = text.decode("utf-8", errors=errors)

        return text


class CharacterTokenizer:
    """Character tokenizer with an explicit unknown-character token."""

    def __init__(self):
        self.itos = {}
        self.stoi = {}
        self.unk_id = 0
        self.vocab_size = 0

    def fit(self, text):
        if not text:
            raise ValueError("training text must not be empty")

        chars = sorted(list(set(text)))
        self.itos = {idx: ch for idx, ch in enumerate(chars)}
        self.stoi = {ch: idx for idx, ch in enumerate(chars)}

        self.unk_id = len(chars)
        self.itos[self.unk_id] = "<UNK>"
        self.stoi["<UNK>"] = self.unk_id
        self.vocab_size = len(self.itos)

    def encode(self, text):
        ids = []
        for ch in text:
            if ch in self.stoi.keys():
                idx = self.stoi[ch]
                ids.append(idx)

            else:
                ids.append(self.unk_id)

        return ids

    def decode(self, ids):
        text = []
        for idx in ids:
            if idx in self.itos.keys():
                ch = self.itos[idx]
                text.append(ch)
            else:
                idx = self.unk_id
                ch = self.itos[idx]
                text.append(ch)

        text = ''.join(text)

        return text


# Backward-compatible alias for the original project spelling.
CharachterTokenizer = CharacterTokenizer
