import torch


def dataset(text, train: float = 0.8, test: float = 0.1, validation: float = 0.1):
    train_split = int(len(text) * train)
    test_split= int(len(text) * (train + test))

    train_data = text[:train_split]
    test_data = text[train_split:test_split]
    val_data = text[test_split:]

    return train_data, test_data, val_data


def batcher(encoded_ids, block_size: int = 256, batch_size: int = 32):
    X = []
    targets = []

    for i in range(batch_size):
        ix = torch.randint(0, len(encoded_ids) - (block_size + 1), (1, ))

        X.append(encoded_ids[ix:ix+block_size])
        targets.append(encoded_ids[ix+1:ix+block_size+1])

    X = torch.tensor(X)
    targets = torch.tensor(targets)

    return X, targets


def get_stats(ids):

    stats = {}

    for pair in zip(ids, ids[1:]):
        if pair in stats:
            stats[pair] += 1
        else:
            stats[pair] = 1

    return stats


def merge(ids, pair, idx):

    new_ids = []

    ix = 0
    while ix < len(ids):

        if pair[0] == ids[ix] and ix < len(ids) - 1 and pair[1] == ids[ix+1]:
            new_ids.append(idx)
            ix += 2

        else:
            new_ids.append(ids[ix])
            ix += 1
    return new_ids


def train_bpe(text, num_merge):
    ids = text.encode('utf-8')

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
    encoded_ids = text.encode('utf-8')

    for pair, token_id in merges.items():
        encoded_ids = merge(encoded_ids, pair, token_id)

    return encoded_ids


def decoder(encoded_ids, vocab):
    text = []
    for idx in encoded_ids:
        ch = vocab[idx]
        text.append(ch)

    text = b''.join(text)
    text = text.decode('utf-8')

    return text


def create_vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for pair, token_id in merges.items():
        vocab[token_id] = vocab[pair[0]] + vocab[pair[1]]

    return vocab


class BPETokenizer():
    def fit(self, text, num_merge):
        self.text = text
        ids = text.encode('utf-8')
        merges = {}
        idx = 256

        for i in range(num_merge):
            stats = get_stats(ids)
            if not stats:
                break        

            target_pair = max(stats, key=lambda pair: stats[pair])
            ids = merge(ids, target_pair, idx)
            merges[target_pair] = idx
            idx += 1

        vocab = {i: bytes([i]) for i in range(256)}
        for pair, token_id in merges.items():
            vocab[token_id] = vocab[pair[0]] + vocab[pair[1]]

        self.merges = merges
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def encode(self, text):
        encoded_ids = text.encode('utf-8')

        for pair, token_id in self.merges.items():
            encoded_ids = merge(encoded_ids, pair, token_id)

        return encoded_ids

    def decode(self, encoded_ids):
        text = []

        for idx in encoded_ids:
            ch = self.vocab[idx]
            text.append(ch)

        text = b''.join(text)
        text = text.decode('utf-8')

        return text


class CharachterTokenizer():
    def fit(self, text):
        chars = sorted(list(set(text)))
        self.itos = {idx: ch for idx, ch in enumerate(chars)}
        self.stoi = {ch: idx for idx, ch in enumerate(chars)}

        self.unk_ids = len(chars)
        self.itos[self.unk_ids] = '<UNK>'
        self.stoi['<UNK>'] = self.unk_ids
        self.vocab_size = len(self.itos)

    def encode(self, text):
        ids = []
        for ch in text:
            if ch in self.stoi.keys():
                idx = self.stoi[ch]
                ids.append(idx)

            else:
                ids.append(self.unk_ids)

        return ids

    def decode(self, ids):
        text = []
        for idx in ids:
            if idx in self.itos.keys():
                ch = self.itos[idx]
                text.append(ch)
            else:
                idx = self.unk_ids
                ch = self.itos[idx]
                text.append(ch)

        text = ''.join(text)

        return text