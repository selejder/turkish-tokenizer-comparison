"""Fast end-to-end checks for the educational tokenizer comparison."""

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY_SOURCE = PROJECT_ROOT / "notebooks" / "py"
sys.path.insert(0, str(PY_SOURCE))

from generate import generate  # noqa: E402
from gpt import GPT  # noqa: E402
from tokenizer import BPETokenizer, CharacterTokenizer  # noqa: E402
from training import bits_per_byte, evaluation, training  # noqa: E402


class TokenizerTests(unittest.TestCase):
    def test_bpe_round_trip_and_decode_policy(self):
        tokenizer = BPETokenizer()
        tokenizer.fit("Türkçe metin ve emoji 😊", num_merge=10)
        sample = "Türkçe 😊"

        self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)
        with self.assertRaises(UnicodeDecodeError):
            tokenizer.decode([0xDA])
        self.assertEqual(tokenizer.decode([0xDA], errors="replace"), "�")

    def test_character_unknown_token(self):
        tokenizer = CharacterTokenizer()
        tokenizer.fit("abc ")

        self.assertEqual(tokenizer.decode(tokenizer.encode("abz")), "ab<UNK>")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.text = ("merhaba dünya küçük model " * 20).strip()
        self.tokenizer = CharacterTokenizer()
        self.tokenizer.fit(self.text)
        self.ids = self.tokenizer.encode(self.text)
        self.model = GPT(
            vocab_size=self.tokenizer.vocab_size,
            block_size=8,
            batch_size=2,
            embed_dim=16,
            head_size=8,
            num_head=2,
            num_transformer=1,
        )

    def test_forward_training_evaluation_and_generation(self):
        logits = self.model(torch.tensor([self.ids[:8]], dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (1, 8, self.tokenizer.vocab_size))

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        losses = training(
            self.ids,
            self.tokenizer.vocab_size,
            self.model,
            optimizer,
            loss_fn,
            iteration=1,
            block_size=8,
            batch_size=2,
            generator=torch.Generator().manual_seed(8),
        )
        self.assertEqual(len(losses), 1)
        self.assertTrue(math.isfinite(losses[0]))

        validation_loss = evaluation(
            self.model,
            self.tokenizer.vocab_size,
            self.ids,
            loss_fn,
            iteration0=1,
            block_size=8,
            batch_size=2,
            generator=torch.Generator().manual_seed(9),
        )
        self.assertTrue(math.isfinite(validation_loss))

        bpb = bits_per_byte(self.model, self.ids, self.text, block_size=8)
        self.assertTrue(math.isfinite(bpb))

        # A context longer than block_size must be cropped before the first pass.
        long_context = torch.tensor([self.ids[:12]], dtype=torch.long)
        generated = generate(
            2,
            long_context,
            self.model,
            generator=torch.Generator().manual_seed(10),
        )
        self.assertEqual(tuple(generated.shape), (1, 14))


if __name__ == "__main__":
    unittest.main()
