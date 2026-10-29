# Turkish Tokenizer Comparison with a Small GPT

An educational, from-scratch comparison of a character tokenizer and a
byte-level BPE tokenizer on Turkish news text. Both tokenizers feed the same
small causal GPT architecture so that the tokenizer is the main experimental
variable.

This is an independent learning project. It is not an official implementation
of GPT, minBPE, or nanoGPT, and it is not intended as a production language
model.

## What the project compares

- vocabulary size and encoded sequence length;
- byte-level BPE compression in bytes per token;
- training and validation cross-entropy within each model;
- validation bits-per-byte, which makes the tokenizer runs comparable;
- generated text and invalid UTF-8 replacement counts;
- character-tokenizer unknown characters.

Raw token-level loss values from different tokenizers should not be compared
directly. The experiment reports bits-per-byte for the cross-tokenizer
comparison.

## Project structure

```text
data/
  raw/                         # local OPUS download (not committed)
  interim/                     # locally cleaned text (not committed)
notebooks/
  ipnyb/                       # corpus, tokenizer, and model learning notebooks
  py/
    corpus.py                  # minimal Unicode cleaning
    tokenizer.py               # character and byte-level BPE tokenizers
    gpt.py                     # shared causal GPT model
    training.py                # training, validation, bits-per-byte
    generate.py                # autoregressive generation
    parameters.py              # shared experiment settings
    experiment.py              # end-to-end comparison
tests/
  test_smoke.py                # fast tokenizer/model pipeline checks
```

The Python files retain the same component boundaries developed during the
learning project. `parameters.py` only centralizes values that were previously
repeated across several files. The notebooks preserve the earlier learning and
exploration steps; the Python scripts are the canonical reproducible workflow.

## Dataset

The project uses the Turkish side of the **Bianet v1** English-Turkish corpus
distributed by [OPUS](https://opus.nlpl.eu/datasets/Bianet). OPUS describes it
as a Turkish, Kurdish, and English parallel news corpus and distributes it
under **CC BY-SA 4.0**.

Download the English-Turkish Moses package from the OPUS page, extract the
Turkish file, and place it at:

```text
data/raw/Bianet.en-tr.tr
```

The corpus is not covered by this repository's MIT license. If you use it,
follow its CC BY-SA 4.0 terms and cite:

> Ataman, D. (2018). Bianet: A Parallel News Corpus in Turkish, Kurdish and
> English. Proceedings of the LREC 2018 Workshop MLP-Moment, 14-17.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Prepare the corpus:

```bash
python notebooks/py/corpus.py
```

Run the CPU-friendly comparison:

```bash
python notebooks/py/experiment.py
```

Run the smoke tests:

```bash
python -m unittest discover -s tests -v
```

## Configuration

All shared values live in `notebooks/py/parameters.py`. The committed defaults
are deliberately small enough for a quick CPU run. For a more meaningful text
generation experiment, increase `TRAINING_STEPS` gradually and then increase
`SAMPLE_CHARS`; verify the small run before scaling.

Both tokenizers always receive the same raw train/validation/test split and the
same GPT architecture. Each tokenizer determines only its own vocabulary size
and encoded sequence length. The test split is held out from the current
development metrics.

## UTF-8 behavior

Byte-level BPE round trips use strict UTF-8 decoding. A model, however, can
sample byte tokens in an invalid UTF-8 order. Generated BPE text therefore uses
replacement decoding and reports the number of `�` characters. This keeps
tokenizer correctness checks strict while allowing imperfect model output to
be inspected. Valid but non-printing control bytes are escaped in the console
report (for example, `\x05`) instead of being sent directly to the terminal.

The character tokenizer cannot create invalid UTF-8, but a character absent
from the training vocabulary is represented by `<UNK>`.

## Current limitations

- The default run is a pipeline demonstration, not a trained Turkish language
  model; generated text will usually be low quality.
- The BPE implementation is intentionally educational and optimized for
  clarity rather than speed.
- Bits-per-byte is evaluated in independent context blocks, so context resets
  at block boundaries.
- Final test-set evaluation and long training are intentionally left out of the
  quick public run.

## License

Project code is released under the [MIT License](LICENSE). Dataset licensing is
described separately above.
