# arc-to-god

The major LLM architectures, implemented from scratch at mini scale and compared on equal terms. One architecture a day.

A repository to teach Arnav Rastogi how to become god by creating sentient AIs.

Kickoff post: https://arnav.rastogi.net.in/blog/2026/one-architecture-a-day/

## What's held fixed

Every architecture runs through the same harness (`harness/`) with the same settings (`configs/base.yaml`):

- **Data:** enwik8, using the standard 90M / 5M / 5M byte split. It's character (byte) level, so the vocabulary is 256.
- **Budget:** 100M training tokens per run, at sequence length 512 with an effective batch of 32.
- **Size:** two parameter budgets counted without embeddings, `s` ≈ 20M and `m` ≈ 60M. Each architecture picks its own shape to hit them, within ±10%.
- **Optimizer:** AdamW with linear warmup then cosine decay. Same learning rate, weight decay and gradient clipping for every model.
- **Evaluation:**
  - **test BPC:** over every non-overlapping 512-byte window of the test split.
  - **val BPC:** on a fixed set of validation batches.
  - **copy BPC:** a zero-shot probe. The model sees a random string twice and is scored on predicting the second copy. Lower means better copying from context.
  - **Speed and memory:** training tokens/s and peak GPU memory.

The only thing that changes between runs is the model file.

## Results

`python -m harness.summarize` prints the table below from `results/*.json`.

<!-- results table goes here -->

## Running

Locally (pixi, for development):

```bash
pixi run smoke                       # 2M-token sanity run, not recorded
pixi run train gpt2 --size s         # full run -> runs/<id>/ and results/gpt2-s.json
pixi run summarize
```

On Kaggle's GPUs, launched from your own machine:

```bash
pixi run kg push gpt2 --size s --smoke    # quick sanity run on a Kaggle T4
pixi run kg push gpt2 --size s            # full run
pixi run kg status gpt2 --size s
pixi run kg pull gpt2 --size s            # download outputs, record results/gpt2-s.json
```

`kg push` bundles the local `harness/`, `configs/` and `pyproject.toml` into one script and runs it as a private Kaggle kernel on a T4, so nothing has to be committed first. Each arch/size pair gets its own kernel, and every push creates a new version of it. All recorded results use the T4, so speed numbers are comparable. Setup: put an API token in `~/.kaggle/kaggle.json` (kaggle.com, then Settings > API > Create New Token). Your Kaggle account must be phone-verified for GPU and internet access.

## Adding an architecture

1. Create `harness/models/<name>.py`. Register a `nn.Module` with `@register("<name>")` whose `forward(idx)` returns logits of shape `(B, T, 256)`. It can also return `(logits, aux_loss)`, e.g. for MoE load balancing.
2. Create `configs/<name>.yaml` with a shape for each size, plus any architecture-specific `model:` settings.
3. Run `pixi run smoke` with the name swapped in, then do the full runs.

## Caveats

These results are at small scale and character level, so they may not transfer to large BPE-tokenized models. BPC on enwik8 is technically bits per byte, which is how the literature reports it.
