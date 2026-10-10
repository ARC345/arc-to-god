# arc-to-god

Controlled comparison of LLM architectures at mini scale, character level (enwik8), one per day. Public repo, so no private notes go in tracked files.

## Layout
- `harness/train.py`: the shared loop. It handles data, AdamW plus warmup/cosine, AMP, grad accumulation, evaluation and results. It's the same for every architecture.
- `harness/models/<arch>.py`: one file per architecture, registered with `@register`. The file only defines the architecture. No loss, optimizer or data code goes there.
- `configs/base.yaml`: everything shared. `configs/<arch>.yaml`: only a shape per size plus `model:` settings.
- `results/<arch>-<size>.json`: the committed record of each run. `runs/` and `data/` are git-ignored.

## Fairness rules (the point of the project)
- Never change `configs/base.yaml` or `harness/train.py` for one architecture. A change there has to apply to all of them, and earlier runs then need re-running.
- Match the non-embedding parameter target to within 10%. The harness warns when a run is outside it.
- If an architecture runs out of memory, lower `train.micro_batch_size` only. The effective `batch_size` stays fixed.
- Run every recorded result on the same hardware (Kaggle T4, fp16). Local runs on the RTX 4060 use bf16, so they're for development only. Throughput numbers from different GPUs aren't comparable.
- Models return logits, or `(logits, aux_loss)`. Reported BPC never includes the auxiliary loss.

## Commands
- `pixi run smoke`: 2M-token GPT-2-s sanity run, not recorded.
- `pixi run train <arch> --size s|m [--set key=value ...]`: full run.
- `pixi run summarize`: markdown results table.
- `pixi run kg push|status|pull <arch> --size s|m [--smoke]`: run on a Kaggle T4 from here (`scripts/kaggle.py`). It bundles the working tree into one script kernel. `pull` copies `results.json` into `results/`. Recorded results come from `kg`; local runs are for development only.

## Notes
- The GPT-2 baseline is ported from ARC345/llm-lab's GPT and kept faithful to GPT-2: pre-LN, learned positions, GELU, tied embeddings.
- Published enwik8 reference: a 12-layer char transformer (~44M params) reaches ~1.1 test BPC (Al-Rfou et al. 2018), but it was trained far longer than 100M tokens. Expect our baseline to land noticeably higher. Use the reference as a ballpark sanity check, not a target.
- The `[system-requirements]` warning from pixi 0.81 is harmless. The newer per-platform syntax is rejected by this version.
