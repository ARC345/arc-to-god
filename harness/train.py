"""Train one architecture at one size under the shared budget, then evaluate.

    python -m harness.train gpt2 --size s
    python -m harness.train gpt2 --size s --set train.tokens=2e6   # smoke test

Writes runs/<arch>-<size>-<timestamp>/{config.yaml,metrics.jsonl,results.json,ckpt.pt}
and copies results.json to results/<arch>-<size>.json (the committed record).
"""

import argparse
import contextlib
import json
import math
import os
import shutil
import subprocess
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from harness import config as config_lib
from harness import data, probes
from harness.registry import ModelConfig, build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LN2 = math.log(2)


def split_output(out):
    """Models return logits, or (logits, aux_loss)."""
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, None


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    emb_ids = {id(p) for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters()}
    emb = sum(p.numel() for p in model.parameters() if id(p) in emb_ids)
    return total, total - emb


@torch.no_grad()
def evaluate(model, batches, device, amp_ctx) -> float:
    """Mean BPC over the given (x, y) batches."""
    model.eval()
    total, count = 0.0, 0
    for x, y in batches:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with amp_ctx:
            logits, _ = split_output(model(x))
        total += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum").item()
        count += y.numel()
    model.train()
    return total / count / LN2


def git_commit() -> str | None:
    if os.environ.get("ARC_TO_GOD_COMMIT"):  # set by scripts/kaggle.py, which has no .git to read
        return os.environ["ARC_TO_GOD_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arch")
    ap.add_argument("--size", default="s")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config, e.g. train.tokens=2e6")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--no-record", action="store_true", help="don't copy results into results/ (use for smoke tests)")
    args = ap.parse_args()

    cfg = config_lib.load(args.arch, args.size, args.set)
    t = cfg["train"]
    seq_len, batch_size, micro = int(t["seq_len"]), int(t["batch_size"]), int(t["micro_batch_size"])
    assert batch_size % micro == 0, "batch_size must be a multiple of micro_batch_size"
    accum = batch_size // micro

    torch.manual_seed(t["seed"])
    gen = torch.Generator().manual_seed(t["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Native bf16 needs compute capability 8.0+ (Ampere). is_bf16_supported() also counts
        # slow emulation, which made the T4 run in emulated bf16 at ~1/3 speed.
        native_bf16 = torch.cuda.get_device_capability()[0] >= 8
        amp_dtype = torch.bfloat16 if native_bf16 else torch.float16  # T4 (sm75): fp16 + GradScaler
        amp_ctx = torch.autocast("cuda", dtype=amp_dtype)
    else:
        amp_dtype = torch.float32
        amp_ctx = contextlib.nullcontext()

    # --- data ---
    paths = data.prepare(os.path.join(ROOT, cfg["data"]["dir"]) if not os.path.isabs(cfg["data"]["dir"]) else cfg["data"]["dir"])
    train_split, val_split, test_split = (data.Split(paths[s]) for s in ("train", "val", "test"))
    val_gen = torch.Generator().manual_seed(10_000)
    val_batches = [val_split.sample(micro, seq_len, val_gen) for _ in range(int(t["eval_batches"]))]

    # --- model ---
    m = cfg["model"]
    mcfg = ModelConfig(
        vocab_size=data.VOCAB_SIZE,
        seq_len=seq_len,
        d_model=m["d_model"],
        n_layer=m["n_layer"],
        n_head=m.get("n_head", 1),
        dropout=m.get("dropout", 0.0),
        extra={k: v for k, v in m.items() if k not in {"d_model", "n_layer", "n_head", "dropout"}},
    )
    model = build(args.arch, mcfg).to(device)
    n_total, n_nonemb = count_params(model)
    target = cfg["target_params"]
    print(f"{args.arch}-{args.size}: {n_total/1e6:.2f}M params ({n_nonemb/1e6:.2f}M non-embedding, target {target/1e6:.0f}M)")
    if abs(n_nonemb / target - 1) > 0.10:
        print(f"WARNING: non-embedding params are {100*(n_nonemb/target-1):+.0f}% off target; adjust configs/{args.arch}.yaml")

    # --- optimizer: decay matrices only ---
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": t["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=t["lr"],
        betas=tuple(t["betas"]),
        fused=device == "cuda",
    )
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)
    fwd_model = torch.compile(model) if t["compile"] else model

    tokens_per_step = batch_size * seq_len
    total_steps = max(1, int(t["tokens"]) // tokens_per_step)
    warmup_steps = max(1, int(t["warmup_tokens"]) // tokens_per_step)
    eval_every = max(1, int(t["eval_every_tokens"]) // tokens_per_step)

    def lr_at(step):
        if step < warmup_steps:
            return t["lr"] * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        floor = t["lr"] * t["min_lr_frac"]
        return floor + (t["lr"] - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    # --- run dir ---
    run_name = f"{args.arch}-{args.size}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = os.path.join(args.out, run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    log = open(os.path.join(run_dir, "metrics.jsonl"), "w")

    def emit(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    print(f"{total_steps} steps x {tokens_per_step} tokens, amp={str(amp_dtype).split('.')[-1]}, device={device}, run={run_dir}")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # --- train ---
    train_time = 0.0
    window_start, window_tokens = time.perf_counter(), 0
    wall_start = time.perf_counter()
    model.train()
    for step in range(total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        loss_sum = torch.zeros((), device=device)
        for _ in range(accum):
            x, y = train_split.sample(micro, seq_len, gen)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with amp_ctx:
                logits, aux = split_output(fwd_model(x))
                lm_loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
                loss = lm_loss + (aux if aux is not None else 0.0)
            scaler.scale(loss / accum).backward()
            loss_sum += lm_loss.detach() / accum

        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        window_tokens += tokens_per_step

        is_eval = (step + 1) % eval_every == 0 or step + 1 == total_steps
        if (step + 1) % t["log_every_steps"] == 0 or is_eval:
            if device == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            train_time += now - window_start
            tok_s = window_tokens / (now - window_start)
            loss_val = loss_sum.item()
            rec = {"step": step + 1, "tokens": (step + 1) * tokens_per_step, "train_bpc": loss_val / LN2,
                   "lr": lr_at(step), "grad_norm": grad_norm.item(), "tok_per_s": tok_s}
            if is_eval:
                rec["val_bpc"] = evaluate(model, val_batches, device, amp_ctx)
            emit(rec)
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
            if not math.isfinite(loss_val):
                raise RuntimeError("loss is not finite; stopping")
            window_start, window_tokens = time.perf_counter(), 0

    # --- final eval ---
    print("final eval: val (fixed batches), test (all non-overlapping windows), copy probe")
    val_bpc = evaluate(model, val_batches, device, amp_ctx)
    test_bpc = evaluate(model, test_split.windows(micro, seq_len), device, amp_ctx)
    copy = probes.copy_bpc(model, seq_len, device, amp_ctx, batch_size=micro)

    results = {
        "arch": args.arch,
        "size": args.size,
        "params_total": n_total,
        "params_non_embedding": n_nonemb,
        "target_params": target,
        "tokens": total_steps * tokens_per_step,
        "steps": total_steps,
        "val_bpc": val_bpc,
        "test_bpc": test_bpc,
        "copy_bpc": copy,
        "train_tokens_per_s": total_steps * tokens_per_step / train_time,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
        "train_time_s": train_time,
        "wall_time_s": time.perf_counter() - wall_start,
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "amp": str(amp_dtype).split(".")[-1],
        "torch": torch.__version__,
        "git_commit": git_commit(),
        "run": run_name,
        "config": cfg,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), os.path.join(run_dir, "ckpt.pt"))
    log.close()

    if not args.no_record:
        os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
        shutil.copy(os.path.join(run_dir, "results.json"), os.path.join(ROOT, "results", f"{args.arch}-{args.size}.json"))

    print(f"val_bpc={val_bpc:.4f} test_bpc={test_bpc:.4f} copy_bpc={copy:.4f} "
          f"tok/s={results['train_tokens_per_s']:.0f} peak_mem={results['peak_mem_gb']}GB")


if __name__ == "__main__":
    main()
