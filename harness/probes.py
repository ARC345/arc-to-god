"""Zero-shot probes run on the trained model. No extra training.

copy_bpc: a random printable-ASCII string of length L, then the same string
again. The first half is unpredictable (~6.6 bits/char); the second half is
fully predictable by copying from context. BPC on the second half measures
in-context copying (induction heads in transformers; a known weak spot for
fixed-size recurrent state). Lower is better.
"""

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def copy_bpc(model, seq_len: int, device, amp_ctx, n_seqs: int = 64, batch_size: int = 32, seed: int = 1234):
    gen = torch.Generator().manual_seed(seed)
    half = seq_len // 2
    total, count = 0.0, 0
    model.eval()
    for start in range(0, n_seqs, batch_size):
        b = min(batch_size, n_seqs - start)
        s = torch.randint(32, 127, (b, half), generator=gen)
        seq = torch.cat([s, s], dim=1)[:, :seq_len].to(device)
        x, y = seq[:, :-1], seq[:, 1:]
        with amp_ctx:
            out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        # y[:, half:] are the second copy minus its first char, which the model
        # can't know is the start of a repeat until it has seen it.
        loss = F.cross_entropy(
            logits[:, half:].float().reshape(-1, logits.shape[-1]),
            y[:, half:].reshape(-1),
            reduction="sum",
        )
        total += loss.item()
        count += y[:, half:].numel()
    model.train()
    return total / count / math.log(2)
