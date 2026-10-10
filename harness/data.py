"""enwik8: download once, split 90M/5M/5M bytes (the standard split), serve batches.

Every byte is a token, so the vocabulary is 256. Literature reports this as
bits per character (BPC) even though it is technically bits per byte.
"""

import os
import urllib.request
import zipfile

import numpy as np
import torch

URL = "http://mattmahoney.net/dc/enwik8.zip"
VOCAB_SIZE = 256
SPLITS = {
    "train": (0, 90_000_000),
    "val": (90_000_000, 95_000_000),
    "test": (95_000_000, 100_000_000),
}


def prepare(data_dir: str) -> dict[str, str]:
    """Download enwik8 if needed and write one uint8 .bin file per split."""
    os.makedirs(data_dir, exist_ok=True)
    paths = {s: os.path.join(data_dir, f"{s}.bin") for s in SPLITS}
    if all(os.path.exists(p) for p in paths.values()):
        return paths

    zpath = os.path.join(data_dir, "enwik8.zip")
    if not os.path.exists(zpath):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        raw = z.read("enwik8")
    assert len(raw) == 100_000_000, f"unexpected enwik8 size {len(raw)}"

    for split, (a, b) in SPLITS.items():
        np.frombuffer(raw[a:b], dtype=np.uint8).tofile(paths[split])
    return paths


class Split:
    def __init__(self, path: str):
        self.data = np.memmap(path, dtype=np.uint8, mode="r")

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int, seq_len: int, gen: torch.Generator):
        """Random windows. Returns CPU int64 tensors x, y of shape (B, T)."""
        ix = torch.randint(len(self.data) - seq_len - 1, (batch_size,), generator=gen)
        chunks = np.stack([self.data[i : i + seq_len + 1] for i in ix.tolist()])
        chunks = torch.from_numpy(chunks.astype(np.int64))
        return chunks[:, :-1], chunks[:, 1:]

    def windows(self, batch_size: int, seq_len: int):
        """Every non-overlapping window over the split, in order (for test eval)."""
        n = (len(self.data) - 1) // seq_len
        for start in range(0, n, batch_size):
            idx = range(start, min(start + batch_size, n))
            chunks = np.stack([self.data[i * seq_len : i * seq_len + seq_len + 1] for i in idx])
            chunks = torch.from_numpy(chunks.astype(np.int64))
            yield chunks[:, :-1], chunks[:, 1:]
