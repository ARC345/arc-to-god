"""Print a markdown table of results/*.json, ready to paste into a post.

    python -m harness.summarize
"""

import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLUMNS = [
    ("arch", "arch", "{}"),
    ("size", "size", "{}"),
    ("params (non-emb)", "params_non_embedding", "{:.1f}M"),
    ("val BPC", "val_bpc", "{:.3f}"),
    ("test BPC", "test_bpc", "{:.3f}"),
    ("copy BPC", "copy_bpc", "{:.3f}"),
    ("tok/s", "train_tokens_per_s", "{:,.0f}"),
    ("peak mem (GB)", "peak_mem_gb", "{:.2f}"),
    ("device", "device", "{}"),
]


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "*.json"))):
        with open(path) as f:
            r = json.load(f)
        r["params_non_embedding"] = r["params_non_embedding"] / 1e6
        rows.append(r)
    rows.sort(key=lambda r: (r["size"], r["test_bpc"]))

    print("| " + " | ".join(c[0] for c in COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for r in rows:
        cells = [fmt.format(r[key]) if r.get(key) is not None else "-" for _, key, fmt in COLUMNS]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
