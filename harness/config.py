"""Config loading: configs/base.yaml + configs/<arch>.yaml + --set overrides."""

import copy
import os

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(arch: str, size: str, overrides: list[str]) -> dict:
    with open(os.path.join(CONFIG_DIR, "base.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(CONFIG_DIR, f"{arch}.yaml")) as f:
        arch_cfg = yaml.safe_load(f)

    if size not in cfg["sizes"]:
        raise KeyError(f"unknown size {size!r}; base.yaml defines {sorted(cfg['sizes'])}")
    if size not in arch_cfg.get("sizes", {}):
        raise KeyError(f"configs/{arch}.yaml has no shape for size {size!r}")

    cfg = deep_merge(cfg, {k: v for k, v in arch_cfg.items() if k != "sizes"})
    cfg["arch"] = arch
    cfg["size"] = size
    cfg["target_params"] = cfg["sizes"][size]["target_params"]
    cfg["model"] = deep_merge(cfg.get("model", {}), arch_cfg["sizes"][size])
    del cfg["sizes"]

    # --set train.tokens=2e6 model.dropout=0.1
    for item in overrides:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        value = yaml.safe_load(raw)
        if isinstance(value, str):  # allow 2e6 / 1_000_000 style numbers
            try:
                value = float(value) if any(c in value for c in ".eE") else int(value.replace("_", ""))
            except ValueError:
                pass
        node[leaf] = value
    return cfg
