"""Architecture registry.

Every architecture lives in harness/models/<name>.py and registers itself:

    @register("gpt2")
    class GPT2(nn.Module):
        def __init__(self, cfg: ModelConfig): ...
        def forward(self, idx):  # idx: (B, T) int64
            return logits         # (B, T, vocab_size)
            # or (logits, aux_loss) -- aux_loss is added to the training loss
            # only (e.g. MoE load balancing); reported BPC never includes it.

The harness owns the loss, optimizer, data, and evaluation, so a model file
contains nothing but the architecture.
"""

from dataclasses import dataclass, field

MODELS: dict[str, type] = {}


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int
    d_model: int
    n_layer: int
    n_head: int = 1
    dropout: float = 0.0
    extra: dict = field(default_factory=dict)  # architecture-specific knobs


def register(name: str):
    def deco(cls):
        if name in MODELS:
            raise ValueError(f"architecture {name!r} registered twice")
        MODELS[name] = cls
        return cls

    return deco


def build(name: str, cfg: ModelConfig):
    import harness.models  # noqa: F401  (imports every model module)

    if name not in MODELS:
        raise KeyError(f"unknown architecture {name!r}; registered: {sorted(MODELS)}")
    return MODELS[name](cfg)
