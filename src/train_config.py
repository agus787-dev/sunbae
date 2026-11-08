"""HOW we train (model_config.py is WHAT we train).

Read from the same run YAML as ModelConfig:
    data:      where train.bin / val.bin / meta.json are
    training:  batch size, schedule, optimizer, logging, checkpoints
    run.seed, run.device
"""

from dataclasses import asdict, dataclass, fields

import yaml


@dataclass
class TrainConfig:
    # ---- data (paths relative to the project root, or absolute) ----------
    train_bin: str = "dataset/train.bin"
    val_bin: str = "dataset/val.bin"
    meta: str = "dataset/meta.json"

    # ---- batching ---------------------------------------------------------
    # Micro-batch: sequences per forward pass (limited by GPU memory).
    batch_size: int = 32
    # Effective batch = batch_size * grad_accum_steps sequences per optimizer step.
    grad_accum_steps: int = 4

    # ---- schedule ---------------------------------------------------------
    max_steps: int = 8000
    warmup_steps: int = 200
    lr_max: float = 3e-4
    min_lr_ratio: float = 0.1

    # ---- optimizer --------------------------------------------------------
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # ---- logging / checkpoints --------------------------------------------
    log_interval: int = 20          # print every N optimizer steps
    eval_interval: int = 250        # measure val loss every N steps
    eval_iters: int = 50            # batches averaged per loss estimate
    checkpoint_interval: int = 250  # save last.pt every N steps

    # ---- misc -------------------------------------------------------------
    seed: int = 42                  # from run.seed
    device: str = "auto"            # from run.device; "auto" = cuda -> mps -> cpu
    # "auto": bf16 if the GPU supports it, else fp16 on CUDA (T4), else fp32.
    precision: str = "auto"
    out_dir: str = "checkpoints"

    def tokens_per_step(self, block_size):
        return self.batch_size * self.grad_accum_steps * block_size

    def to_dict(self):
        return asdict(self)


def load_train_config(path):
    """Read a TrainConfig from the "data:", "training:" and "run:" sections."""
    with open(path, "r", encoding="utf-8") as f:
        values = yaml.safe_load(f)

    merged = {**values.get("data", {}), **values.get("training", {})}
    run = values.get("run", {})
    for key in ("seed", "device"):
        if key in run:
            merged[key] = run[key]

    # Cast to the field's type, so e.g. "3e-4" (a string in YAML) still works.
    config = TrainConfig()
    for field in fields(TrainConfig):
        if field.name in merged:
            setattr(config, field.name, field.type(merged[field.name]))
    return config
