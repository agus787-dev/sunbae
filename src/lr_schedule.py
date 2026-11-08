"""The learning-rate schedule: linear warmup, then cosine decay.

    lr
    |        ____
    |      /      \\___
    |    /             \\___
    |  /                    \\________  <- lr_max * min_lr_ratio
    | /
    |/________________________________ step
     0   warmup_steps            max_steps

- Warmup: a random model should not take big steps at the start.
- Decay: a smaller lr at the end lets the weights settle into a minimum.
"""

import math


def get_lr(step, config):
    """Learning rate for one optimizer step.

    A pure function of `step`, so resuming only needs the step number.
    """
    if step < config.warmup_steps:
        # Linear ramp from ~0 up to lr_max, reached at warmup_steps.
        return config.lr_max * (step + 1) / config.warmup_steps

    lr_min = config.lr_max * config.min_lr_ratio
    if step >= config.max_steps:
        return lr_min

    # Cosine from lr_max down to lr_min over the steps after warmup.
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    return lr_min + coeff * (config.lr_max - lr_min)
