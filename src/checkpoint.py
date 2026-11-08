"""Save and resume a training run.

On Colab the session can disappear at any time. A checkpoint with only the
weights would lose the run's PLACE: the optimizer's moments, the fp16 loss
scale, the step (= where we are on the lr curve) and the random state (which
batches come next). So every checkpoint holds all of them.
"""

import os
import tempfile
from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, scaler, step, best_val_loss,
                    model_config, train_config, rng_state):
    """Write a checkpoint atomically: temp file first, then os.replace.

    If the write is cut off halfway, `path` still holds the previous good
    checkpoint instead of a broken file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "best_val_loss": best_val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
        "rng_state": rng_state,
    }

    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)  # atomic rename
    except BaseException:
        os.unlink(tmp_path)
        raise


def load_checkpoint(path, model, optimizer=None, scaler=None, map_location=None):
    """Restore model (and optimizer/scaler if given). Return the whole payload."""
    # weights_only=True: only tensors and plain Python values are allowed,
    # so loading a checkpoint (e.g. copied from Drive) cannot run code.
    payload = torch.load(path, map_location=map_location, weights_only=True)

    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scaler is not None and payload.get("scaler_state"):
        scaler.load_state_dict(payload["scaler_state"])
    return payload
