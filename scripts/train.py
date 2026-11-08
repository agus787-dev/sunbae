"""Train Sunbae.

    python scripts/train.py --config configs/run_01.yaml
    python scripts/train.py --config configs/run_01.yaml --resume checkpoints/last.pt
    python scripts/train.py --max_steps 30 --out_dir checkpoints/smoke     # short test

Run tests/test_model.py first: a bug found there takes seconds, a bug found
during training costs GPU hours.

Colab: put --data_dir and --out_dir on Google Drive, and after a disconnect
run the same command with --resume <out_dir>/last.pt.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.checkpoint import load_checkpoint  # noqa: E402
from src.model import Sunbae  # noqa: E402
from src.model_config import load_config  # noqa: E402
from src.seed import set_seed  # noqa: E402
from src.train_config import load_train_config  # noqa: E402
from src.trainer import Trainer, pick_device  # noqa: E402


def project_path(path):
    """Relative paths are relative to the project root; absolute ones stay as they are."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description="Train Sunbae.")
    parser.add_argument("--config", default="configs/run_01.yaml")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from, e.g. checkpoints/last.pt")
    parser.add_argument("--data_dir", default=None, help="folder with train.bin, val.bin, meta.json")
    parser.add_argument("--out_dir", default=None, help="folder for checkpoints and the log")
    parser.add_argument("--max_steps", type=int, default=None, help="override, e.g. for short tests")
    parser.add_argument("--device", default=None, help="auto, cuda, mps or cpu (default: run.device)")
    args = parser.parse_args()

    config_path = project_path(args.config)
    model_config = load_config(config_path)
    train_config = load_train_config(config_path)

    # Command-line values override the config.
    if args.data_dir:
        train_config.train_bin = str(Path(args.data_dir) / "train.bin")
        train_config.val_bin = str(Path(args.data_dir) / "val.bin")
        train_config.meta = str(Path(args.data_dir) / "meta.json")
    if args.out_dir:
        train_config.out_dir = args.out_dir
    if args.max_steps:
        train_config.max_steps = args.max_steps
    if args.device:
        train_config.device = args.device

    train_bin = project_path(train_config.train_bin)
    val_bin = project_path(train_config.val_bin)
    # Resolved paths are passed to the Trainer; train_config keeps the paths
    # as written, so checkpoints do not store /Users/... paths.
    out_dir = project_path(train_config.out_dir)

    # The model's vocabulary must be the one the dataset was built with,
    # otherwise token ids and embeddings do not match.
    meta = json.loads(project_path(train_config.meta).read_text(encoding="utf-8"))
    if meta["vocab_size"] != model_config.vocab_size:
        raise ValueError(
            f"config vocab_size ({model_config.vocab_size}) != meta.json vocab_size "
            f"({meta['vocab_size']}) - rebuild the dataset or fix the config"
        )

    set_seed(train_config.seed)
    device = pick_device(train_config.device)
    model = Sunbae(model_config)
    trainer = Trainer(model, model_config, train_config, train_bin, val_bin, out_dir, device)

    start_step, best_val_loss = 0, float("inf")
    if args.resume:
        # Load to CPU: RNG states must stay CPU tensors, and load_state_dict
        # copies weights and optimizer state to the model's device anyway.
        payload = load_checkpoint(project_path(args.resume), model, trainer.optimizer,
                                  trainer.scaler, map_location="cpu")
        trainer.set_rng_state(payload["rng_state"])
        start_step = payload["step"] + 1  # the saved step was already finished
        best_val_loss = payload["best_val_loss"]
        print(f"Resumed from {args.resume} at step {start_step}, best val loss {best_val_loss:.4f}")

    if start_step >= train_config.max_steps:
        print(f"Nothing to do: already at step {start_step}, max_steps is {train_config.max_steps}.")
        return

    print(f"=== Training Sunbae: {model.num_parameters():,} parameters, "
          f"{meta['n_tokens_train']:,} train tokens ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    best_val_loss = trainer.train(start_step, best_val_loss, log_path)

    print()
    print(f"Training finished. Best val loss: {best_val_loss:.4f}")
    print("Checkpoints and log in:", out_dir)


if __name__ == "__main__":
    main()
