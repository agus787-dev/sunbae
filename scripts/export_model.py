"""Export a training checkpoint to a small file for serving.

    python scripts/export_model.py --input checkpoints/best.pt
    python scripts/export_model.py --input checkpoints/best.pt --output models/sunbae.pt

A training checkpoint holds the weights plus AdamW's two moment tensors,
the scaler and RNG state - about 4x the model. Serving only needs the
weights, so we keep those (in float16, half the size again), the model
config, and the step / val loss so /health can say which model is running.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.model import Sunbae  # noqa: E402
from src.model_config import ModelConfig  # noqa: E402


def project_path(path):
    """Relative paths are relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description="Export model weights for serving.")
    parser.add_argument("--input", required=True, help="training checkpoint, e.g. checkpoints/best.pt")
    parser.add_argument("--output", default="models/sunbae.pt")
    args = parser.parse_args()
    input_path, output_path = project_path(args.input), project_path(args.output)

    checkpoint = torch.load(input_path, map_location="cpu", weights_only=True)

    # Load into a real model first, so tied weights stay ONE tensor:
    # token_embedding and lm_head share storage, and torch.save writes it once.
    config = ModelConfig(**checkpoint["model_config"])
    model = Sunbae(config)
    model.load_state_dict(checkpoint["model_state"])
    model.half()  # float32 -> float16 (only floating-point tensors change)

    export = {
        "model_state": model.state_dict(),
        "model_config": config.to_dict(),
        "step": checkpoint["step"],
        "val_loss": checkpoint["best_val_loss"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export, output_path)

    print(f"Step {export['step']}, val loss {export['val_loss']:.4f}, "
          f"{model.num_parameters():,} parameters")
    print(f"Input : {args.input:<24} {input_path.stat().st_size / 1e6:>8.1f} MB")
    print(f"Output: {args.output:<24} {output_path.stat().st_size / 1e6:>8.1f} MB")


if __name__ == "__main__":
    main()
