"""The training loop.

    get a batch -> forward -> loss -> backward (accumulated) -> clip -> step
    -> sometimes: log, evaluate, save a checkpoint
"""

import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from src.checkpoint import save_checkpoint
from src.dataset_builder import DatasetBuilder
from src.lr_schedule import get_lr


def pick_device(requested):
    """"auto" -> cuda, then mps (Mac GPU), then cpu. Otherwise use what was asked."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested, but no CUDA GPU is available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' requested, but MPS is not available")
    return torch.device(requested)


def pick_precision(device, requested):
    """Turn "auto" (or an explicit choice) into a torch dtype.

    Mixed precision is only used on CUDA. A T4 (free Colab GPU) has no fast
    bf16, so "auto" picks fp16 there and bf16 on newer GPUs (A100, L4).
    """
    if device.type != "cuda":
        if requested not in ("auto", "fp32"):
            print(f"No CUDA device - ignoring precision={requested!r}, using fp32.")
        return torch.float32
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    if requested == "fp32":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def format_minutes(minutes):
    """95.5 -> '1h36m', 12.3 -> '12m'."""
    hours, mins = divmod(int(minutes), 60)
    return f"{hours}h{mins:02d}m" if hours else f"{mins}m"


class Trainer:
    def __init__(self, model, model_config, train_config, train_bin, val_bin, out_dir, device):
        self.model = model.to(device)
        self.model_config = model_config
        self.train_config = train_config
        self.train_bin = train_bin
        self.val_bin = val_bin
        self.out_dir = Path(out_dir)  # where last.pt / best.pt go
        self.device = device

        # Batch positions come from this generator, so its state can be saved
        # and a resumed run continues with the same batches it would have seen.
        self.rng = np.random.default_rng(train_config.seed)

        self.dtype = pick_precision(device, train_config.precision)
        self.use_amp = device.type == "cuda" and self.dtype != torch.float32
        # fp16 has a small range, so tiny gradients can become 0: GradScaler
        # scales the loss up first. bf16/fp32 do not need it (enabled=False
        # makes the scaler do nothing, so the loop has only one code path).
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.dtype == torch.float16)

        self.optimizer = model.configure_optimizers(
            weight_decay=train_config.weight_decay,
            lr=train_config.lr_max,
            betas=(train_config.beta1, train_config.beta2),
        )
        print("Device:", device, "| precision:", self.dtype)

    # ---- random state (saved in checkpoints) ------------------------------

    def get_rng_state(self):
        state = {"torch": torch.get_rng_state(), "numpy_batches": self.rng.bit_generator.state}
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        if torch.backends.mps.is_available():
            state["mps"] = torch.mps.get_rng_state()
        return state

    def set_rng_state(self, state):
        torch.set_rng_state(state["torch"])
        self.rng.bit_generator.state = state["numpy_batches"]
        # A checkpoint from another machine may not have these; skip them then.
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
        if "mps" in state and torch.backends.mps.is_available():
            torch.mps.set_rng_state(state["mps"])

    # ---- batches and evaluation -------------------------------------------

    def get_batch(self, split):
        path = self.train_bin if split == "train" else self.val_bin
        x, y = DatasetBuilder.get_batch(
            path, self.train_config.batch_size, self.model_config.block_size, rng=self.rng
        )
        x = torch.from_numpy(x).to(self.device, non_blocking=True)
        y = torch.from_numpy(y).to(self.device, non_blocking=True)
        return x, y

    def autocast(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    @torch.no_grad()
    def estimate_loss(self):
        """Average loss over eval_iters batches, for train and val."""
        self.model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(self.train_config.eval_iters)
            for i in range(self.train_config.eval_iters):
                x, y = self.get_batch(split)
                with self.autocast():
                    _, loss = self.model(x, y)
                losses[i] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def save(self, name, step, best_val_loss):
        path = self.out_dir / name
        save_checkpoint(
            path, self.model, self.optimizer, self.scaler, step, best_val_loss,
            self.model_config, self.train_config, self.get_rng_state(),
        )
        return path

    # ---- the loop ---------------------------------------------------------

    def train(self, start_step=0, best_val_loss=float("inf"), log_path=None):
        cfg = self.train_config
        tokens_per_step = cfg.tokens_per_step(self.model_config.block_size)
        print(f"Tokens per optimizer step: {tokens_per_step:,}")
        print(f"Planned total tokens     : {tokens_per_step * cfg.max_steps:,}")
        print(f"Steps                    : {start_step} -> {cfg.max_steps - 1}")
        print()

        def log(record):
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

        self.model.train()
        t_last, steps_since_log = time.time(), 0

        for step in range(start_step, cfg.max_steps):
            lr = get_lr(step, cfg)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            # Accumulate gradients over several micro-batches = one big batch.
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(cfg.grad_accum_steps):
                x, y = self.get_batch("train")
                with self.autocast():
                    _, loss = self.model(x, y)
                    # Divide so the summed gradients equal the AVERAGE over the big batch.
                    loss = loss / cfg.grad_accum_steps
                self.scaler.scale(loss).backward()
                step_loss += loss.item()

            # Unscale first, so grad_clip is compared with the real gradient size.
            self.scaler.unscale_(self.optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            steps_since_log += 1

            is_last = step == cfg.max_steps - 1
            if step % cfg.log_interval == 0 or is_last:
                now = time.time()
                tok_per_sec = steps_since_log * tokens_per_step / max(now - t_last, 1e-6)
                t_last, steps_since_log = now, 0
                eta_min = (cfg.max_steps - step - 1) * tokens_per_step / tok_per_sec / 60
                print(f"step {step:5d} | loss {step_loss:.4f} | lr {lr:.2e} | grad_norm {grad_norm:.2f} "
                      f"| tok/s {tok_per_sec:,.0f} | eta {format_minutes(eta_min)}")
                log({"step": step, "train_loss": step_loss, "lr": lr,
                     "grad_norm": grad_norm, "tokens_per_sec": tok_per_sec})

            # step > start_step: no eval/save right after a resume (nothing new yet).
            do_eval = (step % cfg.eval_interval == 0 and step > start_step) or is_last
            do_ckpt = (step % cfg.checkpoint_interval == 0 and step > start_step) or is_last
            t_pause = time.time()

            if do_eval:
                losses = self.estimate_loss()
                print(f"  eval @ step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
                log({"step": step, "eval": losses})
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    path = self.save("best.pt", step, best_val_loss)
                    print(f"  -> new best val loss, saved {path}")

            if do_ckpt:
                path = self.save("last.pt", step, best_val_loss)
                print(f"  -> saved {path}")

            # Eval/save time should not count against tokens/sec.
            t_last += time.time() - t_pause

        return best_val_loss
