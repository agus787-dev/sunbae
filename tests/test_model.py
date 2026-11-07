""" python tests/test_model.py

These are not style checks. Each one catches a specific, common bug that
would otherwise only show up hours into a real training run, after Mac GPU
time is already spent. Do not start training run_01 until every check below prints
PASS.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.model_config import load_config
from src.model import Sunbae

CONFIG_FILE = Path("configs/run_01.yaml")


def shape_test(model, config):
    """A random batch of shape (B, T) must produce logits (B, T, vocab_size)."""
    B, T = 4, 32
    batch = torch.randint(0, config.vocab_size, (B, T))
    logits, _ = model(batch)

    expected = (B, T, config.vocab_size)
    actual = tuple(logits.shape)
    assert actual == expected, "expected logits shape {}, got {}".format(
        expected, actual
    )


def random_init_loss_test(model, config):
    """Before any training, loss should be close to ln(vocab_size).

    An untrained model assigns roughly equal probability to every token, so
    its cross-entropy loss should equal the entropy of a uniform distribution
    over the vocabulary: -log(1 / vocab_size) = ln(vocab_size). A very
    different number means the loss, the input/target shift, or the causal
    mask is wrong.
    """
    B, T = 8, config.block_size
    batch = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    _, loss = model(batch, targets)
    expected = math.log(config.vocab_size)

    print("  random-init loss = {:.3f}   (expected ~= {:.3f})".format(
        loss.item(), expected))

    # A generous tolerance: this is a sanity check, not a precise prediction.
    assert abs(loss.item() - expected) < 1.0, (
        "loss is too far from ln(vocab_size) - check the loss, the "
        "input/target shift, or the causal mask"
    )


def overfit_one_batch_test(model, config):
    """Train on a single batch for 200 steps; loss must approach zero.

    If the model cannot memorise one batch, no amount of real data will fix
    it later - the bug is in the model or the optimiser, not the data.

    Dropout is switched off for this one check with model.eval(). Dropout is
    a regulariser: it exists to stop the model from memorising, which is
    exactly what this test asks the model to do. Gradients still flow fine
    in eval mode - only Dropout (and BatchNorm, which we don't use) behave
    differently between train and eval.
    """
    torch.manual_seed(0)
    model.eval()

    B, T = 4, 32
    batch = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    loss = None
    for _ in range(200):
        _, loss = model(batch, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("  final loss after 200 steps on one batch = {:.4f}".format(loss.item()))
    assert loss.item() < 0.1, (
        "loss did not collapse toward zero - the model or optimiser is broken"
    )


def causality_test(model, config):
    """Changing token t must not change logits at positions before t.

    This is the direct proof that the causal mask works: a position can only
    be influenced by itself and earlier positions, never a later one.
    """
    B, T = 1, 16
    batch = torch.randint(0, config.vocab_size, (B, T))

    model.eval()
    with torch.no_grad():
        logits_before, _ = model(batch)

        changed = batch.clone()
        t = T // 2
        # +1 mod vocab_size guarantees an actually different token, never the
        # same one by chance.
        changed[0, t] = (changed[0, t] + 1) % config.vocab_size

        logits_after, _ = model(changed)

    # Only positions before t are checked here - position t itself, and
    # everything after it, are allowed (and expected) to change.
    unaffected_before = torch.allclose(
        logits_before[:, :t], logits_after[:, :t], atol=1e-5
    )
    assert unaffected_before, (
        "logits before the changed position moved - the causal mask is leaking"
    )


def main():
    config = load_config(CONFIG_FILE)

    # Each check gets its own freshly-initialised model. The overfit test
    # trains its model to near-zero loss, and that must not contaminate the
    # random-init loss test that runs after it.
    checks = [
        ("Shape test", shape_test),
        ("Random-init loss test", random_init_loss_test),
        ("Overfit-one-batch test", overfit_one_batch_test),
        ("Causality test", causality_test),
    ]

    print()

    failures = 0
    for name, check in checks:
        print(name + "...")
        try:
            check(Sunbae(config), config)
            print("  PASS")
        except AssertionError as e:
            failures += 1
            print("  FAIL:", e)
        print()

    if failures:
        print("{} check(s) failed - fix".format(failures))
        sys.exit(1)
    else:
        print("All checks passed")


if __name__ == "__main__":
    main()