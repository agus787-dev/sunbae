"""One set_seed() used by every training script.

A fixed seed does not make training bit-for-bit identical (GPU kernels can
still vary), but it removes the biggest source of "did my change help, or
was I just lucky with the random init and batch order?".
"""

import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS
    torch.cuda.manual_seed_all(seed)
