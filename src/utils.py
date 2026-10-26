# src/utils.py
import os
import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def seed_everything(seed: int = 42):
    """Sets random seed for complete reproducibility across libraries."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
