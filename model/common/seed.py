"""Global seeding utility used by every entry point."""
from __future__ import annotations

import os
import random


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python ``random``, NumPy, and Torch (CPU+CUDA if available)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass
