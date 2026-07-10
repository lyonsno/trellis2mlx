#!/usr/bin/env python3
"""CLI shim for Kaggle CUDA witness packets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trellmlx.kaggle_cuda_witness import main


if __name__ == "__main__":
    raise SystemExit(main())
