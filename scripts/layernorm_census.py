#!/usr/bin/env python3
"""CLI wrapper for TRELLIS LayerNorm witness/census diagnostics."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trellmlx.layernorm_census import main


if __name__ == "__main__":
    raise SystemExit(main())
