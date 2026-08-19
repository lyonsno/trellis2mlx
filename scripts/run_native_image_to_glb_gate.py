#!/usr/bin/env python3
"""Run one report-bearing local gate for the native image-to-GLB witness."""

from __future__ import annotations

import argparse
from pathlib import Path

from trellmlx.witness_gate import (
    native_image_to_glb_gate_commands,
    native_image_to_glb_source_paths,
    run_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--profile", choices=("focused", "final"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    source_paths = native_image_to_glb_source_paths(root)
    report = run_gate(
        native_image_to_glb_gate_commands(args.profile),
        cwd=root,
        report_path=args.report,
        source_paths=source_paths,
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
