#!/usr/bin/env python3
"""Kaggle entrypoint for the exact saved-source-field native o-voxel control."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import traceback


EXPECTED_SHA256 = "ddd8cfd389c73e30e254df3066b09d9043bc06929e1df901f9ee61e69940cad5"
EXPECTED_KERNEL_SLUG = "t2mlx-kiln-1024-exact-slat-cuda-decode"
MOUNT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
RECEIPT = OUT / "native_grid_kernel_receipt.json"


def write_receipt(payload: dict) -> None:
    RECEIPT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    receipt = {
        "schema": "trellis2mlx.kaggle_native_grid_extract_receipt.v1",
        "status": "failed", "failure_phase": "mount_preflight",
        "expected_input_sha256": EXPECTED_SHA256,
        "expected_kernel_slug": EXPECTED_KERNEL_SLUG,
        "native_report": str(OUT / "native_grid_report.json"),
    }
    write_receipt(receipt)
    try:
        candidates = sorted(MOUNT.rglob("cuda_decoder_state.npz"))
        if len(candidates) != 1 or EXPECTED_KERNEL_SLUG not in candidates[0].parts:
            raise ValueError(f"expected one mounted prior-kernel decoder state under {EXPECTED_KERNEL_SLUG}, found {candidates}")
        decoder = candidates[0]
        receipt["mounted_input"] = str(decoder)
        receipt["failure_phase"] = "native_assay"
        write_receipt(receipt)
        command = [
            sys.executable, str(Path(__file__).with_name("source_cuda_native_grid_extract.py")),
            "--input-npz", str(decoder), "--expected-sha256", EXPECTED_SHA256,
            "--output-npz", str(OUT / "native_grid_mesh.npz"),
            "--report", str(OUT / "native_grid_report.json"),
            "--work-dir", str(OUT / "native-grid-work"),
            "--child-report", str(OUT / "native_grid_child.json"),
            "--child-stdout", str(OUT / "native_grid_child.stdout.log"),
            "--child-stderr", str(OUT / "native_grid_child.stderr.log"),
            "--resolution", "1024",
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        (OUT / "native_grid_parent.stdout.log").write_text(result.stdout)
        (OUT / "native_grid_parent.stderr.log").write_text(result.stderr)
        receipt["child_returncode"] = result.returncode
        native = json.loads((OUT / "native_grid_report.json").read_text())
        receipt["native_status"] = native["status"]
        if result.returncode or native["status"] != "done":
            raise RuntimeError(f"native assay failed: process={result.returncode}, phase={native.get('failure_phase')}")
        receipt.update(status="done", failure_phase=None)
        write_receipt(receipt)
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write_receipt(receipt)
    # Publish the failure receipt and child logs even for a native crash.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
