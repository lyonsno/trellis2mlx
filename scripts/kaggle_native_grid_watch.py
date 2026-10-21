"""Retain the terminal Kaggle native-grid result, including failed runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import traceback

import numpy as np


KERNEL_ID = "noahboo/t2mlx-kiln-1024-native-cuda-grid-r6"
DECLARED_KERNEL_ID = "noahboo/t2mlx-kiln-1024-native-cuda-grid-r6-1021"
EXPECTED_ASSAY_ID = "kiln-1024-full-cuda-grid-r6-20261021"
# Digest of the frozen single-file r6 kernel; the kernel measures its own bytes.
EXPECTED_KERNEL_SOURCE_SHA256 = "10b92362b58af0d9901d1178035e7c0f777ad2e0e96e946d2ef38211409de0f7"
EXPECTED_INPUT_SHA256 = "ddd8cfd389c73e30e254df3066b09d9043bc06929e1df901f9ee61e69940cad5"
EXPECTED_SOURCE_COMMIT = "5565d240c4a494caaf9ece7a554542b76ffa36d3"
EXPECTED_NATIVE_SOURCE_HEADS = {
    "trellis": EXPECTED_SOURCE_COMMIT,
    "cumesh": "c4ad6125924fcedfd13f0bd61520ca2d24eb7a87",
    "flex_gemm": "6dd94a859c26ee8246888502eada3dd8ad85532e",
    "nvdiffrast": "253ac4fcea7de5f396371124af597e6cc957bfae",
}
TERMINAL = {"COMPLETE", "ERROR", "FAILED", "CANCELLED", "CANCELED"}
PENDING = {"QUEUED", "RUNNING", "PENDING", "PREPARING"}


def status_from_text(output: str) -> str:
    matches = re.findall(r"KernelWorkerStatus\.([A-Z_]+)", output)
    if len(matches) != 1 or matches[0] not in TERMINAL | PENDING:
        raise ValueError(f"unrecognized Kaggle status: {output!r}")
    return matches[0]


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_native_result(output_dir: Path) -> dict:
    """Admit only a current, hash-bound mesh from the named native assay."""
    receipt = json.loads((output_dir / "native_grid_kernel_receipt.json").read_text())
    native = json.loads((output_dir / "native_grid_report.json").read_text())
    mesh_path = output_dir / "native_grid_mesh.npz"
    if receipt.get("status") != "done" or native.get("status") != "done":
        raise ValueError("native assay did not report completion")
    if receipt.get("native_status") != "done" or receipt.get("assay_returncode") != 0 or native.get("preflight_returncode") != 0 or native.get("child_returncode") != 0:
        raise ValueError("native execution status or return code contradicts completion")
    if receipt.get("kernel_id") != DECLARED_KERNEL_ID or receipt.get("kernel_source_sha256") != EXPECTED_KERNEL_SOURCE_SHA256:
        raise ValueError("kernel declared identity or executed script mismatch")
    if receipt.get("assay_id") != EXPECTED_ASSAY_ID:
        raise ValueError("native receipt assay identity mismatch")
    if receipt.get("expected_input_sha256") != EXPECTED_INPUT_SHA256 or native.get("expected_sha256") != EXPECTED_INPUT_SHA256:
        raise ValueError("native decoder input identity mismatch")
    if native.get("source_commit") != EXPECTED_SOURCE_COMMIT or native.get("source_head") != EXPECTED_SOURCE_COMMIT:
        raise ValueError("native source identity mismatch")
    if native.get("native_source_heads") != EXPECTED_NATIVE_SOURCE_HEADS:
        raise ValueError("native dependency source identity mismatch")
    if native.get("resolution") != 1024:
        raise ValueError("native grid resolution mismatch")
    route = native.get("effective_route") or {}
    if "T4" not in str(route.get("cuda_device", "")) or "/o-voxel/" not in str(route.get("convert_path", "")) or not route.get("extension_path"):
        raise ValueError("native converter route identity missing")
    mesh = native.get("mesh") or {}
    if not mesh_path.is_file() or mesh_path.stat().st_size == 0:
        raise ValueError("native mesh missing or empty")
    if not isinstance(mesh.get("vertices"), int) or mesh["vertices"] < 3 or not isinstance(mesh.get("faces"), int) or mesh["faces"] < 1:
        raise ValueError("native mesh counts missing")
    with mesh_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if mesh.get("output_sha256") != digest:
        raise ValueError("native mesh SHA256 mismatch")
    with np.load(mesh_path, allow_pickle=False) as saved:
        if set(saved.files) != {"vertices", "faces"}:
            raise ValueError("native mesh arrays missing or unexpected")
        vertices, faces = saved["vertices"], saved["faces"]
    if vertices.shape != (mesh["vertices"], 3) or faces.shape != (mesh["faces"], 3):
        raise ValueError("native mesh array counts mismatch")
    if not np.issubdtype(vertices.dtype, np.floating) or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("native mesh arrays have invalid coordinate or triangle-index dtypes")
    if not np.isfinite(vertices).all() or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("native mesh arrays are invalid")
    return {"mesh_path": str(mesh_path), "mesh_sha256": digest, "vertices": mesh["vertices"], "faces": mesh["faces"], "effective_route": route,
            "declared_kernel_id": DECLARED_KERNEL_ID, "effective_kernel_id": KERNEL_ID,
            "kernel_alias_drift": DECLARED_KERNEL_ID != KERNEL_ID,
            "executed_script_sha256": receipt["kernel_source_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-id", default=KERNEL_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    report = {"schema": "trellis2mlx.kaggle_native_grid_watch.v1", "status": "running",
              "phase": "status", "kernel_id": args.kernel_id, "output_dir": str(args.output_dir)}
    write_report(args.report, report)
    try:
        # Kaggle CLI accepts /version syntax but ignores it for both status and
        # output requests. This assay uses one isolated kernel slug and exact
        # executed-script hash; a fabricated version suffix is never evidence.
        if args.kernel_id != KERNEL_ID:
            raise ValueError("kernel identity mismatch; use the isolated assay slug without a version suffix")
        if args.poll_seconds <= 0:
            raise ValueError("poll interval must be positive")
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("nonempty output directory may contain stale Kaggle evidence")
        while True:
            observed = subprocess.run(["kaggle", "kernels", "status", args.kernel_id],
                                      capture_output=True, text=True, check=False)
            if observed.returncode:
                raise RuntimeError(f"Kaggle status exited {observed.returncode}: {observed.stderr}")
            state = status_from_text(observed.stdout)
            report.update(kernel_status=state, last_trustworthy_phase="remote_status_observed")
            write_report(args.report, report)
            if state in TERMINAL:
                break
            time.sleep(args.poll_seconds)
        report["phase"] = "download"
        write_report(args.report, report)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        download = subprocess.run(["kaggle", "kernels", "output", args.kernel_id, "--path", str(args.output_dir)],
                                  capture_output=True, text=True, check=False)
        report.update(download_returncode=download.returncode, download_stdout=download.stdout,
                      download_stderr=download.stderr)
        write_report(args.report, report)
        receipt_path = args.output_dir / "native_grid_kernel_receipt.json"
        if not receipt_path.exists():
            raise RuntimeError("kernel terminated without its durable receipt")
        receipt = json.loads(receipt_path.read_text())
        report["child_status"] = receipt.get("status")
        report["child_report"] = receipt.get("report")
        if state != "COMPLETE" or download.returncode or receipt.get("status") != "done":
            raise RuntimeError(f"kernel={state}, download={download.returncode}, child={receipt.get('status')}")
        verified = verify_native_result(args.output_dir)
        report.update(status="done", phase="complete", last_trustworthy_phase="native_mesh_hash_verified", **verified)
        write_report(args.report, report)
        return 0
    except BaseException as exc:
        report.update(status="failed", failure_phase=report["phase"], error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        write_report(args.report, report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
