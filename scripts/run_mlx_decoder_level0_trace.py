"""Capture an evidence-bound MLX shape-decoder level-zero trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import traceback
from typing import Any

import numpy as np

from scripts.decoder_level0_trace_contract import (
    decoder_trace_input_sha256,
    write_decoder_level0_trace_npz,
)


SCHEMA = "trellis2mlx.decoder_level0_trace_run.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-slat-sample", required=True, type=Path)
    parser.add_argument("--expected-shape-slat-sha256", required=True)
    parser.add_argument("--shape-decoder-checkpoint", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--torso-dtype", required=True, choices=("fp16", "fp32"))
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _validate_digest(value: str, label: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be canonical lowercase SHA256")


def _load_shape_slat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"shape SLat sample does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = {"feats", "coords"} - set(archive.files)
        if missing:
            raise KeyError(
                "shape SLat sample missing required arrays: "
                + ", ".join(sorted(missing))
            )
        feats = np.asarray(archive["feats"])
        coords = np.asarray(archive["coords"])
    if feats.dtype != np.dtype(np.float32):
        raise ValueError(f"shape SLat feats must have dtype float32, got {feats.dtype}")
    if feats.ndim != 2 or feats.shape[1] != 32:
        raise ValueError(f"shape SLat feats must have shape [N, 32], got {feats.shape}")
    if coords.dtype != np.dtype(np.int32):
        raise ValueError(f"shape SLat coords must have dtype int32, got {coords.dtype}")
    if coords.ndim != 2 or coords.shape != (feats.shape[0], 4):
        raise ValueError(
            f"shape SLat coords must have shape [{feats.shape[0]}, 4], got {coords.shape}"
        )
    if feats.shape[0] == 0:
        raise ValueError("shape SLat sample must be nonempty")
    if not np.isfinite(feats).all():
        raise ValueError("shape SLat feats contain non-finite values")
    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise ValueError("shape SLat coords contain duplicate rows")
    return np.ascontiguousarray(feats), np.ascontiguousarray(coords)


def _validate_repo_state(expected_commit: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("--expected-repo-commit must be a full lowercase commit SHA")
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit:
        raise ValueError(
            f"repo commit mismatch: expected={expected_commit}, actual={actual}"
        )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError("repo worktree is dirty; refusing evidence capture")
    return {
        "root": str(REPO_ROOT),
        "commit": actual,
        "clean": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "failure_phase": None,
        "last_trustworthy_phase": "request_received",
        "requested_route": (
            f"mlx-shape-decoder-level0-trace-{args.torso_dtype}"
        ),
        "effective_route": None,
        "input_slat_sha256": None,
        "input_tensor_sha256": None,
        "checkpoint_sha256": None,
        "stale_primary_invalidated": False,
        "primary": {
            "path": str(args.output_npz),
            "status": "not_written",
            "sha256": None,
        },
    }
    phase = "request_validation"
    try:
        protected = (
            args.shape_slat_sample.resolve(),
            args.shape_decoder_checkpoint.resolve(),
            args.output_json.resolve(),
        )
        if args.output_npz.resolve() in protected:
            raise ValueError("--output-npz collides with an input or report path")
        if args.output_json.resolve() in (
            args.shape_slat_sample.resolve(),
            args.shape_decoder_checkpoint.resolve(),
        ):
            raise ValueError("--output-json collides with an input path")
        _validate_digest(
            args.expected_shape_slat_sha256,
            "--expected-shape-slat-sha256",
        )
        _validate_digest(
            args.expected_checkpoint_sha256,
            "--expected-checkpoint-sha256",
        )
        if args.output_npz.exists():
            args.output_npz.unlink()
            report["stale_primary_invalidated"] = True

        phase = "input_validation"
        feats, coords = _load_shape_slat(args.shape_slat_sample)
        input_sha = _sha256_file(args.shape_slat_sample)
        if input_sha != args.expected_shape_slat_sha256:
            raise ValueError(
                "shape SLat digest mismatch: "
                f"expected={args.expected_shape_slat_sha256}, actual={input_sha}"
            )
        report["input_slat_sha256"] = input_sha
        report["input_tensor_sha256"] = decoder_trace_input_sha256(
            feats,
            coords,
        )
        report["input"] = {
            "path": str(args.shape_slat_sample),
            "sha256": input_sha,
            "feats_shape": list(feats.shape),
            "coords_shape": list(coords.shape),
            "tensor_sha256": report["input_tensor_sha256"],
        }
        report["last_trustworthy_phase"] = phase

        phase = "checkpoint_validation"
        if not args.shape_decoder_checkpoint.is_file():
            raise FileNotFoundError(
                f"shape decoder checkpoint does not exist: {args.shape_decoder_checkpoint}"
            )
        checkpoint_sha = _sha256_file(args.shape_decoder_checkpoint)
        if checkpoint_sha != args.expected_checkpoint_sha256:
            raise ValueError(
                "shape decoder checkpoint digest mismatch: "
                f"expected={args.expected_checkpoint_sha256}, actual={checkpoint_sha}"
            )
        report["checkpoint_sha256"] = checkpoint_sha
        report["checkpoint"] = {
            "path": str(args.shape_decoder_checkpoint),
            "sha256": checkpoint_sha,
            "size_bytes": args.shape_decoder_checkpoint.stat().st_size,
        }
        report["last_trustworthy_phase"] = phase

        phase = "repo_validation"
        report["repo"] = _validate_repo_state(args.expected_repo_commit)
        report["last_trustworthy_phase"] = phase

        phase = "runtime_validation"
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        effective_device = str(mx.default_device())
        if "gpu" not in effective_device.lower():
            raise RuntimeError(
                f"MLX decoder trace requires Metal GPU, got {effective_device}"
            )
        effective_route = (
            f"mlx-shape-decoder-level0-trace-{args.torso_dtype}"
        )
        report["effective_route"] = effective_route
        report["effective_device"] = effective_device
        report["last_trustworthy_phase"] = phase

        phase = "model_load"
        from trellmlx.models.shape_slat_decoder import SLatDecoder
        from trellmlx.weight_loader import load_weights

        decoder = SLatDecoder(
            out_channels=7,
            pred_subdiv=True,
            use_fp16=args.torso_dtype == "fp16",
        )
        unloaded = load_weights(
            decoder,
            str(args.shape_decoder_checkpoint),
            verbose=False,
        )
        if unloaded:
            raise ValueError(
                f"shape decoder checkpoint has {len(unloaded)} unloaded keys"
            )
        report["last_trustworthy_phase"] = phase

        phase = "trace_capture"
        from trellmlx.decoder_level0_trace import (
            capture_mlx_decoder_level0_trace,
        )

        arrays = capture_mlx_decoder_level0_trace(
            decoder,
            mx.array(feats),
            mx.array(coords),
        )
        torso_dtype = np.float16 if args.torso_dtype == "fp16" else np.float32
        validation = write_decoder_level0_trace_npz(
            args.output_npz,
            arrays,
            latent_channels=32,
            channels=1024,
            torso_dtype=torso_dtype,
        )
        primary_sha = _sha256_file(args.output_npz)
        report["primary"] = {
            "path": str(args.output_npz),
            "status": "written",
            "sha256": primary_sha,
            "size_bytes": args.output_npz.stat().st_size,
            "validation": validation,
        }
        report.update(
            {
                "status": "passed",
                "failure_phase": None,
                "last_trustworthy_phase": "trace_primary_reopened_exact",
            }
        )
        _write_report(args.output_json, report)
        return 0
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _write_report(args.output_json, report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
