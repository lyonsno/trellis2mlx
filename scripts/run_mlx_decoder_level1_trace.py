"""Capture an evidence-bound MLX first-upsample and level-one decoder trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import traceback
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.decoder_level1_trace_contract import (
    LEVEL1_HASH_LEDGER_SCHEMA,
    decoder_level1_hash_entry,
    decoder_level1_trace_input_sha256,
    validate_decoder_level1_hash_ledger,
    write_decoder_level1_trace_npz,
)


SCHEMA = "trellis2mlx.decoder_level1_trace_run.v1"
ROUTE = "mlx-shape-decoder-level1-trace"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level0-trace", required=True, type=Path)
    parser.add_argument("--expected-level0-trace-sha256", required=True)
    parser.add_argument("--shape-decoder-checkpoint", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--decoder-silu-lut", required=True, type=Path)
    parser.add_argument("--expected-decoder-silu-lut-sha256", required=True)
    parser.add_argument("--turing-rsqrt-lut", required=True, type=Path)
    parser.add_argument("--expected-turing-rsqrt-lut-sha256", required=True)
    parser.add_argument("--expected-repo-commit", required=True)
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


def _load_turing_rsqrt_lut(
    path: Path,
    expected_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    _validate_digest(
        expected_sha256,
        "--expected-turing-rsqrt-lut-sha256",
    )
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Turing rsqrt LUT does not exist: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Turing rsqrt LUT digest mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    with np.load(path, allow_pickle=False) as archive:
        if "normalized_delta" not in archive.files:
            raise ValueError("Turing rsqrt LUT NPZ omits normalized_delta")
        correction = np.asarray(archive["normalized_delta"])
    if correction.dtype != np.dtype(np.int8) or correction.shape != (1 << 24,):
        raise ValueError(
            "Turing rsqrt LUT normalized_delta must be int8[16777216], "
            f"got dtype={correction.dtype}, shape={correction.shape}"
        )
    correction = np.ascontiguousarray(correction)
    return correction, {
        "path": str(path.resolve()),
        "sha256": actual_sha256,
        "normalized_delta_sha256": hashlib.sha256(
            correction.tobytes()
        ).hexdigest(),
        "entries": int(correction.size),
        "dtype": str(correction.dtype),
    }


def _failure_sibling(requested: Path, protected: set[Path]) -> Path:
    for index in range(len(protected) + 1):
        suffix = ".failure.json" if index == 0 else f".failure.{index}.json"
        candidate = requested.with_name(requested.name + suffix)
        if candidate.resolve() not in protected:
            return candidate
    raise RuntimeError("could not derive a non-colliding failure report path")


def _load_parent_state(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"level-zero trace does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = {"coords", "block3_output"} - set(archive.files)
        if missing:
            raise KeyError(
                "level-zero trace missing required arrays: "
                + ", ".join(sorted(missing))
            )
        coords = np.asarray(archive["coords"])
        level0_output = np.asarray(archive["block3_output"])
    if coords.dtype != np.dtype(np.int32):
        raise ValueError(f"parent coords must have dtype int32, got {coords.dtype}")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"parent coords must have shape [N, 4], got {coords.shape}")
    if level0_output.dtype != np.dtype(np.float16):
        raise ValueError(
            "level-zero output must have dtype float16, "
            f"got {level0_output.dtype}"
        )
    if level0_output.ndim != 2 or level0_output.shape != (
        coords.shape[0],
        1024,
    ):
        raise ValueError(
            "level-zero output must have shape "
            f"[{coords.shape[0]}, 1024], got {level0_output.shape}"
        )
    if coords.shape[0] == 0:
        raise ValueError("level-zero parent state must be nonempty")
    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise ValueError("parent coords contain duplicate rows")
    if not np.isfinite(level0_output).all():
        raise ValueError("level-zero output contains non-finite values")
    return (
        np.ascontiguousarray(level0_output),
        np.ascontiguousarray(coords),
    )


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
    effective_report_path = args.output_json
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "failure_phase": None,
        "last_trustworthy_phase": "request_received",
        "requested_route": {
            "route": ROUTE,
            "device_type": "metal",
            "decoder_linear_backend": "turing_fda",
            "sparse_conv_matmul_backend": "turing_fda",
            "decoder_layernorm_backend": "cuda-welford-turing-t4",
            "decoder_silu_backend": "cuda-turing-t4-fp16-lut",
            "parent_state": "externally-captured-level0-trace",
        },
        "effective_route": None,
        "input_tensor_sha256": None,
        "requested_report_path": str(args.output_json),
        "effective_report_path": str(effective_report_path),
        "stale_primary_invalidated": False,
        "primary": {
            "path": str(args.output_npz),
            "status": "not_written",
            "sha256": None,
        },
    }
    phase = "request_validation"
    previous_linear = os.environ.get("TRELLIS2MLX_DECODER_LINEAR_BACKEND")
    previous_sparse = os.environ.get("TRELLIS2MLX_SPARSE_CONV_MATMUL_BACKEND")
    try:
        protected_inputs = {
            args.level0_trace.resolve(),
            args.shape_decoder_checkpoint.resolve(),
            args.decoder_silu_lut.resolve(),
            args.turing_rsqrt_lut.resolve(),
        }
        primary_path = args.output_npz.resolve()
        protected_report_paths = protected_inputs | {args.output_npz.resolve()}
        if args.output_json.resolve() in protected_report_paths:
            effective_report_path = _failure_sibling(
                args.output_json,
                protected_report_paths,
            )
            report["effective_report_path"] = str(effective_report_path)
        if primary_path in protected_inputs:
            raise ValueError("--output-npz collides with an input or report path")
        if args.output_npz.exists():
            args.output_npz.unlink()
            report["stale_primary_invalidated"] = True
        if args.output_json.resolve() in protected_report_paths:
            raise ValueError("--output-json collides with an input or primary path")
        for value, label in (
            (
                args.expected_level0_trace_sha256,
                "--expected-level0-trace-sha256",
            ),
            (args.expected_checkpoint_sha256, "--expected-checkpoint-sha256"),
            (
                args.expected_decoder_silu_lut_sha256,
                "--expected-decoder-silu-lut-sha256",
            ),
            (
                args.expected_turing_rsqrt_lut_sha256,
                "--expected-turing-rsqrt-lut-sha256",
            ),
        ):
            _validate_digest(value, label)
        report["last_trustworthy_phase"] = phase

        phase = "parent_trace_validation"
        level0_output, parent_coords = _load_parent_state(args.level0_trace)
        parent_sha = _sha256_file(args.level0_trace)
        if parent_sha != args.expected_level0_trace_sha256:
            raise ValueError(
                "level-zero trace digest mismatch: "
                f"expected={args.expected_level0_trace_sha256}, actual={parent_sha}"
            )
        input_identity = decoder_level1_trace_input_sha256(
            level0_output,
            parent_coords,
        )
        report["input_tensor_sha256"] = input_identity
        report["parent_trace"] = {
            "path": str(args.level0_trace.resolve()),
            "sha256": parent_sha,
            "level0_output_shape": list(level0_output.shape),
            "parent_coords_shape": list(parent_coords.shape),
            "input_tensor_sha256": input_identity,
        }
        report["last_trustworthy_phase"] = phase

        phase = "layernorm_lut_validation"
        turing_rsqrt_lut, turing_rsqrt_lut_identity = _load_turing_rsqrt_lut(
            args.turing_rsqrt_lut,
            args.expected_turing_rsqrt_lut_sha256,
        )
        report["turing_rsqrt_lut"] = turing_rsqrt_lut_identity
        report["last_trustworthy_phase"] = phase

        phase = "checkpoint_validation"
        if not args.shape_decoder_checkpoint.is_file():
            raise FileNotFoundError(
                "shape decoder checkpoint does not exist: "
                f"{args.shape_decoder_checkpoint}"
            )
        checkpoint_sha = _sha256_file(args.shape_decoder_checkpoint)
        if checkpoint_sha != args.expected_checkpoint_sha256:
            raise ValueError(
                "shape decoder checkpoint digest mismatch: "
                f"expected={args.expected_checkpoint_sha256}, "
                f"actual={checkpoint_sha}"
            )
        report["checkpoint"] = {
            "path": str(args.shape_decoder_checkpoint.resolve()),
            "sha256": checkpoint_sha,
            "size_bytes": args.shape_decoder_checkpoint.stat().st_size,
        }
        report["last_trustworthy_phase"] = phase

        phase = "repo_validation"
        report["repo"] = _validate_repo_state(args.expected_repo_commit)
        report["last_trustworthy_phase"] = phase

        phase = "runtime_validation"
        import mlx.core as mx

        from trellmlx.decoder_turing_layernorm import (
            CUDA_WELFORD_TURING_T4_BACKEND as LAYERNORM_BACKEND,
            configure_decoder_layernorm_backend,
            decoder_layernorm_backend_identity,
        )
        from trellmlx.decoder_turing_silu import (
            CUDA_TURING_T4_LUT_BACKEND,
            configure_decoder_silu_backend,
            decoder_silu_backend_identity,
        )

        mx.set_default_device(mx.gpu)
        effective_device = str(mx.default_device())
        if "gpu" not in effective_device.lower():
            raise RuntimeError(
                f"MLX decoder trace requires Metal GPU, got {effective_device}"
            )
        os.environ["TRELLIS2MLX_DECODER_LINEAR_BACKEND"] = "turing_fda"
        os.environ["TRELLIS2MLX_SPARSE_CONV_MATMUL_BACKEND"] = "turing_fda"
        configure_decoder_layernorm_backend(
            LAYERNORM_BACKEND,
            turing_rsqrt_delta_lut=mx.array(turing_rsqrt_lut),
            turing_rsqrt_lut_artifact_sha256_attested=(
                args.expected_turing_rsqrt_lut_sha256
            ),
        )
        configure_decoder_silu_backend(
            CUDA_TURING_T4_LUT_BACKEND,
            output_lut_artifact_path=args.decoder_silu_lut,
            output_lut_artifact_sha256_attested=(
                args.expected_decoder_silu_lut_sha256
            ),
        )
        effective_route = {
            "route": ROUTE,
            "device_type": "metal",
            "device": effective_device,
            "decoder_linear_backend": os.environ[
                "TRELLIS2MLX_DECODER_LINEAR_BACKEND"
            ],
            "sparse_conv_matmul_backend": os.environ[
                "TRELLIS2MLX_SPARSE_CONV_MATMUL_BACKEND"
            ],
            "decoder_layernorm": decoder_layernorm_backend_identity(),
            "decoder_layernorm_lut": turing_rsqrt_lut_identity,
            "decoder_silu": decoder_silu_backend_identity(),
            "parent_state": {
                "path": str(args.level0_trace.resolve()),
                "sha256": parent_sha,
                "input_tensor_sha256": input_identity,
            },
        }
        report["effective_route"] = effective_route
        report["last_trustworthy_phase"] = phase

        phase = "model_load"
        from trellmlx.models.shape_slat_decoder import SLatDecoder
        from trellmlx.weight_loader import load_weights

        decoder = SLatDecoder(
            out_channels=7,
            pred_subdiv=True,
            use_fp16=True,
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
        from trellmlx.decoder_level1_trace import (
            capture_mlx_decoder_level1_trace,
        )

        arrays, hash_entries = capture_mlx_decoder_level1_trace(
            decoder,
            mx.array(level0_output),
            mx.array(parent_coords),
            hash_entry=decoder_level1_hash_entry,
        )
        hash_ledger = validate_decoder_level1_hash_ledger(
            {
                "schema": LEVEL1_HASH_LEDGER_SCHEMA,
                "entries": hash_entries,
            }
        )
        validation = write_decoder_level1_trace_npz(
            args.output_npz,
            arrays,
            torso_dtype=np.float16,
        )
        primary_sha = _sha256_file(args.output_npz)
        report["primary"] = {
            "path": str(args.output_npz.resolve()),
            "status": "written",
            "sha256": primary_sha,
            "size_bytes": args.output_npz.stat().st_size,
            "validation": validation,
            "hash_ledger": hash_ledger,
        }
        report.update(
            {
                "status": "done",
                "failure_phase": None,
                "last_trustworthy_phase": "trace_primary_reopened_exact",
            }
        )
        _write_report(effective_report_path, report)
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
        _write_report(effective_report_path, report)
        return 1
    finally:
        if previous_linear is None:
            os.environ.pop("TRELLIS2MLX_DECODER_LINEAR_BACKEND", None)
        else:
            os.environ["TRELLIS2MLX_DECODER_LINEAR_BACKEND"] = previous_linear
        if previous_sparse is None:
            os.environ.pop("TRELLIS2MLX_SPARSE_CONV_MATMUL_BACKEND", None)
        else:
            os.environ["TRELLIS2MLX_SPARSE_CONV_MATMUL_BACKEND"] = previous_sparse


if __name__ == "__main__":
    raise SystemExit(main())
