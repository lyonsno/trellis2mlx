#!/usr/bin/env python3
"""Expose source-CUDA arithmetic inside the guided x0-to-pred conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import traceback
from typing import Any

import numpy as np


SCHEMA = "trellis2mlx.cuda_flow_conversion_witness.v3"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_DEVICE = "Tesla T4"
STEP_INDEX = 1
STEPS = 8
RESCALE_T = 3.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _float32_bits(value: float | np.float32) -> str:
    bits = int(np.asarray(np.float32(value)).view(np.uint32))
    return f"0x{bits:08x}"


def _float64_bits(value: float) -> str:
    bits = int(np.asarray(np.float64(value)).view(np.uint64))
    return f"0x{bits:016x}"


def _exact_metric(
    candidate: np.ndarray, reference: np.ndarray
) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if candidate.shape != reference.shape:
        return {
            "shape_match": False,
            "candidate_shape": list(candidate.shape),
            "reference_shape": list(reference.shape),
            "exact": False,
            "nonzero": None,
            "max_abs": None,
            "mean_abs": None,
        }
    difference = np.abs(
        candidate.astype(np.float64) - reference.astype(np.float64)
    )
    return {
        "shape_match": True,
        "shape": list(candidate.shape),
        "exact": bool(np.array_equal(candidate, reference)),
        "nonzero": int(np.count_nonzero(candidate != reference)),
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
    }


def analyze_conversion(
    *,
    sample: np.ndarray,
    x0_after_rescale: np.ndarray,
    scaled_sample: np.ndarray,
    numerator: np.ndarray,
    pred_recomputed: np.ndarray,
    source_pred_final: np.ndarray,
) -> dict[str, Any]:
    arrays = {
        "sample": np.asarray(sample, dtype=np.float32),
        "x0_after_rescale": np.asarray(x0_after_rescale, dtype=np.float32),
        "scaled_sample": np.asarray(scaled_sample, dtype=np.float32),
        "numerator": np.asarray(numerator, dtype=np.float32),
        "pred_recomputed": np.asarray(pred_recomputed, dtype=np.float32),
        "source_pred_final": np.asarray(source_pred_final, dtype=np.float32),
    }
    shape = arrays["sample"].shape
    for name, array in arrays.items():
        if array.shape != shape:
            raise ValueError(f"{name} shape does not match sample")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
    prediction_metric = _exact_metric(
        arrays["pred_recomputed"], arrays["source_pred_final"]
    )
    if not prediction_metric["exact"]:
        raise ValueError(
            "recomputed prediction does not reproduce source "
            f"(nonzero={prediction_metric['nonzero']}, "
            f"max_abs={prediction_metric['max_abs']})"
        )
    return {
        "self_authentication": {"pred_recomputed_exact": True},
        "intermediate_capture": {
            "status": "captured_not_independently_recomputed",
            "arrays": ["scaled_sample", "numerator"],
        },
        "pred_recomputed_vs_source": prediction_metric,
        "array_sha256": {
            name: sha256_array(array) for name, array in arrays.items()
        },
    }


def analyze_reciprocal_schedule(
    *,
    step_indices: np.ndarray,
    coefficient_float64: np.ndarray,
    coefficient_float32: np.ndarray,
    native_reciprocals: np.ndarray,
    host_float64_reciprocals: np.ndarray,
    pred_recomputed: np.ndarray,
    source_pred_final: np.ndarray,
) -> dict[str, Any]:
    step_indices = np.asarray(step_indices)
    coefficient_float64 = np.asarray(coefficient_float64)
    coefficient_float32 = np.asarray(coefficient_float32)
    native_reciprocals = np.asarray(native_reciprocals)
    host_float64_reciprocals = np.asarray(
        host_float64_reciprocals
    )
    pred_recomputed = np.asarray(pred_recomputed)
    source_pred_final = np.asarray(source_pred_final)
    active_count = step_indices.size
    if (
        step_indices.dtype != np.int32
        or step_indices.ndim != 1
        or not np.array_equal(step_indices, np.unique(step_indices))
    ):
        raise ValueError(
            "schedule step indices must be unique sorted int32 values"
        )
    if (
        coefficient_float64.dtype != np.float64
        or coefficient_float64.shape != (active_count,)
        or not np.isfinite(coefficient_float64).all()
        or np.any(coefficient_float64 <= 0)
    ):
        raise ValueError(
            "coefficient_float64 must be positive finite float64 "
            "with one value per active step"
        )
    for name, array in (
        ("coefficient_float32", coefficient_float32),
        ("native_reciprocals", native_reciprocals),
        ("host_float64_reciprocals", host_float64_reciprocals),
    ):
        if array.dtype != np.float32 or array.shape != (active_count,):
            raise ValueError(
                f"{name} must be float32 with one value per active step"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
    expected_coefficient_float32 = coefficient_float64.astype(np.float32)
    if not np.array_equal(
        coefficient_float32, expected_coefficient_float32
    ):
        raise ValueError("float32 coefficient reference is stale")
    expected_host_reciprocals = (1.0 / coefficient_float64).astype(
        np.float32
    )
    if not np.array_equal(
        host_float64_reciprocals, expected_host_reciprocals
    ):
        raise ValueError("host float64 reciprocal reference is stale")
    if not np.array_equal(native_reciprocals, host_float64_reciprocals):
        mismatch_count = int(
            np.count_nonzero(
                native_reciprocals != host_float64_reciprocals
            )
        )
        raise ValueError(
            "native scalar reciprocal differs from host float64 reference "
            f"(nonzero={mismatch_count})"
        )
    if (
        pred_recomputed.dtype != np.float32
        or source_pred_final.dtype != np.float32
        or pred_recomputed.ndim < 2
        or pred_recomputed.shape != source_pred_final.shape
        or pred_recomputed.shape[0] != active_count
    ):
        raise ValueError(
            "schedule predictions must be shape-matched float32 active rows"
        )
    if (
        not np.isfinite(pred_recomputed).all()
        or not np.isfinite(source_pred_final).all()
    ):
        raise ValueError("schedule predictions contain non-finite values")

    coefficient_float64_bits = coefficient_float64.view(np.uint64)
    coefficient_float32_bits = coefficient_float32.view(np.uint32)
    native_bits = native_reciprocals.view(np.uint32)
    host_bits = host_float64_reciprocals.view(np.uint32)
    signed_delta = (
        native_bits.astype(np.int64) - host_bits.astype(np.int64)
    )
    rows = []
    for active_row, step_index in enumerate(step_indices.tolist()):
        exact = bool(
            np.array_equal(
                pred_recomputed[active_row],
                source_pred_final[active_row],
            )
        )
        if not exact:
            raise ValueError(
                f"schedule step {step_index} does not reproduce source"
            )
        rows.append(
            {
                "active_row": active_row,
                "step_index": step_index,
                "coefficient_float64_bits": (
                    f"0x{int(coefficient_float64_bits[active_row]):016x}"
                ),
                "coefficient_float32_bits": (
                    f"0x{int(coefficient_float32_bits[active_row]):08x}"
                ),
                "native_reciprocal_bits": (
                    f"0x{int(native_bits[active_row]):08x}"
                ),
                "host_float64_reciprocal_bits": (
                    f"0x{int(host_bits[active_row]):08x}"
                ),
                "native_minus_host_float64_ulp": int(
                    signed_delta[active_row]
                ),
                "prediction_exact": True,
            }
        )
    return {
        "all_active_predictions_exact": True,
        "all_native_reciprocals_match_host_float64": True,
        "active_step_count": active_count,
        "rows": rows,
    }


def analyze_output_contract(
    *,
    source_pred_final: np.ndarray,
    step_indices: np.ndarray,
    coefficient_float64: np.ndarray,
    coefficient_float32: np.ndarray,
    pred_direct: np.ndarray,
    pred_recomputed: np.ndarray,
    native_reciprocals: np.ndarray,
    host_float64_reciprocals: np.ndarray,
) -> dict[str, Any]:
    source_pred_final = np.asarray(source_pred_final)
    pred_direct = np.asarray(pred_direct)
    if (
        source_pred_final.dtype != np.float32
        or pred_direct.dtype != np.float32
        or pred_direct.shape != source_pred_final.shape
        or not np.isfinite(pred_direct).all()
        or not np.isfinite(source_pred_final).all()
    ):
        raise ValueError(
            "direct and source schedule predictions must be finite "
            "shape-matched float32 arrays"
        )
    direct_metric = _exact_metric(pred_direct, source_pred_final)
    if not direct_metric["exact"]:
        raise ValueError(
            "direct schedule prediction does not reproduce source "
            f"(nonzero={direct_metric['nonzero']}, "
            f"max_abs={direct_metric['max_abs']})"
        )
    schedule = analyze_reciprocal_schedule(
        step_indices=step_indices,
        coefficient_float64=coefficient_float64,
        coefficient_float32=coefficient_float32,
        native_reciprocals=native_reciprocals,
        host_float64_reciprocals=host_float64_reciprocals,
        pred_recomputed=pred_recomputed,
        source_pred_final=source_pred_final,
    )
    return {
        "pred_direct_vs_source": direct_metric,
        "reciprocal_schedule": schedule,
    }


def _requested_source_identity(expected_sha256: str | None) -> dict[str, str]:
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError(
            "expected source recurrence sha256 must be 64 lowercase "
            "hexadecimal characters"
        )
    return {"sha256": expected_sha256}


def validate_source_chain(
    *,
    source_identity: dict[str, Any],
    source_tar_path: Path,
    expected_source_tar_sha256: str | None,
) -> dict[str, Any]:
    requested_tar = _requested_source_identity(expected_source_tar_sha256)
    route = source_identity.get("route", {})
    if route.get("cuda_device") != EXPECTED_DEVICE:
        raise ValueError(
            f"source recurrence route must be {EXPECTED_DEVICE}, "
            f"got {route.get('cuda_device')!r}"
        )
    effective_tar_sha256 = sha256_file(source_tar_path)
    if effective_tar_sha256 != requested_tar["sha256"]:
        raise ValueError(
            "source tar sha256 mismatch: "
            f"expected {requested_tar['sha256']}, got {effective_tar_sha256}"
        )
    claimed_tar_sha256 = source_identity.get("source_tar_sha256_claimed")
    if claimed_tar_sha256 != effective_tar_sha256:
        raise ValueError(
            "source tar claim mismatch: "
            f"recurrence claims {claimed_tar_sha256!r}, "
            f"effective tar is {effective_tar_sha256}"
        )
    return {
        "path": str(source_tar_path),
        "sha256": effective_tar_sha256,
        "recurrence_claim_exact": True,
    }


def _source_validator():
    try:
        from scripts.source_cuda_shape_flow_transition0_recoverability import (
            validate_source_recurrence_artifact,
        )
    except ImportError:
        from source_cuda_shape_flow_transition0_recoverability import (  # type: ignore[no-redef]
            validate_source_recurrence_artifact,
        )

    return validate_source_recurrence_artifact


def _source_schedule() -> list[float]:
    t_seq = np.linspace(1, 0, STEPS + 1)
    t_seq = RESCALE_T * t_seq / (1 + (RESCALE_T - 1) * t_seq)
    return t_seq.tolist()


def _load_source_guidance_rows(
    path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    validation = _source_validator()(path)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        guidance_indices = np.asarray(
            archive["guidance_step_indices"], dtype=np.int32
        )
        if (
            guidance_indices.ndim != 1
            or guidance_indices.size == 0
            or not np.array_equal(
                guidance_indices,
                np.unique(guidance_indices),
            )
        ):
            raise ValueError(
                "source recurrence must contain unique sorted guidance steps"
            )
        step_one_rows = np.flatnonzero(guidance_indices == STEP_INDEX)
        if step_one_rows.size != 1:
            raise ValueError(
                f"source recurrence must contain guided step {STEP_INDEX} once"
            )
        schedule = np.asarray(_source_schedule(), dtype=np.float64)
        active_t = schedule[guidance_indices]
        saved_t = np.asarray(archive["t"], dtype=np.float32)[
            guidance_indices
        ]
        if not np.array_equal(saved_t, active_t.astype(np.float32)):
            raise ValueError("source recurrence saved t does not match source schedule")
        sigma_min = float(metadata["sampler_args"]["sigma_min"])
        arrays = {
            "sample": np.asarray(
                archive["sample_in"][guidance_indices], dtype=np.float32
            ),
            "x0_after_rescale": np.asarray(
                archive["x0_after_rescale"], dtype=np.float32
            ),
            "source_pred_final": np.asarray(
                archive["pred_final"][guidance_indices], dtype=np.float32
            ),
        }
        if (
            arrays["sample"].shape != arrays["x0_after_rescale"].shape
            or arrays["sample"].shape != arrays["source_pred_final"].shape
        ):
            raise ValueError(
                "source recurrence guided arrays are not row aligned"
            )
    one_minus_sigma = 1.0 - sigma_min
    scalars = {
        "t": active_t,
        "sigma_min": sigma_min,
        "one_minus_sigma": one_minus_sigma,
        "coefficient": sigma_min + one_minus_sigma * active_t,
    }
    route = metadata["effective_route"]
    source_tar_sha256_claimed = metadata["inputs"]["expected_digests"][
        "source tar"
    ]
    source_identity = {
        "validation": validation,
        "route": route,
        "source_tar_sha256_claimed": source_tar_sha256_claimed,
        "sampler_name": metadata["sampler_name"],
        "sampler_args": metadata["sampler_args"],
        "sampler_params": metadata["sampler_params"],
        "guidance_step_indices": guidance_indices.tolist(),
        "step_one_active_row": int(step_one_rows[0]),
    }
    return source_identity, arrays, scalars


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-recurrence", default="source_recurrence.npz")
    parser.add_argument("--expected-source-recurrence-sha256")
    parser.add_argument("--source-tar", default="trellis2_source_tarball.bin")
    parser.add_argument("--expected-source-tar-sha256")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-npz", required=True)
    args = parser.parse_args()

    started = time.time()
    source_path = Path(args.source_recurrence)
    source_tar_path = Path(args.source_tar)
    output_json = Path(args.output_json)
    output_npz = Path(args.output_npz)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "failure_phase": "request_validation",
        "last_trustworthy_phase": "request_received",
        "source_recurrence": str(source_path),
        "source_tar": str(source_tar_path),
        "output_json": str(output_json),
        "output_npz": str(output_npz),
    }
    try:
        if output_json.resolve() == output_npz.resolve():
            raise ValueError("output JSON and NPZ paths must be distinct")
        if output_npz.exists():
            output_npz.unlink()
        report["last_trustworthy_phase"] = "output_path_validated"
        requested_identity = _requested_source_identity(
            args.expected_source_recurrence_sha256
        )
        report["source_identity_requested"] = requested_identity
        report["source_tar_identity_requested"] = _requested_source_identity(
            args.expected_source_tar_sha256
        )
        report["last_trustworthy_phase"] = "request_validated"
        report["failure_phase"] = "input_validation"

        effective_digest = sha256_file(source_path)
        report["source_identity_effective"] = {"sha256": effective_digest}
        if effective_digest != requested_identity["sha256"]:
            raise ValueError(
                "source recurrence sha256 mismatch: "
                f"expected {requested_identity['sha256']}, got {effective_digest}"
            )
        source_identity, source_arrays, scalars = (
            _load_source_guidance_rows(source_path)
        )
        report["source_tar_identity_effective"] = validate_source_chain(
            source_identity=source_identity,
            source_tar_path=source_tar_path,
            expected_source_tar_sha256=args.expected_source_tar_sha256,
        )
        report["source_identity_effective"].update(source_identity)
        report["source_array_sha256"] = {
            name: sha256_array(array)
            for name, array in source_arrays.items()
        }
        report["scalar_identity"] = {
            "sigma_min": {
                "value": scalars["sigma_min"],
                "float64_bits": _float64_bits(scalars["sigma_min"]),
                "float32_bits": _float32_bits(scalars["sigma_min"]),
            },
            "one_minus_sigma": {
                "value": scalars["one_minus_sigma"],
                "float64_bits": _float64_bits(
                    scalars["one_minus_sigma"]
                ),
                "float32_bits": _float32_bits(
                    scalars["one_minus_sigma"]
                ),
            },
            "active_schedule": [
                {
                    "active_row": active_row,
                    "step_index": step_index,
                    "t": {
                        "value": float(scalars["t"][active_row]),
                        "float64_bits": _float64_bits(
                            float(scalars["t"][active_row])
                        ),
                        "float32_bits": _float32_bits(
                            scalars["t"][active_row]
                        ),
                    },
                    "coefficient": {
                        "value": float(
                            scalars["coefficient"][active_row]
                        ),
                        "float64_bits": _float64_bits(
                            float(scalars["coefficient"][active_row])
                        ),
                        "float32_bits": _float32_bits(
                            scalars["coefficient"][active_row]
                        ),
                    },
                }
                for active_row, step_index in enumerate(
                    source_identity["guidance_step_indices"]
                )
            ],
        }
        report["last_trustworthy_phase"] = "input_validated"
        report["failure_phase"] = "cuda_route_validation"

        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device"] = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        )
        if report["torch"] != EXPECTED_TORCH:
            raise RuntimeError(
                f"expected Torch {EXPECTED_TORCH}, got {report['torch']}"
            )
        if report["cuda_device"] != EXPECTED_DEVICE:
            raise RuntimeError(
                f"expected CUDA device {EXPECTED_DEVICE}, "
                f"got {report['cuda_device']!r}"
            )
        report["last_trustworthy_phase"] = "cuda_route_validated"
        report["failure_phase"] = "oracle_execution"

        sample = torch.from_numpy(source_arrays["sample"]).to(
            device="cuda", dtype=torch.float32
        )
        x0_after_rescale = torch.from_numpy(
            source_arrays["x0_after_rescale"]
        ).to(device="cuda", dtype=torch.float32)
        execution_started = time.time()
        scaled_rows = []
        numerator_rows = []
        pred_direct_rows = []
        pred_recomputed_rows = []
        native_reciprocal_rows = []
        for active_row, coefficient in enumerate(scalars["coefficient"]):
            scaled_row = scalars["one_minus_sigma"] * sample[active_row]
            numerator_row = scaled_row - x0_after_rescale[active_row]
            pred_direct_row = numerator_row / float(coefficient)
            native_reciprocal = (
                torch.ones((), device="cuda", dtype=torch.float32)
                / float(coefficient)
            )
            pred_recomputed_row = numerator_row * native_reciprocal
            scaled_rows.append(scaled_row)
            numerator_rows.append(numerator_row)
            pred_direct_rows.append(pred_direct_row)
            pred_recomputed_rows.append(pred_recomputed_row)
            native_reciprocal_rows.append(native_reciprocal)
        scaled_sample = torch.stack(scaled_rows)
        numerator = torch.stack(numerator_rows)
        pred_direct = torch.stack(pred_direct_rows)
        pred_recomputed = torch.stack(pred_recomputed_rows)
        native_reciprocals = torch.stack(native_reciprocal_rows)
        torch.cuda.synchronize()
        report["cuda_schedule_execution_seconds"] = (
            time.time() - execution_started
        )

        def as_float32(value: Any) -> np.ndarray:
            return (
                value.detach()
                .to(dtype=torch.float32, device="cpu")
                .numpy()
                .astype(np.float32, copy=False)
            )

        coefficient_float64 = np.asarray(
            scalars["coefficient"], dtype=np.float64
        )
        coefficient_float32 = np.asarray(
            scalars["coefficient"], dtype=np.float32
        )
        host_float64_reciprocals = (
            1.0 / coefficient_float64
        ).astype(np.float32)
        step_indices = np.asarray(
            source_identity["guidance_step_indices"], dtype=np.int32
        )
        output_arrays = {
            "scaled_sample": as_float32(scaled_sample),
            "numerator": as_float32(numerator),
            "pred_direct": as_float32(pred_direct),
            "pred_recomputed": as_float32(pred_recomputed),
            "native_reciprocals": as_float32(native_reciprocals),
            "host_float64_reciprocals": host_float64_reciprocals,
            "step_indices": step_indices,
            "coefficient_float64": coefficient_float64,
            "coefficient_float32": coefficient_float32,
        }
        report["last_trustworthy_phase"] = "oracle_executed"
        report["failure_phase"] = "self_authentication"
        output_contract = analyze_output_contract(
            source_pred_final=source_arrays["source_pred_final"],
            step_indices=step_indices,
            coefficient_float64=coefficient_float64,
            coefficient_float32=coefficient_float32,
            pred_direct=output_arrays["pred_direct"],
            pred_recomputed=output_arrays["pred_recomputed"],
            native_reciprocals=output_arrays["native_reciprocals"],
            host_float64_reciprocals=host_float64_reciprocals,
        )
        report.update(output_contract)
        step_one_active_row = source_identity["step_one_active_row"]
        report["step_one_conversion"] = analyze_conversion(
            sample=source_arrays["sample"][step_one_active_row],
            x0_after_rescale=source_arrays["x0_after_rescale"][
                step_one_active_row
            ],
            scaled_sample=output_arrays["scaled_sample"][
                step_one_active_row
            ],
            numerator=output_arrays["numerator"][step_one_active_row],
            pred_recomputed=output_arrays["pred_recomputed"][
                step_one_active_row
            ],
            source_pred_final=source_arrays["source_pred_final"][
                step_one_active_row
            ],
        )
        report["last_trustworthy_phase"] = "self_authenticated"
        report["failure_phase"] = "output_write"

        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_npz,
            **source_arrays,
            **output_arrays,
            t=np.asarray(scalars["t"], dtype=np.float64),
            sigma_min=np.asarray(scalars["sigma_min"], dtype=np.float64),
            one_minus_sigma=np.asarray(
                scalars["one_minus_sigma"], dtype=np.float64
            ),
        )
        with np.load(output_npz, allow_pickle=False) as archive:
            for name, expected in {**source_arrays, **output_arrays}.items():
                written = np.asarray(archive[name])
                if (
                    written.dtype != expected.dtype
                    or written.shape != expected.shape
                    or not np.array_equal(written, expected)
                ):
                    raise ValueError(
                        f"written output array {name} differs from "
                        "self-authenticated memory"
                    )
            reloaded_contract = analyze_output_contract(
                source_pred_final=np.asarray(
                    archive["source_pred_final"]
                ),
                step_indices=np.asarray(archive["step_indices"]),
                coefficient_float64=np.asarray(
                    archive["coefficient_float64"]
                ),
                coefficient_float32=np.asarray(
                    archive["coefficient_float32"]
                ),
                pred_direct=np.asarray(archive["pred_direct"]),
                pred_recomputed=np.asarray(archive["pred_recomputed"]),
                native_reciprocals=np.asarray(
                    archive["native_reciprocals"]
                ),
                host_float64_reciprocals=np.asarray(
                    archive["host_float64_reciprocals"]
                ),
            )
            if reloaded_contract != output_contract:
                raise ValueError(
                    "written output contract differs from "
                    "self-authenticated memory"
                )
        report["output_npz_sha256"] = sha256_file(output_npz)
        report["output_npz_size_bytes"] = output_npz.stat().st_size
        report["status"] = "done"
        report["failure_phase"] = None
        report["last_trustworthy_phase"] = "output_validated"
        report["elapsed_seconds"] = time.time() - started
        _write_report(output_json, report)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["elapsed_seconds"] = time.time() - started
        report["primary_output"] = {
            "exists": output_npz.is_file(),
            "sha256": sha256_file(output_npz)
            if output_npz.is_file()
            else None,
            "size_bytes": output_npz.stat().st_size
            if output_npz.is_file()
            else None,
        }
        _write_report(output_json, report)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
