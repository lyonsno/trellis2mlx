#!/usr/bin/env python3
"""Compare source-CUDA timestep modulation with the local reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np


SCHEMA = "trellis2mlx.cuda_timestep_modulation_witness.v1"
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_DEVICE = "Tesla T4"
PROJECTION_BATCH_MODES = (
    "batched-eight",
    "independent-singletons",
)
EXPECTED_STEP_INDICES = tuple(range(8))
SCHEDULE_TIMESTEP_FLOAT32_BITS = {
    "shape-slat-rescale-3": (
        0x447A0000,
        0x446EA2E9,
        0x44610000,
        0x44505555,
        0x443B8000,
        0x4420B6DB,
        0x43FA0000,
        0x43960000,
    ),
    "sparse-structure-rescale-5": (
        0x447A0000,
        0x44730E39,
        0x446A6000,
        0x445F36DB,
        0x44505555,
        0x443B8000,
        0x441C4000,
        0x43D05555,
    ),
}
STAGES = (
    "embedding",
    "linear0",
    "silu0",
    "linear1",
    "silu1",
    "modulation_float32",
    "modulation_bfloat16_bits",
)
WEIGHT_SHAPES = {
    "linear0_weight": (1536, 256),
    "linear0_bias": (1536,),
    "linear1_weight": (1536, 1536),
    "linear1_bias": (1536,),
    "modulation_weight": (9216, 1536),
    "modulation_bias": (9216,),
}


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _same_path(first: Path, second: Path) -> bool:
    if Path(first).resolve() == Path(second).resolve():
        return True
    try:
        return os.path.samefile(first, second)
    except FileNotFoundError:
        return False
    except OSError:
        # Existing paths whose identity cannot be established are unsafe.
        return True


def _safe_failure_report_path(
    requested: Path,
    *,
    protected: tuple[Path, ...],
) -> Path:
    if not any(_same_path(requested, path) for path in protected):
        return requested
    candidate = protected[0].with_name(
        protected[0].name + ".timestep-modulation-failure.json"
    )
    while any(_same_path(candidate, path) for path in protected):
        candidate = candidate.with_name(candidate.name + ".failure.json")
    return candidate


def validate_route(*, torch_version: str, cuda_device: str) -> None:
    if torch_version != EXPECTED_TORCH:
        raise ValueError(
            f"Torch route must be {EXPECTED_TORCH}, got {torch_version}"
        )
    if cuda_device != EXPECTED_DEVICE:
        raise ValueError(
            f"CUDA device route must be {EXPECTED_DEVICE}, got {cuda_device}"
        )


def validate_schedule(
    *,
    step_indices: np.ndarray,
    timesteps: np.ndarray,
    schedule_profile: str = "shape-slat-rescale-3",
) -> dict[str, Any]:
    step_indices = np.asarray(step_indices)
    timesteps = np.asarray(timesteps)
    expected_steps = np.asarray(EXPECTED_STEP_INDICES, dtype=np.int32)
    if schedule_profile not in SCHEDULE_TIMESTEP_FLOAT32_BITS:
        raise ValueError(f"unsupported schedule profile {schedule_profile!r}")
    expected_timestep_bits = SCHEDULE_TIMESTEP_FLOAT32_BITS[schedule_profile]
    expected_bits = np.asarray(expected_timestep_bits, dtype=np.uint32)
    if (
        step_indices.dtype != np.int32
        or step_indices.shape != expected_steps.shape
        or not np.array_equal(step_indices, expected_steps)
    ):
        raise ValueError(
            "candidate must use the canonical eight-step schedule indices"
        )
    if (
        timesteps.dtype != np.float32
        or timesteps.shape != expected_steps.shape
        or not np.array_equal(timesteps.view(np.uint32), expected_bits)
    ):
        raise ValueError(
            "candidate must use the canonical eight-step schedule timesteps"
        )
    return {
        "profile_expected": schedule_profile,
        "profile_effective": schedule_profile,
        "step_indices_expected": list(EXPECTED_STEP_INDICES),
        "step_indices_effective": step_indices.tolist(),
        "timestep_float32_bits_expected": [
            f"0x{bits:08x}" for bits in expected_timestep_bits
        ],
        "timestep_float32_bits_effective": [
            f"0x{int(bits):08x}" for bits in timesteps.view(np.uint32)
        ],
    }


def validate_provenance(
    *,
    source_checkpoint_sha256: str,
    expected_source_checkpoint_sha256: str,
    candidate_route: str,
    expected_candidate_route: str,
    candidate_projection_batch_mode: str,
    requested_projection_batch_mode: str,
) -> None:
    if source_checkpoint_sha256 != expected_source_checkpoint_sha256:
        raise ValueError(
            "source checkpoint sha256 mismatch: "
            f"expected {expected_source_checkpoint_sha256}, "
            f"got {source_checkpoint_sha256}"
        )
    if candidate_route != expected_candidate_route:
        raise ValueError(
            "candidate route mismatch: "
            f"expected {expected_candidate_route!r}, got {candidate_route!r}"
        )
    if candidate_projection_batch_mode != requested_projection_batch_mode:
        raise ValueError(
            "projection batch mode mismatch: "
            f"requested {requested_projection_batch_mode!r}, "
            f"candidate records {candidate_projection_batch_mode!r}"
        )


def _scalar_text(value: np.ndarray, *, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a scalar string")
    result = str(array.item())
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _validate_step_indices(step_indices: np.ndarray) -> np.ndarray:
    value = np.asarray(step_indices)
    if (
        value.dtype != np.int32
        or value.ndim != 1
        or value.size == 0
        or not np.array_equal(value, np.arange(value.size, dtype=np.int32))
    ):
        raise ValueError(
            "step_indices must be contiguous zero-based int32 values"
        )
    return value


def _metric(candidate: np.ndarray, source: np.ndarray) -> dict[str, Any]:
    candidate = np.asarray(candidate)
    source = np.asarray(source)
    if candidate.shape != source.shape:
        return {
            "shape_match": False,
            "candidate_shape": list(candidate.shape),
            "source_shape": list(source.shape),
            "exact": False,
            "nonzero": None,
            "max_abs": None,
            "mean_abs": None,
        }
    if candidate.dtype == np.uint16 and source.dtype == np.uint16:
        difference = np.abs(
            candidate.astype(np.int64) - source.astype(np.int64)
        )
    else:
        difference = np.abs(
            candidate.astype(np.float64) - source.astype(np.float64)
        )
    return {
        "shape_match": True,
        "shape": list(candidate.shape),
        "exact": bool(np.array_equal(candidate, source)),
        "nonzero": int(np.count_nonzero(candidate != source)),
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
    }


def analyze_modulation(
    *,
    step_indices: np.ndarray,
    candidate_arrays: dict[str, np.ndarray],
    source_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    step_indices = _validate_step_indices(step_indices)
    step_count = int(step_indices.size)
    stages: dict[str, Any] = {}
    first_float32 = None
    first_bfloat16 = None

    for stage in STAGES:
        if stage not in candidate_arrays:
            raise ValueError(f"candidate missing {stage}")
        if stage not in source_arrays:
            raise ValueError(f"source missing {stage}")
        candidate = np.asarray(candidate_arrays[stage])
        source = np.asarray(source_arrays[stage])
        expected_dtype = (
            np.dtype(np.uint16)
            if stage == "modulation_bfloat16_bits"
            else np.dtype(np.float32)
        )
        if candidate.dtype != expected_dtype or source.dtype != expected_dtype:
            raise ValueError(
                f"{stage} dtype mismatch: expected {expected_dtype}, "
                f"candidate={candidate.dtype}, source={source.dtype}"
            )
        if candidate.ndim != 2 or candidate.shape[0] != step_count:
            raise ValueError(
                f"{stage} candidate shape must begin with {step_count} steps"
            )
        if source.shape != candidate.shape:
            raise ValueError(
                f"{stage} shape mismatch: candidate={candidate.shape}, "
                f"source={source.shape}"
            )
        if stage != "modulation_bfloat16_bits":
            if not np.isfinite(candidate).all():
                raise ValueError(f"candidate {stage} contains non-finite values")
            if not np.isfinite(source).all():
                raise ValueError(f"{stage} contains non-finite values")

        per_step = []
        for row, step_index in enumerate(step_indices.tolist()):
            metric = _metric(candidate[row], source[row])
            per_step.append({"step_index": step_index, **metric})
            if not metric["exact"]:
                if stage == "modulation_bfloat16_bits":
                    if first_bfloat16 is None:
                        first_bfloat16 = {
                            "stage": stage,
                            "step_index": step_index,
                        }
                elif first_float32 is None:
                    first_float32 = {
                        "stage": stage,
                        "step_index": step_index,
                    }
        stages[stage] = {
            **_metric(candidate, source),
            "per_step": per_step,
        }

    return {
        "all_float32_exact": all(
            stages[stage]["exact"] for stage in STAGES[:-1]
        ),
        "all_bfloat16_modulation_exact": stages[
            "modulation_bfloat16_bits"
        ]["exact"],
        "first_float32_divergence": first_float32,
        "first_bfloat16_modulation_divergence": first_bfloat16,
        "stages": stages,
    }


def _load_weights(path: Path) -> tuple[dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as data:
        required = {*WEIGHT_SHAPES, "source_checkpoint_sha256"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"weights missing required arrays: {missing}")
        weights = {
            name: np.asarray(data[name])
            for name in WEIGHT_SHAPES
        }
        source_checkpoint_sha256 = _scalar_text(
            data["source_checkpoint_sha256"],
            name="source_checkpoint_sha256",
        )
    for name, shape in WEIGHT_SHAPES.items():
        value = weights[name]
        if value.dtype != np.float32 or value.shape != shape:
            raise ValueError(
                f"{name} must be float32{shape}, got {value.dtype}{value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    return weights, source_checkpoint_sha256


def _load_candidate(
    path: Path,
    *,
    expected_schedule_profile: str = "shape-slat-rescale-3",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], str, str, str]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "step_indices",
            "timestep_float32",
            "candidate_route",
            "projection_batch_mode",
            "schedule_profile",
            *STAGES,
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"candidate missing required arrays: {missing}")
        step_indices = np.asarray(data["step_indices"])
        timesteps = np.asarray(data["timestep_float32"])
        arrays = {stage: np.asarray(data[stage]) for stage in STAGES}
        candidate_route = _scalar_text(
            data["candidate_route"], name="candidate_route"
        )
        projection_batch_mode = _scalar_text(
            data["projection_batch_mode"], name="projection_batch_mode"
        )
        schedule_profile = _scalar_text(
            data["schedule_profile"], name="schedule_profile"
        )
    if schedule_profile != expected_schedule_profile:
        raise ValueError(
            "candidate schedule profile mismatch: "
            f"expected {expected_schedule_profile!r}, got {schedule_profile!r}"
        )
    validate_schedule(
        step_indices=step_indices,
        timesteps=timesteps,
        schedule_profile=schedule_profile,
    )
    # Reuse the full analysis validator against itself.
    analyze_modulation(
        step_indices=step_indices,
        candidate_arrays=arrays,
        source_arrays={name: value.copy() for name, value in arrays.items()},
    )
    return (
        step_indices,
        timesteps,
        arrays,
        candidate_route,
        projection_batch_mode,
        schedule_profile,
    )


def validate_written_primary(
    path: Path,
    *,
    step_indices: np.ndarray,
    timesteps: np.ndarray,
    source_arrays: dict[str, np.ndarray],
    projection_batch_mode: str,
    schedule_profile: str = "shape-slat-rescale-3",
) -> dict[str, str]:
    expected = {
        "step_indices": np.asarray(step_indices),
        "timestep_float32": np.asarray(timesteps),
        "projection_batch_mode": np.asarray(projection_batch_mode),
        "schedule_profile": np.asarray(schedule_profile),
        **{
            f"source_{name}": np.asarray(value)
            for name, value in source_arrays.items()
        },
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(expected).difference(data.files))
        if missing:
            raise ValueError(f"primary missing required arrays: {missing}")
        written = {name: np.asarray(data[name]) for name in expected}
    for name, expected_value in expected.items():
        written_value = written[name]
        if (
            written_value.dtype != expected_value.dtype
            or written_value.shape != expected_value.shape
            or not np.array_equal(written_value, expected_value)
        ):
            raise ValueError(
                f"primary array {name} differs from authenticated memory"
            )
    return {
        name: sha256_array(value)
        for name, value in written.items()
    }


def _projection_batches(
    timesteps: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, ...]:
    value = np.asarray(timesteps)
    if value.ndim != 1 or value.size == 0:
        raise ValueError("timesteps must be a non-empty one-dimensional array")
    if mode == "batched-eight":
        if value.size != len(EXPECTED_STEP_INDICES):
            raise ValueError("batched-eight requires exactly eight timesteps")
        return (value,)
    if mode == "independent-singletons":
        return tuple(value[index : index + 1] for index in range(value.size))
    raise ValueError(
        f"unsupported projection batch mode {mode!r}; "
        f"expected one of {PROJECTION_BATCH_MODES}"
    )


def _compute_source_cuda(
    *,
    torch: Any,
    weights: dict[str, np.ndarray],
    timesteps: np.ndarray,
    projection_batch_mode: str,
) -> dict[str, np.ndarray]:
    device = torch.device("cuda")

    def tensor(name: str):
        return torch.from_numpy(weights[name]).to(device=device)

    linear0_weight = tensor("linear0_weight")
    linear0_bias = tensor("linear0_bias")
    linear1_weight = tensor("linear1_weight")
    linear1_bias = tensor("linear1_bias")
    modulation_weight = tensor("modulation_weight")
    modulation_bias = tensor("modulation_bias")

    half = 128
    freqs = torch.exp(
        -np.log(10000)
        * torch.arange(0, half, dtype=torch.float32)
        / half
    ).to(device=device)

    def compute_batch(batch_timesteps: np.ndarray) -> dict[str, np.ndarray]:
        t = torch.from_numpy(batch_timesteps).to(device=device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        linear0 = torch.nn.functional.linear(
            embedding, linear0_weight, linear0_bias
        )
        silu0 = torch.nn.functional.silu(linear0)
        linear1 = torch.nn.functional.linear(
            silu0, linear1_weight, linear1_bias
        )
        silu1 = torch.nn.functional.silu(linear1)
        modulation = torch.nn.functional.linear(
            silu1, modulation_weight, modulation_bias
        )
        modulation_bfloat16_bits = (
            modulation.to(torch.bfloat16)
            .view(torch.int16)
            .cpu()
            .numpy()
            .view(np.uint16)
        )
        return {
            "embedding": embedding.cpu().numpy().astype(np.float32),
            "linear0": linear0.cpu().numpy().astype(np.float32),
            "silu0": silu0.cpu().numpy().astype(np.float32),
            "linear1": linear1.cpu().numpy().astype(np.float32),
            "silu1": silu1.cpu().numpy().astype(np.float32),
            "modulation_float32": (
                modulation.cpu().numpy().astype(np.float32)
            ),
            "modulation_bfloat16_bits": modulation_bfloat16_bits,
        }

    batches = [
        compute_batch(batch)
        for batch in _projection_batches(
            timesteps, mode=projection_batch_mode
        )
    ]
    return {
        stage: np.concatenate(
            [batch[stage] for batch in batches],
            axis=0,
        )
        for stage in STAGES
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--expected-source-checkpoint-sha256", required=True)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--expected-candidate-route", required=True)
    parser.add_argument(
        "--projection-batch-mode",
        required=True,
        choices=PROJECTION_BATCH_MODES,
    )
    parser.add_argument(
        "--schedule-profile",
        required=True,
        choices=tuple(SCHEDULE_TIMESTEP_FLOAT32_BITS),
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    return parser


def main() -> int:
    started = time.perf_counter()
    args = _parser().parse_args()
    protected_inputs = (args.weights, args.candidate)
    requested_output_json = args.output_json
    output_json = _safe_failure_report_path(
        requested_output_json,
        protected=(*protected_inputs, args.output_npz),
    )
    path_collisions = []
    if any(_same_path(args.output_npz, path) for path in protected_inputs):
        path_collisions.append("output NPZ aliases protected input")
    if any(
        _same_path(requested_output_json, path)
        for path in protected_inputs
    ):
        path_collisions.append("output JSON aliases protected input")
    if _same_path(requested_output_json, args.output_npz):
        path_collisions.append("output JSON aliases output NPZ")
    primary_is_protected = any(
        _same_path(args.output_npz, path) for path in protected_inputs
    )
    phase = "request_validation"
    last_trustworthy_phase = "request_received"
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "failure_phase": phase,
        "last_trustworthy_phase": last_trustworthy_phase,
        "primary_output_status": (
            "protected_input"
            if primary_is_protected
            else (
                "preexisting_untrusted"
                if args.output_npz.exists()
                else "missing"
            )
        ),
        "output_json_requested": str(requested_output_json),
        "output_json_effective": str(output_json),
        "output_npz": str(args.output_npz),
        "requested_route": {
            "torch": EXPECTED_TORCH,
            "cuda_device": EXPECTED_DEVICE,
            "projection_batch_mode": args.projection_batch_mode,
            "schedule_profile": args.schedule_profile,
        },
        "inputs": {
            "weights": str(args.weights),
            "weights_sha256_requested": args.expected_weights_sha256,
            "source_checkpoint_sha256_requested": (
                args.expected_source_checkpoint_sha256
            ),
            "candidate": str(args.candidate),
            "candidate_sha256_requested": args.expected_candidate_sha256,
            "candidate_route_requested": args.expected_candidate_route,
            "projection_batch_mode_requested": args.projection_batch_mode,
            "schedule_profile_requested": args.schedule_profile,
        },
    }
    try:
        if path_collisions:
            raise ValueError("; ".join(path_collisions))
        output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        args.output_npz.unlink(missing_ok=True)
        report["primary_output_status"] = "missing"
        phase = "input_validation"
        last_trustworthy_phase = "request_validated"
        weights_sha256 = sha256_file(args.weights)
        candidate_sha256 = sha256_file(args.candidate)
        report["inputs"].update(
            {
                "weights_sha256_effective": weights_sha256,
                "candidate_sha256_effective": candidate_sha256,
            }
        )
        if weights_sha256 != args.expected_weights_sha256:
            raise ValueError(
                "weights sha256 mismatch: "
                f"expected {args.expected_weights_sha256}, got {weights_sha256}"
            )
        if candidate_sha256 != args.expected_candidate_sha256:
            raise ValueError(
                "candidate sha256 mismatch: "
                f"expected {args.expected_candidate_sha256}, got {candidate_sha256}"
            )
        weights, source_checkpoint_sha256 = _load_weights(args.weights)
        (
            step_indices,
            timesteps,
            candidate_arrays,
            candidate_route,
            candidate_projection_batch_mode,
            candidate_schedule_profile,
        ) = _load_candidate(
            args.candidate,
            expected_schedule_profile=args.schedule_profile,
        )
        validate_provenance(
            source_checkpoint_sha256=source_checkpoint_sha256,
            expected_source_checkpoint_sha256=(
                args.expected_source_checkpoint_sha256
            ),
            candidate_route=candidate_route,
            expected_candidate_route=args.expected_candidate_route,
            candidate_projection_batch_mode=(
                candidate_projection_batch_mode
            ),
            requested_projection_batch_mode=args.projection_batch_mode,
        )
        report["schedule_identity"] = validate_schedule(
            step_indices=step_indices,
            timesteps=timesteps,
            schedule_profile=candidate_schedule_profile,
        )
        report["inputs"].update(
            {
                "source_checkpoint_sha256_effective": (
                    source_checkpoint_sha256
                ),
                "candidate_route_effective": candidate_route,
                "projection_batch_mode_effective": (
                    candidate_projection_batch_mode
                ),
                "schedule_profile_effective": candidate_schedule_profile,
            }
        )
        last_trustworthy_phase = phase

        phase = "cuda_route"
        import torch

        if not torch.cuda.is_available():
            raise ValueError("CUDA route is unavailable")
        cuda_device = torch.cuda.get_device_name(0)
        validate_route(
            torch_version=torch.__version__,
            cuda_device=cuda_device,
        )
        report["effective_route"] = {
            "torch": torch.__version__,
            "cuda_device": cuda_device,
            "device_type": "cuda",
            "projection_batch_mode": args.projection_batch_mode,
            "schedule_profile": candidate_schedule_profile,
        }
        last_trustworthy_phase = phase

        phase = "source_cuda_modulation"
        with torch.inference_mode():
            source_arrays = _compute_source_cuda(
                torch=torch,
                weights=weights,
                timesteps=timesteps,
                projection_batch_mode=args.projection_batch_mode,
            )
        analysis = analyze_modulation(
            step_indices=step_indices,
            candidate_arrays=candidate_arrays,
            source_arrays=source_arrays,
        )
        last_trustworthy_phase = phase

        phase = "write_primary"
        np.savez(
            args.output_npz,
            step_indices=step_indices,
            timestep_float32=timesteps,
            projection_batch_mode=np.asarray(args.projection_batch_mode),
            schedule_profile=np.asarray(candidate_schedule_profile),
            **{
                f"source_{name}": np.ascontiguousarray(value)
                for name, value in source_arrays.items()
            },
        )
        report["primary_output_status"] = "written_unverified"
        last_trustworthy_phase = phase
        phase = "validate_primary"
        primary_array_sha256 = validate_written_primary(
            args.output_npz,
            step_indices=step_indices,
            timesteps=timesteps,
            source_arrays=source_arrays,
            projection_batch_mode=args.projection_batch_mode,
            schedule_profile=candidate_schedule_profile,
        )
        primary_sha256 = sha256_file(args.output_npz)
        report.update(
            {
                "status": "done",
                "failure_phase": None,
                "last_trustworthy_phase": "primary_validated",
                "primary_output_status": "written",
                "primary_output": {
                    "path": str(args.output_npz),
                    "sha256": primary_sha256,
                    "size_bytes": args.output_npz.stat().st_size,
                    "array_sha256": primary_array_sha256,
                },
                "analysis": analysis,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        _write_json(output_json, report)
        return 0
    except Exception as exc:
        if not primary_is_protected:
            args.output_npz.unlink(missing_ok=True)
        report.update(
            {
                "status": "failed",
                "failure_phase": phase,
                "last_trustworthy_phase": last_trustworthy_phase,
                "primary_output_status": (
                    "protected_input" if primary_is_protected else "missing"
                ),
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        _write_json(output_json, report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
