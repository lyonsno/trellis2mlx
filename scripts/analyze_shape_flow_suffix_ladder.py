#!/usr/bin/env python3
"""Admit and map an official-source CUDA shape-flow suffix ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

try:
    from scripts.source_cuda_shape_flow_suffix_ladder import (
        SWITCH_STEPS,
        classify_anchor,
        validate_result_manifest,
        validate_saved_artifact,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from source_cuda_shape_flow_suffix_ladder import (  # type: ignore[no-redef]
        SWITCH_STEPS,
        classify_anchor,
        validate_result_manifest,
        validate_saved_artifact,
    )


SCHEMA = "trellis2mlx.shape_flow_suffix_ladder_analysis.v1"
FLOAT32_REDUCTION_ULPS = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite constant {value}")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} is blank: {path}")
    payload = json.loads(path.read_text(), parse_constant=_reject_json_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _require_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} must be {expected!r}, got {actual!r}")


def _validate_receipt(
    receipt: dict[str, Any],
    report: dict[str, Any],
    packet_manifest: dict[str, Any],
    *,
    result_json: Path,
    result_npz: Path,
    packet_manifest_path: Path,
) -> None:
    _require_equal(
        receipt.get("schema"),
        "trellis2mlx.kaggle_cuda_witness.receipt.v1",
        label="receipt schema",
    )
    if receipt.get("status") != "done":
        raise ValueError("Kaggle receipt is not done")
    _require_equal(
        receipt.get("requested_dataset_id"),
        packet_manifest.get("dataset_id"),
        label="receipt dataset id",
    )
    _require_equal(
        receipt.get("requested_kernel_id"),
        packet_manifest.get("kernel_id"),
        label="receipt kernel id",
    )
    _require_equal(
        receipt.get("requested_accelerator"),
        "NvidiaTeslaT4",
        label="receipt requested accelerator",
    )
    _require_equal(receipt.get("cuda_available"), True, label="receipt CUDA availability")
    _require_equal(receipt.get("cuda_device"), "Tesla T4", label="receipt Tesla T4 device")
    torch_version = receipt.get("torch")
    if not isinstance(torch_version, str) or "+cu" not in torch_version:
        raise ValueError(f"receipt torch identity is not CUDA-enabled: {torch_version!r}")
    _require_equal(receipt.get("exit_code"), 0, label="receipt exit code")
    expected_source_identity = {
        "dataset_sources": [packet_manifest.get("dataset_id")],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    _require_equal(
        receipt.get("source_identity"),
        expected_source_identity,
        label="receipt source identity",
    )
    manifest_record = receipt.get("input_manifest", {})
    _require_equal(
        manifest_record.get("sha256"),
        _sha256(packet_manifest_path),
        label="mounted input manifest digest",
    )
    _require_equal(
        manifest_record.get("size_bytes"),
        packet_manifest_path.stat().st_size,
        label="mounted input manifest size",
    )
    outputs = receipt.get("outputs", {})
    for name, path in (
        ("cuda_result.json", result_json),
        ("cuda_result.npz", result_npz),
    ):
        record = outputs.get(name, {})
        _require_equal(record.get("exists"), True, label=f"receipt output existence {name}")
        _require_equal(
            record.get("sha256"),
            _sha256(path),
            label=f"receipt output digest {name}",
        )
        _require_equal(
            record.get("size_bytes"),
            path.stat().st_size,
            label=f"receipt output size {name}",
        )

    route = report.get("effective_route", {})
    _require_equal(route.get("cuda_device"), receipt.get("cuda_device"), label="report CUDA device")
    command = receipt.get("effective_command", [])
    if not isinstance(command, list) or not command or not isinstance(command[0], str):
        raise ValueError("receipt effective command is not a list")
    outputs_declared = packet_manifest.get("outputs")
    if outputs_declared != ["cuda_result.json", "cuda_result.npz"]:
        raise ValueError(f"packet outputs are not the exact suffix pair: {outputs_declared!r}")
    expected_command_tail = [
        packet_manifest.get("entrypoint"),
        "--output-json",
        outputs_declared[0],
        "--output-npz",
        outputs_declared[1],
        *packet_manifest.get("entrypoint_args", []),
    ]
    _require_equal(command[1:], expected_command_tail, label="receipt effective command tail")

    inputs = report.get("inputs", {})
    mounted = receipt.get("inputs", {})
    manifest_files = packet_manifest.get("files")
    if not isinstance(manifest_files, dict) or not manifest_files:
        raise ValueError("packet manifest has no input files")
    if set(mounted) != set(manifest_files):
        raise ValueError(
            f"mounted input set differs from packet manifest: {sorted(mounted)} versus {sorted(manifest_files)}"
        )
    for name, expected_record in manifest_files.items():
        record = mounted.get(name, {})
        _require_equal(
            record.get("sha256"),
            expected_record.get("sha256"),
            label=f"mounted packet digest {name}",
        )
        _require_equal(
            record.get("size_bytes"),
            expected_record.get("size_bytes"),
            label=f"mounted packet size {name}",
        )
    expected_inputs = {
        "shape_flow_steps.npz": inputs.get("mlx", {}).get("capture_sha256"),
        "accepted_source_baseline.npz": inputs.get("accepted_source", {}).get("baseline_sha256"),
        "accepted_source_report.json": inputs.get("accepted_source", {}).get("report_sha256"),
        "trellis2_source_tarball.bin": inputs.get("source_tar_sha256"),
        "conditioning.npz": inputs.get("conditioning_sha256"),
    }
    for name, expected_sha in expected_inputs.items():
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(f"result report omits admitted digest for {name}")
        actual_sha = mounted.get(name, {}).get("sha256")
        _require_equal(actual_sha, expected_sha, label=f"mounted input digest {name}")

    dataset_id = str(packet_manifest["dataset_id"])
    owner, slug = dataset_id.split("/", 1)
    effective_dataset_dir = receipt.get("effective_dataset_dir")
    allowed_dataset_dirs = {
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/{owner}/{slug}",
    }
    if effective_dataset_dir not in allowed_dataset_dirs:
        raise ValueError(
            f"effective dataset directory is not canonical for {dataset_id}: {effective_dataset_dir!r}"
        )
    dataset_root = PurePosixPath(str(effective_dataset_dir))
    input_root = PurePosixPath("/kaggle/input")
    expected_dirs: set[str] = set()
    cursor = dataset_root
    while cursor != input_root:
        expected_dirs.add(str(cursor))
        if input_root not in cursor.parents:
            raise ValueError(f"effective dataset directory escapes /kaggle/input: {dataset_root}")
        cursor = cursor.parent
    expected_files = {
        str(dataset_root / relative_name)
        for relative_name in (*manifest_files, "witness-manifest.json")
    }
    for relative_name in manifest_files:
        parent = (dataset_root / relative_name).parent
        while parent != dataset_root:
            expected_dirs.add(str(parent))
            parent = parent.parent
    snapshot = receipt.get("mounted_input_snapshot", {})
    _require_equal(
        snapshot.get("mounted_input_root_exists"),
        True,
        label="mounted input root existence",
    )
    observed_dirs = snapshot.get("mounted_input_dirs")
    observed_files = snapshot.get("mounted_input_files")
    if not isinstance(observed_dirs, list) or set(observed_dirs) != expected_dirs:
        raise ValueError(
            f"mounted input directory set differs: {observed_dirs!r} versus {sorted(expected_dirs)!r}"
        )
    if not isinstance(observed_files, list) or set(observed_files) != expected_files:
        raise ValueError(
            f"mounted input file set differs: {observed_files!r} versus {sorted(expected_files)!r}"
        )


def _validate_download_report(
    download_report: dict[str, Any],
    *,
    result_json: Path,
    result_npz: Path,
    receipt_json: Path,
) -> None:
    _require_equal(
        download_report.get("schema"),
        "trellis2mlx.kaggle_cuda_witness.command_report.v1",
        label="download report schema",
    )
    _require_equal(download_report.get("phase"), "kernel_output", label="download report phase")
    _require_equal(download_report.get("status"), "done", label="download report status")
    _require_equal(download_report.get("failure_phase"), None, label="download failure phase")
    _require_equal(download_report.get("exit_code"), 0, label="download exit code")
    records = download_report.get("downloaded_outputs", {})
    expected_paths = {
        "cuda_result.json": result_json,
        "cuda_result.npz": result_npz,
        "kaggle_cuda_witness_receipt.json": receipt_json,
    }
    if set(records) != set(expected_paths):
        raise ValueError("download report does not bind exactly the result, NPZ, and receipt")
    for name, path in expected_paths.items():
        label = "downloaded receipt" if name == "kaggle_cuda_witness_receipt.json" else f"downloaded {name}"
        _require_equal(records[name].get("sha256"), _sha256(path), label=f"{label} digest")
        _require_equal(records[name].get("size_bytes"), path.stat().st_size, label=f"{label} size")


def _validate_packet_manifest(packet_manifest: dict[str, Any]) -> None:
    _require_equal(
        packet_manifest.get("schema"),
        "trellis2mlx.kaggle_cuda_witness.inputs.v1",
        label="packet manifest schema",
    )
    _require_equal(
        packet_manifest.get("accelerator"),
        "NvidiaTeslaT4",
        label="packet accelerator",
    )
    _require_equal(
        packet_manifest.get("entrypoint"),
        "source_cuda_shape_flow_suffix_ladder.py",
        label="packet entrypoint",
    )
    for name in ("dataset_id", "kernel_id"):
        value = packet_manifest.get(name)
        if not isinstance(value, str) or value.count("/") != 1:
            raise ValueError(f"packet {name} is invalid: {value!r}")


def _metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError(f"metric arrays differ in shape: {left.shape} versus {right.shape}")
    diff = np.abs(left.astype(np.float32, copy=False) - right.astype(np.float32, copy=False))
    return {
        "shape_match": True,
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
        "nonzero": int(np.count_nonzero(diff)),
        "exact": bool(np.array_equal(left, right)),
    }


def _float32_metrics_close(left: float, right: float) -> bool:
    left32 = np.float32(left)
    right32 = np.float32(right)
    if not np.isfinite(left32) or not np.isfinite(right32):
        return False
    if left32 == right32:
        return True
    if left32 < 0 or right32 < 0:
        return False
    left_bits = int(left32.view(np.uint32))
    right_bits = int(right32.view(np.uint32))
    return abs(left_bits - right_bits) <= FLOAT32_REDUCTION_ULPS


def _validate_metrics(reported: dict[str, Any], actual: dict[str, Any], *, label: str) -> None:
    for key in ("shape_match", "nonzero", "exact"):
        _require_equal(reported.get(key), actual[key], label=f"{label} {key}")
    for key in ("mean_abs", "max_abs"):
        value = reported.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{label} {key} is not finite")
        if not _float32_metrics_close(float(value), actual[key]):
            raise ValueError(f"{label} {key} differs: {value!r} versus {actual[key]!r}")


def _load_and_validate_arrays(
    result_npz: Path, report: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    primary = report.get("primary_output", {})
    actual_sha = _sha256(result_npz)
    if primary.get("sha256") != actual_sha:
        raise ValueError("primary NPZ digest differs from the result report")
    if primary.get("size_bytes") != result_npz.stat().st_size:
        raise ValueError("primary NPZ size differs from the result report")
    validation = primary.get("validation", {})
    if validation.get("point_arrays_bound") is not True or validation.get("switch_count") != 9:
        raise ValueError("primary NPZ validation does not bind all nine point arrays")

    points = report["points"]
    canonical_keys = {f"switch_{step}_shape_slat" for step in SWITCH_STEPS}
    observed_keys = [str(point.get("output_key")) for point in points]
    if observed_keys != [f"switch_{step}_shape_slat" for step in SWITCH_STEPS]:
        raise ValueError(f"point output keys do not match the canonical output key order: {observed_keys}")
    validate_saved_artifact(result_npz, points=points)
    with np.load(result_npz, allow_pickle=False) as archive:
        required = {
            "coords",
            "switch_steps",
            "accepted_source_anchor_shape_slat",
            "mlx_anchor_shape_slat",
        } | canonical_keys
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"suffix artifact is missing analysis arrays: {missing}")
        coords = np.asarray(archive["coords"])
        switch_steps = np.asarray(archive["switch_steps"])
        source = np.asarray(archive["accepted_source_anchor_shape_slat"])
        mlx = np.asarray(archive["mlx_anchor_shape_slat"])
        endpoints = {
            step: np.asarray(archive[f"switch_{step}_shape_slat"])
            for step in SWITCH_STEPS
        }
        raw_metadata = np.asarray(archive["metadata_json"])
        metadata = json.loads(str(raw_metadata.item()), parse_constant=_reject_json_constant)

    if coords.dtype != np.int32 or coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"coords must be nonempty int32 [N,4], got {coords.dtype} {coords.shape}")
    if coords.shape[0] == 0:
        raise ValueError("coords are empty")
    if switch_steps.dtype != np.int32 or not np.array_equal(
        switch_steps, np.asarray(SWITCH_STEPS, dtype=np.int32)
    ):
        raise ValueError("artifact switch_steps do not preserve the full uncapped ladder")
    if source.dtype != np.float32 or mlx.dtype != np.float32 or source.shape != mlx.shape:
        raise ValueError("source and MLX anchors must be shape-matched float32 arrays")
    if source.ndim != 2 or source.shape[0] != coords.shape[0]:
        raise ValueError("anchor shape does not match coordinates")
    if not np.isfinite(source).all() or not np.isfinite(mlx).all():
        raise ValueError("anchor arrays contain non-finite values")
    if np.array_equal(source, mlx):
        raise ValueError("source and MLX anchors are byte-identical; no anchor axis exists")

    for point in points:
        step = int(point["switch_step"])
        endpoint = endpoints[step]
        if endpoint.dtype != np.float32 or endpoint.shape != source.shape:
            raise ValueError(f"switch {step} does not match anchor dtype and shape")
        if not np.isfinite(endpoint).all():
            raise ValueError(f"switch {step} contains non-finite values")
        if point.get("sha256") != _array_sha(endpoint):
            raise ValueError(f"switch {step} array digest differs from the report")
        vs_source = _metrics(endpoint, source)
        vs_mlx = _metrics(endpoint, mlx)
        _validate_metrics(point.get("vs_source_anchor", {}), vs_source, label=f"switch {step} vs source")
        _validate_metrics(point.get("vs_mlx_anchor", {}), vs_mlx, label=f"switch {step} vs MLX")
        expected_class = classify_anchor(vs_source["mean_abs"], vs_mlx["mean_abs"])
        _require_equal(point.get("nearest_anchor"), expected_class, label=f"switch {step} nearest anchor")

    route = report.get("effective_route", {})
    if metadata.get("effective_route") != route:
        raise ValueError("artifact metadata effective route differs from the result report")
    if metadata.get("inputs") != report.get("inputs"):
        raise ValueError("artifact metadata inputs differ from the result report")
    if metadata.get("points") != points:
        raise ValueError("artifact metadata point identities differ from the result report")
    if metadata.get("pairwise") != report.get("pairwise"):
        raise ValueError("artifact metadata pairwise matrix differs from the result report")
    if metadata.get("timing") != report.get("timing"):
        raise ValueError("artifact metadata timing differs from the result report")
    return source, mlx, endpoints


def _validate_pairwise(report: dict[str, Any], endpoints: dict[int, np.ndarray]) -> None:
    reported = report.get("pairwise", {})
    expected_keys = {f"{left}:{right}" for left in SWITCH_STEPS for right in SWITCH_STEPS}
    if set(reported) != expected_keys:
        raise ValueError("reported pairwise matrix does not contain exactly all 81 switch pairs")
    for left in SWITCH_STEPS:
        for right in SWITCH_STEPS:
            key = f"{left}:{right}"
            _validate_metrics(
                reported[key],
                _metrics(endpoints[left], endpoints[right]),
                label=f"pairwise {key}",
            )


def _exact_quotients(
    points: list[dict[str, Any]], endpoints: dict[int, np.ndarray]
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    classes: list[dict[str, Any]] = []
    by_digest: dict[str, dict[str, Any]] = {}
    representative_by_step: dict[int, int] = {}
    for point in points:
        step = int(point["switch_step"])
        digest = _array_sha(endpoints[step])
        entry = by_digest.get(digest)
        if entry is None:
            entry = {
                "class_index": len(classes),
                "representative_switch": step,
                "switch_steps": [],
                "shape_slat_sha256": digest,
            }
            by_digest[digest] = entry
            classes.append(entry)
        entry["switch_steps"].append(step)
        representative_by_step[step] = int(entry["representative_switch"])
    return classes, representative_by_step


def _finite_scalar(value: float, *, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"derived {label} is non-finite")
    return value


def _build_geometry(
    report: dict[str, Any], source: np.ndarray, mlx: np.ndarray, endpoints: dict[int, np.ndarray]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source64 = source.astype(np.float64).reshape(-1)
    mlx64 = mlx.astype(np.float64).reshape(-1)
    chord = mlx64 - source64
    chord_l2 = float(np.linalg.norm(chord))
    chord_l1 = float(np.mean(np.abs(chord)))
    if not math.isfinite(chord_l2) or not math.isfinite(chord_l1) or chord_l2 <= 0 or chord_l1 <= 0:
        raise ValueError("source-to-MLX anchor separation is invalid")
    chord_sq = float(np.dot(chord, chord))

    point_records: list[dict[str, Any]] = []
    classes: list[str] = []
    for point in report["points"]:
        step = int(point["switch_step"])
        vector = endpoints[step].astype(np.float64).reshape(-1)
        relative = vector - source64
        projection = float(np.dot(relative, chord) / chord_sq)
        residual = relative - projection * chord
        source_l2 = float(np.linalg.norm(vector - source64) / chord_l2)
        mlx_l2 = float(np.linalg.norm(vector - mlx64) / chord_l2)
        nearest = str(point["nearest_anchor"])
        classes.append(nearest)
        point_records.append(
            {
                "switch_step": step,
                "nearest_anchor_l1": nearest,
                "anchor_axis_projection": _finite_scalar(projection, label=f"switch {step} projection"),
                "transverse_l2_ratio": _finite_scalar(
                    np.linalg.norm(residual) / chord_l2,
                    label=f"switch {step} transverse ratio",
                ),
                "source_l1_ratio": _finite_scalar(
                    point["vs_source_anchor"]["mean_abs"] / chord_l1,
                    label=f"switch {step} source L1 ratio",
                ),
                "mlx_l1_ratio": _finite_scalar(
                    point["vs_mlx_anchor"]["mean_abs"] / chord_l1,
                    label=f"switch {step} MLX L1 ratio",
                ),
                "source_l2_ratio": _finite_scalar(source_l2, label=f"switch {step} source L2 ratio"),
                "mlx_l2_ratio": _finite_scalar(mlx_l2, label=f"switch {step} MLX L2 ratio"),
                "shape_slat_sha256": point["sha256"],
            }
        )

    adjacent: list[dict[str, Any]] = []
    deltas: list[np.ndarray] = []
    for left, right in zip(SWITCH_STEPS[:-1], SWITCH_STEPS[1:]):
        delta = endpoints[right].astype(np.float64).reshape(-1) - endpoints[left].astype(np.float64).reshape(-1)
        deltas.append(delta)
        adjacent.append(
            {
                "left_switch": left,
                "right_switch": right,
                "mean_abs": _finite_scalar(np.mean(np.abs(delta)), label=f"adjacent {left}:{right} mean abs"),
                "l2_anchor_ratio": _finite_scalar(
                    np.linalg.norm(delta) / chord_l2,
                    label=f"adjacent {left}:{right} L2 ratio",
                ),
                "exact": bool(np.array_equal(endpoints[left], endpoints[right])),
            }
        )
    turns: list[dict[str, Any]] = []
    for index in range(1, len(deltas)):
        left_delta = deltas[index - 1]
        right_delta = deltas[index]
        denominator = float(np.linalg.norm(left_delta) * np.linalg.norm(right_delta))
        cosine = None
        if denominator > 0 and math.isfinite(denominator):
            cosine = _finite_scalar(
                np.dot(left_delta, right_delta) / denominator,
                label=f"turn at switch {index} cosine",
            )
            cosine = max(-1.0, min(1.0, cosine))
        turns.append({"switch_step": index, "cosine": cosine})
    return point_records, adjacent, turns


def _separatrix(
    point_records: list[dict[str, Any]], representative_by_step: dict[int, int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    classes = [str(point["nearest_anchor_l1"]) for point in point_records]
    for left, right in zip(point_records[:-1], point_records[1:]):
        if left["nearest_anchor_l1"] != right["nearest_anchor_l1"]:
            transitions.append(
                {
                    "left_switch": int(left["switch_step"]),
                    "right_switch": int(right["switch_step"]),
                    "left_class": str(left["nearest_anchor_l1"]),
                    "right_class": str(right["nearest_anchor_l1"]),
                }
            )
    rank = {"source": 0, "equidistant": 1, "mlx": 2}
    monotonic = all(rank[classes[index]] <= rank[classes[index + 1]] for index in range(len(classes) - 1))
    disposition = (
        "no_classification_crossing"
        if not transitions
        else "single_crossing"
        if len(transitions) == 1
        else "multiple_crossings"
    )
    requested_steps: list[int] = []
    for transition in transitions:
        for key in ("left_switch", "right_switch"):
            representative = representative_by_step[int(transition[key])]
            if representative not in requested_steps:
                requested_steps.append(representative)
    requested_steps.sort()
    separatrix = {
        "metric": "recomputed mean-absolute distance to exact source and MLX anchors",
        "classification_sequence": classes,
        "classification_transitions": transitions,
        "monotonic_source_to_mlx": monotonic,
        "disposition": disposition,
    }
    recommendation = {
        "switch_steps": requested_steps,
        "anchor_context_steps": [representative_by_step[0], representative_by_step[8]],
        "selection_rule": "byte-exact quotient representatives at every nearest-anchor classification boundary",
        "requires_visual_decode": bool(requested_steps),
    }
    return separatrix, recommendation


def analyze_suffix_ladder(
    result_json: Path | str,
    result_npz: Path | str,
    receipt_json: Path | str,
    download_report_json: Path | str,
    packet_manifest_json: Path | str,
) -> dict[str, Any]:
    result_json = Path(result_json)
    result_npz = Path(result_npz)
    receipt_json = Path(receipt_json)
    download_report_json = Path(download_report_json)
    packet_manifest_json = Path(packet_manifest_json)
    report = _read_json(result_json, label="CUDA suffix result report")
    receipt = _read_json(receipt_json, label="Kaggle CUDA receipt")
    download_report = _read_json(download_report_json, label="Kaggle output download report")
    packet_manifest = _read_json(packet_manifest_json, label="CUDA witness packet manifest")
    if not result_npz.is_file() or result_npz.stat().st_size == 0:
        raise ValueError(f"CUDA suffix result NPZ is not a nonblank file: {result_npz}")
    validate_result_manifest(report)
    _validate_packet_manifest(packet_manifest)
    _validate_receipt(
        receipt,
        report,
        packet_manifest,
        result_json=result_json,
        result_npz=result_npz,
        packet_manifest_path=packet_manifest_json,
    )
    _validate_download_report(
        download_report,
        result_json=result_json,
        result_npz=result_npz,
        receipt_json=receipt_json,
    )
    source, mlx, endpoints = _load_and_validate_arrays(result_npz, report)
    _validate_pairwise(report, endpoints)
    quotients, representative_by_step = _exact_quotients(report["points"], endpoints)
    points, adjacent, turns = _build_geometry(report, source, mlx, endpoints)
    separatrix, recommendation = _separatrix(points, representative_by_step)
    return {
        "schema": SCHEMA,
        "status": "done",
        "analysis_status": "written",
        "failure_phase": None,
        "switch_steps": list(SWITCH_STEPS),
        "effective_route": report["effective_route"],
        "evidence": {
            "result_json": {"path": str(result_json), "sha256": _sha256(result_json)},
            "result_npz": {"path": str(result_npz), "sha256": _sha256(result_npz)},
            "receipt_json": {"path": str(receipt_json), "sha256": _sha256(receipt_json)},
            "download_report_json": {
                "path": str(download_report_json),
                "sha256": _sha256(download_report_json),
            },
            "packet_manifest_json": {
                "path": str(packet_manifest_json),
                "sha256": _sha256(packet_manifest_json),
            },
            "torch": receipt["torch"],
            "cuda_device": receipt["cuda_device"],
        },
        "points": points,
        "adjacent_steps": adjacent,
        "turn_cosines": turns,
        "exact_quotient_classes": quotients,
        "separatrix": separatrix,
        "decode_recommendation": recommendation,
        "timing": report["timing"],
        "forbidden_inferences": [
            "anchor-axis projection is a direct witness coordinate, not a learned-manifold embedding",
            "nearest-anchor crossing is not visual-basin proof until selected endpoints are decoded",
            "exact quotient equality does not imply topology, winding, texture, or GLB correctness",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--result-npz", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--download-report", required=True, type=Path)
    parser.add_argument("--packet-manifest", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    phase = "request_validation"
    failure: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "analysis_status": "missing",
        "failure_phase": phase,
    }
    try:
        output_resolved = args.output_json.resolve()
        for label, path in (
            ("result report", args.result_json),
            ("result NPZ", args.result_npz),
            ("receipt", args.receipt),
            ("download report", args.download_report),
            ("packet manifest", args.packet_manifest),
        ):
            if output_resolved == path.resolve():
                raise ValueError(f"output JSON collides with {label}: {path}")
        phase = "input_validation"
        analysis = analyze_suffix_ladder(
            args.result_json,
            args.result_npz,
            args.receipt,
            args.download_report,
            args.packet_manifest,
        )
        phase = "write_analysis"
        _write_json(args.output_json, analysis)
        return 0
    except Exception as exc:
        failure.update(
            {
                "failure_phase": phase,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        try:
            if args.output_json.resolve() not in {
                args.result_json.resolve(),
                args.result_npz.resolve(),
                args.receipt.resolve(),
                args.download_report.resolve(),
                args.packet_manifest.resolve(),
            }:
                _write_json(args.output_json, failure)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
