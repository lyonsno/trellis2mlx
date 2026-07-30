"""Compare official CUDA and MLX raw TRELLIS shape-decoder states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


LEVEL_COUNT = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Align official CUDA and MLX shape-decoder subdivision states by "
            "voxel coordinate and locate their first decision fork"
        )
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-input-slat", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = compare_decoder_states(
        args.source,
        args.candidate,
        args.candidate_input_slat,
    )
    _write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "first_numeric_fork_level": report["first_numeric_fork_level"],
                "first_threshold_fork_level": report[
                    "first_threshold_fork_level"
                ],
                "first_support_fork_after_level": report[
                    "first_support_fork_after_level"
                ],
                "final_coord_jaccard": report["final"]["coords"]["jaccard"],
                "final_feature_max_abs_diff": report["final"]["features"][
                    "max_abs_diff"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


def compare_decoder_states(
    source_path: Path,
    candidate_path: Path,
    candidate_input_path: Path,
) -> dict[str, Any]:
    source_path = Path(source_path)
    candidate_path = Path(candidate_path)
    candidate_input_path = Path(candidate_input_path)
    required_source = {"feats", "coords"}
    required_candidate = {"feats", "coords"}
    for level in range(LEVEL_COUNT):
        required_source.update(
            {f"shape_subs_{level}", f"shape_subs_{level}_coords"}
        )
        required_candidate.add(f"shape_subs_{level}")

    with (
        np.load(source_path, allow_pickle=False) as source,
        np.load(candidate_path, allow_pickle=False) as candidate,
        np.load(candidate_input_path, allow_pickle=False) as candidate_input,
    ):
        _require_arrays(source, required_source, "source")
        _require_arrays(candidate, required_candidate, "candidate")
        _require_arrays(
            candidate_input,
            {"coords", "feats"},
            "candidate input SLat",
        )

        source_final_coords = _validate_coords(
            source["coords"], "source final coords"
        )
        source_final_feats = _validate_features(
            source["feats"],
            source_final_coords.shape[0],
            np.float32,
            7,
            "source final feats",
        )
        candidate_final_coords = _validate_coords(
            candidate["coords"], "candidate final coords"
        )
        candidate_final_feats = _validate_features(
            candidate["feats"],
            candidate_final_coords.shape[0],
            np.float32,
            7,
            "candidate final feats",
        )
        candidate_coords = _validate_coords(
            candidate_input["coords"], "candidate input SLat coords"
        )
        _validate_features(
            candidate_input["feats"],
            candidate_coords.shape[0],
            np.float32,
            32,
            "candidate input SLat feats",
        )

        levels: list[dict[str, Any]] = []
        first_numeric_fork: int | None = None
        first_threshold_fork: int | None = None
        first_support_fork: int | None = None

        for level in range(LEVEL_COUNT):
            source_coords = _validate_coords(
                source[f"shape_subs_{level}_coords"],
                f"source subdivision level {level} coords",
            )
            source_logits = _validate_logits(
                source[f"shape_subs_{level}"],
                source_coords.shape[0],
                np.float16,
                f"source subdivision level {level} logits",
            )
            candidate_logits = _validate_logits(
                candidate[f"shape_subs_{level}"],
                candidate_coords.shape[0],
                np.float16,
                f"candidate subdivision level {level} logits",
            )

            input_support, source_order, candidate_order = _coord_overlap(
                source_coords,
                candidate_coords,
            )
            source_common = source_logits[source_order]
            candidate_common = candidate_logits[candidate_order]
            logits_report = _numeric_delta(source_common, candidate_common)
            source_decisions = source_common > 0
            candidate_decisions = candidate_common > 0
            mismatched_decisions = source_decisions != candidate_decisions
            mismatch_count = int(np.count_nonzero(mismatched_decisions))
            decision_count = int(mismatched_decisions.size)

            if (
                first_numeric_fork is None
                and logits_report["nonzero_count"] > 0
            ):
                first_numeric_fork = level
            if (
                first_threshold_fork is None
                and mismatch_count > 0
            ):
                first_threshold_fork = level

            source_next = _upsample_coords(source_coords, source_logits)
            candidate_next = _upsample_coords(candidate_coords, candidate_logits)
            source_recorded_next = (
                source_final_coords
                if level == LEVEL_COUNT - 1
                else _validate_coords(
                    source[f"shape_subs_{level + 1}_coords"],
                    f"source subdivision level {level + 1} coords",
                )
            )
            source_chain, _, _ = _coord_overlap(
                source_recorded_next,
                source_next,
            )
            if source_chain["jaccard"] != 1.0:
                raise ValueError(
                    "source subdivision level "
                    f"{level} reconstructed support does not match recorded next support"
                )

            next_support, _, _ = _coord_overlap(
                source_recorded_next,
                candidate_next,
            )
            if first_support_fork is None and next_support["jaccard"] != 1.0:
                first_support_fork = level

            levels.append(
                {
                    "level": level,
                    "input_support": input_support,
                    "logits": logits_report,
                    "threshold_decisions": {
                        "common_shape": list(source_decisions.shape),
                        "decision_count": decision_count,
                        "mismatched_count": mismatch_count,
                        "mismatch_rate": (
                            mismatch_count / decision_count
                            if decision_count
                            else 0.0
                        ),
                        "reference_positive_count": int(
                            np.count_nonzero(source_decisions)
                        ),
                        "candidate_positive_count": int(
                            np.count_nonzero(candidate_decisions)
                        ),
                    },
                    "source_chain": source_chain,
                    "next_support": next_support,
                }
            )
            candidate_coords = candidate_next

        candidate_chain, _, _ = _coord_overlap(
            candidate_final_coords,
            candidate_coords,
        )
        if candidate_chain["jaccard"] != 1.0:
            raise ValueError(
                "candidate reconstructed final support does not match candidate final coords"
            )

        final_coords, source_order, candidate_order = _coord_overlap(
            source_final_coords,
            candidate_final_coords,
        )
        final_features = _numeric_delta(
            source_final_feats[source_order],
            candidate_final_feats[candidate_order],
        )

    return {
        "schema": "trellis2mlx.decoder_state_comparison.v1",
        "source": _artifact_identity(source_path),
        "candidate": _artifact_identity(candidate_path),
        "candidate_input_slat": _artifact_identity(candidate_input_path),
        "subdivision_levels": LEVEL_COUNT,
        "first_numeric_fork_level": first_numeric_fork,
        "first_threshold_fork_level": first_threshold_fork,
        "first_support_fork_after_level": first_support_fork,
        "levels": levels,
        "final": {
            "coords": final_coords,
            "features": final_features,
            "candidate_reconstruction": candidate_chain,
        },
    }


def _require_arrays(
    archive: Any,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(archive.files))
    if missing:
        raise KeyError(f"{label} artifact missing required arrays: {', '.join(missing)}")


def _validate_coords(array: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(array)
    if values.dtype != np.dtype(np.int32):
        raise ValueError(f"{label} must have dtype int32, got {values.dtype}")
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"{label} must have shape [N, 4], got {values.shape}")
    if values.shape[0] == 0:
        raise ValueError(f"{label} must be nonempty")
    contiguous = np.ascontiguousarray(values)
    if np.unique(_row_keys(contiguous)).size != contiguous.shape[0]:
        raise ValueError(f"{label} contains duplicate coordinates")
    return contiguous


def _validate_features(
    array: np.ndarray,
    rows: int,
    dtype: np.dtype[Any] | type[np.generic],
    width: int,
    label: str,
) -> np.ndarray:
    values = np.asarray(array)
    expected_dtype = np.dtype(dtype)
    if values.dtype != expected_dtype:
        raise ValueError(
            f"{label} must have dtype {expected_dtype}, got {values.dtype}"
        )
    if values.ndim != 2 or values.shape != (rows, width):
        raise ValueError(
            f"{label} must have shape [{rows}, {width}], got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")
    return np.ascontiguousarray(values)


def _validate_logits(
    array: np.ndarray,
    rows: int,
    dtype: np.dtype[Any] | type[np.generic],
    label: str,
) -> np.ndarray:
    values = np.asarray(array)
    expected_dtype = np.dtype(dtype)
    if values.dtype != expected_dtype:
        raise ValueError(
            f"{label} must have dtype {expected_dtype}, got {values.dtype}"
        )
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError(f"{label} must have shape [N, 8], got {values.shape}")
    if values.shape[0] != rows:
        owner = "candidate" if label.startswith("candidate") else "source"
        level = label.split("level ", 1)[1].split(" ", 1)[0]
        raise ValueError(
            f"{owner} subdivision level {level} row count does not match "
            f"{'reconstructed support' if owner == 'candidate' else 'recorded support'}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")
    return np.ascontiguousarray(values)


def _upsample_coords(coords: np.ndarray, logits: np.ndarray) -> np.ndarray:
    parent, child = np.nonzero(logits > 0)
    if parent.size == 0:
        raise ValueError("subdivision decisions produce empty support")
    counts = np.bincount(parent, minlength=coords.shape[0])
    output = np.repeat(coords, counts, axis=0).astype(np.int32, copy=True)
    output[:, 1:] *= 2
    output[:, 1] += child % 2
    output[:, 2] += (child // 2) % 2
    output[:, 3] += child // 4
    return output


def _coord_overlap(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    exact_order_match = bool(
        reference.shape == candidate.shape and np.array_equal(reference, candidate)
    )
    if exact_order_match:
        common_count = int(reference.shape[0])
        reference_order = np.arange(common_count, dtype=np.int64)
        candidate_order = reference_order
    else:
        _, reference_order, candidate_order = np.intersect1d(
            _row_keys(reference),
            _row_keys(candidate),
            assume_unique=True,
            return_indices=True,
        )
        common_count = int(reference_order.size)
    reference_count = int(reference.shape[0])
    candidate_count = int(candidate.shape[0])
    union_count = reference_count + candidate_count - common_count
    report = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_count": reference_count,
        "candidate_count": candidate_count,
        "common_count": common_count,
        "reference_only_count": reference_count - common_count,
        "candidate_only_count": candidate_count - common_count,
        "union_count": union_count,
        "jaccard": common_count / union_count if union_count else 1.0,
        "exact_order_match": exact_order_match,
    }
    return report, reference_order, candidate_order


def _row_keys(coords: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(coords)
    row_dtype = np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    return contiguous.view(row_dtype).reshape(-1)


def _numeric_delta(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(
            "aligned numeric arrays must have matching shapes, "
            f"got {reference.shape} and {candidate.shape}"
        )
    if reference.size == 0:
        return {
            "common_shape": list(reference.shape),
            "reference_dtype": str(reference.dtype),
            "candidate_dtype": str(candidate.dtype),
            "nonzero_count": 0,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "rms_diff": None,
        }
    delta = reference.astype(np.float64) - candidate.astype(np.float64)
    absolute = np.abs(delta)
    return {
        "common_shape": list(reference.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "nonzero_count": int(np.count_nonzero(delta)),
        "max_abs_diff": float(np.max(absolute)),
        "mean_abs_diff": float(np.mean(absolute)),
        "rms_diff": float(np.sqrt(np.mean(np.square(delta)))),
    }


def _artifact_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
