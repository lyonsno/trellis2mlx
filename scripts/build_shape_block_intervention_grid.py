"""Build an uncapped Cartesian manifest plan for block29 residual interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping


SCHEMA = "trellis2mlx.shape_block_intervention_grid_plan.v1"
MANIFEST_SCHEMA = "trellis2mlx.shape_block_injection_manifest.v1"
COMPARISON_CLASS = "block29_after_self_cross_attention_raw_delta_grid"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--index-json", required=True, type=Path)
    parser.add_argument("--prefix-trace", required=True, type=Path)
    parser.add_argument("--block29-trace", required=True, type=Path)
    parser.add_argument("--alphas", required=True)
    parser.add_argument("--betas", required=True)
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        metavar="ALPHA,BETA=TRACE",
        help="Exact accepted trace for one semantic corner; repeat for each corner.",
    )
    return parser


def parse_axis(value: str, *, name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{name} axis must contain at least one value")
    return _validate_axis((float(part) for part in parts), name=name)


def build_grid_plan(
    *,
    manifest_dir: Path,
    run_root: Path,
    index_path: Path,
    prefix_trace: Path,
    block29_trace: Path,
    alphas: Iterable[float],
    betas: Iterable[float],
    control_references: Mapping[tuple[float, float], Path],
) -> dict:
    alpha_values = _validate_axis(alphas, name="alpha")
    beta_values = _validate_axis(betas, name="beta")
    controls = {
        (float(alpha), float(beta)): Path(path).resolve()
        for (alpha, beta), path in control_references.items()
    }
    requested = {(alpha, beta) for alpha in alpha_values for beta in beta_values}
    unknown_controls = sorted(set(controls) - requested)
    if unknown_controls:
        raise ValueError(f"control coordinates are outside the requested grid: {unknown_controls}")

    manifest_dir = Path(manifest_dir).resolve()
    run_root = Path(run_root).resolve()
    index_path = Path(index_path).resolve()
    prefix_trace = Path(prefix_trace).resolve()
    block29_trace = Path(block29_trace).resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    points = []
    for alpha in alpha_values:
        for beta in beta_values:
            name = f"alpha-{_float_slug(alpha)}_beta-{_float_slug(beta)}"
            manifest_path = manifest_dir / f"{name}.json"
            output_dir = run_root / name
            coordinate = {"alpha": alpha, "beta": beta}
            manifest = {
                "schema": MANIFEST_SCHEMA,
                "comparison_class": COMPARISON_CLASS,
                "grid_coordinate": coordinate,
                "sites": [
                    _site(prefix_trace, block_index=28, stage="after_mlp", scale=1.0),
                    _site(block29_trace, block_index=29, stage="after_self", scale=alpha),
                    _site(
                        block29_trace,
                        block_index=29,
                        stage="cross_attention_raw",
                        scale=beta,
                    ),
                ],
            }
            _write_json(manifest_path, manifest)
            point = {
                "name": name,
                "coordinate": coordinate,
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "output_dir": str(output_dir),
                "expected_trace_path": str(
                    output_dir / "checkpoints" / "shape_flow_block_trace.npz"
                ),
            }
            role = _control_role(alpha, beta)
            if role is not None:
                point["control_role"] = role
            if (alpha, beta) in controls:
                point["control_reference"] = str(controls[(alpha, beta)])
            points.append(point)

    index = {
        "schema": SCHEMA,
        "status": "planned",
        "comparison_class": COMPARISON_CLASS,
        "axes": {"alpha": list(alpha_values), "beta": list(beta_values)},
        "axis_semantics": {
            "alpha": "source_delta_scale at block29 after_self",
            "beta": "source_delta_scale at block29 cross_attention_raw",
        },
        "source_traces": {
            "prefix28": str(prefix_trace),
            "block29": str(block29_trace),
        },
        "point_count": len(points),
        "points": points,
    }
    _write_json(index_path, index)
    return index


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    controls = dict(_parse_control(value) for value in args.control)
    build_grid_plan(
        manifest_dir=args.manifest_dir,
        run_root=args.run_root,
        index_path=args.index_json,
        prefix_trace=args.prefix_trace,
        block29_trace=args.block29_trace,
        alphas=parse_axis(args.alphas, name="alpha"),
        betas=parse_axis(args.betas, name="beta"),
        control_references=controls,
    )
    return 0


def _validate_axis(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized:
        raise ValueError(f"{name} axis must contain at least one value")
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} axis values must be finite")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} axis contains duplicate values")
    return normalized


def _parse_control(value: str) -> tuple[tuple[float, float], Path]:
    try:
        coordinate, path = value.split("=", 1)
        alpha_raw, beta_raw = coordinate.split(",", 1)
        alpha, beta = float(alpha_raw), float(beta_raw)
    except ValueError as exc:
        raise ValueError(f"invalid control {value!r}; expected ALPHA,BETA=TRACE") from exc
    if not path or not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError(f"invalid control {value!r}; coordinate must be finite and path non-empty")
    return (alpha, beta), Path(path)


def _site(trace_path: Path, *, block_index: int, stage: str, scale: float) -> dict:
    return {
        "trace_path": str(trace_path),
        "branch": "both",
        "step_index": 0,
        "block_index": block_index,
        "stage": stage,
        "source_delta_scale": scale,
    }


def _control_role(alpha: float, beta: float) -> str | None:
    return {
        (0.0, 0.0): "zero_correction_instrumentation_control",
        (1.0, 0.0): "after_self_exact_control",
        (0.0, 1.0): "cross_attention_raw_exact_control",
        (1.0, 1.0): "exact_join_control",
    }.get((alpha, beta))


def _float_slug(value: float) -> str:
    text = format(value, ".17g")
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
