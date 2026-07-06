#!/usr/bin/env python
"""Compare local QEM first-step state against a direct source readback NPZ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trellmlx.qem_source_readback import (  # noqa: E402
    REPORT_SCHEMA,
    SOURCE_DISTRIBUTION_SCHEMA,
    _jsonable,
    build_qem_source_distribution_report,
    build_qem_source_readback_report,
    failure_report,
    load_mesh_npz,
    load_source_readback_npz,
)


REQUESTED_ROUTE = "qem-source-readback-compare"
EFFECTIVE_ROUTE = "local-qem-step-vs-source-readback"
DISTRIBUTION_REQUESTED_ROUTE = "qem-source-distribution-compare"
DISTRIBUTION_EFFECTIVE_ROUTE = "local-qem-step-vs-source-distribution"


def _route_identity(source_count: int) -> tuple[str, str, str]:
    if source_count > 1:
        return DISTRIBUTION_REQUESTED_ROUTE, DISTRIBUTION_EFFECTIVE_ROUTE, SOURCE_DISTRIBUTION_SCHEMA
    return REQUESTED_ROUTE, EFFECTIVE_ROUTE, REPORT_SCHEMA


def _failure_report(
    *,
    failure_phase: str,
    error: Exception,
    mesh_path: Path | None,
    source_readback_path: Path | None,
    source_readback_paths: list[Path],
    qem_backend: str,
) -> dict:
    requested_route, effective_route, schema = _route_identity(len(source_readback_paths))
    return failure_report(
        requested_route=requested_route,
        effective_route=effective_route,
        failure_phase=failure_phase,
        error=error,
        mesh_path=mesh_path,
        source_readback_path=source_readback_path,
        schema=schema,
        source_readback_paths=source_readback_paths if len(source_readback_paths) > 1 else None,
        settings={"qem_backend": qem_backend},
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, help="NPZ containing vertices and faces arrays")
    parser.add_argument(
        "--source-readback",
        type=Path,
        action="append",
        required=True,
        help="NPZ containing source edges, costs, props, and optional qems arrays",
    )
    parser.add_argument("--lambda-edge-length", type=float, default=1e-2)
    parser.add_argument("--lambda-skinny", type=float, default=1e-3)
    parser.add_argument("--collapse-thresh", type=float, default=1e-8)
    parser.add_argument(
        "--local-qem-backend",
        choices=["cpu-vectorized", "mlx-metal-source"],
        default="cpu-vectorized",
        help="Local QEM/base-cost backend to compare against source readbacks",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overwrite-report", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n")


def _ensure_report_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"report already exists: {path}; pass --overwrite-report to replace it")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _ensure_report_writable(args.report, args.overwrite_report)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        vertices, faces = load_mesh_npz(args.mesh)
    except Exception as exc:
        _write_json(
            args.report,
            _failure_report(
                failure_phase="mesh",
                error=exc,
                mesh_path=args.mesh,
                source_readback_path=None,
                source_readback_paths=args.source_readback,
                qem_backend=args.local_qem_backend,
            ),
        )
        return 1

    sources = []
    try:
        for source_readback_path in args.source_readback:
            sources.append(load_source_readback_npz(source_readback_path))
    except Exception as exc:
        _write_json(
            args.report,
            _failure_report(
                failure_phase="source_readback",
                error=exc,
                mesh_path=args.mesh,
                source_readback_path=source_readback_path,
                source_readback_paths=args.source_readback,
                qem_backend=args.local_qem_backend,
            ),
        )
        return 1

    try:
        if len(sources) == 1:
            report = build_qem_source_readback_report(
                requested_route=REQUESTED_ROUTE,
                effective_route=EFFECTIVE_ROUTE,
                mesh_path=args.mesh,
                source_readback_path=args.source_readback[0],
                vertices=vertices,
                faces=faces,
                source=sources[0],
                lambda_edge_length=args.lambda_edge_length,
                lambda_skinny=args.lambda_skinny,
                collapse_thresh=args.collapse_thresh,
                qem_backend=args.local_qem_backend,
            )
        else:
            report = build_qem_source_distribution_report(
                requested_route=DISTRIBUTION_REQUESTED_ROUTE,
                effective_route=DISTRIBUTION_EFFECTIVE_ROUTE,
                mesh_path=args.mesh,
                source_readback_paths=args.source_readback,
                vertices=vertices,
                faces=faces,
                sources=sources,
                lambda_edge_length=args.lambda_edge_length,
                lambda_skinny=args.lambda_skinny,
                collapse_thresh=args.collapse_thresh,
                qem_backend=args.local_qem_backend,
            )
    except Exception as exc:
        _write_json(
            args.report,
            _failure_report(
                failure_phase="compare",
                error=exc,
                mesh_path=args.mesh,
                source_readback_path=args.source_readback[0] if len(args.source_readback) == 1 else None,
                source_readback_paths=args.source_readback,
                qem_backend=args.local_qem_backend,
            ),
        )
        return 1

    _write_json(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
