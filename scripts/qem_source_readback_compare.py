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
    _jsonable,
    build_qem_source_readback_report,
    failure_report,
    load_mesh_npz,
    load_source_readback_npz,
)


REQUESTED_ROUTE = "qem-source-readback-compare"
EFFECTIVE_ROUTE = "local-qem-step-vs-source-readback"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, help="NPZ containing vertices and faces arrays")
    parser.add_argument(
        "--source-readback",
        type=Path,
        required=True,
        help="NPZ containing source edges, costs, props, and optional qems arrays",
    )
    parser.add_argument("--lambda-edge-length", type=float, default=1e-2)
    parser.add_argument("--lambda-skinny", type=float, default=1e-3)
    parser.add_argument("--collapse-thresh", type=float, default=1e-8)
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
            failure_report(
                requested_route=REQUESTED_ROUTE,
                effective_route=EFFECTIVE_ROUTE,
                failure_phase="mesh",
                error=exc,
                mesh_path=args.mesh,
                source_readback_path=args.source_readback,
            ),
        )
        return 1

    try:
        source = load_source_readback_npz(args.source_readback)
    except Exception as exc:
        _write_json(
            args.report,
            failure_report(
                requested_route=REQUESTED_ROUTE,
                effective_route=EFFECTIVE_ROUTE,
                failure_phase="source_readback",
                error=exc,
                mesh_path=args.mesh,
                source_readback_path=args.source_readback,
            ),
        )
        return 1

    try:
        report = build_qem_source_readback_report(
            requested_route=REQUESTED_ROUTE,
            effective_route=EFFECTIVE_ROUTE,
            mesh_path=args.mesh,
            source_readback_path=args.source_readback,
            vertices=vertices,
            faces=faces,
            source=source,
            lambda_edge_length=args.lambda_edge_length,
            lambda_skinny=args.lambda_skinny,
            collapse_thresh=args.collapse_thresh,
        )
    except Exception as exc:
        _write_json(
            args.report,
            failure_report(
                requested_route=REQUESTED_ROUTE,
                effective_route=EFFECTIVE_ROUTE,
                failure_phase="compare",
                error=exc,
                mesh_path=args.mesh,
                source_readback_path=args.source_readback,
            ),
        )
        return 1

    _write_json(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
