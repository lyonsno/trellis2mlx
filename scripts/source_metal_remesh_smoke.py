"""Cheap live conformance witness for the optional original Metal finalizer.

The solid cube is intentionally plain: this proves composition, not asset quality.
Run under Greenroom with the same backend manifest as the natural inputs.
"""
import argparse
import json
from pathlib import Path
import traceback
import uuid

from scripts.source_metal_remesh_finishing import digest, run, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backend-manifest", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    report = {"schema": "trellis2mlx.metal-remesh-composition.v1", "status": "failed",
              "failure_phase": "input_preparation", "last_trustworthy_phase": "request",
              "requested": {key: str(value) for key, value in vars(args).items()}}
    args.root.mkdir(parents=True, exist_ok=True)
    report_path = args.root / "smoke-report.json"
    if report_path.exists():
        report_path = args.root / f"smoke-report-{uuid.uuid4().hex}.json"
    write_report(report_path, report)
    try:
        result = prepare_and_run(args)
        report.update(status=result["status"], failure_phase=result["failure_phase"],
                      last_trustworthy_phase=result["last_trustworthy_phase"],
                      finishing_report=str(args.root / "finishing" / "report.json"))
    except Exception as exc:
        report.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    write_report(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path)}), flush=True)
    return 0 if report["status"] == "complete" else 1


def prepare_and_run(args):
    import numpy as np
    import trimesh
    inputs = args.root / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    cube = trimesh.creation.box(extents=[.6] * 3)
    grid_size = 32
    coords = np.indices((grid_size,) * 3).reshape(3, -1).T.astype(np.int32)
    attrs = np.column_stack([coords / grid_size, np.full(len(coords), .2),
                             np.full(len(coords), .7), np.ones(len(coords))]).astype(np.float32)
    np.savez_compressed(inputs / "mesh_raw.npz", vertices=cube.vertices.astype(np.float32),
                        faces=cube.faces.astype(np.int32))
    np.savez_compressed(inputs / "texture.npz", tex_np=attrs, tex_coords_spatial=coords,
                        mesh_grid_size=grid_size)
    request = argparse.Namespace(checkpoints=inputs, output_dir=args.root / "finishing",
                                 mesh_sha256=digest(inputs / "mesh_raw.npz"),
                                 texture_sha256=digest(inputs / "texture.npz"),
                                 backend_manifest=args.backend_manifest,
                                 expected_revision=args.expected_revision, grid_size=grid_size,
                                 target_faces=2000, texture_size=64, band=1., project_back=.9)
    return run(request)


if __name__ == "__main__":
    raise SystemExit(main())
