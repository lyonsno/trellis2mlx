"""Cheap live conformance witness for the optional original Metal finalizer.

The solid cube is intentionally plain: this proves composition, not asset quality.
Run under Greenroom with the same backend manifest as the natural inputs.
"""
import argparse
from pathlib import Path

import numpy as np
import trimesh

from scripts.source_metal_remesh_finishing import digest, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backend-manifest", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
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
    return 0 if run(request)["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
