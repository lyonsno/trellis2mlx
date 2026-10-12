"""Finish saved MLX raw mesh + appearance with the original Metal remesh branch.

No inference, local cleanup, backend fallback or production-default change.
Use an existing optional Torch/Metal interpreter, with this repo on PYTHONPATH.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np

from trellmlx.source_metal_remesh import load_backend, to_glb


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_inputs(checkpoints, mesh_sha, texture_sha, grid_size):
    mesh_path, texture_path = checkpoints / "mesh_raw.npz", checkpoints / "texture.npz"
    for path, expected in ((mesh_path, mesh_sha), (texture_path, texture_sha)):
        if digest(path) != expected:
            raise ValueError(f"input digest mismatch: {path}")
    with np.load(mesh_path, allow_pickle=False) as data:
        vertices, faces = data["vertices"], data["faces"]
    with np.load(texture_path, allow_pickle=False) as data:
        attrs, coords = data["tex_np"], data["tex_coords_spatial"]
        # Historical checkpoints predate embedded grid metadata. Their explicit
        # grid remains caller-owned provenance, bound with the input hashes.
        if "mesh_grid_size" in data and int(data["mesh_grid_size"].item()) != grid_size:
            raise ValueError("appearance grid does not match saved mesh_grid_size")
    for name, value, width in (("vertices", vertices, 3), ("faces", faces, 3),
                               ("attrs", attrs, 6), ("coords", coords, 3)):
        if value.ndim != 2 or value.shape[1] != width or not len(value) or not np.isfinite(value).all():
            raise ValueError(f"invalid {name} shape/values")
    if faces.dtype.kind not in "iu" or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("invalid face indices")
    if coords.dtype.kind not in "iu" or len(coords) != len(attrs) or coords.min() < 0 or coords.max() >= grid_size:
        raise ValueError("invalid sparse appearance coordinates")
    return vertices, faces, attrs, coords


def verify_backend(backend, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["modules"]
    if set(expected) != set(backend.identity):
        raise ValueError("backend module set mismatch")
    observed = {}
    for name, path in backend.identity.items():
        record = {"path": str(Path(path).resolve()), "sha256": digest(path)}
        if record != expected[name]:
            raise ValueError(f"backend source identity mismatch: {name}")
        observed[name] = record
    binaries = manifest["binaries"]
    required = set(backend.native_libraries) | set(backend.declared_metallibs)
    if {record["path"] for record in binaries} != required or len(binaries) != len(required):
        raise ValueError("backend native library set mismatch")
    for record in binaries:
        if digest(record["path"]) != record["sha256"]:
            raise ValueError(f"backend library identity mismatch: {record['path']}")
    return {"modules": observed,
            "imported_native_libraries": [r for r in binaries if r["path"] in backend.native_libraries],
            "declared_metallibs": [r for r in binaries if r["path"] in backend.declared_metallibs],
            "metallib_authority": "adjacent-file provenance; runtime loader object not authenticated",
            "host_tensors": "torch-cpu",
            "accelerator": "Metal custom kernels", "python": sys.executable}


def validate_composition(path, revision, backend_manifest):
    """Gate queued natural runs on the same-revision live primitive witness."""
    witness = json.loads(path.read_text())
    if (witness.get("schema") != "trellis2mlx.original-metal-remesh.v1"
            or witness.get("status") != "complete"
            or witness.get("source", {}).get("revision") != revision
            or witness.get("backend_manifest_sha256") != digest(backend_manifest)):
        raise ValueError("missing or mismatched complete composition witness")
    effective = witness.get("effective", {})
    expected = {"accelerator": "Metal custom kernels", "host_tensors": "torch-cpu",
                "grid_size": 32, "target_faces": 2000, "texture_size": 64,
                "band": 1., "project_back": .9, "alpha_mode": "OPAQUE", "double_sided": False}
    if any(effective.get(key) != value for key, value in expected.items()):
        raise ValueError("composition effective configuration mismatch")
    if [s["name"] for s in witness.get("stages", [])] != ["raw", "filled_original", "rebuilt", "reduced", "uv", "normals"]:
        raise ValueError("composition stages incomplete")
    output = witness["output"]
    if digest(output["path"]) != output["sha256"]:
        raise ValueError("composition asset digest mismatch")
    return {"report": str(path), "sha256": digest(path), "output_sha256": output["sha256"]}


def run(args, *, backend_loader=load_backend, finalizer=to_glb):
    started = time.perf_counter()
    report = {"schema": "trellis2mlx.original-metal-remesh.v1", "status": "failed",
              "failure_phase": None, "last_trustworthy_phase": "request",
              "requested": vars(args).copy(), "effective": None, "stages": [],
              "scope": "saved MLX prediction; local Torch-facing Metal finishing; no CUDA parity claim"}
    report["requested"] = {k: str(v) if isinstance(v, Path) else v for k, v in report["requested"].items()}
    phase, owns_output = "output_admission", False
    report_path = args.output_dir / "report.json"
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        owns_output = True
        write_report(report_path, report)
        phase = "source_identity"
        source_root = Path(__file__).resolve().parents[1]
        revision = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(source_root), "status", "--porcelain"], text=True).strip()
        if revision != args.expected_revision or dirty:
            raise ValueError("finishing worktree must be clean at the expected revision")
        report["source"] = {"root": str(source_root), "revision": revision}
        if getattr(args, "require_composition_report", None):
            phase = "composition_admission"
            report["composition"] = validate_composition(
                args.require_composition_report, revision, args.backend_manifest)
            report["last_trustworthy_phase"] = phase
        phase = "input_validation"
        arrays = load_inputs(args.checkpoints, args.mesh_sha256, args.texture_sha256, args.grid_size)
        report["last_trustworthy_phase"] = phase
        phase = "backend_identity"
        backend = backend_loader()
        report["effective"] = verify_backend(backend, args.backend_manifest)
        report["effective"].update({"grid_size": args.grid_size, "target_faces": args.target_faces,
                                     "texture_size": args.texture_size, "band": args.band,
                                     "project_back": args.project_back, "alpha_mode": "OPAQUE",
                                     "double_sided": False})
        report["backend_manifest_sha256"] = digest(args.backend_manifest)
        with np.load(args.checkpoints / "texture.npz", allow_pickle=False) as checkpoint:
            report["effective"]["grid_identity_source"] = (
                "embedded mesh_grid_size" if "mesh_grid_size" in checkpoint else "caller declaration; historical checkpoint")
        report["last_trustworthy_phase"] = phase
        write_report(report_path, report)

        def set_phase(value):
            nonlocal phase
            phase = value
            report["active_phase"] = value
            report["failure_phase"] = value
            write_report(report_path, report)
            print(f"phase: {value}", flush=True)

        def capture(name, tensors):
            set_phase(name)
            path = args.output_dir / f"{len(report['stages']):02d}-{name}.npz"
            payload = {k: v.detach().cpu().numpy() for k, v in tensors.items()}
            np.savez_compressed(path, **payload)
            record = {"name": name, "path": str(path), "sha256": digest(path),
                      "shapes": {k: list(v.shape) for k, v in payload.items()},
                      "elapsed_seconds": time.perf_counter() - started}
            report["stages"].append(record)
            report["last_trustworthy_phase"] = name
            write_report(report_path, report)
            print(json.dumps(record), flush=True)

        vertices, faces, attrs, coords = [backend.torch.from_numpy(v) for v in arrays]
        with backend.torch.no_grad():
            mesh = finalizer(vertices, faces, attrs, coords, backend=backend,
                             grid_size=args.grid_size, target_faces=args.target_faces,
                             texture_size=args.texture_size, band=args.band,
                             project_back=args.project_back, capture=capture,
                             phase=set_phase, verbose=True)
        set_phase("export")
        output = args.output_dir / "output.glb"
        mesh.export(output)
        # Independent reopen checks the consumer artifact, not just the producer.
        import trimesh
        reopened = trimesh.load(output, force="scene", process=False)
        geometries = list(reopened.geometry.values())
        if not geometries or any(not len(g.faces) or not np.isfinite(g.vertices).all() for g in geometries):
            raise ValueError("empty or nonfinite exported geometry")
        for g in geometries:
            material = g.visual.material
            if material.alphaMode != "OPAQUE" or material.doubleSided:
                raise ValueError("exported material policy mismatch")
            for image in (material.baseColorTexture, material.metallicRoughnessTexture):
                if image is None or image.size != (args.texture_size, args.texture_size):
                    raise ValueError("exported texture resolution mismatch")
        report["output"] = {"path": str(output), "sha256": digest(output), "size_bytes": output.stat().st_size,
                            "vertices": sum(len(g.vertices) for g in geometries),
                            "faces": sum(len(g.faces) for g in geometries)}
        report.update(status="complete", last_trustworthy_phase="export", active_phase=None, failure_phase=None)
    except Exception as exc:
        report.update(failure_phase=phase, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        if not owns_output:
            # Never overwrite an earlier attempt or the source to report refusal.
            report_path = args.output_dir.parent / f"{args.output_dir.name}-failure-{uuid.uuid4().hex}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    write_report(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh-sha256", required=True)
    parser.add_argument("--texture-sha256", required=True)
    parser.add_argument("--backend-manifest", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--require-composition-report", type=Path,
                        help="Require the same-revision cube32/2000/64 live witness before a queued natural run")
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--target-faces", type=int, default=1000000)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--band", type=float, default=1.)
    parser.add_argument("--project-back", type=float, default=.9)
    args = parser.parse_args()
    return 0 if run(args)["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
