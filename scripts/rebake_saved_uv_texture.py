#!/usr/bin/env python3
"""Compare direct/projected texture lookup on one saved UV mesh and appearance.

No inference, cleanup, simplification, UV unwrap, or normal recalculation.
The input GLB is a geometry/material template; only its two texture images change.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import struct
import sys
import time

import numpy as np
from PIL import Image
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trellmlx.checkpoint import load_checkpoint
from trellmlx.texture_bake import bake_texture


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_checkpoint(checkpoint_dir, stage, report):
    """Accept legacy checkpoints, but verify completion hashes when provided."""
    files = [checkpoint_dir / (stage + suffix) for suffix in (".npz", ".json")]
    if (checkpoint_dir / (stage + ".pending")).exists() or not all(p.is_file() for p in files):
        raise ValueError("invalid or missing checkpoint: " + stage)
    complete = checkpoint_dir / (stage + ".complete.json")
    if complete.exists():
        manifest = json.loads(complete.read_text())
        if manifest.get("schema") != 1 or any(
                manifest.get("files", {}).get(p.name) != sha(p) for p in files):
            raise ValueError("invalid or missing checkpoint completion hashes: " + stage)
        files.append(complete)
    report.setdefault("checkpoint_completion_verified", {})[stage] = complete.exists()
    report["inputs_sha256"].update({str(p): sha(p) for p in files})
    return load_checkpoint(str(checkpoint_dir), stage)


def replace_texture_images(template_path, color, mr):
    """Keep the original GLB buffers/scene intact; replace only embedded images."""
    source = template_path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", source)
    json_size, json_type = struct.unpack_from("<II", source, 12)
    bin_offset = 20 + json_size
    bin_size, bin_type = struct.unpack_from("<II", source, bin_offset)
    if (magic != b"glTF" or version != 2 or total != len(source)
            or json_type != 0x4E4F534A or bin_type != 0x004E4942
            or bin_offset + 8 + bin_size != total):
        raise ValueError("template must be a GLB with one JSON and one BIN chunk")
    document = json.loads(source[20:bin_offset])
    if len(document["buffers"]) != 1 or len(document["materials"]) != 1:
        raise ValueError("template must have one buffer and one material")
    binary = bytearray(source[bin_offset + 8:])
    pbr = document["materials"][0]["pbrMetallicRoughness"]
    indices = [document["textures"][pbr[key]["index"]]["source"]
               for key in ("baseColorTexture", "metallicRoughnessTexture")]
    if indices[0] == indices[1]:
        raise ValueError("template color and metallic/roughness images must be distinct")
    for index, pixels in zip(indices, (color, mr)):
        encoded = io.BytesIO()
        Image.fromarray(pixels).save(encoded, format="PNG")
        data = encoded.getvalue()
        binary.extend(b"\0" * (-len(binary) % 4))
        document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
        document["images"][index] = {"bufferView": len(document["bufferViews"]) - 1, "mimeType": "image/png"}
        binary.extend(data)
    document["buffers"][0]["byteLength"] = len(binary)
    binary.extend(b"\0" * (-len(binary) % 4))
    header = json.dumps(document, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 4)
    return (struct.pack("<4sII", b"glTF", 2, 28 + len(header) + len(binary))
            + struct.pack("<II", len(header), json_type) + header
            + struct.pack("<II", len(binary), bin_type) + binary)


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if report_path.exists() or any((output / name).exists() for name in ("control.glb", "projected.glb")):
        raise ValueError("use a new output directory; existing results are preserved")
    report = {"status": "running", "phase": "input_load", "started_at": time.time(),
              "backend": args.backend, "texture_size": args.texture_size,
              "projection_backend": "libigl CPU AABB", "inference_rerun": False}
    try:
        checkpoint_dir = args.checkpoint_dir.resolve()
        repo = Path(__file__).resolve().parents[1]
        report["repo_root"] = str(repo)
        report["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        report["source_sha256"] = {name: sha(repo / name) for name in
            ("scripts/rebake_saved_uv_texture.py", "trellmlx/texture_bake.py")}
        report["inputs_sha256"] = {str(args.template_glb.resolve()): sha(args.template_glb)}
        uv = checked_checkpoint(checkpoint_dir, "mesh_uv", report)
        raw = checked_checkpoint(checkpoint_dir, "mesh_raw", report)
        texture = checked_checkpoint(checkpoint_dir, "texture", report)
        if not (uv["mesh_grid_size"] == raw["mesh_grid_size"] == texture["mesh_grid_size"]):
            raise ValueError("checkpoint coordinate grid sizes differ")
        scene = trimesh.load(args.template_glb, force="scene", process=False)
        if len(scene.geometry) != 1:
            raise ValueError("template must contain exactly one mesh")
        template = next(iter(scene.geometry.values()))
        export_vertices = uv["vertices"][:, [0, 2, 1]].copy()
        export_vertices[:, 2] *= -1
        export_uvs = uv["uvs"].copy()
        export_uvs[:, 1] = 1 - export_uvs[:, 1]
        if (template.vertices.shape != export_vertices.shape
                or template.faces.shape != uv["faces"].shape
                or not np.array_equal(template.faces, uv["faces"])
                or not np.allclose(template.vertices, export_vertices, atol=1e-7, rtol=0)
                or not np.allclose(template.visual.uv, export_uvs, atol=1e-7, rtol=0)):
            raise ValueError("template GLB does not match the saved UV mesh")
        images = {}
        report["outputs"] = {}
        for mode in ("control", "projected"):
            report["phase"] = mode + "_bake"
            report_path.write_text(json.dumps(report, indent=2))
            kwargs = {} if mode == "control" else {
                "original_vertices": raw["vertices"], "original_faces": raw["faces"]}
            color, mr, _ = bake_texture(
                uv["vertices"], uv["faces"], uv["uvs"], uv["vmapping"],
                texture["tex_coords_spatial"], texture["tex_np"], texture["mesh_grid_size"],
                texture_size=args.texture_size, backend=args.backend, **kwargs)
            images[mode] = color
            path = output / (mode + ".glb")
            path.write_bytes(replace_texture_images(args.template_glb, color, mr))
            color_path = output / (mode + "-basecolor.png")
            Image.fromarray(color).save(color_path)
            decoded = next(iter(trimesh.load(path, force="scene", process=False).geometry.values()))
            for field in ("vertices", "faces", "vertex_normals"):
                if not np.array_equal(getattr(decoded, field), getattr(template, field)):
                    raise ValueError("rebake changed template " + field)
            if not np.array_equal(decoded.visual.uv, template.visual.uv):
                raise ValueError("rebake changed template UVs")
            report["outputs"][mode] = {"glb": str(path), "sha256": sha(path),
                "basecolor": str(color_path), "vertices": len(decoded.vertices),
                "faces": len(decoded.faces), "geometry_normals_uv_exact": True}
        delta = np.abs(images["projected"].astype(float) - images["control"].astype(float))
        report["basecolor_mean_absolute_byte_delta"] = delta[:, :, :3].mean(axis=(0, 1)).tolist()
        report["changed_basecolor_pixels"] = int(np.any(delta[:, :, :3] != 0, axis=2).sum())
        if any(sha(Path(path)) != digest for path, digest in report["inputs_sha256"].items()):
            raise ValueError("saved inputs changed during rebake")
        report.update(status="done", phase="complete", finished_at=time.time())
    except Exception as error:
        report.update(status="failed", error=repr(error), finished_at=time.time())
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--template-glb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    run(parser.parse_args())
