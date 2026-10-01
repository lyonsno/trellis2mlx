"""A fixed-geometry comparison cannot silently accept another mesh."""
from argparse import Namespace
import json

import numpy as np
import pytest
from PIL import Image
import trimesh

from scripts.rebake_saved_uv_texture import run
from trellmlx.checkpoint import save_checkpoint


def inputs(tmp_path):
    checkpoint = tmp_path / "checkpoints"
    vertices = np.array([[0., 0., 0.2], [0.1, 0., 0.2], [0., 0.1, 0.2]])
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    uvs = np.array([[0., 0.], [1., 0.], [0., 1.]], dtype=np.float32)
    save_checkpoint(str(checkpoint), "mesh_uv", vertices=vertices, faces=faces,
                    uvs=uvs, vmapping=np.arange(3), mesh_grid_size=4)
    save_checkpoint(str(checkpoint), "mesh_raw", vertices=vertices - [0., 0., 0.2],
                    faces=faces, mesh_grid_size=4)
    coords = np.indices((4, 4, 4)).reshape(3, -1).T.astype(np.int32)
    attrs = np.tile([0.2, 0.4, 0.6, 0.1, 0.7, 1.], (len(coords), 1))
    save_checkpoint(str(checkpoint), "texture", tex_np=attrs,
                    tex_coords_spatial=coords, mesh_grid_size=4)
    exported = vertices[:, [0, 2, 1]].copy()
    exported[:, 2] *= -1
    export_uvs = uvs.copy()
    export_uvs[:, 1] = 1 - export_uvs[:, 1]
    template = tmp_path / "template.glb"
    mesh = trimesh.Trimesh(vertices=exported, faces=faces, process=False,
        visual=trimesh.visual.TextureVisuals(uv=export_uvs,
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.new("RGB", (8, 8)),
                metallicRoughnessTexture=Image.new("RGB", (8, 8)))))
    # Authored normals deliberately differ from normals recalculated from faces.
    mesh.vertex_normals = np.tile([1., 0., 0.], (3, 1))
    mesh.export(template, include_normals=True)
    return Namespace(checkpoint_dir=checkpoint, template_glb=template,
                     output_dir=tmp_path / "result", backend="cpu", texture_size=8)


def test_pair_preserves_geometry_and_records_effective_route(tmp_path):
    args = inputs(tmp_path)
    report = run(args)
    assert report["status"] == "done"
    assert report["backend"] == "cpu"
    assert report["inference_rerun"] is False
    assert all(item["geometry_normals_uv_exact"] for item in report["outputs"].values())
    assert len(report["inputs_sha256"]) == 7
    assert not any(report["checkpoint_completion_verified"].values())


def test_wrong_template_fails_before_bake_with_durable_report(tmp_path):
    args = inputs(tmp_path)
    scene = trimesh.load(args.template_glb, force="scene", process=False)
    next(iter(scene.geometry.values())).vertices += [0., 1., 0.]
    scene.export(args.template_glb)
    with pytest.raises(ValueError, match="does not match"):
        run(args)
    report = json.loads((args.output_dir / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["phase"] == "input_load"
    assert not (args.output_dir / "control.glb").exists()


def test_incomplete_checkpoint_fails_with_durable_report(tmp_path):
    args = inputs(tmp_path)
    (args.checkpoint_dir / "texture.pending").touch()
    with pytest.raises(ValueError, match="invalid or missing"):
        run(args)
    report = json.loads((args.output_dir / "report.json").read_text())
    assert report["status"] == "failed"
    assert not (args.output_dir / "projected.glb").exists()
