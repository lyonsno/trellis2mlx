"""Focused contracts for the Trellis Mac fallback postprocess witness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh


SCRIPT = Path("scripts/trellis_mac_fallback_postprocess_witness.py")


def load_witness_module():
    spec = importlib.util.spec_from_file_location("trellis_mac_fallback_witness", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_oriented_edge_summary_distinguishes_consistent_and_flipped_faces():
    witness = load_witness_module()
    consistent = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)
    flipped = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)

    consistent_summary = witness.oriented_edge_summary(consistent)
    flipped_summary = witness.oriented_edge_summary(flipped)

    assert consistent_summary["inconsistently_oriented_shared_edges"] == 0
    assert flipped_summary["inconsistently_oriented_shared_edges"] == 1


def test_face_normal_comparison_detects_per_face_reversal():
    witness = load_witness_module()
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    forward = np.array([[0, 1, 2]], dtype=np.int32)
    reverse = np.array([[0, 2, 1]], dtype=np.int32)

    report = witness.face_normal_comparison(vertices, forward, vertices, reverse)

    assert report["opposite_direction_faces"] == 1
    assert report["same_direction_faces"] == 0


def test_load_mesh_reads_pipeline_checkpoint(tmp_path):
    witness = load_witness_module()
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    checkpoint = tmp_path / "mesh_raw.npz"
    np.savez_compressed(checkpoint, vertices=vertices, faces=faces)

    loaded_vertices, loaded_faces = witness.load_mesh(checkpoint)

    assert loaded_vertices.dtype == np.float32
    assert loaded_faces.dtype == np.int32
    np.testing.assert_array_equal(loaded_vertices, vertices.astype(np.float32))
    np.testing.assert_array_equal(loaded_faces, faces.astype(np.int32))


def test_load_mesh_rejects_checkpoint_without_faces(tmp_path):
    witness = load_witness_module()
    checkpoint = tmp_path / "mesh_raw.npz"
    np.savez_compressed(checkpoint, vertices=np.zeros((1, 3), dtype=np.float32))

    with np.testing.assert_raises_regex(ValueError, "missing.*faces"):
        witness.load_mesh(checkpoint)


def test_undo_glb_axis_transform_recovers_pipeline_vertices():
    witness = load_witness_module()
    pipeline = np.array(
        [[1.0, 2.0, 3.0], [-4.0, -5.0, -6.0]], dtype=np.float32
    )
    exported = pipeline[:, [0, 2, 1]].copy()
    exported[:, 2] *= -1

    restored = witness.undo_glb_axis_transform(exported)

    np.testing.assert_array_equal(restored, pipeline)


def test_checkpoint_hourglass_exports_all_mesh_waists(tmp_path):
    witness = load_witness_module()
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    raw = tmp_path / "mesh_raw.npz"
    clean = tmp_path / "mesh_clean.npz"
    uv = tmp_path / "mesh_uv.npz"
    final = tmp_path / "output.glb"
    np.savez_compressed(raw, vertices=vertices, faces=faces)
    np.savez_compressed(clean, vertices=vertices, faces=faces)
    np.savez_compressed(uv, vertices=vertices, faces=faces)
    exported = vertices[:, [0, 2, 1]].copy()
    exported[:, 2] *= -1
    trimesh.Trimesh(vertices=exported, faces=faces, process=False).export(final)

    report = witness.run_checkpoint_hourglass(
        raw_checkpoint=raw,
        clean_checkpoint=clean,
        uv_checkpoint=uv,
        final_glb=final,
        output_dir=tmp_path / "hourglass",
    )

    assert report["status"] == "done"
    assert set(report["stages"]) == {"raw", "clean", "uv", "final"}
    assert report["comparisons"]["uv_vs_final_direct"]["faces_exact"]
    assert report["comparisons"]["uv_vs_final_direct"][
        "vertices_allclose_after_axis_restore"
    ]
    for stage in report["stages"].values():
        assert Path(stage["artifact"]).is_file()


def test_product_route_selects_reference_shaped_fast_branch(tmp_path):
    witness = load_witness_module()
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    source = tmp_path / "mesh.npz"
    np.savez_compressed(source, vertices=vertices, faces=faces)
    observed = {}

    def fake_postprocess(in_vertices, in_faces, **kwargs):
        observed.update(kwargs)
        return in_vertices, in_faces

    report = witness.run_product_route(
        input_mesh=source,
        output_dir=tmp_path / "out",
        target_faces=100,
        route="reference-fast",
        postprocess=fake_postprocess,
    )

    assert report["status"] == "done"
    assert observed["reference_cleanup"] is True
    assert observed["simplify_first"] is False
    assert observed["qem_simplify"] is False
    assert Path(report["output"]["artifact"]).is_file()


def test_component_outward_sign_flips_inward_tetrahedron_only():
    witness = load_witness_module()
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    outward = np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32
    )
    inward = outward[:, ::-1]

    repaired_outward, outward_records = witness.orient_components_outward(
        vertices, outward
    )
    repaired_inward, inward_records = witness.orient_components_outward(
        vertices, inward
    )

    np.testing.assert_array_equal(repaired_outward, outward)
    np.testing.assert_array_equal(repaired_inward, outward)
    assert outward_records[0]["flipped"] is False
    assert inward_records[0]["flipped"] is True


def test_reference_stage_sign_ladder_captures_each_operation(tmp_path):
    witness = load_witness_module()
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32
    )
    source = tmp_path / "mesh.npz"
    np.savez_compressed(source, vertices=vertices, faces=faces)

    def passthrough(stage_vertices, stage_faces, *args, **kwargs):
        return stage_vertices, stage_faces

    def fake_postprocess(stage_vertices, stage_faces, **kwargs):
        stage_vertices, stage_faces = kwargs["simplify"](
            stage_vertices, stage_faces
        )
        stage_vertices, stage_faces = kwargs["cleanup_mesh"](
            stage_vertices, stage_faces
        )
        stage_vertices, stage_faces = kwargs["simplify"](
            stage_vertices, stage_faces
        )
        stage_vertices, stage_faces = kwargs["cleanup_mesh"](
            stage_vertices, stage_faces
        )
        return kwargs["orient_faces_by_adjacency"](
            stage_vertices, stage_faces
        )

    report = witness.run_reference_stage_sign_ladder(
        input_mesh=source,
        output_dir=tmp_path / "ladder",
        target_faces=4,
        postprocess=fake_postprocess,
        simplify=passthrough,
        cleanup=passthrough,
        orient=passthrough,
    )

    assert report["status"] == "done"
    assert list(report["stages"]) == [
        "01-coarse-simplify",
        "02-initial-cleanup",
        "03-final-simplify",
        "04-final-cleanup",
        "05-adjacency-orientation",
    ]
    assert report["final_matches_last_capture"] == {
        "vertices_exact": True,
        "faces_exact": True,
    }
    assert all(stage["inward_face_count"] == 0 for stage in report["stages"].values())
    assert all(Path(stage["artifact"]).is_file() for stage in report["stages"].values())


def test_radial_flux_summary_distinguishes_tetrahedron_sign():
    witness = load_witness_module()
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    outward = np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32
    )

    outward_report = witness.radial_flux_summary(vertices, outward, chunk_size=2)
    inward_report = witness.radial_flux_summary(
        vertices, outward[:, ::-1], chunk_size=2
    )

    assert outward_report["predominant_sign"] == "outward"
    assert inward_report["predominant_sign"] == "inward"
    assert outward_report["radial_score"] == -inward_report["radial_score"]


def test_trimesh_fix_normals_witness_changes_only_face_sign(tmp_path):
    witness = load_witness_module()
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    outward = np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32
    )
    source = tmp_path / "inward.npz"
    np.savez_compressed(source, vertices=vertices, faces=outward[:, ::-1])

    report = witness.run_trimesh_fix_normals_witness(
        input_mesh=source,
        output_dir=tmp_path / "out",
    )

    assert report["status"] == "done"
    assert report["vertices_exact"]
    assert report["face_rows_reversed"] == len(outward)
    assert report["face_rows_other"] == 0
    assert report["output"]["inward_face_count"] == 0
    assert Path(report["output"]["artifact"]).is_file()


def test_trimesh_fix_normals_preserve_visuals_witness_keeps_uv_and_texture(tmp_path):
    witness = load_witness_module()
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    outward = np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32
    )
    uv = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    texture = Image.fromarray(
        np.array(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [255, 255, 255]],
            ],
            dtype=np.uint8,
        ),
        mode="RGB",
    )
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture,
        metallicFactor=0.0,
        roughnessFactor=0.8,
        doubleSided=False,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=outward[:, ::-1],
        visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
        process=False,
    )
    source = tmp_path / "textured-inward.glb"
    source.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))

    report = witness.run_trimesh_fix_normals_preserve_visuals_witness(
        input_mesh=source,
        output_dir=tmp_path / "out",
    )

    assert report["status"] == "done"
    assert report["vertices_exact"]
    assert report["face_rows_reversed"] == len(outward)
    assert report["face_rows_other"] == 0
    assert report["output"]["inward_face_count"] == 0
    assert report["visual_payload_exact"]
    assert report["input"]["visual"]["visual_kind"] == "texture"
    assert report["input"]["visual"]["texture_digests"]
    assert Path(report["output"]["artifact"]).is_file()
