"""Generate.py mesh postprocess sequencing contracts."""

import numpy as np
import pytest
from types import SimpleNamespace


class FaceBag:
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count


def test_export_material_is_double_sided_only_without_voxel_remesh():
    from generate import _export_material_double_sided

    assert _export_material_double_sided(SimpleNamespace(voxel_remesh_pitch=0.0)) is True
    assert _export_material_double_sided(SimpleNamespace(voxel_remesh_pitch=None)) is True
    assert _export_material_double_sided(SimpleNamespace(voxel_remesh_pitch=1.0 / 128.0)) is False


def test_unusable_resume_checkpoint_dir_refuses_full_pipeline_fallback():
    from generate import _raise_unusable_resume_checkpoint_dir

    with pytest.raises(ValueError, match="Refusing to run full pipeline fallback"):
        _raise_unusable_resume_checkpoint_dir(
            "/tmp/checkpoints",
            ["mesh_clean", "mesh_uv"],
            reason="no texture and mesh_raw checkpoint pair",
        )


def test_uv_visible_orientation_result_records_post_repair_provenance(monkeypatch):
    import trellmlx.mesh_cleanup
    from generate import _orient_uv_faces_for_export

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)

    def fake_visible_orient(export_vertices, input_faces, *, image_size, verbose):
        assert image_size == 17
        assert verbose is True
        oriented = input_faces.copy()
        oriented[1] = oriented[1][::-1]
        return export_vertices, oriented

    monkeypatch.setattr(
        trellmlx.mesh_cleanup,
        "orient_uv_islands_by_visible_exterior",
        fake_visible_orient,
    )

    result = _orient_uv_faces_for_export(
        vertices,
        faces,
        SimpleNamespace(no_uv_visible_orient=False, uv_visible_orient_size=17),
    )

    np.testing.assert_array_equal(result.faces, np.array([[0, 1, 2], [2, 3, 1]]))
    assert result.metadata == {
        "uv_face_orientation_provenance": "post_visible_exterior_orient",
        "uv_visible_orient_applied": 1,
        "uv_visible_orient_image_size": 17,
        "uv_visible_orient_input_faces": 2,
        "uv_visible_orient_changed_faces": 1,
    }


def test_uv_visible_orientation_result_records_skip_provenance():
    from generate import _orient_uv_faces_for_export

    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    result = _orient_uv_faces_for_export(
        vertices,
        faces,
        SimpleNamespace(no_uv_visible_orient=True, uv_visible_orient_size=17),
    )

    assert result.faces is faces
    assert result.metadata == {
        "uv_face_orientation_provenance": "raw_uv_unwrap_no_visible_orient",
        "uv_visible_orient_applied": 0,
        "uv_visible_orient_image_size": 17,
        "uv_visible_orient_input_faces": 1,
        "uv_visible_orient_changed_faces": 0,
    }


def test_postprocess_skips_final_simplify_if_cleanup_drops_below_target():
    from generate import _cleanup_and_simplify_mesh

    vertices = FaceBag(10)
    cleanup_outputs = [FaceBag(1_000_000), FaceBag(150_000), FaceBag(150_000)]
    simplify_calls = []
    cleanup_calls = []

    def cleanup_mesh(v, faces, keep_largest=False, do_fix_normals=True, verbose=True):
        cleanup_calls.append((len(faces), do_fix_normals, verbose))
        return v, cleanup_outputs.pop(0)

    def simplify(v, faces, target_reduction):
        simplify_calls.append(target_reduction)
        if target_reduction < 0:
            raise AssertionError(f"negative reduction {target_reduction}")
        return v, FaceBag(600_000)

    out_vertices, out_faces = _cleanup_and_simplify_mesh(
        vertices,
        FaceBag(1_000_000),
        target_faces=200_000,
        no_cleanup=False,
        cleanup_mesh=cleanup_mesh,
        simplify=simplify,
        log=lambda *args, **kwargs: None,
    )

    assert out_vertices is vertices
    assert len(out_faces) == 150_000
    assert simplify_calls == [pytest.approx(0.4)]
    assert cleanup_calls == [
        (1_000_000, False, True),
        (600_000, False, False),
        (150_000, True, False),
    ]


def test_postprocess_target_faces_zero_still_runs_final_normals_cleanup():
    from generate import _cleanup_and_simplify_mesh

    vertices = FaceBag(10)
    cleanup_outputs = [FaceBag(500_000), FaceBag(500_000)]
    cleanup_calls = []

    def cleanup_mesh(v, faces, keep_largest=False, do_fix_normals=True, verbose=True):
        cleanup_calls.append((len(faces), do_fix_normals, verbose))
        return v, cleanup_outputs.pop(0)

    out_vertices, out_faces = _cleanup_and_simplify_mesh(
        vertices,
        FaceBag(500_000),
        target_faces=0,
        no_cleanup=False,
        cleanup_mesh=cleanup_mesh,
        simplify=None,
        log=lambda *args, **kwargs: None,
    )

    assert out_vertices is vertices
    assert len(out_faces) == 500_000
    assert cleanup_calls == [
        (500_000, False, True),
        (500_000, True, False),
    ]


def test_postprocess_no_cleanup_still_simplifies_without_cleanup_import():
    from generate import _cleanup_and_simplify_mesh

    vertices = FaceBag(10)
    simplify_calls = []

    def cleanup_mesh(*args, **kwargs):
        raise AssertionError("cleanup should not run with no_cleanup=True")

    def simplify(v, faces, target_reduction):
        simplify_calls.append(target_reduction)
        if len(simplify_calls) == 1:
            return v, FaceBag(600_000)
        return v, FaceBag(200_000)

    out_vertices, out_faces = _cleanup_and_simplify_mesh(
        vertices,
        FaceBag(1_000_000),
        target_faces=200_000,
        no_cleanup=True,
        cleanup_mesh=cleanup_mesh,
        simplify=simplify,
        log=lambda *args, **kwargs: None,
    )

    assert out_vertices is vertices
    assert len(out_faces) == 200_000
    assert simplify_calls == [pytest.approx(0.4), pytest.approx(2 / 3)]


def test_postprocess_reference_cleanup_fills_before_staged_simplification():
    from generate import _cleanup_and_simplify_mesh

    vertices = FaceBag(10)
    cleanup_outputs = [FaceBag(500_000), FaceBag(190_000)]
    fill_calls = []
    cleanup_calls = []
    simplify_calls = []

    def fill_holes(v, faces, max_hole_perimeter=3e-2, verbose=True):
        fill_calls.append((len(faces), max_hole_perimeter, verbose))
        return v, faces

    def cleanup_mesh(v, faces, keep_largest=False, do_fix_normals=True, verbose=True):
        cleanup_calls.append((len(faces), do_fix_normals, verbose))
        return v, cleanup_outputs.pop(0)

    def simplify(v, faces, target_reduction):
        simplify_calls.append(target_reduction)
        if len(simplify_calls) == 1:
            return v, FaceBag(600_000)
        return v, FaceBag(200_000)

    out_vertices, out_faces = _cleanup_and_simplify_mesh(
        vertices,
        FaceBag(1_000_000),
        target_faces=200_000,
        no_cleanup=False,
        reference_cleanup=True,
        cleanup_mesh=cleanup_mesh,
        fill_holes=fill_holes,
        simplify=simplify,
        log=lambda *args, **kwargs: None,
    )

    assert out_vertices is vertices
    assert len(out_faces) == 190_000
    assert fill_calls == [(1_000_000, pytest.approx(3e-2), True)]
    assert simplify_calls == [pytest.approx(0.4), pytest.approx(0.6)]
    assert cleanup_calls == [
        (600_000, False, True),
        (200_000, True, False),
    ]


def test_postprocess_reference_cleanup_records_stage_boundaries():
    from generate import _cleanup_and_simplify_mesh

    vertices = FaceBag(10)
    cleanup_outputs = [FaceBag(500_000), FaceBag(190_000)]
    records = []

    def fill_holes(v, faces, max_hole_perimeter=3e-2, verbose=True):
        return v, FaceBag(1_010_000)

    def cleanup_mesh(v, faces, keep_largest=False, do_fix_normals=True, verbose=True):
        return v, cleanup_outputs.pop(0)

    def simplify(v, faces, target_reduction):
        if len(records) <= 1:
            return v, FaceBag(600_000)
        return v, FaceBag(200_000)

    def save_stage(stage, v, faces):
        records.append((stage, len(faces)))

    _cleanup_and_simplify_mesh(
        vertices,
        FaceBag(1_000_000),
        target_faces=200_000,
        no_cleanup=False,
        reference_cleanup=True,
        cleanup_mesh=cleanup_mesh,
        fill_holes=fill_holes,
        simplify=simplify,
        save_postprocess_stage=save_stage,
        log=lambda *args, **kwargs: None,
    )

    assert records == [
        ("mesh_after_initial_fill", 1_010_000),
        ("mesh_after_coarse_simplify", 600_000),
        ("mesh_after_cleanup_pass1", 500_000),
        ("mesh_after_final_simplify", 200_000),
        ("mesh_after_cleanup_final", 190_000),
    ]


def test_postprocess_stage_saver_writes_mesh_checkpoint(tmp_path):
    from generate import _make_postprocess_stage_saver, _mesh_coord_space
    from trellmlx.checkpoint import load_checkpoint

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    save_stage = _make_postprocess_stage_saver(str(tmp_path), mesh_grid_size=512)
    save_stage("mesh_after_cleanup_pass1", vertices, faces)

    data = load_checkpoint(str(tmp_path), "mesh_after_cleanup_pass1")
    np.testing.assert_array_equal(data["vertices"], vertices)
    np.testing.assert_array_equal(data["faces"], faces)
    assert data["mesh_grid_size"] == 512
    assert data["mesh_coord_space"] == _mesh_coord_space()
    assert data["postprocess_stage"] == "mesh_after_cleanup_pass1"


def test_voxel_remesh_runs_final_cleanup_when_cleanup_enabled():
    from generate import _apply_voxel_remesh_if_requested

    vertices = FaceBag(10)
    input_faces = FaceBag(100)
    remesh_faces = FaceBag(80)
    cleaned_faces = FaceBag(75)
    cleanup_calls = []

    def voxel_remesh(v, faces, *, pitch):
        assert v is vertices
        assert faces is input_faces
        assert pitch == pytest.approx(1.0 / 128.0)
        return v, remesh_faces

    def cleanup_mesh(v, faces, keep_largest=False, do_fix_normals=True, verbose=True):
        cleanup_calls.append((faces, keep_largest, do_fix_normals, verbose))
        return v, cleaned_faces

    out_vertices, out_faces = _apply_voxel_remesh_if_requested(
        vertices,
        input_faces,
        pitch=1.0 / 128.0,
        no_cleanup=False,
        keep_largest=True,
        cleanup_mesh=cleanup_mesh,
        voxel_remesh=voxel_remesh,
        log=lambda *args, **kwargs: None,
    )

    assert out_vertices is vertices
    assert out_faces is cleaned_faces
    assert cleanup_calls == [(remesh_faces, True, True, False)]


def test_mesh_checkpoint_validator_rejects_wrong_coord_space_metadata():
    from generate import _validate_mesh_checkpoint_vertices

    vertices = np.zeros((1, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="unsupported mesh_coord_space"):
        _validate_mesh_checkpoint_vertices(
            vertices,
            mesh_grid_size=512,
            coord_space="voxel_index_space",
            stage="mesh_raw",
        )


def test_mesh_checkpoint_validator_rejects_legacy_out_of_domain_vertices():
    from generate import _validate_mesh_checkpoint_vertices

    vertices = np.array([[121.5, 0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="outside TRELLIS world coordinate domain"):
        _validate_mesh_checkpoint_vertices(
            vertices,
            mesh_grid_size=512,
            coord_space=None,
            stage="mesh_raw",
        )
