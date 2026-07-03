"""Generate.py mesh postprocess sequencing contracts."""

import numpy as np
import pytest


class FaceBag:
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count


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
