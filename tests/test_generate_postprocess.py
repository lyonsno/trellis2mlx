"""Generate.py mesh postprocess sequencing contracts."""

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


def test_postprocess_reference_cleanup_final_cleanup_matches_source_orientation():
    from generate import _cleanup_and_simplify_mesh

    vertices = FaceBag(10)
    cleanup_outputs = [FaceBag(500_000), FaceBag(190_000)]
    cleanup_calls = []
    simplify_calls = []
    orient_calls = []

    def fill_holes(v, faces, max_hole_perimeter=3e-2, verbose=True):
        return v, FaceBag(1_000_000)

    def cleanup_mesh(v, faces, keep_largest=False, do_fix_normals=True, verbose=True):
        cleanup_calls.append((len(faces), do_fix_normals, verbose))
        return v, cleanup_outputs.pop(0)

    def simplify(v, faces, target_reduction=None, target_count=None):
        simplify_calls.append((target_reduction, target_count))
        if target_count == 600_000:
            return v, FaceBag(600_000)
        if target_count == 200_000:
            return v, FaceBag(200_000)
        raise AssertionError(f"unexpected simplify target {target_count}")

    def orient_faces_by_adjacency(v, faces, verbose=True):
        orient_calls.append((len(faces), verbose))
        return v, faces

    out_vertices, out_faces = _cleanup_and_simplify_mesh(
        vertices,
        FaceBag(1_000_000),
        target_faces=200_000,
        no_cleanup=False,
        reference_cleanup=True,
        cleanup_mesh=cleanup_mesh,
        fill_holes=fill_holes,
        simplify=simplify,
        orient_faces_by_adjacency=orient_faces_by_adjacency,
        log=lambda *args, **kwargs: None,
    )

    assert out_vertices is vertices
    assert len(out_faces) == 190_000
    assert simplify_calls == [(None, 600_000), (None, 200_000)]
    assert cleanup_calls == [
        (600_000, False, True),
        (200_000, False, False),
    ]
    assert orient_calls == [(190_000, False)]
