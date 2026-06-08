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
    cleanup_outputs = [FaceBag(1_000_000), FaceBag(150_000)]
    simplify_calls = []
    cleanup_calls = []

    def cleanup_mesh(v, faces, verbose=True):
        cleanup_calls.append((len(faces), verbose))
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
    assert cleanup_calls == [(1_000_000, True), (600_000, False)]


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
