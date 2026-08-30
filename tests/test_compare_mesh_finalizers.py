from pathlib import Path

import numpy as np
import trimesh

from scripts.compare_mesh_finalizers import build_report, load_mesh


def _sheet() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return vertices, faces


def test_load_mesh_preserves_npz_arrays(tmp_path: Path):
    vertices, faces = _sheet()
    path = tmp_path / "mesh.npz"
    np.savez(path, vertices=vertices, faces=faces)
    loaded_vertices, loaded_faces = load_mesh(path)
    assert np.array_equal(loaded_vertices, vertices)
    assert np.array_equal(loaded_faces, faces)


def test_build_report_compares_npz_and_ply_without_claiming_lineage(tmp_path: Path):
    vertices, faces = _sheet()
    before = tmp_path / "before.npz"
    after = tmp_path / "after.ply"
    np.savez(before, vertices=vertices, faces=faces)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(after)

    report = build_report(
        before,
        after,
        before_label="old",
        after_label="new",
        fixed_raw_sha256="abc123",
        max_faces_per_side=100,
    )

    assert report["claim_scope"]["fixed_raw_specimen_sha256"] == "abc123"
    assert report["claim_scope"]["surface_comparison"].endswith("not ancestry")
    assert report["before"]["topology"]["faces"] == 2
    assert report["after"]["topology"]["faces"] == 2
    assert report["delta_after_minus_before"]["faces"]["absolute"] == 0
    assert report["surface_proximity"]["claim"] == (
        "nearest-face-centroid-proximity-not-lineage"
    )


def test_build_report_rejects_nonpositive_sample_limit(tmp_path: Path):
    vertices, faces = _sheet()
    path = tmp_path / "mesh.npz"
    np.savez(path, vertices=vertices, faces=faces)
    try:
        build_report(
            path,
            path,
            before_label="old",
            after_label="new",
            fixed_raw_sha256=None,
            max_faces_per_side=0,
        )
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")
