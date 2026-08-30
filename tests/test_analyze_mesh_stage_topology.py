from pathlib import Path

import numpy as np

from scripts.analyze_mesh_stage_topology import (
    _validate_stage_files,
    analyze_orientation_transition,
    analyze_stage,
)


def test_analyze_stage_reports_orientable_pair(monkeypatch, tmp_path: Path):
    vertices = np.zeros((6, 3), dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    path = tmp_path / "mesh.ply"
    path.write_bytes(b"mesh")
    monkeypatch.setattr(
        "scripts.analyze_mesh_stage_topology.read_binary_ply",
        lambda _: (vertices, faces),
    )

    report = analyze_stage(path, index=2, operation="simplify")

    assert report["vertices"] == 6
    assert report["faces"] == 2
    assert report["edge_groups"] == 5
    assert report["boundary_edges"] == 4
    assert report["manifold_edges"] == 1
    assert report["same_direction_manifold_edges"] == 0
    assert report["contradictory_components"] == 0
    assert report["contradictory_face_fraction"] == 0.0


def test_orientation_transition_counts_only_face_reversals(
    monkeypatch,
    tmp_path: Path,
):
    vertices = np.arange(18, dtype=np.float32).reshape(6, 3)
    before_faces = np.asarray(
        [[0, 1, 2], [1, 3, 2], [3, 4, 5]],
        dtype=np.int32,
    )
    after_faces = np.asarray(
        [[1, 2, 0], [1, 2, 3], [3, 4, 0]],
        dtype=np.int32,
    )
    before_path = tmp_path / "11_before.ply"
    after_path = tmp_path / "12_after.ply"
    meshes = {
        before_path: (vertices, before_faces),
        after_path: (vertices.copy(), after_faces),
    }
    monkeypatch.setattr(
        "scripts.analyze_mesh_stage_topology.read_binary_ply",
        lambda path: meshes[path],
    )

    report = analyze_orientation_transition(before_path, after_path)

    assert report["vertices_exact"] is True
    assert report["same_orientation_rows"] == 1
    assert report["reversed_orientation_rows"] == 1
    assert report["noncorresponding_rows"] == 1
    assert report["topology_preserved_row_for_row"] is False


def test_stage_validation_rejects_twelve_files_with_wrong_indices(tmp_path: Path):
    stages = [
        (index, f"operation-{index}", tmp_path / f"{index}.ply")
        for index in range(2, 14)
    ]

    try:
        _validate_stage_files(stages, tmp_path)
    except ValueError as exc:
        assert "1..12" in str(exc)
        assert "[2, 3" in str(exc)
    else:
        raise AssertionError("wrong stage indices were accepted")
