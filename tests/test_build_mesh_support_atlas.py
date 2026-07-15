import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest


def test_mesh_support_atlas_reports_multiscale_spatial_differences(tmp_path):
    import scripts.build_mesh_support_atlas as build_mesh_support_atlas
    from scripts.postprocess_raw_cuda_mesh import write_binary_ply

    vertices_a = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    vertices_b = vertices_a.copy()
    vertices_b[3] = [1, 1, 1]
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh_a = tmp_path / "a.ply"
    mesh_b = tmp_path / "b.ply"
    output_json = tmp_path / "atlas.json"
    output_png = tmp_path / "atlas.png"
    write_binary_ply(mesh_a, vertices_a, faces)
    write_binary_ply(mesh_b, vertices_b, faces)

    report = build_mesh_support_atlas.build_mesh_support_atlas(
        meshes={"alpha-0_beta-0": mesh_a, "alpha-1_beta-1": mesh_b},
        grid_sizes=[4, 8],
        reference="alpha-0_beta-0",
        output_json=output_json,
        output_png=output_png,
    )

    assert json.loads(output_json.read_text()) == report
    assert report["status"] == "done"
    assert report["route"] == "shared_grid_vertex_surface_support"
    assert report["embedding_authority"] == "none"
    assert report["grid_sizes"] == [4, 8]
    assert report["reference"] == "alpha-0_beta-0"
    assert [source["name"] for source in report["sources"]] == [
        "alpha-0_beta-0",
        "alpha-1_beta-1",
    ]
    for grid_size in (4, 8):
        matrix = report["scales"][str(grid_size)]["pairwise_jaccard"]
        assert matrix[0][0] == 1.0
        assert matrix[1][1] == 1.0
        assert matrix[0][1] == matrix[1][0]
        assert 0.0 < matrix[0][1] < 1.0
    assert report["forbidden_inferences"] == [
        "vertex support occupancy is not watertight volume occupancy",
        "Jaccard distance is not global learned-manifold distance",
        "projected support deltas are not topology or winding evidence",
    ]
    assert output_png.exists()
    with Image.open(output_png) as image:
        assert image.width > 0
        assert image.height > 0
        assert np.asarray(image.convert("RGB")).std() > 0


def test_mesh_support_atlas_persists_failure_before_visual_output(tmp_path):
    import scripts.build_mesh_support_atlas as build_mesh_support_atlas

    output_json = tmp_path / "atlas.json"
    output_png = tmp_path / "atlas.png"

    with pytest.raises(FileNotFoundError):
        build_mesh_support_atlas.build_mesh_support_atlas(
            meshes={"missing": tmp_path / "missing.ply", "also-missing": tmp_path / "other.ply"},
            grid_sizes=[8],
            reference="missing",
            output_json=output_json,
            output_png=output_png,
        )

    report = json.loads(output_json.read_text())
    assert report["status"] == "failed"
    assert report["phase"] == "read_source_identity"
    assert report["last_trustworthy_evidence"]["validated_request"] is True
    assert report["error_type"] == "FileNotFoundError"
    assert not output_png.exists()


def test_mesh_support_atlas_cli_runs_as_direct_script(tmp_path):
    import scripts.build_mesh_support_atlas as build_mesh_support_atlas
    from scripts.postprocess_raw_cuda_mesh import write_binary_ply

    mesh_a = tmp_path / "a.ply"
    mesh_b = tmp_path / "b.ply"
    output_json = tmp_path / "atlas.json"
    output_png = tmp_path / "atlas.png"
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    write_binary_ply(mesh_a, vertices, faces)
    write_binary_ply(mesh_b, vertices + 0.1, faces)

    result = subprocess.run(
        [
            sys.executable,
            str(Path(build_mesh_support_atlas.__file__)),
            "--mesh",
            f"a={mesh_a}",
            "--mesh",
            f"b={mesh_b}",
            "--grid-size",
            "8",
            "--reference",
            "a",
            "--output-json",
            str(output_json),
            "--output-png",
            str(output_png),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output_json.read_text())["status"] == "done"
    assert output_png.exists()


def test_mesh_support_atlas_cli_persists_malformed_mesh_failure(tmp_path):
    import scripts.build_mesh_support_atlas as build_mesh_support_atlas

    output_json = tmp_path / "atlas.json"
    output_png = tmp_path / "atlas.png"
    output_json.write_text('{"status": "done", "stale": true}\n')
    output_png.write_bytes(b"stale-png")

    result = subprocess.run(
        [
            sys.executable,
            str(Path(build_mesh_support_atlas.__file__)),
            "--mesh",
            "missing-separator",
            "--mesh",
            f"valid={tmp_path / 'valid.ply'}",
            "--grid-size",
            "8",
            "--reference",
            "valid",
            "--output-json",
            str(output_json),
            "--output-png",
            str(output_png),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(output_json.read_text())
    assert report["status"] == "failed"
    assert report["phase"] == "request_validation"
    assert report["requested_output_json"] == str(output_json)
    assert report["output_png"] == str(output_png)
    assert report["error_type"] == "ValueError"
    assert "NAME=PLY" in report["error"]
    assert not output_png.exists()
