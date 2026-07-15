import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_build_preview_records_source_and_effective_route(tmp_path):
    import scripts.build_raw_mesh_preview as build_raw_mesh_preview

    input_ply = tmp_path / "raw.ply"
    output_glb = tmp_path / "preview.glb"
    report_json = tmp_path / "preview.json"
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [1, 4, 2], [1, 5, 4]],
        dtype=np.int32,
    )
    build_raw_mesh_preview.write_binary_ply(input_ply, vertices, faces)

    report = build_raw_mesh_preview.build_raw_mesh_preview(
        input_ply=input_ply,
        output_glb=output_glb,
        report_json=report_json,
        face_stride=2,
    )

    persisted = json.loads(report_json.read_text())
    assert persisted == report
    assert report["status"] == "done"
    assert report["route"] == "deterministic_face_stride_preview"
    assert report["requested_face_stride"] == 2
    assert report["effective_face_stride"] == 2
    assert report["source_mesh"]["vertices"] == 6
    assert report["source_mesh"]["faces"] == 4
    assert report["preview_mesh"]["faces"] == 2
    assert report["preview_mesh"]["vertices"] == 4
    assert report["source_read_mode"] == "numpy_memmap"
    assert report["source_mapped_bytes"] == (6 * 3 * 4) + (4 * 13)
    assert report["preview_materialized_bytes"] == (4 * 3 * 4) + (2 * 3 * 4)
    assert report["input_ply_sha256"] == build_raw_mesh_preview.sha256_file(input_ply)
    assert report["output_glb_sha256"] == build_raw_mesh_preview.sha256_file(output_glb)
    assert report["output_glb_size_bytes"] > 0
    assert report["forbidden_inferences"] == [
        "preview is not the complete raw mesh",
        "preview topology metrics are not full-mesh topology evidence",
        "preview is not cleanup, hole-fill, UV, texture, or final-GLB evidence",
    ]


def test_build_preview_persists_failure_before_primary_output(tmp_path):
    import scripts.build_raw_mesh_preview as build_raw_mesh_preview

    input_ply = tmp_path / "empty.ply"
    output_glb = tmp_path / "preview.glb"
    report_json = tmp_path / "preview.json"
    build_raw_mesh_preview.write_binary_ply(
        input_ply,
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.empty((0, 3), dtype=np.int32),
    )

    with pytest.raises(ValueError, match="no triangular faces"):
        build_raw_mesh_preview.build_raw_mesh_preview(
            input_ply=input_ply,
            output_glb=output_glb,
            report_json=report_json,
            face_stride=4,
        )

    report = json.loads(report_json.read_text())
    assert report["status"] == "failed"
    assert report["phase"] == "validate_source_mesh"
    assert report["last_trustworthy_evidence"]["input_ply_sha256"]
    assert report["error_type"] == "ValueError"
    assert not output_glb.exists()


@pytest.mark.parametrize("collision_target", ["input", "output"])
def test_build_preview_rejects_report_path_alias_without_destroying_evidence(
    tmp_path,
    collision_target,
):
    import scripts.build_raw_mesh_preview as build_raw_mesh_preview

    input_ply = tmp_path / "raw.ply"
    output_glb = tmp_path / "preview.glb"
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    build_raw_mesh_preview.write_binary_ply(input_ply, vertices, faces)
    input_sha256 = build_raw_mesh_preview.sha256_file(input_ply)
    report_json = input_ply if collision_target == "input" else output_glb
    failure_report = report_json.with_name(report_json.name + ".failure.json")

    with pytest.raises(ValueError, match="requested paths must be distinct"):
        build_raw_mesh_preview.build_raw_mesh_preview(
            input_ply=input_ply,
            output_glb=output_glb,
            report_json=report_json,
            face_stride=1,
        )

    assert build_raw_mesh_preview.sha256_file(input_ply) == input_sha256
    assert not output_glb.exists()
    report = json.loads(failure_report.read_text())
    assert report["status"] == "failed"
    assert report["phase"] == "validate_request"
    assert report["requested_report_json"] == str(report_json)
    assert report["effective_report_json"] == str(failure_report)
    assert report["path_collisions"]


def test_build_preview_removes_invalid_post_write_output(monkeypatch, tmp_path):
    import scripts.build_raw_mesh_preview as build_raw_mesh_preview

    input_ply = tmp_path / "raw.ply"
    output_glb = tmp_path / "preview.glb"
    report_json = tmp_path / "preview.json"
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    build_raw_mesh_preview.write_binary_ply(input_ply, vertices, faces)

    def fail_reload(*args, **kwargs):
        raise ValueError("synthetic GLB reload failure")

    monkeypatch.setattr(build_raw_mesh_preview.trimesh, "load", fail_reload)

    with pytest.raises(ValueError, match="synthetic GLB reload failure"):
        build_raw_mesh_preview.build_raw_mesh_preview(
            input_ply=input_ply,
            output_glb=output_glb,
            report_json=report_json,
            face_stride=1,
        )

    report = json.loads(report_json.read_text())
    assert report["status"] == "failed"
    assert report["phase"] == "validate_preview_glb"
    assert report["invalid_output_observed"]["size_bytes"] > 0
    assert report["invalid_output_observed"]["sha256"]
    assert report["invalid_output_removed"] is True
    assert not output_glb.exists()


def test_build_preview_failure_sidecar_cannot_alias_input(tmp_path):
    import scripts.build_raw_mesh_preview as build_raw_mesh_preview

    output_glb = tmp_path / "preview.glb"
    report_json = output_glb
    input_ply = tmp_path / "preview.glb.failure.json"
    expected_failure_report = tmp_path / "preview.glb.failure.1.json"
    build_raw_mesh_preview.write_binary_ply(
        input_ply,
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.int32),
    )
    input_sha256 = build_raw_mesh_preview.sha256_file(input_ply)

    with pytest.raises(ValueError, match="requested paths must be distinct"):
        build_raw_mesh_preview.build_raw_mesh_preview(
            input_ply=input_ply,
            output_glb=output_glb,
            report_json=report_json,
            face_stride=1,
        )

    assert build_raw_mesh_preview.sha256_file(input_ply) == input_sha256
    assert not output_glb.exists()
    report = json.loads(expected_failure_report.read_text())
    assert report["effective_report_json"] == str(expected_failure_report)


def test_build_preview_cli_runs_as_a_direct_script(tmp_path):
    import scripts.build_raw_mesh_preview as build_raw_mesh_preview

    input_ply = tmp_path / "raw.ply"
    output_glb = tmp_path / "preview.glb"
    report_json = tmp_path / "preview.json"
    build_raw_mesh_preview.write_binary_ply(
        input_ply,
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.int32),
    )
    script = Path(build_raw_mesh_preview.__file__)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input-ply",
            str(input_ply),
            "--output-glb",
            str(output_glb),
            "--report-json",
            str(report_json),
            "--face-stride",
            "1",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(report_json.read_text())["status"] == "done"
    assert output_glb.exists()
