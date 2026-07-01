"""Contracts for GLB witness rendering."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import trimesh


SCRIPT = Path("scripts/render_glb_witness.py")


def _write_colored_box(path: Path):
    mesh = trimesh.creation.box(extents=(1.0, 0.5, 0.3))
    colors = np.array(
        [
            [220, 20, 40, 255],
            [220, 20, 40, 255],
            [40, 40, 40, 255],
            [40, 40, 40, 255],
            [245, 245, 245, 255],
            [245, 245, 245, 255],
            [160, 15, 30, 255],
            [160, 15, 30, 255],
        ],
        dtype=np.uint8,
    )
    mesh.visual.vertex_colors = colors
    mesh.export(path)


def _run_witness(input_glb: Path, output_png: Path, report_json: Path, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(input_glb),
            "--output",
            str(output_png),
            "--report",
            str(report_json),
            *extra,
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_witness_renderer_writes_nonblank_png_and_report(tmp_path):
    glb = tmp_path / "box.glb"
    output = tmp_path / "witness.png"
    report = tmp_path / "witness.json"
    _write_colored_box(glb)

    result = _run_witness(glb, output, report)

    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert report.exists()

    image = Image.open(output).convert("RGB")
    pixels = np.asarray(image)
    assert pixels.shape == (720, 2160, 3)
    assert pixels.std() > 1.0

    data = json.loads(report.read_text())
    assert data["status"] == "ok"
    assert data["route"] == "software_projected_mesh_witness"
    assert data["input_glb"] == str(glb)
    assert data["output_png"] == str(output)
    assert data["source_artifacts"]["input_glb"] == str(glb)
    assert data["source_artifacts"]["control_output_png"] == str(output)
    assert data["source_artifacts"]["report_json"] == str(report)
    assert data["mesh"]["vertices"] == 8
    assert data["mesh"]["faces"] == 12
    assert data["witness"]["nonblank"] is True
    assert data["witness"]["pixel_std"] > 1.0
    assert data["witness"]["panels"] == ["front_xz", "side_yz", "top_xy"]
    assert data["witness"]["culling_modes"] == ["double_sided", "front_faces", "back_faces"]
    assert data["witness"]["culling_summary"]["route"] == "software_projected_winding_cull"
    assert data["witness"]["culling_summary"]["front_face"] == "ccw"
    assert data["witness"]["culling_summary"]["orientation_basis"] == (
        "projected_triangle_signed_area_after_panel_projection"
    )
    for mode in data["witness"]["culling_modes"]:
        culling_path = Path(data["source_artifacts"]["culling_output_pngs"][mode])
        assert culling_path.exists()
        assert Path(data["witness"]["culling_reports"][mode]["output_png"]) == culling_path
        culling_image = Image.open(culling_path).convert("RGB")
        assert np.asarray(culling_image).shape == (720, 2160, 3)


def test_witness_renderer_culling_modes_split_projected_winding(tmp_path):
    glb = tmp_path / "box.glb"
    output = tmp_path / "witness.png"
    report = tmp_path / "witness.json"
    culling_dir = tmp_path / "culling"
    _write_colored_box(glb)

    result = _run_witness(glb, output, report, "--culling-dir", str(culling_dir), "--front-face", "cw")

    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text())
    reports = data["witness"]["culling_reports"]
    double_sided = reports["double_sided"]
    front_faces = reports["front_faces"]
    back_faces = reports["back_faces"]

    assert double_sided["faces_drawn"] > 0
    assert front_faces["faces_drawn"] > 0
    assert back_faces["faces_drawn"] > 0
    assert front_faces["faces_culled"] > 0
    assert back_faces["faces_culled"] > 0
    assert front_faces["faces_drawn"] + back_faces["faces_drawn"] == double_sided["faces_drawn"]
    assert data["witness"]["culling_summary"]["front_face"] == "cw"
    assert set(p.name for p in culling_dir.iterdir()) == {
        "double_sided.png",
        "front_faces.png",
        "back_faces.png",
    }


@pytest.mark.parametrize(
    ("input_name", "expected_phase"),
    [
        ("missing.glb", "load_mesh"),
        ("empty.glb", "validate_mesh"),
    ],
)
def test_witness_renderer_failure_writes_report(tmp_path, input_name, expected_phase):
    glb = tmp_path / input_name
    output = tmp_path / "witness.png"
    report = tmp_path / "witness.json"
    if input_name == "empty.glb":
        trimesh.Trimesh().export(glb)

    result = _run_witness(glb, output, report)

    assert result.returncode != 0
    assert not output.exists()
    assert report.exists()

    data = json.loads(report.read_text())
    assert data["status"] == "error"
    assert data["phase"] == expected_phase
    assert data["input_glb"] == str(glb)
    assert data["output_png"] == str(output)
    assert data["last_trustworthy_evidence"]["input_exists"] == glb.exists()


def test_witness_renderer_removes_stale_output_on_argument_failure(tmp_path):
    glb = tmp_path / "missing.glb"
    output = tmp_path / "stale.png"
    report = tmp_path / "witness.json"
    culling_dir = tmp_path / "stale"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(output)
    culling_dir.mkdir()
    for name in ("double_sided.png", "front_faces.png", "back_faces.png"):
        Image.new("RGB", (8, 8), (255, 0, 0)).save(culling_dir / name)

    result = _run_witness(glb, output, report, "--size", "1")

    assert result.returncode == 2
    assert not output.exists()
    assert not (culling_dir / "double_sided.png").exists()
    assert not (culling_dir / "front_faces.png").exists()
    assert not (culling_dir / "back_faces.png").exists()
    assert report.exists()

    data = json.loads(report.read_text())
    assert data["status"] == "error"
    assert data["phase"] == "parse_args"
    assert data["output_png"] == str(output)
    assert data["last_trustworthy_evidence"]["input_exists"] is False
    assert data["last_trustworthy_evidence"]["output_exists"] is True
    assert data["last_trustworthy_evidence"]["output_size_bytes"] > 0
    assert {entry["mode"] for entry in data["stale_culling_outputs_removed"]} == {
        "double_sided",
        "front_faces",
        "back_faces",
    }


def test_witness_renderer_rejects_blank_geometry_render(tmp_path):
    glb = tmp_path / "degenerate.glb"
    output = tmp_path / "witness.png"
    report = tmp_path / "witness.json"
    mesh = trimesh.Trimesh(
        vertices=np.zeros((3, 3), dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        process=False,
    )
    mesh.export(glb)

    result = _run_witness(glb, output, report)

    assert result.returncode != 0
    assert not output.exists()
    assert report.exists()

    data = json.loads(report.read_text())
    assert data["status"] == "error"
    assert data["phase"] == "validate_witness"
    assert data["last_trustworthy_evidence"]["input_exists"] is True
