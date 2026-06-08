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
    assert data["mesh"]["vertices"] == 8
    assert data["mesh"]["faces"] == 12
    assert data["witness"]["nonblank"] is True
    assert data["witness"]["pixel_std"] > 1.0
    assert data["witness"]["panels"] == ["front_xz", "side_yz", "top_xy"]


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
    Image.new("RGB", (8, 8), (255, 0, 0)).save(output)

    result = _run_witness(glb, output, report, "--size", "1")

    assert result.returncode == 2
    assert not output.exists()
    assert report.exists()

    data = json.loads(report.read_text())
    assert data["status"] == "error"
    assert data["phase"] == "parse_args"
    assert data["output_png"] == str(output)
    assert data["last_trustworthy_evidence"]["input_exists"] is False
    assert data["last_trustworthy_evidence"]["output_exists"] is True
    assert data["last_trustworthy_evidence"]["output_size_bytes"] > 0


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
