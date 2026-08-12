"""Contracts for deterministic, provenance-bearing GLB volume crops."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh


SCRIPT = Path("scripts/extract_glb_aabb_crop.py")


def _write_three_region_mesh(path: Path) -> None:
    vertices = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [0.00, 0.10, 0.00],
            [0.30, 0.00, 0.00],
            [0.35, 0.00, 0.00],
            [0.30, 0.05, 0.00],
            [0.80, 0.00, 0.00],
            [0.85, 0.00, 0.00],
            [0.80, 0.05, 0.00],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(
        path, file_type="glb"
    )


def _run_crop(
    input_glb: Path,
    output_glb: Path,
    report_json: Path,
    provenance_dir: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(input_glb),
            "--output",
            str(output_glb),
            "--report",
            str(report_json),
            "--provenance-dir",
            str(provenance_dir),
            "--core-min",
            "-0.2",
            "-0.2",
            "-0.2",
            "--core-max",
            "0.2",
            "0.2",
            "0.2",
            "--halo-fraction",
            "1.0",
            "--chunk-faces",
            "2",
            *extra,
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _mutate_glb_document(
    path: Path,
    mutate: Callable[[dict, int], bytes | None],
) -> None:
    raw = path.read_bytes()
    magic, version, _declared_length = struct.unpack_from("<4sII", raw, 0)
    json_length, json_type = struct.unpack_from("<II", raw, 12)
    json_start = 20
    json_end = json_start + json_length
    binary_length, _binary_type = struct.unpack_from("<II", raw, json_end)
    document = json.loads(raw[json_start:json_end].rstrip(b" \t\r\n\x00"))
    trailing = mutate(document, binary_length) or b""
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    chunks = struct.pack("<II", len(encoded), json_type) + encoded + raw[json_end:] + trailing
    path.write_bytes(struct.pack("<4sII", magic, version, 12 + len(chunks)) + chunks)


def test_crop_preserves_core_halo_identity_and_source_order(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    output = tmp_path / "crop.glb"
    report = tmp_path / "crop.report.json"
    provenance = tmp_path / "crop-provenance"
    _write_three_region_mesh(source)

    result = _run_crop(source, output, report, provenance)

    assert result.returncode == 0, result.stderr
    loaded = trimesh.load(output, force="mesh", process=False)
    assert len(loaded.faces) == 2
    assert len(loaded.vertices) == 6

    source_faces = np.load(provenance / "source_face_indices.npy")
    source_vertices = np.load(provenance / "source_vertex_indices.npy")
    core_mask = np.load(provenance / "core_face_mask.npy")
    np.testing.assert_array_equal(source_faces, np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(source_vertices, np.arange(6, dtype=np.int64))
    np.testing.assert_array_equal(core_mask, np.array([True, False]))

    data = json.loads(report.read_text())
    assert data["status"] == "ok"
    assert data["route"] == "glb-mmap-triangle-bounds-aabb-v1"
    assert data["selection"]["rule"] == "triangle_bounds_overlap_outer_aabb"
    assert data["selection"]["source_faces"] == 3
    assert data["selection"]["selected_faces"] == 2
    assert data["selection"]["core_faces"] == 1
    assert data["selection"]["halo_only_faces"] == 1
    assert data["effective_config"]["chunk_faces"] == 2
    assert data["effective_config"]["face_limit"] is None
    assert data["source"]["sha256"] == _sha256(source)
    assert data["output"]["sha256"] == _sha256(output)
    assert data["report"] == {
        "requested_path": str(report),
        "effective_path": str(report),
    }


def test_crop_is_byte_deterministic_across_repeated_runs(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    _write_three_region_mesh(source)

    outputs = []
    provenance_arrays = []
    for run in (1, 2):
        output = tmp_path / f"crop-{run}.glb"
        report = tmp_path / f"crop-{run}.report.json"
        provenance = tmp_path / f"crop-{run}-provenance"
        result = _run_crop(source, output, report, provenance)
        assert result.returncode == 0, result.stderr
        outputs.append(_sha256(output))
        provenance_arrays.append(
            (
                np.load(provenance / "source_face_indices.npy"),
                np.load(provenance / "source_vertex_indices.npy"),
                np.load(provenance / "core_face_mask.npy"),
            )
        )

    assert outputs[0] == outputs[1]
    for left, right in zip(provenance_arrays[0], provenance_arrays[1], strict=True):
        np.testing.assert_array_equal(left, right)


def test_empty_crop_removes_stale_outputs_and_writes_failure_report(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    output = tmp_path / "stale.glb"
    report = tmp_path / "crop.report.json"
    provenance = tmp_path / "crop-provenance"
    _write_three_region_mesh(source)
    output.write_bytes(b"stale-output")
    provenance.mkdir()
    (provenance / "source_face_indices.npy").write_bytes(b"stale-provenance")

    result = _run_crop(
        source,
        output,
        report,
        provenance,
        "--core-min",
        "2.0",
        "2.0",
        "2.0",
        "--core-max",
        "2.1",
        "2.1",
        "2.1",
    )

    assert result.returncode != 0
    assert not output.exists()
    assert not provenance.exists()
    assert report.exists()

    data = json.loads(report.read_text())
    assert data["status"] == "error"
    assert data["phase"] == "select_faces"
    assert data["last_trustworthy_evidence"]["source_faces"] == 3
    assert data["last_trustworthy_evidence"]["preexisting_output_removed"] is True
    assert data["last_trustworthy_evidence"]["preexisting_provenance_removed"] is True


def test_invalid_aabb_fails_before_reading_mesh_and_reports_effective_request(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    output = tmp_path / "crop.glb"
    report = tmp_path / "crop.report.json"
    provenance = tmp_path / "crop-provenance"
    _write_three_region_mesh(source)

    result = _run_crop(
        source,
        output,
        report,
        provenance,
        "--core-min",
        "0.2",
        "0.2",
        "0.2",
        "--core-max",
        "0.1",
        "0.1",
        "0.1",
    )

    assert result.returncode != 0
    data = json.loads(report.read_text())
    assert data["status"] == "error"
    assert data["phase"] == "validate_request"
    assert data["request"]["core_min"] == [0.2, 0.2, 0.2]
    assert data["request"]["core_max"] == [0.1, 0.1, 0.1]
    assert data["request"]["halo_fraction"] == 1.0


def test_input_output_collision_fails_without_mutating_source(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    report = tmp_path / "crop.report.json"
    provenance = tmp_path / "crop-provenance"
    _write_three_region_mesh(source)
    source_sha256 = _sha256(source)

    result = _run_crop(source, source, report, provenance)

    assert result.returncode != 0
    assert source.exists()
    assert _sha256(source) == source_sha256
    data = json.loads(report.read_text())
    assert data["status"] == "error"
    assert data["phase"] == "validate_output_paths"
    assert data["last_trustworthy_evidence"]["input_exists"] is True
    assert data["last_trustworthy_evidence"]["source_preserved"] is True


def test_report_input_collision_preserves_source_and_uses_safe_failure_report(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.glb"
    output = tmp_path / "crop.glb"
    provenance = tmp_path / "crop-provenance"
    safe_report = tmp_path / "source.glb.crop-error.json"
    _write_three_region_mesh(source)
    source_sha256 = _sha256(source)

    result = _run_crop(source, output, source, provenance)

    assert result.returncode != 0
    assert source.exists()
    assert _sha256(source) == source_sha256
    assert safe_report.exists()
    data = json.loads(safe_report.read_text())
    assert data["status"] == "error"
    assert data["phase"] == "validate_output_paths"
    assert data["report"]["requested_path"] == str(source)
    assert data["report"]["effective_path"] == str(safe_report)
    assert data["last_trustworthy_evidence"]["source_preserved"] is True


def test_derived_output_temporary_collision_preserves_source(tmp_path: Path) -> None:
    output = tmp_path / "crop.glb"
    source = tmp_path / "crop.glb.tmp"
    report = tmp_path / "crop.report.json"
    provenance = tmp_path / "crop-provenance"
    _write_three_region_mesh(source)
    source_sha256 = _sha256(source)

    result = _run_crop(source, output, report, provenance)

    assert result.returncode != 0
    assert source.exists()
    assert _sha256(source) == source_sha256
    data = json.loads(report.read_text())
    assert data["phase"] == "validate_output_paths"
    assert data["last_trustworthy_evidence"]["source_preserved"] is True


def test_derived_provenance_temporary_collision_preserves_source(tmp_path: Path) -> None:
    output = tmp_path / "crop.glb"
    report = tmp_path / "crop.report.json"
    provenance = tmp_path / "crop-provenance"
    source = tmp_path / "crop-provenance.tmp"
    _write_three_region_mesh(source)
    source_sha256 = _sha256(source)

    result = _run_crop(source, output, report, provenance)

    assert result.returncode != 0
    assert source.exists()
    assert _sha256(source) == source_sha256
    data = json.loads(report.read_text())
    assert data["phase"] == "validate_output_paths"


def test_output_directory_cleanup_failure_writes_durable_report(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    output = tmp_path / "crop.glb"
    report = tmp_path / "crop.report.json"
    provenance = tmp_path / "crop-provenance"
    _write_three_region_mesh(source)
    source_sha256 = _sha256(source)
    output.mkdir()

    result = _run_crop(source, output, report, provenance)

    assert result.returncode != 0
    assert output.is_dir()
    assert _sha256(source) == source_sha256
    data = json.loads(report.read_text())
    assert data["status"] == "error"
    assert data["phase"] == "cleanup_output_surface"


def test_provenance_file_cleanup_failure_writes_durable_report(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    output = tmp_path / "crop.glb"
    report = tmp_path / "crop.report.json"
    provenance = tmp_path / "crop-provenance"
    _write_three_region_mesh(source)
    source_sha256 = _sha256(source)
    provenance.write_text("not-a-directory")

    result = _run_crop(source, output, report, provenance)

    assert result.returncode != 0
    assert provenance.is_file()
    assert _sha256(source) == source_sha256
    data = json.loads(report.read_text())
    assert data["status"] == "error"
    assert data["phase"] == "cleanup_output_surface"


def test_negative_position_accessor_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    output = tmp_path / "crop.glb"
    report = tmp_path / "crop.report.json"
    provenance = tmp_path / "crop-provenance"
    _write_three_region_mesh(source)

    def set_negative_accessor(document: dict, _binary_length: int) -> None:
        document["meshes"][0]["primitives"][0]["attributes"]["POSITION"] = -1

    _mutate_glb_document(source, set_negative_accessor)
    result = _run_crop(source, output, report, provenance)

    assert result.returncode != 0
    data = json.loads(report.read_text())
    assert data["phase"] == "parse_glb"
    assert "accessor" in data["error"]


def test_accessor_extent_must_fit_referenced_buffer_view(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    output = tmp_path / "crop.glb"
    report = tmp_path / "crop.report.json"
    provenance = tmp_path / "crop-provenance"
    _write_three_region_mesh(source)

    def shrink_position_view(document: dict, _binary_length: int) -> None:
        position = document["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
        view = document["accessors"][position]["bufferView"]
        document["bufferViews"][view]["byteLength"] = 4

    _mutate_glb_document(source, shrink_position_view)
    result = _run_crop(source, output, report, provenance)

    assert result.returncode != 0
    data = json.loads(report.read_text())
    assert data["phase"] == "parse_glb"
    assert "bufferView" in data["error"]


def test_accessor_cannot_escape_bin_chunk_into_trailing_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    output = tmp_path / "crop.glb"
    report = tmp_path / "crop.report.json"
    provenance = tmp_path / "crop-provenance"
    _write_three_region_mesh(source)

    def move_positions_after_bin(document: dict, binary_length: int) -> bytes:
        position = document["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
        accessor = document["accessors"][position]
        view = document["bufferViews"][accessor["bufferView"]]
        payload_bytes = int(accessor["count"]) * 3 * 4
        view["byteOffset"] = binary_length
        view["byteLength"] = payload_bytes
        document["buffers"][0]["byteLength"] = binary_length + payload_bytes
        return b"\x00" * payload_bytes

    _mutate_glb_document(source, move_positions_after_bin)
    result = _run_crop(source, output, report, provenance)

    assert result.returncode != 0
    data = json.loads(report.read_text())
    assert data["phase"] == "parse_glb"
    assert "BIN" in data["error"] or "buffer" in data["error"]
