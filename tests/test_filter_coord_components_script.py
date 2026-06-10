"""NPZ artifact coordinate-component filter CLI contracts."""

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path("scripts/filter_coord_components.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("filter_coord_components", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_filter_npz_artifact_writes_filtered_rows_and_report(tmp_path):
    module = _load_module()
    input_path = tmp_path / "input.npz"
    output_path = tmp_path / "filtered.npz"
    report_path = tmp_path / "report.json"
    coords = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [8, 8, 8],
        ],
        dtype=np.int32,
    )
    slat = np.arange(16, dtype=np.float32).reshape(4, 4)
    untouched = np.array([13], dtype=np.int32)
    coords4 = np.column_stack([np.zeros(len(coords), dtype=np.int32), coords])
    np.savez(
        input_path,
        hr_coords_3d_1024=coords,
        hr_coords_quantized_1024=coords4,
        hr_slat=slat,
        untouched=untouched,
    )

    report = module.filter_npz_artifact(
        input_path,
        output_path,
        report_path,
        coords_key="hr_coords_3d_1024",
        features_key="hr_slat",
        aligned_keys=["hr_coords_quantized_1024"],
        mode="largest",
        min_component_ratio=1e-5,
    )

    with np.load(output_path) as data:
        np.testing.assert_array_equal(data["hr_coords_3d_1024"], coords[:3])
        np.testing.assert_array_equal(data["hr_coords_quantized_1024"], coords4[:3])
        np.testing.assert_array_equal(data["hr_slat"], slat[:3])
        np.testing.assert_array_equal(data["untouched"], untouched)
        embedded = json.loads(str(data["component_filter_report_json"]))

    persisted = json.loads(report_path.read_text())
    assert report["schema"] == "trellis2mlx.coord_component_filter.v1"
    assert persisted["status"] == "ok"
    assert persisted["route"] == "sparse-coordinate-component-filter"
    assert persisted["filter"]["mode"] == "largest"
    assert persisted["filter"]["input_count"] == 4
    assert persisted["filter"]["kept_count"] == 3
    assert persisted["filter"]["dropped_count"] == 1
    assert persisted["output"]["filtered_keys"] == [
        "hr_coords_3d_1024",
        "hr_slat",
        "hr_coords_quantized_1024",
    ]
    assert embedded["component_sizes"] == [3, 1]


def test_main_writes_failure_report_when_key_missing(tmp_path):
    module = _load_module()
    input_path = tmp_path / "input.npz"
    output_path = tmp_path / "filtered.npz"
    report_path = tmp_path / "report.json"
    np.savez(input_path, other=np.zeros((1, 3), dtype=np.int32))

    try:
        module.main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--report",
                str(report_path),
                "--coords-key",
                "missing",
            ]
        )
    except KeyError:
        pass
    else:
        raise AssertionError("expected missing coords key to fail")

    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["phase"] == "filter_npz_artifact"
    assert "missing" in report["error"]
    assert report["last_trustworthy_evidence"]["input_exists"] is True
    assert report["last_trustworthy_evidence"]["output_exists"] is False
