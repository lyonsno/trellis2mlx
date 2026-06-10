"""Shape artifact decode/export CLI contracts."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import trimesh

from generate import SHAPE_SLAT_MEAN, SHAPE_SLAT_STD


SCRIPT = Path("scripts/decode_shape_artifact.py")
EXPECTED_DECODER_SLAT = None


def _load_module():
    spec = importlib.util.spec_from_file_location("decode_shape_artifact", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubDecoder:
    def __call__(self, slat, coords, return_subs=False):
        slat_np = np.array(slat)
        coords_np = np.array(coords)
        if slat_np.shape[0] != 3:
            raise AssertionError(f"expected filtered slat rows, got {slat_np.shape[0]}")
        if slat_np.shape[1] != 32:
            raise AssertionError(f"expected 32 slat channels, got {slat_np.shape[1]}")
        if EXPECTED_DECODER_SLAT is not None:
            np.testing.assert_allclose(slat_np, EXPECTED_DECODER_SLAT, rtol=1e-6, atol=1e-6)
        if coords_np.shape != (3, 4):
            raise AssertionError(f"expected filtered coords shape [3,4], got {coords_np.shape}")

        dec_out = np.zeros((4, 7), dtype=np.float32)
        dec_out[:, 3:6] = 1.0
        dec_coords = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
            ],
            dtype=np.int32,
        )
        subs = [np.array([True, False, True], dtype=bool)]
        if return_subs:
            return dec_out, dec_coords, subs
        return dec_out, dec_coords


def _stub_decoder_factory():
    return _StubDecoder()


def _stub_weight_loader(decoder, checkpoint, verbose=False):
    decoder.loaded_checkpoint = str(checkpoint)


def _stub_mesh_converter(dec_out, dec_coords, resolution):
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    return vertices, faces


def test_decode_shape_artifact_filters_rows_exports_glb_and_reports_identity(tmp_path):
    global EXPECTED_DECODER_SLAT
    module = _load_module()
    input_path = tmp_path / "artifact.npz"
    output_glb = tmp_path / "mesh.glb"
    output_artifact = tmp_path / "decoded.npz"
    report_path = tmp_path / "report.json"
    coords3 = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [8, 8, 8],
        ],
        dtype=np.int32,
    )
    coords4 = np.column_stack([np.zeros(len(coords3), dtype=np.int32), coords3])
    slat = (np.arange(128, dtype=np.float32).reshape(4, 32) / 32.0) - 2.0
    EXPECTED_DECODER_SLAT = slat[:3] * SHAPE_SLAT_STD[None, :] + SHAPE_SLAT_MEAN[None, :]
    np.savez(
        input_path,
        hr_slat=slat,
        hr_coords_quantized_1024=coords4,
        hr_coords_3d_1024=coords3,
    )

    report = module.decode_shape_artifact(
        input_path,
        output_glb,
        report_path,
        output_artifact_path=output_artifact,
        slat_key="hr_slat",
        coords_key="hr_coords_quantized_1024",
        spatial_coords_key="hr_coords_3d_1024",
        component_filter_mode="largest",
        component_filter_min_ratio=1e-5,
        slat_normalization="normalized",
        decoder_checkpoint=tmp_path / "shape.safetensors",
        resolution=1024,
        overwrite=False,
        decoder_factory=_stub_decoder_factory,
        weight_loader=_stub_weight_loader,
        mesh_converter=_stub_mesh_converter,
    )

    persisted = json.loads(report_path.read_text())
    assert report["schema"] == "trellis2mlx.shape_artifact_decode.v1"
    assert persisted["status"] == "ok"
    assert persisted["route"] == "shape-artifact-decode"
    assert persisted["component_filter"]["mode"] == "largest"
    assert persisted["component_filter"]["input_count"] == 4
    assert persisted["component_filter"]["kept_count"] == 3
    assert persisted["component_filter"]["dropped_count"] == 1
    assert persisted["decode"]["input_slat_shape"] == [3, 32]
    assert persisted["decode"]["input_coords_shape"] == [3, 4]
    assert persisted["decode"]["slat_normalization"] == "normalized"
    assert persisted["decode"]["decoder_slat_shape"] == [3, 32]
    assert persisted["output"]["glb_exists"] is True
    assert persisted["output"]["artifact_exists"] is True
    assert output_glb.exists()
    assert output_artifact.exists()

    with np.load(output_artifact) as data:
        embedded = json.loads(str(data["component_filter_report_json"]))
        np.testing.assert_array_equal(data["filtered_hr_slat"], slat[:3])
        np.testing.assert_allclose(data["decoder_hr_slat"], EXPECTED_DECODER_SLAT)
        np.testing.assert_array_equal(data["filtered_hr_coords_quantized_1024"], coords4[:3])

    assert embedded == persisted["component_filter"]
    loaded = trimesh.load(output_glb, force="mesh", process=False)
    assert len(loaded.vertices) == 3
    assert len(loaded.faces) == 1
    EXPECTED_DECODER_SLAT = None


def test_main_writes_failure_report_when_spatial_key_missing(tmp_path):
    module = _load_module()
    input_path = tmp_path / "artifact.npz"
    output_glb = tmp_path / "mesh.glb"
    report_path = tmp_path / "report.json"
    np.savez(
        input_path,
        hr_slat=np.zeros((2, 4), dtype=np.float32),
        hr_coords_quantized_1024=np.zeros((2, 4), dtype=np.int32),
    )

    try:
        module.main(
            [
                "--input",
                str(input_path),
                "--output-glb",
                str(output_glb),
                "--report",
                str(report_path),
                "--decoder-checkpoint",
                str(tmp_path / "shape.safetensors"),
                "--spatial-coords-key",
                "hr_coords_3d_1024",
            ]
        )
    except KeyError:
        pass
    else:
        raise AssertionError("expected missing spatial coords key to fail")

    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["phase"] == "decode_shape_artifact"
    assert "hr_coords_3d_1024" in report["error"]
    assert report["last_trustworthy_evidence"]["input_exists"] is True
    assert report["last_trustworthy_evidence"]["glb_exists"] is False
