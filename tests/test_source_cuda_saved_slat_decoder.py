import hashlib
import json

import numpy as np
import pytest

from scripts.source_cuda_saved_slat_decoder import main, validate_saved_slat, validate_request


def _write_slat(path, *, feats=None, coords=None):
    np.savez(
        path,
        feats=np.ones((2, 32), dtype=np.float32) if feats is None else feats,
        coords=np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
        if coords is None else coords,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_saved_slat_rejects_wrong_digest_before_decoder(tmp_path):
    path = tmp_path / "shape_slat.npz"
    _write_slat(path)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_saved_slat(path, "0" * 64, resolution=1024)


def test_saved_slat_rejects_fractional_coordinates(tmp_path):
    path = tmp_path / "shape_slat.npz"
    digest = _write_slat(path, coords=np.array([[0, 1.5, 2, 3], [0, 4, 5, 6]]))
    with pytest.raises(ValueError, match="int32"):
        validate_saved_slat(path, digest, resolution=1024)


def test_saved_slat_rejects_stale_grid_metadata(tmp_path):
    path = tmp_path / "shape_slat.npz"
    digest = _write_slat(path)
    meta = tmp_path / "shape_slat.json"
    meta.write_text(json.dumps({"mesh_grid_size": 768, "cascade": True}))
    with pytest.raises(ValueError, match="mesh_grid_size"):
        validate_request(path, digest, meta, resolution=1024)


def test_saved_slat_accepts_exact_shape_and_grid(tmp_path):
    path = tmp_path / "shape_slat.npz"
    digest = _write_slat(path)
    meta = tmp_path / "shape_slat.json"
    meta.write_text(json.dumps({"mesh_grid_size": 1024, "cascade": True}))
    identity = validate_request(path, digest, meta, resolution=1024)
    assert identity["shape"] == [2, 32]
    assert identity["resolution"] == 1024
    assert identity["sha256"] == digest


def test_failure_before_decode_writes_report_without_overwriting_input(tmp_path):
    path = tmp_path / "shape_slat.npz"
    digest = _write_slat(path)
    original = path.read_bytes()
    meta = tmp_path / "shape_slat.json"
    meta.write_text(json.dumps({"mesh_grid_size": 1024, "cascade": True}))
    source = tmp_path / "source.tar"
    source.write_bytes(b"source")
    override = tmp_path / "override.py"
    override.write_text("# bridge\n")
    result = main([
        "--output-json", str(path), "--output-npz", str(tmp_path / "state.npz"),
        "--shape-slat", str(path), "--shape-slat-metadata", str(meta),
        "--shape-slat-sha256", digest, "--source-tar", str(source),
        "--source-tar-sha256", "0" * 64,
        "--mesh-override", str(override),
        "--mesh-override-sha256", "0" * 64,
        "--model-revision", "a" * 40,
        "--resolution", "1024",
    ])
    assert result == 1
    assert path.read_bytes() == original
    report = json.loads((tmp_path / "shape_slat.npz.failure.json").read_text())
    assert report["failure_phase"] == "request_validation"
    assert not (tmp_path / "state.npz").exists()
