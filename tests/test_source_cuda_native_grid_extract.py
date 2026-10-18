import numpy as np
import pytest
import json

from scripts.source_cuda_native_grid_extract import (
    NativeRouteError,
    require_native_convert,
    validate_decoder_arrays,
    main,
)


def test_decoder_arrays_reject_duplicate_and_nonfinite_rows():
    feats = np.zeros((2, 7), dtype=np.float32)
    coords = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32)
    with pytest.raises(ValueError, match="duplicate"):
        validate_decoder_arrays(feats, coords, resolution=1024)
    coords[1, 3] = 4
    feats[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_decoder_arrays(feats, coords, resolution=1024)


def test_native_convert_rejects_override_and_wrong_source():
    with pytest.raises(NativeRouteError, match="override"):
        require_native_convert("/tmp/stubs/o_voxel_override_convert.py", "/tmp/TRELLIS.2/o-voxel")
    with pytest.raises(NativeRouteError, match="outside"):
        require_native_convert("/tmp/unrelated/o_voxel/convert.py", "/tmp/TRELLIS.2/o-voxel")
    assert require_native_convert("/tmp/TRELLIS.2/o-voxel/o_voxel/convert.py", "/tmp/TRELLIS.2/o-voxel")


def test_missing_input_preserves_failure_report_before_mesh(tmp_path):
    report = tmp_path / "report.json"
    output = tmp_path / "raw.npz"
    result = main([
        "--input-npz", str(tmp_path / "missing.npz"),
        "--expected-sha256", "0" * 64,
        "--output-npz", str(output),
        "--report", str(report),
        "--work-dir", str(tmp_path / "work"),
    ])
    assert result == 1
    saved = json.loads(report.read_text())
    assert saved["status"] == "failed"
    assert saved["failure_phase"] == "request"
    assert "FileNotFoundError" in saved["error"]
    assert not output.exists()


def test_kaggle_mount_miss_cannot_look_like_success(tmp_path, monkeypatch):
    from scripts import source_cuda_native_grid_extract as assay

    assert assay.kaggle_main(mount=tmp_path / "empty-input", out=tmp_path) == 0
    saved = json.loads((tmp_path / "native_grid_kernel_receipt.json").read_text())
    assert saved["status"] == "failed"
    assert saved["failure_phase"] == "mount_preflight"
    assert "expected one mounted" in saved["error"]


def test_report_alias_does_not_overwrite_decoder_input(tmp_path):
    source = tmp_path / "decoder.npz"
    source.write_bytes(b"protected source bytes")
    assert main([
        "--input-npz", str(source), "--expected-sha256", "0" * 64,
        "--output-npz", str(tmp_path / "mesh.npz"),
        "--report", str(source), "--work-dir", str(tmp_path / "work"),
    ]) == 1
    assert source.read_bytes() == b"protected source bytes"
    assert json.loads((tmp_path / "decoder.npz.failure.json").read_text())["status"] == "failed"


def test_native_grid_watch_rejects_unknown_status_and_blank_terminal_output(tmp_path, monkeypatch):
    from scripts import kaggle_native_grid_watch as watch
    from types import SimpleNamespace
    import sys

    with pytest.raises(ValueError, match="unrecognized"):
        watch.status_from_text('kernel is maybe done')

    monkeypatch.setattr(sys, "argv", ["watch", "--kernel-id", watch.KERNEL_ID + "/2",
                                      "--output-dir", str(tmp_path / "output"),
                                      "--report", str(tmp_path / "watch.json")])
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[2] == "status":
            return SimpleNamespace(returncode=0, stdout='status "KernelWorkerStatus.COMPLETE"', stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(watch.subprocess, "run", fake_run)
    assert watch.main() == 1
    saved = json.loads((tmp_path / "watch.json").read_text())
    assert saved["status"] == "failed"
    assert saved["failure_phase"] == "download"
    assert "without its durable receipt" in saved["error"]
    assert [call[2] for call in calls] == ["status", "output"]
    assert all(call[3] == watch.KERNEL_ID + "/2" for call in calls)


def test_single_file_kaggle_entrypoint_runs_without_sibling_script(tmp_path, monkeypatch):
    from scripts import source_cuda_native_grid_extract as assay

    mounted = tmp_path / "input" / "notebooks" / "noahboo" / "t2mlx-kiln-1024-exact-slat-cuda-decode"
    mounted.mkdir(parents=True)
    (mounted / "cuda_decoder_state.npz").write_bytes(b"saved decoder fixture")
    output = tmp_path / "working"
    output.mkdir()
    calls = []

    def fake_parent(argv):
        calls.append(argv)
        (output / "native_grid_report.json").write_text(json.dumps({"status": "done"}))
        return 0

    monkeypatch.setattr(assay, "main", fake_parent)
    assert assay.kaggle_main(mount=tmp_path / "input", out=output) == 0
    receipt = json.loads((output / "native_grid_kernel_receipt.json").read_text())
    assert receipt["status"] == "done"
    assert calls[0][calls[0].index("--input-npz") + 1] == str(mounted / "cuda_decoder_state.npz")
    assert calls[0][calls[0].index("--expected-sha256") + 1] == assay.EXPECTED_DECODER_SHA256
