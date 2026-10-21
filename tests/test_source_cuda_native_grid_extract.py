import numpy as np
import pytest
import json
import hashlib
from pathlib import Path

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

    monkeypatch.setattr(sys, "argv", ["watch", "--kernel-id", watch.KERNEL_ID,
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
    assert all(call[3] == watch.KERNEL_ID for call in calls)


def test_native_grid_watch_refuses_stale_output_before_download(tmp_path, monkeypatch):
    from scripts import kaggle_native_grid_watch as watch
    import sys

    output = tmp_path / "old-output"
    output.mkdir()
    (output / "native_grid_kernel_receipt.json").write_text('{"status":"done"}')
    monkeypatch.setattr(sys, "argv", ["watch", "--output-dir", str(output),
                                      "--report", str(tmp_path / "watch.json")])
    monkeypatch.setattr(watch.subprocess, "run", lambda *_a, **_k: pytest.fail("stale output reached Kaggle"))
    assert watch.main() == 1
    assert "nonempty" in json.loads((tmp_path / "watch.json").read_text())["error"]


@pytest.mark.parametrize("defect", ["missing_mesh", "wrong_hash", "wrong_source", "wrong_resolution", "wrong_input", "wrong_assay", "wrong_route", "wrong_native_status", "wrong_assay_returncode", "wrong_child_returncode", "wrong_preflight_returncode", "missing_preflight_returncode", "noninteger_faces", "nan_faces", "wrong_kernel", "wrong_script"])
def test_native_grid_watch_rejects_false_success(tmp_path, defect):
    from scripts import kaggle_native_grid_watch as watch
    import hashlib

    output = tmp_path / "download"
    output.mkdir()
    mesh = output / "native_grid_mesh.npz"
    if defect != "missing_mesh":
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        if defect == "noninteger_faces":
            faces = np.array([[0, 0.5, 2]], dtype=np.float32)
        if defect == "nan_faces":
            faces = np.array([[0, np.nan, 2]], dtype=np.float32)
        np.savez(mesh, vertices=np.zeros((3, 3), dtype=np.float32), faces=faces)
    expected_hash = hashlib.sha256(mesh.read_bytes()).hexdigest() if mesh.exists() else "0" * 64
    native = {
        "status": "done", "preflight_returncode": 0, "child_returncode": 0, "expected_sha256": watch.EXPECTED_INPUT_SHA256,
        "source_commit": watch.EXPECTED_SOURCE_COMMIT, "source_head": watch.EXPECTED_SOURCE_COMMIT,
        "native_source_heads": watch.EXPECTED_NATIVE_SOURCE_HEADS,
        "resolution": 1024,
        "mesh": {"vertices": 3, "faces": 1, "output_sha256": expected_hash},
        "effective_route": {"cuda_device": "Tesla T4", "convert_path": "/tmp/TRELLIS.2/o-voxel/o_voxel/convert.py",
                            "extension_path": "/tmp/o_voxel/_C.so"},
    }
    if defect == "wrong_hash":
        native["mesh"]["output_sha256"] = "0" * 64
    if defect == "wrong_source":
        native["source_commit"] = "0" * 40
    if defect == "wrong_resolution":
        native["resolution"] = 768
    if defect == "wrong_input":
        native["expected_sha256"] = "0" * 64
    if defect == "wrong_route":
        native["effective_route"]["convert_path"] = "/tmp/stubs/convert.py"
    if defect == "wrong_child_returncode":
        native["child_returncode"] = 1
    if defect == "wrong_preflight_returncode":
        native["preflight_returncode"] = 9
    if defect == "missing_preflight_returncode":
        native.pop("preflight_returncode")
    (output / "native_grid_report.json").write_text(json.dumps(native))
    (output / "native_grid_kernel_receipt.json").write_text(json.dumps({
        "status": "done", "expected_input_sha256": watch.EXPECTED_INPUT_SHA256,
        "assay_id": "other-run" if defect == "wrong_assay" else watch.EXPECTED_ASSAY_ID,
        "native_status": "failed" if defect == "wrong_native_status" else "done",
        "kernel_id": "other/kernel" if defect == "wrong_kernel" else watch.DECLARED_KERNEL_ID,
        "kernel_source_sha256": "0" * 64 if defect == "wrong_script" else watch.EXPECTED_KERNEL_SOURCE_SHA256,
        "assay_returncode": 1 if defect == "wrong_assay_returncode" else 0}))
    with pytest.raises(ValueError):
        watch.verify_native_result(output)


def test_native_grid_watch_accepts_hash_bound_current_mesh(tmp_path):
    from scripts import kaggle_native_grid_watch as watch
    import hashlib

    output = tmp_path / "download"
    output.mkdir()
    mesh = output / "native_grid_mesh.npz"
    np.savez(mesh, vertices=np.zeros((3, 3), dtype=np.float32), faces=np.array([[0, 1, 2]], dtype=np.int32))
    digest = hashlib.sha256(mesh.read_bytes()).hexdigest()
    (output / "native_grid_kernel_receipt.json").write_text(json.dumps({
        "status": "done", "assay_id": watch.EXPECTED_ASSAY_ID,
        "kernel_id": watch.DECLARED_KERNEL_ID, "kernel_source_sha256": watch.EXPECTED_KERNEL_SOURCE_SHA256,
        "native_status": "done", "assay_returncode": 0,
        "expected_input_sha256": watch.EXPECTED_INPUT_SHA256}))
    (output / "native_grid_report.json").write_text(json.dumps({
        "status": "done", "preflight_returncode": 0, "child_returncode": 0, "expected_sha256": watch.EXPECTED_INPUT_SHA256,
        "source_commit": watch.EXPECTED_SOURCE_COMMIT, "source_head": watch.EXPECTED_SOURCE_COMMIT,
        "native_source_heads": watch.EXPECTED_NATIVE_SOURCE_HEADS,
        "resolution": 1024, "mesh": {"vertices": 3, "faces": 1, "output_sha256": digest},
        "effective_route": {"cuda_device": "Tesla T4", "convert_path": "/tmp/TRELLIS.2/o-voxel/o_voxel/convert.py",
                            "extension_path": "/tmp/o_voxel/_C.so"}}))
    assert watch.verify_native_result(output)["mesh_sha256"] == digest


@pytest.mark.parametrize("missing", ["native_status", "assay_returncode", "preflight_returncode", "child_returncode", "kernel_id", "kernel_source_sha256"])
def test_native_grid_watch_requires_complete_execution_identity(tmp_path, missing):
    from scripts import kaggle_native_grid_watch as watch

    output = tmp_path / "download"
    output.mkdir()
    mesh = output / "native_grid_mesh.npz"
    np.savez(mesh, vertices=np.zeros((3, 3), dtype=np.float32), faces=np.array([[0, 1, 2]], dtype=np.int32))
    receipt = {"status": "done", "assay_id": watch.EXPECTED_ASSAY_ID,
               "kernel_id": watch.DECLARED_KERNEL_ID, "kernel_source_sha256": watch.EXPECTED_KERNEL_SOURCE_SHA256,
               "native_status": "done", "assay_returncode": 0,
               "expected_input_sha256": watch.EXPECTED_INPUT_SHA256}
    native = {"status": "done", "preflight_returncode": 0, "child_returncode": 0, "expected_sha256": watch.EXPECTED_INPUT_SHA256,
              "source_commit": watch.EXPECTED_SOURCE_COMMIT, "source_head": watch.EXPECTED_SOURCE_COMMIT,
              "native_source_heads": watch.EXPECTED_NATIVE_SOURCE_HEADS, "resolution": 1024,
              "mesh": {"vertices": 3, "faces": 1, "output_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest()},
              "effective_route": {"cuda_device": "Tesla T4", "convert_path": "/tmp/TRELLIS.2/o-voxel/o_voxel/convert.py",
                                  "extension_path": "/tmp/o_voxel/_C.so"}}
    (native if missing in {"preflight_returncode", "child_returncode"} else receipt).pop(missing)
    (output / "native_grid_kernel_receipt.json").write_text(json.dumps(receipt))
    (output / "native_grid_report.json").write_text(json.dumps(native))
    with pytest.raises(ValueError):
        watch.verify_native_result(output)


def test_native_grid_watch_rejects_version_suffix_before_contact(tmp_path, monkeypatch):
    from scripts import kaggle_native_grid_watch as watch
    import sys

    monkeypatch.setattr(sys, "argv", ["watch", "--kernel-id", watch.KERNEL_ID + "/999",
                                      "--output-dir", str(tmp_path / "output"),
                                      "--report", str(tmp_path / "watch.json")])
    monkeypatch.setattr(watch.subprocess, "run", lambda *_a, **_k: pytest.fail("version suffix reached Kaggle"))
    assert watch.main() == 1
    assert "kernel identity" in json.loads((tmp_path / "watch.json").read_text())["error"]


@pytest.mark.parametrize("preflight", [None, 9, 0])
def test_native_grid_watch_main_requires_successful_preflight(tmp_path, monkeypatch, preflight):
    from scripts import kaggle_native_grid_watch as watch
    from types import SimpleNamespace
    import sys

    output = tmp_path / "fresh-output"
    report = tmp_path / "watch.json"
    monkeypatch.setattr(sys, "argv", ["watch", "--output-dir", str(output), "--report", str(report)])

    def fake_run(command, **_kwargs):
        if command[2] == "status":
            return SimpleNamespace(returncode=0, stdout='status "KernelWorkerStatus.COMPLETE"', stderr="")
        output.mkdir(exist_ok=True)
        mesh = output / "native_grid_mesh.npz"
        np.savez(mesh, vertices=np.zeros((3, 3), dtype=np.float32), faces=np.array([[0, 1, 2]], dtype=np.int32))
        receipt = {"status": "done", "assay_id": watch.EXPECTED_ASSAY_ID,
                   "kernel_id": watch.DECLARED_KERNEL_ID, "kernel_source_sha256": watch.EXPECTED_KERNEL_SOURCE_SHA256,
                   "native_status": "done", "assay_returncode": 0,
                   "expected_input_sha256": watch.EXPECTED_INPUT_SHA256}
        native = {"status": "done", "child_returncode": 0, "expected_sha256": watch.EXPECTED_INPUT_SHA256,
                  "source_commit": watch.EXPECTED_SOURCE_COMMIT, "source_head": watch.EXPECTED_SOURCE_COMMIT,
                  "native_source_heads": watch.EXPECTED_NATIVE_SOURCE_HEADS, "resolution": 1024,
                  "mesh": {"vertices": 3, "faces": 1, "output_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest()},
                  "effective_route": {"cuda_device": "Tesla T4", "convert_path": "/tmp/TRELLIS.2/o-voxel/o_voxel/convert.py",
                                      "extension_path": "/tmp/o_voxel/_C.so"}}
        if preflight is not None:
            native["preflight_returncode"] = preflight
        (output / "native_grid_kernel_receipt.json").write_text(json.dumps(receipt))
        (output / "native_grid_report.json").write_text(json.dumps(native))
        return SimpleNamespace(returncode=0, stdout="downloaded", stderr="")

    monkeypatch.setattr(watch.subprocess, "run", fake_run)
    assert watch.main() == (0 if preflight == 0 else 1)
    saved = json.loads(report.read_text())
    if preflight == 0:
        assert saved["status"] == "done"
        assert saved["mesh_sha256"] == hashlib.sha256((output / "native_grid_mesh.npz").read_bytes()).hexdigest()
    else:
        assert saved["status"] == "failed"
        assert saved["failure_phase"] == "download"
        assert "preflight" not in saved["last_trustworthy_phase"]
    assert (output / "native_grid_mesh.npz").exists(), "reject without destroying retained evidence"


def test_native_grid_runtime_plan_uses_authenticated_native_dependencies(tmp_path):
    from scripts import source_cuda_native_grid_extract as assay

    plan = assay.native_runtime_sources(tmp_path)
    assert plan["trellis"][1] == assay.TRELLIS_COMMIT
    assert plan["flex_gemm"][1] == "6dd94a859c26ee8246888502eada3dd8ad85532e"
    assert plan["cumesh"][1] == "c4ad6125924fcedfd13f0bd61520ca2d24eb7a87"
    assert plan["nvdiffrast"][1] == "253ac4fcea7de5f396371124af597e6cc957bfae"


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
    work_dir = Path(calls[0][calls[0].index("--work-dir") + 1])
    assert not work_dir.is_relative_to(output), "temporary source checkout must not become Kaggle output"


def test_native_bootstrap_installs_required_import_before_child(tmp_path, monkeypatch):
    from scripts import source_cuda_native_grid_extract as assay

    decoder = tmp_path / "decoder.npz"
    decoder.write_bytes(b"saved decoder fixture")
    source_root = tmp_path / "work" / "TRELLIS.2" / "o-voxel"
    source_root.mkdir(parents=True)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            root = Path(kwargs["cwd"])
            return next((commit for _repository, commit, path in assay.native_runtime_sources(tmp_path / "work").values() if path == root), "pinned-o-voxel-submodule")
        if command[:3] == ["git", "status", "--porcelain"]:
            return ""
        if command[:4] == [assay.sys.executable, "-m", "pip", "install"]:
            raise RuntimeError("stop before build")
        return ""

    monkeypatch.setattr(assay, "_run", fake_run)
    assert assay.main([
        "--input-npz", str(decoder),
        "--expected-sha256", hashlib.sha256(decoder.read_bytes()).hexdigest(),
        "--output-npz", str(tmp_path / "mesh.npz"),
        "--report", str(tmp_path / "report.json"),
        "--work-dir", str(tmp_path / "work"),
    ]) == 1
    pip_commands = [command for command in commands if command[:4] == [assay.sys.executable, "-m", "pip", "install"]]
    assert pip_commands and {"ninja", "trimesh", "plyfile"}.issubset(pip_commands[0])


def test_native_grid_preflight_blocks_mesh_call_and_installs_source_backed_o_voxel(tmp_path, monkeypatch):
    from scripts import source_cuda_native_grid_extract as assay
    from types import SimpleNamespace

    decoder = tmp_path / "decoder.npz"
    decoder.write_bytes(b"saved decoder fixture")
    source_root = tmp_path / "work" / "TRELLIS.2" / "o-voxel"
    source_root.mkdir(parents=True)
    commands = []
    children = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            root = Path(kwargs["cwd"])
            return next((commit for _repo, commit, path in assay.native_runtime_sources(tmp_path / "work").values() if path == root), "pinned-o-voxel-submodule")
        return ""

    def fake_child(command, **_kwargs):
        children.append(command)
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(assay, "_run", fake_run)
    monkeypatch.setattr(assay.subprocess, "run", fake_child)
    assert assay.main([
        "--input-npz", str(decoder),
        "--expected-sha256", hashlib.sha256(decoder.read_bytes()).hexdigest(),
        "--output-npz", str(tmp_path / "mesh.npz"),
        "--report", str(tmp_path / "report.json"),
        "--work-dir", str(tmp_path / "work"),
        "--child-report", str(tmp_path / "child.json"),
        "--child-stdout", str(tmp_path / "child.stdout"),
        "--child-stderr", str(tmp_path / "child.stderr"),
    ]) == 1
    assert any(command[-2:] == ["-e", str(source_root)] for command in commands)
    assert len(children) == 1 and "--preflight-only" in children[0]
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["failure_phase"] == "native_import_preflight"
    assert not (tmp_path / "mesh.npz").exists()
