import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

import scripts.run_kaggle_model_source_conformance_lifecycle as lifecycle
from trellmlx.kaggle_model_source_conformance import (
    build_kernel_push_command,
    ModelSourceBlob,
    ModelSourceConformanceError,
    ModelSourceConformancePacket,
    load_prepared_packet,
    prepare_packet,
    validate_downloaded_receipt,
)
from scripts.prepare_kaggle_model_source_conformance import (
    build_packet,
    validate_r11_authority,
)
from scripts.run_kaggle_model_source_conformance_lifecycle import (
    normalize_kernel_status,
    prepare_download_dir,
    select_failure_identity,
)


def _blob(relative_path: str, payload: bytes) -> ModelSourceBlob:
    import hashlib

    return ModelSourceBlob(
        relative_path=relative_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _packet(tmp_path: Path) -> tuple[ModelSourceConformancePacket, dict[str, bytes]]:
    payloads = {
        "models--microsoft--TRELLIS.2-4B/blobs/marker": b"pipeline",
        "models--microsoft--TRELLIS-image-large/blobs/weights": b"weights",
    }
    blobs = tuple(_blob(path, payload) for path, payload in payloads.items())
    return (
        ModelSourceConformancePacket(
            output_dir=tmp_path / "packet",
            kernel_id="operator/source-mount-conformance",
            title="Source Mount Conformance",
            source_kernel="operator/pinned-model-output",
            marker="runtime/huggingface/models--microsoft--TRELLIS.2-4B/blobs/marker",
            marker_sha256=blobs[0].sha256,
            blobs=blobs,
            attempt_id="1" * 32,
        ),
        payloads,
    )


def _run_prepared(
    packet: ModelSourceConformancePacket,
    *,
    input_root: Path,
    working_root: Path,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "TRELLIS2MLX_KAGGLE_INPUT_ROOT": str(input_root),
            "TRELLIS2MLX_KAGGLE_WORKING_ROOT": str(working_root),
        }
    )
    return subprocess.run(
        [sys.executable, str(packet.kernel_dir / packet.code_file)],
        cwd=working_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _materialize_mount(
    input_root: Path,
    payloads: dict[str, bytes],
    *,
    name: str = "mounted-source",
) -> Path:
    mount = input_root / name
    blob_root = mount / "runtime" / "huggingface"
    for relative_path, payload in payloads.items():
        path = blob_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return mount


def test_prepare_packet_binds_cpu_private_kernel_source(tmp_path):
    packet, _payloads = _packet(tmp_path)

    prepared = prepare_packet(packet)
    metadata = json.loads((prepared.kernel_dir / "kernel-metadata.json").read_text())
    manifest = json.loads((prepared.output_dir / "packet.json").read_text())

    assert metadata == {
        "id": packet.kernel_id,
        "title": packet.title,
        "code_file": packet.code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "false",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [packet.source_kernel],
        "model_sources": [],
    }
    assert manifest["source_kernel"] == packet.source_kernel
    assert manifest["requested_accelerator"] is None
    assert manifest["mounted_blob_bytes"] == len(b"pipelineweights")
    assert len(manifest["blobs"]) == 2
    assert load_prepared_packet(packet.output_dir) == packet
    assert build_kernel_push_command(packet) == [
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(packet.kernel_dir),
    ]


def test_real_packet_uses_exact_r11_model_manifest(tmp_path):
    packet = build_packet(
        output_dir=tmp_path / "real-packet",
        kernel_id="noahboo/t2mlx-model-source-conformance-r11",
        source_kernel="noahboo/t2mlx-native-pixal9-t4-f6446f9-r10",
        title="Trellis2MLX R11 Model Source Conformance",
    )

    assert len(packet.blobs) == 17
    assert packet.mounted_blob_bytes == 14_967_470_615
    assert packet.marker == (
        "runtime/huggingface/models--microsoft--TRELLIS.2-4B/"
        "blobs/f5ec14c7f71b3d7f2cb0221c5f568a6871dc5e90"
    )
    assert packet.marker_sha256 == (
        "222c359ab1ed9bc6735a640a34f95d47f8681b9bc4aaa101bfb80274676253c6"
    )
    validate_r11_authority(packet)


def _mutate_real_packet_authority(
    packet: ModelSourceConformancePacket,
    mutation: str,
) -> ModelSourceConformancePacket:
    blobs = list(packet.blobs)
    marker_relative = packet.marker.removeprefix("runtime/huggingface/")
    nonmarker_index = next(
        index
        for index, blob in enumerate(blobs)
        if blob.relative_path != marker_relative
    )
    if mutation == "source_kernel":
        return replace(packet, source_kernel="attacker/substituted-source")
    if mutation == "blob_digest":
        blobs[nonmarker_index] = replace(blobs[nonmarker_index], sha256="0" * 64)
        return replace(packet, blobs=tuple(blobs))
    if mutation == "blob_size":
        blobs[nonmarker_index] = replace(
            blobs[nonmarker_index],
            size_bytes=blobs[nonmarker_index].size_bytes + 1,
        )
        return replace(packet, blobs=tuple(blobs))
    if mutation == "foreign_blob":
        blobs[nonmarker_index] = replace(
            blobs[nonmarker_index],
            relative_path="foreign-model/blobs/substituted",
        )
        return replace(packet, blobs=tuple(blobs))
    if mutation == "marker":
        replacement = blobs[nonmarker_index]
        return replace(
            packet,
            marker=f"runtime/huggingface/{replacement.relative_path}",
            marker_sha256=replacement.sha256,
        )
    raise AssertionError(f"unknown authority mutation: {mutation}")


@pytest.mark.parametrize(
    "mutation",
    ("source_kernel", "blob_digest", "blob_size", "foreign_blob", "marker"),
)
def test_lifecycle_rejects_self_consistent_authority_substitution_before_push(
    tmp_path,
    monkeypatch,
    mutation,
):
    packet = build_packet(
        output_dir=tmp_path / "packet",
        kernel_id="noahboo/t2mlx-model-source-conformance-r11",
        source_kernel="noahboo/t2mlx-native-pixal9-t4-f6446f9-r10",
        title="Trellis2MLX R11 Model Source Conformance",
    )
    packet = _mutate_real_packet_authority(packet, mutation)
    prepare_packet(packet)
    pushes = []

    def reject_push(command, *, phase, report_path):
        pushes.append((command, phase, report_path))
        raise AssertionError("substituted authority reached kernel push")

    monkeypatch.setenv("KAGGLE_API_TOKEN", "test-token")
    monkeypatch.setattr(lifecycle, "run_command", reject_push)
    monkeypatch.setattr(
        lifecycle,
        "verify_local_custody",
        lambda _root, commit: {"object_commit": commit, "object_status": "clean"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_kaggle_model_source_conformance_lifecycle.py",
            "--packet-dir",
            str(packet.output_dir),
            "--download-dir",
            str(tmp_path / "downloads"),
            "--lifecycle-report",
            str(tmp_path / "reports" / "lifecycle.json"),
            "--failure-report",
            str(tmp_path / "reports" / "failure.json"),
            "--object-root",
            str(tmp_path),
            "--object-commit",
            "a" * 40,
            "--expected-kernel-id",
            packet.kernel_id,
        ],
    )

    assert lifecycle.main() == 1
    assert pushes == []


def test_lifecycle_consumes_each_prepared_attempt_before_push(tmp_path, monkeypatch):
    packet = build_packet(
        output_dir=tmp_path / "packet",
        kernel_id="noahboo/t2mlx-model-source-conformance-r11",
        source_kernel="noahboo/t2mlx-native-pixal9-t4-f6446f9-r10",
        title="Trellis2MLX R11 Model Source Conformance",
    )
    prepare_packet(packet)
    pushes = []

    def stop_after_push(command, *, phase, report_path):
        pushes.append((command, phase, report_path))
        raise RuntimeError("stop after observing push")

    monkeypatch.setenv("KAGGLE_API_TOKEN", "test-token")
    monkeypatch.setattr(lifecycle, "run_command", stop_after_push)
    monkeypatch.setattr(
        lifecycle,
        "verify_local_custody",
        lambda _root, commit: {"object_commit": commit, "object_status": "clean"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_kaggle_model_source_conformance_lifecycle.py",
            "--packet-dir",
            str(packet.output_dir),
            "--download-dir",
            str(tmp_path / "downloads"),
            "--lifecycle-report",
            str(tmp_path / "reports" / "lifecycle.json"),
            "--failure-report",
            str(tmp_path / "reports" / "failure.json"),
            "--object-root",
            str(tmp_path),
            "--object-commit",
            "a" * 40,
            "--expected-kernel-id",
            packet.kernel_id,
        ],
    )

    assert lifecycle.main() == 1
    assert lifecycle.main() == 1
    assert len(pushes) == 1


@pytest.mark.parametrize(
    ("raw", "normalized"),
    (
        ("running", "running"),
        ("KernelWorkerStatus.RUNNING", "running"),
        ("complete", "complete"),
        ("error", "error"),
    ),
)
def test_lifecycle_normalizes_kernel_status(raw, normalized):
    assert normalize_kernel_status(raw) == normalized


def test_lifecycle_preserves_remote_terminal_failure_over_recovery_error():
    lifecycle = {
        "primary_failure": {
            "failure_phase": "kernel_terminal",
            "error_type": "RemoteKernelTerminalError",
            "error_message": "kernel terminal status is not complete: error",
        }
    }

    assert select_failure_identity(lifecycle, RuntimeError("download failed")) == (
        "kernel_terminal",
        "RemoteKernelTerminalError",
        "kernel terminal status is not complete: error",
    )


def test_prepare_download_dir_rejects_stale_outputs(tmp_path):
    download = tmp_path / "downloads"
    download.mkdir()
    (download / "stale.json").write_text("{}")

    with pytest.raises(ModelSourceConformanceError, match="not empty"):
        prepare_download_dir(download)


def test_lifecycle_composes_terminal_download_and_semantic_admission(
    tmp_path,
    monkeypatch,
):
    packet, payloads = _packet(tmp_path)
    prepare_packet(packet)
    input_root = tmp_path / "input"
    remote_working = tmp_path / "remote-working"
    input_root.mkdir()
    remote_working.mkdir()
    _materialize_mount(input_root, payloads)
    completed = _run_prepared(
        packet,
        input_root=input_root,
        working_root=remote_working,
    )
    assert completed.returncode == 0
    remote_receipt = json.loads((remote_working / packet.receipt_name).read_text())
    remote_receipt.update(
        {
            "effective_input_root": "/kaggle/input",
            "effective_working_root": "/kaggle/working",
            "effective_mount_root": "/kaggle/input/mounted-source",
            "effective_blob_root": (
                "/kaggle/input/mounted-source/runtime/huggingface"
            ),
        }
    )

    def fake_run_command(command, *, phase, report_path):
        report = {
            "schema": "test.command.v1",
            "phase": phase,
            "command": list(command),
            "status": "done",
            "failure_phase": None,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        }
        if phase == "kernel_output_download":
            download_dir = Path(command[command.index("-p") + 1])
            (download_dir / packet.receipt_name).write_text(json.dumps(remote_receipt))
        Path(report_path).write_text(json.dumps(report))
        return report

    def fake_status(command, **_kwargs):
        assert command == ["kaggle", "kernels", "status", packet.kernel_id]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f'{packet.kernel_id} has status "complete"\n',
            stderr="",
        )

    lifecycle_report = tmp_path / "reports" / "lifecycle.json"
    failure_report = tmp_path / "reports" / "failure.json"
    download_dir = tmp_path / "downloads"
    monkeypatch.setenv("KAGGLE_API_TOKEN", "test-token")
    monkeypatch.setattr(lifecycle, "run_command", fake_run_command)
    monkeypatch.setattr(lifecycle.subprocess, "run", fake_status)
    monkeypatch.setattr(
        lifecycle,
        "verify_local_custody",
        lambda _root, commit: {"object_commit": commit, "object_status": "clean"},
    )
    monkeypatch.setattr(lifecycle, "validate_r11_authority", lambda _packet: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_kaggle_model_source_conformance_lifecycle.py",
            "--packet-dir",
            str(packet.output_dir),
            "--download-dir",
            str(download_dir),
            "--lifecycle-report",
            str(lifecycle_report),
            "--failure-report",
            str(failure_report),
            "--object-root",
            str(tmp_path),
            "--object-commit",
            "a" * 40,
            "--expected-kernel-id",
            packet.kernel_id,
        ],
    )

    assert lifecycle.main() == 0
    report = json.loads(lifecycle_report.read_text())
    assert report["status"] == "completed"
    assert report["last_trustworthy_phase"] == (
        "live_model_source_conformance_admitted"
    )
    assert report["terminal_kernel_status"] == "complete"
    assert report["admitted_blob_count"] == 2
    assert report["writable_model_bytes"] == 0
    assert not failure_report.exists()


def test_runner_preserves_failure_receipt_when_source_mount_is_missing(tmp_path):
    packet, _payloads = _packet(tmp_path)
    prepare_packet(packet)
    input_root = tmp_path / "input"
    working_root = tmp_path / "working"
    input_root.mkdir()
    working_root.mkdir()

    completed = _run_prepared(
        packet,
        input_root=input_root,
        working_root=working_root,
    )
    receipt = json.loads((working_root / packet.receipt_name).read_text())

    assert completed.returncode != 0
    assert receipt["status"] == "failed"
    assert receipt["failure_phase"] == "source_mount"
    assert receipt["candidate_count"] == 0
    assert receipt["requested_source_kernel"] == packet.source_kernel


def test_lifecycle_rejects_prior_remote_receipt_after_terminal_complete(
    tmp_path,
    monkeypatch,
):
    packet, payloads = _packet(tmp_path)
    prepare_packet(packet)
    input_root = tmp_path / "input"
    remote_working = tmp_path / "remote-working"
    input_root.mkdir()
    remote_working.mkdir()
    _materialize_mount(input_root, payloads)
    completed = _run_prepared(
        packet,
        input_root=input_root,
        working_root=remote_working,
    )
    assert completed.returncode == 0
    prior_receipt = json.loads((remote_working / packet.receipt_name).read_text())
    prior_receipt.update(
        {
            "attempt_id": "0" * 32,
            "effective_input_root": "/kaggle/input",
            "effective_working_root": "/kaggle/working",
            "effective_mount_root": "/kaggle/input/mounted-source",
            "effective_blob_root": (
                "/kaggle/input/mounted-source/runtime/huggingface"
            ),
        }
    )

    def fake_run_command(command, *, phase, report_path):
        if phase == "kernel_output_download":
            download_dir = Path(command[command.index("-p") + 1])
            (download_dir / packet.receipt_name).write_text(json.dumps(prior_receipt))
        report = {
            "schema": "test.command.v1",
            "phase": phase,
            "command": list(command),
            "status": "done",
            "failure_phase": None,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        }
        Path(report_path).write_text(json.dumps(report))
        return report

    monkeypatch.setenv("KAGGLE_API_TOKEN", "test-token")
    monkeypatch.setattr(lifecycle, "run_command", fake_run_command)
    monkeypatch.setattr(
        lifecycle.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=f'{packet.kernel_id} has status "complete"\n',
            stderr="",
        ),
    )
    monkeypatch.setattr(
        lifecycle,
        "verify_local_custody",
        lambda _root, commit: {"object_commit": commit, "object_status": "clean"},
    )
    monkeypatch.setattr(lifecycle, "validate_r11_authority", lambda _packet: None)
    lifecycle_report = tmp_path / "reports" / "lifecycle.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_kaggle_model_source_conformance_lifecycle.py",
            "--packet-dir",
            str(packet.output_dir),
            "--download-dir",
            str(tmp_path / "downloads"),
            "--lifecycle-report",
            str(lifecycle_report),
            "--failure-report",
            str(tmp_path / "reports" / "failure.json"),
            "--object-root",
            str(tmp_path),
            "--object-commit",
            "a" * 40,
            "--expected-kernel-id",
            packet.kernel_id,
        ],
    )

    assert lifecycle.main() == 1
    report = json.loads(lifecycle_report.read_text())
    assert report["failure_phase"] == "semantic_validation"
    assert "attempt identity mismatch" in report["error_message"]


def test_runner_rejects_ambiguous_marker_mounts(tmp_path):
    packet, payloads = _packet(tmp_path)
    prepare_packet(packet)
    input_root = tmp_path / "input"
    working_root = tmp_path / "working"
    input_root.mkdir()
    working_root.mkdir()
    _materialize_mount(input_root, payloads, name="first")
    _materialize_mount(input_root, payloads, name="second")

    completed = _run_prepared(
        packet,
        input_root=input_root,
        working_root=working_root,
    )
    receipt = json.loads((working_root / packet.receipt_name).read_text())

    assert completed.returncode != 0
    assert receipt["status"] == "failed"
    assert receipt["failure_phase"] == "source_mount"
    assert receipt["candidate_count"] == 2


def test_runner_hashes_mount_without_writable_model_payload(tmp_path):
    packet, payloads = _packet(tmp_path)
    prepare_packet(packet)
    input_root = tmp_path / "input"
    working_root = tmp_path / "working"
    input_root.mkdir()
    working_root.mkdir()
    mount = _materialize_mount(input_root, payloads)
    (working_root / "preexisting.txt").write_text("keep")

    completed = _run_prepared(
        packet,
        input_root=input_root,
        working_root=working_root,
    )
    receipt_path = working_root / packet.receipt_name
    receipt = json.loads(receipt_path.read_text())

    assert completed.returncode == 0, completed.stderr
    assert receipt["status"] == "completed"
    assert receipt["failure_phase"] is None
    assert receipt["candidate_count"] == 1
    assert receipt["effective_mount_root"] == str(mount)
    assert receipt["effective_blob_root"] == str(mount / "runtime" / "huggingface")
    assert receipt["marker"] == packet.marker
    assert receipt["marker_sha256"] == packet.marker_sha256
    assert receipt["mounted_blob_bytes"] == len(b"pipelineweights")
    assert receipt["writable_model_bytes"] == 0
    assert receipt["working_tree_before"] == receipt["working_tree_after"]
    assert receipt["nvidia_device_nodes"] == []
    assert [record["relative_path"] for record in receipt["blobs"]] == sorted(payloads)
    assert set(path.name for path in working_root.iterdir()) == {
        "preexisting.txt",
        packet.receipt_name,
    }


def test_runner_rejects_preexisting_renamed_model_blob_in_writable_space(tmp_path):
    packet, payloads = _packet(tmp_path)
    prepare_packet(packet)
    input_root = tmp_path / "input"
    working_root = tmp_path / "working"
    input_root.mkdir()
    working_root.mkdir()
    _materialize_mount(input_root, payloads)
    (working_root / "renamed-copy.bin").write_bytes(b"weights")

    completed = _run_prepared(
        packet,
        input_root=input_root,
        working_root=working_root,
    )
    receipt = json.loads((working_root / packet.receipt_name).read_text())

    assert completed.returncode != 0
    assert receipt["status"] == "failed"
    assert receipt["failure_phase"] == "writable_model_payload"
    assert receipt["writable_model_bytes"] == len(b"weights")
    assert receipt["writable_model_matches"] == [
        {
            "relative_path": "renamed-copy.bin",
            "sha256": _blob(
                "models--microsoft--TRELLIS-image-large/blobs/weights",
                b"weights",
            ).sha256,
            "size_bytes": len(b"weights"),
        }
    ]


def test_validate_downloaded_receipt_rejects_false_closure_classes(tmp_path):
    packet, payloads = _packet(tmp_path)
    prepare_packet(packet)
    input_root = tmp_path / "input"
    working_root = tmp_path / "working"
    output_root = tmp_path / "download"
    input_root.mkdir()
    working_root.mkdir()
    output_root.mkdir()
    _materialize_mount(input_root, payloads)
    completed = _run_prepared(
        packet,
        input_root=input_root,
        working_root=working_root,
    )
    assert completed.returncode == 0
    receipt = json.loads((working_root / packet.receipt_name).read_text())
    receipt["effective_input_root"] = "/kaggle/input"
    receipt["effective_working_root"] = "/kaggle/working"
    receipt["effective_mount_root"] = "/kaggle/input/mounted-source"
    receipt["effective_blob_root"] = "/kaggle/input/mounted-source/runtime/huggingface"
    receipt_path = output_root / packet.receipt_name
    receipt_path.write_text(json.dumps(receipt))

    admitted = validate_downloaded_receipt(packet, output_root)
    assert admitted["status"] == "completed"

    mutations = (
        ("status", "failed", "status"),
        ("effective_mount_root", "/tmp/source", "mount root"),
        ("effective_blob_root", "/kaggle/input/other/runtime/huggingface", "blob root"),
        ("candidate_count", 2, "candidate"),
        ("marker_sha256", "0" * 64, "marker"),
        ("mounted_blob_bytes", 0, "mounted blob"),
        ("writable_model_bytes", 1, "writable model"),
        ("nvidia_device_nodes", ["/dev/nvidia0"], "CPU route"),
    )
    for field, value, match in mutations:
        changed = dict(receipt)
        changed[field] = value
        receipt_path.write_text(json.dumps(changed))
        with pytest.raises(ModelSourceConformanceError, match=match):
            validate_downloaded_receipt(packet, output_root)

    changed = dict(receipt)
    changed["blobs"] = [dict(record) for record in receipt["blobs"]]
    changed["blobs"][0]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(changed))
    with pytest.raises(ModelSourceConformanceError, match="blob manifest"):
        validate_downloaded_receipt(packet, output_root)

    stale_attempt = dict(receipt)
    stale_attempt["attempt_id"] = "0" * 32
    receipt_path.write_text(json.dumps(stale_attempt))
    with pytest.raises(ModelSourceConformanceError, match="attempt|contract"):
        validate_downloaded_receipt(packet, output_root)

    stale_contract = dict(receipt)
    stale_contract["packet_contract_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(stale_contract))
    with pytest.raises(ModelSourceConformanceError, match="contract"):
        validate_downloaded_receipt(packet, output_root)

    malformed_receipts = []
    missing_snapshots = dict(receipt)
    missing_snapshots.pop("working_tree_before")
    missing_snapshots.pop("working_tree_after")
    malformed_receipts.append(missing_snapshots)
    for missing_field in ("working_tree_before", "working_tree_after"):
        missing = dict(receipt)
        missing.pop(missing_field)
        malformed_receipts.append(missing)
    malformed_receipts.extend(
        [
            {**receipt, "candidate_count": True},
            {**receipt, "mounted_blob_bytes": True},
            {**receipt, "writable_model_bytes": False},
            {
                **receipt,
                "working_tree_before": {},
                "working_tree_after": {},
            },
            {**receipt, "working_tree_before": None, "working_tree_after": None},
            {**receipt, "working_tree_before": "same", "working_tree_after": "same"},
            {
                **receipt,
                "working_tree_before": [{"relative_path": "partial"}],
                "working_tree_after": [{"relative_path": "partial"}],
            },
            {**receipt, "writable_model_matches": [{"relative_path": "partial"}]},
            {**receipt, "nvidia_device_nodes": [False]},
        ]
    )
    for malformed in malformed_receipts:
        receipt_path.write_text(json.dumps(malformed))
        with pytest.raises(ModelSourceConformanceError, match="schema|field|type"):
            validate_downloaded_receipt(packet, output_root)
