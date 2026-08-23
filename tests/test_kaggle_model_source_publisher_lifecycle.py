from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import scripts.run_kaggle_model_source_publisher_lifecycle as lifecycle
from scripts.prepare_kaggle_model_source_publisher import build_packet
from trellmlx.kaggle_model_source_publisher import (
    ModelSourcePublisherFile,
    ModelSourcePublisherPacket,
    ModelSourcePublisherRepository,
    prepare_publisher_packet,
)


OBJECT_ROOT = Path(__file__).resolve().parents[1]


def _argv(tmp_path: Path, packet: ModelSourcePublisherPacket, suffix: str = "one"):
    return [
        "run_kaggle_model_source_publisher_lifecycle.py",
        "--packet-dir",
        str(packet.output_dir),
        "--download-dir",
        str(tmp_path / f"downloads-{suffix}"),
        "--lifecycle-report",
        str(tmp_path / "reports" / f"lifecycle-{suffix}.json"),
        "--failure-report",
        str(tmp_path / "reports" / f"failure-{suffix}.json"),
        "--attempt-registry-root",
        str(tmp_path / "attempt-registry"),
        "--object-root",
        str(OBJECT_ROOT),
        "--object-commit",
        "a" * 40,
        "--expected-kernel-id",
        packet.kernel_id,
    ]


def _allow_custody(monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "test-token")
    monkeypatch.setattr(
        lifecycle,
        "verify_local_custody",
        lambda _root, commit: {"object_commit": commit, "object_status": "clean"},
    )


def test_lifecycle_rejects_self_consistent_authority_drift_before_push(
    tmp_path,
    monkeypatch,
):
    packet = build_packet(output_dir=tmp_path / "packet", attempt_id="1" * 32)
    first = packet.repositories[0]
    drifted = replace(
        packet,
        repositories=(replace(first, revision="c" * 40), *packet.repositories[1:]),
    )
    prepare_publisher_packet(drifted)
    pushes = []
    _allow_custody(monkeypatch)
    monkeypatch.setattr(
        lifecycle,
        "run_command",
        lambda *args, **kwargs: pushes.append((args, kwargs)),
    )
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, drifted))

    assert lifecycle.main() == 1
    assert pushes == []


def test_lifecycle_rejects_manifest_consistent_runner_substitution_before_claim(
    tmp_path,
    monkeypatch,
):
    packet = build_packet(output_dir=tmp_path / "packet", attempt_id="2" * 32)
    prepare_publisher_packet(packet)
    runner = packet.kernel_dir / packet.code_file
    replacement = runner.read_bytes().replace(b"\n", b"\r\n")
    runner.write_bytes(replacement)
    manifest_path = packet.output_dir / "packet.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runner_sha256"] = hashlib.sha256(replacement).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    pushes = []
    _allow_custody(monkeypatch)
    monkeypatch.setattr(
        lifecycle,
        "run_command",
        lambda *args, **kwargs: pushes.append((args, kwargs)),
    )
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, packet))

    assert lifecycle.main() == 1
    assert pushes == []
    assert list((tmp_path / "attempt-registry").rglob("*.json")) == []


def test_lifecycle_attempt_claim_survives_packet_repreparation(tmp_path, monkeypatch):
    packet = build_packet(output_dir=tmp_path / "packet", attempt_id="3" * 32)
    prepare_publisher_packet(packet)
    pushes = []

    def stop_after_push(command, *, phase, report_path):
        pushes.append((command, phase, report_path))
        raise RuntimeError("stop after push")

    _allow_custody(monkeypatch)
    monkeypatch.setattr(lifecycle, "run_command", stop_after_push)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, packet, "first"))
    assert lifecycle.main() == 1
    owner, slug = packet.kernel_id.split("/", 1)
    claim_path = tmp_path / "attempt-registry" / owner / f"{slug}.json"
    assert claim_path.is_file()
    claim = json.loads(claim_path.read_text())
    assert claim["attempt_id"] == packet.attempt_id
    assert claim["kernel_id"] == packet.kernel_id

    prepare_publisher_packet(packet)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, packet, "second"))
    assert lifecycle.main() == 1
    assert len(pushes) == 1


def test_lifecycle_bootstrap_failure_uses_distinct_failure_sink(
    tmp_path,
    monkeypatch,
):
    packet = build_packet(output_dir=tmp_path / "packet", attempt_id="6" * 32)
    prepare_publisher_packet(packet)
    invalid_parent = tmp_path / "not-a-directory"
    invalid_parent.write_text("occupied")
    failure_path = tmp_path / "reports" / "bootstrap-failure.json"
    argv = _argv(tmp_path, packet)
    argv[argv.index("--lifecycle-report") + 1] = str(
        invalid_parent / "lifecycle.json"
    )
    argv[argv.index("--failure-report") + 1] = str(failure_path)
    _allow_custody(monkeypatch)
    monkeypatch.setattr(sys, "argv", argv)

    assert lifecycle.main() == 1
    failure = json.loads(failure_path.read_text())
    assert failure["status"] == "failed"
    assert failure["current_phase"] == "lifecycle_initialization"
    assert failure["failure_phase"] == "lifecycle_initialization"
    assert failure["last_trustworthy_phase"] is None
    assert failure["terminal_at"]
    assert failure["error_type"] in {"FileExistsError", "NotADirectoryError"}
    assert failure["error_message"]


def test_lifecycle_rejects_registry_owner_symlink_into_packet(tmp_path, monkeypatch):
    packet = build_packet(output_dir=tmp_path / "packet", attempt_id="4" * 32)
    prepare_publisher_packet(packet)
    registry = tmp_path / "attempt-registry"
    registry.mkdir()
    owner = packet.kernel_id.split("/", 1)[0]
    (registry / owner).symlink_to(packet.output_dir, target_is_directory=True)
    argv = _argv(tmp_path, packet)
    argv[argv.index("--attempt-registry-root") + 1] = str(registry)
    pushes = []
    _allow_custody(monkeypatch)
    monkeypatch.setattr(
        lifecycle,
        "run_command",
        lambda *args, **kwargs: pushes.append((args, kwargs)),
    )
    monkeypatch.setattr(sys, "argv", argv)

    assert lifecycle.main() == 1
    assert pushes == []


def _tiny_packet(tmp_path: Path) -> tuple[ModelSourcePublisherPacket, bytes]:
    payload = b"pipeline"
    digest = hashlib.sha256(payload).hexdigest()
    repository = ModelSourcePublisherRepository(
        name="trellis",
        repo_id="operator/trellis",
        revision="d" * 40,
        cache_dir="models--operator--trellis",
        files=(
            ModelSourcePublisherFile(
                coordinate="pipeline.json",
                blob=digest,
                sha256=digest,
                size_bytes=len(payload),
            ),
        ),
    )
    packet = ModelSourcePublisherPacket(
        output_dir=tmp_path / "packet",
        kernel_id="operator/pinned-model-source",
        title="pinned-model-source",
        attempt_id="5" * 32,
        marker=f"runtime/huggingface/{repository.cache_dir}/blobs/{digest}",
        marker_sha256=digest,
        repositories=(repository,),
    )
    return packet, payload


def test_lifecycle_composes_terminal_receipt_without_redownloading_blobs(
    tmp_path,
    monkeypatch,
):
    packet, payload = _tiny_packet(tmp_path)
    prepare_publisher_packet(packet)
    fake_root = tmp_path / "fake-module"
    fake_root.mkdir()
    (fake_root / "huggingface_hub.py").write_text(
        "from pathlib import Path\n"
        "import hashlib\n"
        f"PAYLOAD = bytes.fromhex({payload.hex()!r})\n"
        "def snapshot_download(*, repo_id, revision, cache_dir, allow_patterns):\n"
        "    root = Path(cache_dir) / 'models--operator--trellis'\n"
        "    blob = root / 'blobs' / hashlib.sha256(PAYLOAD).hexdigest()\n"
        "    blob.parent.mkdir(parents=True)\n"
        "    blob.write_bytes(PAYLOAD)\n"
        "    snap = root / 'snapshots' / revision\n"
        "    snap.mkdir(parents=True)\n"
        "    (snap / allow_patterns[0]).symlink_to(blob)\n"
        "    return str(snap)\n"
    )
    remote_working = tmp_path / "remote-working"
    remote_working.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_root)
    env["TRELLIS2MLX_KAGGLE_WORKING_ROOT"] = str(remote_working)
    completed = subprocess.run(
        [sys.executable, str(packet.kernel_dir / packet.code_file)],
        cwd=remote_working,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    remote_receipt = json.loads((remote_working / packet.receipt_name).read_text())
    remote_receipt["effective_working_root"] = "/kaggle/working"
    remote_receipt["effective_cache_root"] = "/kaggle/working/runtime/huggingface"

    def fake_run(command, *, phase, report_path):
        report = {"status": "done", "phase": phase, "command": command}
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report))
        if phase == "kernel_output_download":
            download = Path(command[command.index("-p") + 1])
            (download / packet.receipt_name).write_text(json.dumps(remote_receipt))
        return report

    def fake_status(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f'{packet.kernel_id} has status "complete"\n',
            stderr="",
        )

    _allow_custody(monkeypatch)
    monkeypatch.setattr(lifecycle, "run_command", fake_run)
    monkeypatch.setattr(lifecycle.subprocess, "run", fake_status)
    monkeypatch.setattr(lifecycle, "validate_r11_publisher_authority", lambda _packet: None)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, packet))

    assert lifecycle.main() == 0
    report = json.loads((tmp_path / "reports" / "lifecycle-one.json").read_text())
    assert report["last_trustworthy_phase"] == "live_model_source_publication_admitted"
    assert report["admitted_blob_count"] == 1
    assert list((tmp_path / "downloads-one").iterdir()) == [
        tmp_path / "downloads-one" / packet.receipt_name
    ]
