import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from trellmlx.kaggle_model_source_publisher import (
    load_publisher_packet,
    ModelSourcePublisherError,
    ModelSourcePublisherFile,
    ModelSourcePublisherPacket,
    ModelSourcePublisherRepository,
    prepare_publisher_packet,
    validate_publisher_receipt,
)


def _file(coordinate: str, payload: bytes) -> ModelSourcePublisherFile:
    digest = hashlib.sha256(payload).hexdigest()
    return ModelSourcePublisherFile(
        coordinate=coordinate,
        blob=digest,
        sha256=digest,
        size_bytes=len(payload),
    )


def _packet(tmp_path: Path) -> tuple[ModelSourcePublisherPacket, dict[str, bytes]]:
    payloads = {"pipeline.json": b"pipeline", "decoder.bin": b"decoder"}
    repositories = (
        ModelSourcePublisherRepository(
            name="trellis",
            repo_id="operator/trellis",
            revision="a" * 40,
            cache_dir="models--operator--trellis",
            files=(_file("pipeline.json", payloads["pipeline.json"]),),
        ),
        ModelSourcePublisherRepository(
            name="decoder",
            repo_id="operator/decoder",
            revision="b" * 40,
            cache_dir="models--operator--decoder",
            files=(_file("decoder.bin", payloads["decoder.bin"]),),
        ),
    )
    marker_file = repositories[0].files[0]
    return (
        ModelSourcePublisherPacket(
            output_dir=tmp_path / "packet",
            kernel_id="operator/pinned-model-source",
            title="pinned-model-source",
            attempt_id="1" * 32,
            marker=(
                f"runtime/huggingface/{repositories[0].cache_dir}/"
                f"blobs/{marker_file.blob}"
            ),
            marker_sha256=marker_file.sha256,
            repositories=repositories,
        ),
        payloads,
    )


def _write_fake_huggingface_hub(
    root: Path,
    payloads: dict[str, bytes],
    *,
    omit: str | None = None,
) -> None:
    module = root / "huggingface_hub.py"
    encoded = {key: value.hex() for key, value in payloads.items() if key != omit}
    module.write_text(
        "from pathlib import Path\n"
        f"PAYLOADS = {encoded!r}\n"
        "def snapshot_download(*, repo_id, revision, cache_dir, allow_patterns):\n"
        "    owner, slug = repo_id.split('/', 1)\n"
        "    family = Path(cache_dir) / f'models--{owner}--{slug}'\n"
        "    snapshot = family / 'snapshots' / revision\n"
        "    for coordinate in allow_patterns:\n"
        "        if coordinate not in PAYLOADS:\n"
        "            continue\n"
        "        payload = bytes.fromhex(PAYLOADS[coordinate])\n"
        "        import hashlib\n"
        "        blob = family / 'blobs' / hashlib.sha256(payload).hexdigest()\n"
        "        blob.parent.mkdir(parents=True, exist_ok=True)\n"
        "        blob.write_bytes(payload)\n"
        "        link = snapshot / coordinate\n"
        "        link.parent.mkdir(parents=True, exist_ok=True)\n"
        "        link.symlink_to(blob)\n"
        "    return str(snapshot)\n"
    )


def _run_packet(
    packet: ModelSourcePublisherPacket,
    tmp_path: Path,
    payloads: dict[str, bytes],
    *,
    omit: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    fake_module_root = tmp_path / "fake-module"
    fake_module_root.mkdir()
    _write_fake_huggingface_hub(fake_module_root, payloads, omit=omit)
    working_root = tmp_path / "working"
    working_root.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_module_root)
    env["TRELLIS2MLX_KAGGLE_WORKING_ROOT"] = str(working_root)
    completed = subprocess.run(
        [sys.executable, str(packet.output_dir / "kernel" / packet.code_file)],
        cwd=working_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, working_root


def test_prepare_publisher_packet_binds_private_cpu_internet_route(tmp_path):
    packet, _payloads = _packet(tmp_path)
    prepare_publisher_packet(packet)

    metadata = json.loads(
        (packet.output_dir / "kernel" / "kernel-metadata.json").read_text()
    )
    manifest = json.loads((packet.output_dir / "packet.json").read_text())
    assert metadata == {
        "id": packet.kernel_id,
        "title": packet.title,
        "code_file": packet.code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    assert load_publisher_packet(packet.output_dir) == packet
    assert manifest["attempt_id"] == packet.attempt_id
    assert manifest["repository_count"] == 2
    assert manifest["blob_count"] == 2
    assert manifest["published_blob_bytes"] == len(b"pipelinedecoder")


def test_publisher_runner_writes_minimal_verified_blob_output(tmp_path):
    packet, payloads = _packet(tmp_path)
    prepare_publisher_packet(packet)

    completed, working_root = _run_packet(packet, tmp_path, payloads)

    assert completed.returncode == 0, completed.stderr
    receipt = validate_publisher_receipt(packet, working_root)
    assert receipt["status"] == "completed"
    assert receipt["effective_cache_root"] == str(
        working_root / "runtime" / "huggingface"
    )
    assert receipt["published_blob_bytes"] == len(b"pipelinedecoder")
    assert len(receipt["blobs"]) == 2
    cache_root = working_root / "runtime" / "huggingface"
    assert sorted(
        path.relative_to(cache_root).as_posix()
        for path in cache_root.rglob("*")
        if path.is_file()
    ) == sorted(
        f"{repository.cache_dir}/blobs/{file.blob}"
        for repository in packet.repositories
        for file in repository.files
    )


def test_publisher_runner_preserves_failure_receipt_for_missing_blob(tmp_path):
    packet, payloads = _packet(tmp_path)
    prepare_publisher_packet(packet)

    completed, working_root = _run_packet(
        packet,
        tmp_path,
        payloads,
        omit="decoder.bin",
    )

    assert completed.returncode == 1
    receipt = json.loads((working_root / packet.receipt_name).read_text())
    assert receipt["status"] == "failed"
    assert receipt["failure_phase"] == "blob_validation"
    assert receipt["attempt_id"] == packet.attempt_id
    assert receipt["packet_contract_sha256"]
    with pytest.raises(ModelSourcePublisherError, match="not completed"):
        validate_publisher_receipt(packet, working_root)


def test_prepare_publisher_packet_rejects_title_route_mismatch(tmp_path):
    packet, _payloads = _packet(tmp_path)
    packet = ModelSourcePublisherPacket(
        **{**packet.__dict__, "title": "Human Friendly Title"}
    )

    with pytest.raises(ModelSourcePublisherError, match="exact kernel slug"):
        prepare_publisher_packet(packet)

    assert not packet.output_dir.exists()
