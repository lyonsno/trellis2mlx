"""Private CPU Kaggle publisher for pinned Hugging Face model blobs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any


SCHEMA = "trellis2mlx.kaggle_model_source_publisher.v1"
PACKET_SCHEMA = "trellis2mlx.kaggle_model_source_publisher.packet.v1"
_REF_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_HEX_40_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_HEX_64_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ModelSourcePublisherError(ValueError):
    """Raised when publisher authority or evidence is incomplete."""


@dataclass(frozen=True, order=True)
class ModelSourcePublisherFile:
    coordinate: str
    blob: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, order=True)
class ModelSourcePublisherRepository:
    name: str
    repo_id: str
    revision: str
    cache_dir: str
    files: tuple[ModelSourcePublisherFile, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(sorted(self.files)))


@dataclass(frozen=True)
class ModelSourcePublisherPacket:
    output_dir: Path
    kernel_id: str
    title: str
    attempt_id: str
    marker: str
    marker_sha256: str
    repositories: tuple[ModelSourcePublisherRepository, ...]
    code_file: str = "run_model_source_publisher.py"
    receipt_name: str = "model_source_publisher.json"

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "repositories", tuple(sorted(self.repositories)))

    @property
    def kernel_dir(self) -> Path:
        return self.output_dir / "kernel"

    @property
    def published_blob_bytes(self) -> int:
        return sum(
            file.size_bytes
            for repository in self.repositories
            for file in repository.files
        )

    @property
    def blob_count(self) -> int:
        return sum(len(repository.files) for repository in self.repositories)

    @property
    def packet_contract_sha256(self) -> str:
        encoded = json.dumps(
            _packet_contract(self), separators=(",", ":"), sort_keys=True
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_publisher_packet(
    packet: ModelSourcePublisherPacket,
) -> ModelSourcePublisherPacket:
    """Write a private internet-enabled CPU Kaggle script-kernel packet."""

    _validate_packet(packet)
    if packet.output_dir.exists():
        shutil.rmtree(packet.output_dir)
    packet.kernel_dir.mkdir(parents=True)
    runner_path = packet.kernel_dir / packet.code_file
    runner_path.write_text(render_publisher_runner(packet))
    metadata = _kernel_metadata(packet)
    _write_json(packet.kernel_dir / "kernel-metadata.json", metadata)
    manifest = _packet_manifest(packet)
    manifest["runner_sha256"] = sha256_file(runner_path)
    manifest["kernel_metadata"] = metadata
    _write_json(packet.output_dir / "packet.json", manifest)
    return packet


def load_publisher_packet(output_dir: Path) -> ModelSourcePublisherPacket:
    """Reload and authenticate a prepared publisher packet."""

    output_dir = Path(output_dir)
    manifest_path = output_dir / "packet.json"
    if not manifest_path.is_file():
        raise ModelSourcePublisherError(f"missing packet manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelSourcePublisherError(f"invalid packet manifest: {exc}") from exc
    if manifest.get("schema") != PACKET_SCHEMA:
        raise ModelSourcePublisherError("packet manifest schema is invalid")
    packet = ModelSourcePublisherPacket(
        output_dir=output_dir,
        kernel_id=manifest["kernel_id"],
        title=manifest["title"],
        attempt_id=manifest["attempt_id"],
        marker=manifest["marker"],
        marker_sha256=manifest["marker_sha256"],
        repositories=tuple(
            ModelSourcePublisherRepository(
                name=record["name"],
                repo_id=record["repo_id"],
                revision=record["revision"],
                cache_dir=record["cache_dir"],
                files=tuple(
                    ModelSourcePublisherFile(**file) for file in record["files"]
                ),
            )
            for record in manifest["repositories"]
        ),
        code_file=manifest["code_file"],
        receipt_name=manifest["receipt_name"],
    )
    _validate_packet(packet)
    expected = _packet_manifest(packet)
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ModelSourcePublisherError(
                f"prepared packet manifest {field} mismatch"
            )
    metadata_path = packet.kernel_dir / "kernel-metadata.json"
    runner_path = packet.kernel_dir / packet.code_file
    if not metadata_path.is_file() or not runner_path.is_file():
        raise ModelSourcePublisherError("prepared kernel packet is incomplete")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelSourcePublisherError(f"invalid kernel metadata: {exc}") from exc
    if metadata != _kernel_metadata(packet) or manifest.get("kernel_metadata") != metadata:
        raise ModelSourcePublisherError("prepared kernel metadata mismatch")
    if manifest.get("runner_sha256") != sha256_file(runner_path):
        raise ModelSourcePublisherError("prepared runner digest mismatch")
    return packet


def render_publisher_runner(packet: ModelSourcePublisherPacket) -> str:
    """Render exact runner bytes for this authenticated publisher packet."""

    _validate_packet(packet)
    config = {
        "schema": SCHEMA,
        "attempt_id": packet.attempt_id,
        "packet_contract_sha256": packet.packet_contract_sha256,
        "requested_accelerator": None,
        "marker": packet.marker,
        "marker_sha256": packet.marker_sha256,
        "published_blob_bytes": packet.published_blob_bytes,
        "repositories": _repository_records(packet),
        "receipt_name": packet.receipt_name,
    }
    return _RUNNER_TEMPLATE.replace(
        "__CONFIG_JSON__", repr(json.dumps(config, sort_keys=True))
    )


def build_kernel_push_command(packet: ModelSourcePublisherPacket) -> list[str]:
    return ["kaggle", "kernels", "push", "-p", str(packet.kernel_dir)]


def build_kernel_status_command(packet: ModelSourcePublisherPacket) -> list[str]:
    return ["kaggle", "kernels", "status", packet.kernel_id]


def build_kernel_output_command(
    packet: ModelSourcePublisherPacket,
    output_dir: Path,
) -> list[str]:
    return [
        "kaggle",
        "kernels",
        "output",
        packet.kernel_id,
        "-p",
        str(Path(output_dir)),
        "-o",
        "--file-pattern",
        rf"\A{re.escape(packet.receipt_name)}\Z",
        "--page-size",
        "100",
    ]


def validate_publisher_receipt(
    packet: ModelSourcePublisherPacket,
    output_dir: Path,
) -> dict[str, Any]:
    """Admit a publisher result only when receipt and blob tree agree."""

    _validate_packet(packet)
    output_dir = Path(output_dir)
    receipt_path = output_dir / packet.receipt_name
    if not receipt_path.is_file() or receipt_path.stat().st_size <= 0:
        raise ModelSourcePublisherError("publisher receipt is missing or blank")
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelSourcePublisherError(f"publisher receipt is invalid: {exc}") from exc
    _validate_receipt_identity(
        packet,
        receipt,
        expected_working_root=str(output_dir),
        expected_cache_root=str(output_dir / "runtime" / "huggingface"),
    )
    cache_root = output_dir / "runtime" / "huggingface"
    expected_blobs = _blob_records(packet)
    actual_files = (
        sorted(
            path.relative_to(cache_root).as_posix()
            for path in cache_root.rglob("*")
            if path.is_file()
        )
        if cache_root.is_dir()
        else []
    )
    expected_files = [record["relative_path"] for record in expected_blobs]
    if actual_files != expected_files:
        raise ModelSourcePublisherError("published tree is not the exact blob manifest")
    for record in expected_blobs:
        path = cache_root / record["relative_path"]
        if (
            path.stat().st_size != record["size_bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ModelSourcePublisherError(
                f"published blob identity mismatch: {record['relative_path']}"
            )
    return receipt


def validate_downloaded_publisher_receipt(
    packet: ModelSourcePublisherPacket,
    output_dir: Path,
) -> dict[str, Any]:
    """Admit the small remote receipt without redownloading the 15 GB payload."""

    _validate_packet(packet)
    receipt_path = Path(output_dir) / packet.receipt_name
    if not receipt_path.is_file() or receipt_path.stat().st_size <= 0:
        raise ModelSourcePublisherError("downloaded publisher receipt is missing or blank")
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelSourcePublisherError(
            f"downloaded publisher receipt is invalid: {exc}"
        ) from exc
    _validate_receipt_identity(
        packet,
        receipt,
        expected_working_root="/kaggle/working",
        expected_cache_root="/kaggle/working/runtime/huggingface",
    )
    return receipt


def _validate_receipt_identity(
    packet: ModelSourcePublisherPacket,
    receipt: Any,
    *,
    expected_working_root: str,
    expected_cache_root: str,
) -> None:
    if type(receipt) is not dict or receipt.get("schema") != SCHEMA:
        raise ModelSourcePublisherError("publisher receipt schema is invalid")
    if (
        receipt.get("status") != "completed"
        or receipt.get("failure_phase") is not None
    ):
        raise ModelSourcePublisherError("publisher receipt status is not completed")
    checks = {
        "attempt_id": packet.attempt_id,
        "packet_contract_sha256": packet.packet_contract_sha256,
        "requested_accelerator": None,
        "marker": packet.marker,
        "marker_sha256": packet.marker_sha256,
        "repository_count": len(packet.repositories),
        "blob_count": packet.blob_count,
        "published_blob_bytes": packet.published_blob_bytes,
        "repositories": _repository_records(packet),
        "nvidia_device_nodes": [],
        "promotion_mode": "same-filesystem-hardlink",
        "huggingface_hub_disable_xet": "1",
    }
    for field, value in checks.items():
        if receipt.get(field) != value:
            raise ModelSourcePublisherError(f"publisher receipt {field} mismatch")
    if receipt.get("effective_working_root") != expected_working_root:
        raise ModelSourcePublisherError("publisher receipt working root mismatch")
    if receipt.get("effective_cache_root") != expected_cache_root:
        raise ModelSourcePublisherError("publisher receipt cache root mismatch")
    expected_blobs = _blob_records(packet)
    if receipt.get("blobs") != expected_blobs:
        raise ModelSourcePublisherError("publisher receipt blob manifest mismatch")


def _safe_relative(value: str, field: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
        raise ModelSourcePublisherError(f"{field} is unsafe")
    return path


def _validate_packet(packet: ModelSourcePublisherPacket) -> None:
    if not _REF_PATTERN.fullmatch(packet.kernel_id):
        raise ModelSourcePublisherError("kernel_id must be an owner/slug reference")
    slug = packet.kernel_id.split("/", 1)[1]
    if packet.title != slug:
        raise ModelSourcePublisherError(
            "title must equal the exact kernel slug so Kaggle preserves route identity"
        )
    if not re.fullmatch(r"[0-9a-f]{32}", packet.attempt_id):
        raise ModelSourcePublisherError("attempt_id must be 32 lowercase hex characters")
    if PurePosixPath(packet.code_file).name != packet.code_file:
        raise ModelSourcePublisherError("code_file must be a filename")
    if PurePosixPath(packet.receipt_name).name != packet.receipt_name:
        raise ModelSourcePublisherError("receipt_name must be a filename")
    if not packet.repositories:
        raise ModelSourcePublisherError("at least one repository is required")
    names: set[str] = set()
    cache_dirs: set[str] = set()
    blob_paths: set[str] = set()
    marker_matches = []
    marker_relative = _safe_relative(packet.marker, "marker")
    prefix = PurePosixPath("runtime/huggingface")
    try:
        marker_blob = marker_relative.relative_to(prefix).as_posix()
    except ValueError as exc:
        raise ModelSourcePublisherError(
            "marker must be beneath runtime/huggingface"
        ) from exc
    if not _HEX_64_PATTERN.fullmatch(packet.marker_sha256):
        raise ModelSourcePublisherError("marker_sha256 is invalid")
    for repository in packet.repositories:
        if not repository.name or repository.name in names:
            raise ModelSourcePublisherError("repository name is blank or duplicated")
        names.add(repository.name)
        if not _REF_PATTERN.fullmatch(repository.repo_id):
            raise ModelSourcePublisherError("repository repo_id is invalid")
        if not _HEX_40_PATTERN.fullmatch(repository.revision):
            raise ModelSourcePublisherError("repository revision must be pinned")
        owner, slug = repository.repo_id.split("/", 1)
        expected_cache = f"models--{owner}--{slug}"
        if repository.cache_dir != expected_cache or repository.cache_dir in cache_dirs:
            raise ModelSourcePublisherError(
                "repository cache_dir is invalid or duplicated"
            )
        cache_dirs.add(repository.cache_dir)
        if not repository.files:
            raise ModelSourcePublisherError("repository files must not be empty")
        coordinates: set[str] = set()
        for file in repository.files:
            _safe_relative(file.coordinate, "file coordinate")
            if file.coordinate in coordinates:
                raise ModelSourcePublisherError("file coordinate is duplicated")
            coordinates.add(file.coordinate)
            if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", file.blob):
                raise ModelSourcePublisherError("file blob identity is invalid")
            if not _HEX_64_PATTERN.fullmatch(file.sha256) or file.size_bytes <= 0:
                raise ModelSourcePublisherError("file content identity is invalid")
            relative = f"{repository.cache_dir}/blobs/{file.blob}"
            if relative in blob_paths:
                raise ModelSourcePublisherError("published blob path is duplicated")
            blob_paths.add(relative)
            if relative == marker_blob:
                marker_matches.append(file)
    if len(marker_matches) != 1 or marker_matches[0].sha256 != packet.marker_sha256:
        raise ModelSourcePublisherError("marker is not an exact published blob identity")


def _repository_records(packet: ModelSourcePublisherPacket) -> list[dict[str, Any]]:
    return [
        {
            "name": repository.name,
            "repo_id": repository.repo_id,
            "revision": repository.revision,
            "cache_dir": repository.cache_dir,
            "files": [
                {
                    "coordinate": file.coordinate,
                    "blob": file.blob,
                    "sha256": file.sha256,
                    "size_bytes": file.size_bytes,
                }
                for file in repository.files
            ],
        }
        for repository in packet.repositories
    ]


def _blob_records(packet: ModelSourcePublisherPacket) -> list[dict[str, Any]]:
    records = [
        {
            "relative_path": f"{repository.cache_dir}/blobs/{file.blob}",
            "sha256": file.sha256,
            "size_bytes": file.size_bytes,
        }
        for repository in packet.repositories
        for file in repository.files
    ]
    return sorted(records, key=lambda record: record["relative_path"])


def _packet_contract(packet: ModelSourcePublisherPacket) -> dict[str, Any]:
    return {
        "schema": PACKET_SCHEMA,
        "kernel_id": packet.kernel_id,
        "title": packet.title,
        "attempt_id": packet.attempt_id,
        "requested_accelerator": None,
        "marker": packet.marker,
        "marker_sha256": packet.marker_sha256,
        "repository_count": len(packet.repositories),
        "blob_count": packet.blob_count,
        "published_blob_bytes": packet.published_blob_bytes,
        "repositories": _repository_records(packet),
        "code_file": packet.code_file,
        "receipt_name": packet.receipt_name,
    }


def _packet_manifest(packet: ModelSourcePublisherPacket) -> dict[str, Any]:
    return {
        **_packet_contract(packet),
        "packet_contract_sha256": packet.packet_contract_sha256,
    }


def _kernel_metadata(packet: ModelSourcePublisherPacket) -> dict[str, Any]:
    return {
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


_RUNNER_TEMPLATE = r'''#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback


CONFIG = json.loads(__CONFIG_JSON__)
WORKING_ROOT = Path(os.environ.get("TRELLIS2MLX_KAGGLE_WORKING_ROOT", "/kaggle/working"))
DOWNLOAD_ROOT = WORKING_ROOT / ".model-source-download"
CACHE_ROOT = WORKING_ROOT / "runtime" / "huggingface"
RECEIPT = WORKING_ROOT / CONFIG["receipt_name"]
TEMP_RECEIPT = WORKING_ROOT / (CONFIG["receipt_name"] + ".tmp")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_receipt(payload: dict) -> None:
    WORKING_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_RECEIPT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(TEMP_RECEIPT, RECEIPT)


def expected_blob_records() -> list[dict]:
    records = []
    for repository in CONFIG["repositories"]:
        for file in repository["files"]:
            records.append({
                "relative_path": (
                    f"{repository['cache_dir']}/blobs/{file['blob']}"
                ),
                "sha256": file["sha256"],
                "size_bytes": file["size_bytes"],
            })
    return sorted(records, key=lambda record: record["relative_path"])


def published_files() -> list[str]:
    if not CACHE_ROOT.is_dir():
        return []
    return sorted(
        path.relative_to(CACHE_ROOT).as_posix()
        for path in CACHE_ROOT.rglob("*")
        if path.is_file()
    )


def main() -> int:
    phase = "preflight"
    nvidia_nodes = sorted(str(path) for path in Path("/dev").glob("nvidia*"))
    context = {
        "schema": CONFIG["schema"],
        "status": "running",
        "failure_phase": None,
        "requested_accelerator": CONFIG["requested_accelerator"],
        "attempt_id": CONFIG["attempt_id"],
        "packet_contract_sha256": CONFIG["packet_contract_sha256"],
        "effective_working_root": str(WORKING_ROOT),
        "effective_cache_root": str(CACHE_ROOT),
        "marker": CONFIG["marker"],
        "marker_sha256": CONFIG["marker_sha256"],
        "repository_count": len(CONFIG["repositories"]),
        "blob_count": sum(
            len(repository["files"])
            for repository in CONFIG["repositories"]
        ),
        "published_blob_bytes": 0,
        "repositories": CONFIG["repositories"],
        "blobs": [],
        "nvidia_device_nodes": nvidia_nodes,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "promotion_mode": "same-filesystem-hardlink",
        "huggingface_hub_disable_xet": "1",
    }
    try:
        WORKING_ROOT.mkdir(parents=True, exist_ok=True)
        phase = "cpu_route"
        if nvidia_nodes:
            raise RuntimeError("CPU publisher unexpectedly exposes Nvidia device nodes")

        phase = "output_preflight"
        if CACHE_ROOT.exists() or DOWNLOAD_ROOT.exists():
            raise RuntimeError("publisher output or download root already exists")
        CACHE_ROOT.mkdir(parents=True)
        DOWNLOAD_ROOT.mkdir(parents=True)

        phase = "dependency_import"
        os.environ["HF_HOME"] = str(DOWNLOAD_ROOT)
        os.environ["HF_HUB_CACHE"] = str(DOWNLOAD_ROOT)
        os.environ["HF_XET_CACHE"] = str(DOWNLOAD_ROOT / "xet")
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        from huggingface_hub import snapshot_download

        phase = "blob_validation"
        records = []
        published_bytes = 0
        for repository in CONFIG["repositories"]:
            snapshot = Path(snapshot_download(
                repo_id=repository["repo_id"],
                revision=repository["revision"],
                cache_dir=str(DOWNLOAD_ROOT),
                allow_patterns=[file["coordinate"] for file in repository["files"]],
            ))
            for expected in repository["files"]:
                source = snapshot / expected["coordinate"]
                if not source.is_file():
                    raise RuntimeError(
                        "downloaded coordinate is missing: "
                        f"repo={repository['repo_id']}, coordinate={expected['coordinate']}"
                    )
                actual_size = source.stat().st_size
                actual_sha256 = sha256_file(source)
                if (
                    actual_size != expected["size_bytes"]
                    or actual_sha256 != expected["sha256"]
                ):
                    raise RuntimeError(
                        "downloaded coordinate identity mismatch: "
                        f"repo={repository['repo_id']}, coordinate={expected['coordinate']}, "
                        f"expected_sha256={expected['sha256']}, actual_sha256={actual_sha256}, "
                        f"expected_size={expected['size_bytes']}, actual_size={actual_size}"
                    )
                relative_path = (
                    f"{repository['cache_dir']}/blobs/{expected['blob']}"
                )
                destination = CACHE_ROOT / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".tmp")
                os.link(source.resolve(strict=True), temporary)
                if (
                    temporary.stat().st_size != expected["size_bytes"]
                    or sha256_file(temporary) != expected["sha256"]
                ):
                    raise RuntimeError(
                        f"copied blob identity mismatch: {relative_path}"
                    )
                os.replace(temporary, destination)
                record = {
                    "relative_path": relative_path,
                    "sha256": expected["sha256"],
                    "size_bytes": expected["size_bytes"],
                }
                records.append(record)
                published_bytes += expected["size_bytes"]

        records.sort(key=lambda record: record["relative_path"])
        if records != expected_blob_records():
            raise RuntimeError("published blob records diverge from packet")
        if published_bytes != CONFIG["published_blob_bytes"]:
            raise RuntimeError("published blob byte total mismatch")
        marker = WORKING_ROOT / CONFIG["marker"]
        if not marker.is_file() or sha256_file(marker) != CONFIG["marker_sha256"]:
            raise RuntimeError("published marker identity mismatch")

        phase = "minimal_output"
        shutil.rmtree(DOWNLOAD_ROOT)
        expected_files = [record["relative_path"] for record in records]
        if published_files() != expected_files:
            raise RuntimeError("published tree contains noncanonical files")

        context.update({
            "status": "completed",
            "failure_phase": None,
            "published_blob_bytes": published_bytes,
            "blobs": records,
        })
        write_receipt(context)
        return 0
    except BaseException as exc:
        context.update({
            "status": "failed",
            "failure_phase": phase,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        })
        try:
            write_receipt(context)
        except BaseException as receipt_exc:
            print(
                f"failed to preserve receipt after {type(exc).__name__}: {exc}; "
                f"receipt failure={type(receipt_exc).__name__}: {receipt_exc}",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
'''
