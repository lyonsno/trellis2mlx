"""CPU-only Kaggle witness for a mounted model-kernel source."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any


SCHEMA = "trellis2mlx.kaggle_model_source_conformance.v1"
PACKET_SCHEMA = "trellis2mlx.kaggle_model_source_conformance.packet.v1"
_REF_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9_-]*$")


class ModelSourceConformanceError(ValueError):
    """Raised when model-source conformance evidence is incomplete or mixed."""


@dataclass(frozen=True, order=True)
class ModelSourceBlob:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ModelSourceConformancePacket:
    output_dir: Path
    kernel_id: str
    title: str
    source_kernel: str
    marker: str
    marker_sha256: str
    blobs: tuple[ModelSourceBlob, ...]
    attempt_id: str
    code_file: str = "run_model_source_conformance.py"
    receipt_name: str = "model_source_conformance.json"

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "blobs", tuple(sorted(self.blobs)))

    @property
    def kernel_dir(self) -> Path:
        return self.output_dir / "kernel"

    @property
    def mounted_blob_bytes(self) -> int:
        return sum(blob.size_bytes for blob in self.blobs)

    @property
    def packet_contract_sha256(self) -> str:
        encoded = json.dumps(
            _packet_contract(self),
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_packet(
    packet: ModelSourceConformancePacket,
) -> ModelSourceConformancePacket:
    """Write a private CPU Kaggle script-kernel packet."""

    _validate_packet(packet)
    if packet.output_dir.exists():
        shutil.rmtree(packet.output_dir)
    packet.kernel_dir.mkdir(parents=True)

    runner_path = packet.kernel_dir / packet.code_file
    runner_path.write_text(render_runner(packet))
    metadata = _kernel_metadata(packet)
    _write_json(packet.kernel_dir / "kernel-metadata.json", metadata)
    manifest = _packet_manifest(packet)
    manifest["runner_sha256"] = sha256_file(runner_path)
    manifest["kernel_metadata"] = metadata
    _write_json(packet.output_dir / "packet.json", manifest)
    return packet


def load_prepared_packet(output_dir: Path) -> ModelSourceConformancePacket:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "packet.json"
    if not manifest_path.is_file():
        raise ModelSourceConformanceError(f"missing packet manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelSourceConformanceError(f"invalid packet manifest: {exc}") from exc
    if manifest.get("schema") != PACKET_SCHEMA:
        raise ModelSourceConformanceError("packet manifest schema is invalid")
    packet = ModelSourceConformancePacket(
        output_dir=output_dir,
        kernel_id=manifest["kernel_id"],
        title=manifest["title"],
        source_kernel=manifest["source_kernel"],
        marker=manifest["marker"],
        marker_sha256=manifest["marker_sha256"],
        blobs=tuple(ModelSourceBlob(**record) for record in manifest["blobs"]),
        attempt_id=manifest["attempt_id"],
        code_file=manifest["code_file"],
        receipt_name=manifest["receipt_name"],
    )
    _validate_packet(packet)
    expected = _packet_manifest(packet)
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ModelSourceConformanceError(
                f"prepared packet manifest {field} mismatch"
            )
    metadata_path = packet.kernel_dir / "kernel-metadata.json"
    runner_path = packet.kernel_dir / packet.code_file
    if not metadata_path.is_file() or not runner_path.is_file():
        raise ModelSourceConformanceError("prepared kernel packet is incomplete")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelSourceConformanceError(f"invalid kernel metadata: {exc}") from exc
    if metadata != _kernel_metadata(packet) or manifest.get("kernel_metadata") != metadata:
        raise ModelSourceConformanceError("prepared kernel metadata mismatch")
    if manifest.get("runner_sha256") != sha256_file(runner_path):
        raise ModelSourceConformanceError("prepared runner digest mismatch")
    return packet


def build_kernel_push_command(packet: ModelSourceConformancePacket) -> list[str]:
    return ["kaggle", "kernels", "push", "-p", str(packet.kernel_dir)]


def build_kernel_status_command(packet: ModelSourceConformancePacket) -> list[str]:
    return ["kaggle", "kernels", "status", packet.kernel_id]


def build_kernel_output_command(
    packet: ModelSourceConformancePacket,
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


def validate_downloaded_receipt(
    packet: ModelSourceConformancePacket,
    output_dir: Path,
) -> dict[str, Any]:
    """Admit a live receipt only when every route and payload carrier agrees."""

    _validate_packet(packet)
    receipt_path = Path(output_dir) / packet.receipt_name
    if not receipt_path.is_file() or receipt_path.stat().st_size <= 0:
        raise ModelSourceConformanceError("downloaded receipt is missing or blank")
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelSourceConformanceError(f"downloaded receipt is invalid: {exc}") from exc
    _validate_completed_receipt_schema(receipt)
    if receipt.get("schema") != SCHEMA:
        raise ModelSourceConformanceError("downloaded receipt schema is invalid")
    if receipt.get("status") != "completed" or receipt.get("failure_phase") is not None:
        raise ModelSourceConformanceError("downloaded receipt status is not completed")
    if receipt.get("requested_source_kernel") != packet.source_kernel:
        raise ModelSourceConformanceError("downloaded receipt source kernel mismatch")
    if receipt.get("attempt_id") != packet.attempt_id:
        raise ModelSourceConformanceError("downloaded receipt attempt identity mismatch")
    if receipt.get("packet_contract_sha256") != packet.packet_contract_sha256:
        raise ModelSourceConformanceError("downloaded receipt packet contract mismatch")
    if receipt.get("requested_accelerator") is not None:
        raise ModelSourceConformanceError("downloaded receipt CPU route requested an accelerator")
    if receipt.get("effective_input_root") != "/kaggle/input":
        raise ModelSourceConformanceError("downloaded receipt input root is not canonical")
    if receipt.get("effective_working_root") != "/kaggle/working":
        raise ModelSourceConformanceError("downloaded receipt working root is not canonical")

    mount_value = receipt.get("effective_mount_root")
    if not isinstance(mount_value, str):
        raise ModelSourceConformanceError("downloaded receipt mount root is missing")
    mount = PurePosixPath(mount_value)
    if not mount.is_absolute() or mount.parent != PurePosixPath("/kaggle/input"):
        raise ModelSourceConformanceError("downloaded receipt mount root is invalid")
    expected_blob_root = str(mount / "runtime" / "huggingface")
    if receipt.get("effective_blob_root") != expected_blob_root:
        raise ModelSourceConformanceError("downloaded receipt blob root is invalid")
    if receipt["candidate_count"] != 1:
        raise ModelSourceConformanceError("downloaded receipt candidate count is not one")
    if receipt.get("marker") != packet.marker:
        raise ModelSourceConformanceError("downloaded receipt marker coordinate mismatch")
    if receipt.get("marker_sha256") != packet.marker_sha256:
        raise ModelSourceConformanceError("downloaded receipt marker digest mismatch")
    if receipt["mounted_blob_bytes"] != packet.mounted_blob_bytes:
        raise ModelSourceConformanceError("downloaded receipt mounted blob byte total mismatch")
    if receipt["writable_model_bytes"] != 0:
        raise ModelSourceConformanceError("downloaded receipt writable model payload is nonzero")
    if receipt["writable_model_matches"] != []:
        raise ModelSourceConformanceError("downloaded receipt writable model matches are nonempty")
    snapshot_fields = ("relative_path", "size_bytes", "is_symlink")
    if _project_records(
        receipt["working_tree_before"], snapshot_fields
    ) != _project_records(receipt["working_tree_after"], snapshot_fields):
        raise ModelSourceConformanceError("downloaded receipt working tree changed during hashing")
    if receipt["nvidia_device_nodes"] != []:
        raise ModelSourceConformanceError("downloaded receipt CPU route exposed Nvidia devices")
    blob_fields = ("relative_path", "sha256", "size_bytes")
    if _project_records(receipt["blobs"], blob_fields) != _blob_records(packet):
        raise ModelSourceConformanceError("downloaded receipt blob manifest mismatch")
    return receipt


def _require_field_type(
    payload: dict[str, Any],
    field: str,
    expected_type: type,
) -> Any:
    if field not in payload:
        raise ModelSourceConformanceError(
            f"downloaded receipt schema missing field: {field}"
        )
    value = payload[field]
    if type(value) is not expected_type:
        raise ModelSourceConformanceError(
            f"downloaded receipt field type is invalid: {field}"
        )
    return value


def _validate_record_list(
    payload: dict[str, Any],
    field: str,
    required: dict[str, type],
) -> None:
    records = _require_field_type(payload, field, list)
    for index, record in enumerate(records):
        if type(record) is not dict:
            raise ModelSourceConformanceError(
                f"downloaded receipt field type is invalid: {field}[{index}]"
            )
        for key, expected_type in required.items():
            if key not in record or type(record[key]) is not expected_type:
                raise ModelSourceConformanceError(
                    "downloaded receipt record field type is invalid: "
                    f"{field}[{index}].{key}"
                )


def _validate_completed_receipt_schema(receipt: Any) -> None:
    if type(receipt) is not dict:
        raise ModelSourceConformanceError("downloaded receipt schema is not an object")
    for field in (
        "schema",
        "status",
        "requested_source_kernel",
        "attempt_id",
        "packet_contract_sha256",
        "effective_input_root",
        "effective_working_root",
        "effective_mount_root",
        "effective_blob_root",
        "marker",
        "marker_sha256",
    ):
        _require_field_type(receipt, field, str)
    if "failure_phase" not in receipt or receipt["failure_phase"] is not None:
        raise ModelSourceConformanceError(
            "downloaded receipt schema requires a null failure_phase"
        )
    if "requested_accelerator" not in receipt or receipt["requested_accelerator"] is not None:
        raise ModelSourceConformanceError(
            "downloaded receipt schema requires a null requested_accelerator"
        )
    for field in ("candidate_count", "mounted_blob_bytes", "writable_model_bytes"):
        _require_field_type(receipt, field, int)
    _validate_record_list(
        receipt,
        "blobs",
        {"relative_path": str, "sha256": str, "size_bytes": int},
    )
    snapshot_fields = {
        "relative_path": str,
        "size_bytes": int,
        "is_symlink": bool,
    }
    _validate_record_list(receipt, "working_tree_before", snapshot_fields)
    _validate_record_list(receipt, "working_tree_after", snapshot_fields)
    _validate_record_list(
        receipt,
        "writable_model_matches",
        {"relative_path": str, "sha256": str, "size_bytes": int},
    )
    device_nodes = _require_field_type(receipt, "nvidia_device_nodes", list)
    if any(type(value) is not str for value in device_nodes):
        raise ModelSourceConformanceError(
            "downloaded receipt field type is invalid: nvidia_device_nodes"
        )


def _project_records(
    records: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    return [{field: record[field] for field in fields} for record in records]


def _validate_packet(packet: ModelSourceConformancePacket) -> None:
    if not _REF_PATTERN.fullmatch(packet.kernel_id):
        raise ModelSourceConformanceError("kernel_id must be an owner/slug reference")
    if not _REF_PATTERN.fullmatch(packet.source_kernel):
        raise ModelSourceConformanceError("source_kernel must be an owner/slug reference")
    if not packet.title.strip():
        raise ModelSourceConformanceError("title must be nonblank")
    if not re.fullmatch(r"[0-9a-f]{32}", packet.attempt_id):
        raise ModelSourceConformanceError("attempt_id must be 32 lowercase hex characters")
    if PurePosixPath(packet.code_file).name != packet.code_file:
        raise ModelSourceConformanceError("code_file must be a filename")
    if PurePosixPath(packet.receipt_name).name != packet.receipt_name:
        raise ModelSourceConformanceError("receipt_name must be a filename")
    if not packet.blobs:
        raise ModelSourceConformanceError("at least one mounted blob is required")
    if len(packet.marker_sha256) != 64:
        raise ModelSourceConformanceError("marker_sha256 is invalid")
    try:
        int(packet.marker_sha256, 16)
    except ValueError as exc:
        raise ModelSourceConformanceError("marker_sha256 is invalid") from exc

    paths: set[str] = set()
    marker_relative = PurePosixPath(packet.marker)
    prefix = PurePosixPath("runtime/huggingface")
    try:
        marker_blob_relative = marker_relative.relative_to(prefix).as_posix()
    except ValueError as exc:
        raise ModelSourceConformanceError(
            "marker must be beneath runtime/huggingface"
        ) from exc
    marker_records = []
    for blob in packet.blobs:
        path = PurePosixPath(blob.relative_path)
        if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
            raise ModelSourceConformanceError("blob relative_path is unsafe")
        if blob.relative_path in paths:
            raise ModelSourceConformanceError("blob relative_path is duplicated")
        paths.add(blob.relative_path)
        if len(blob.sha256) != 64 or blob.size_bytes <= 0:
            raise ModelSourceConformanceError("blob identity is invalid")
        try:
            int(blob.sha256, 16)
        except ValueError as exc:
            raise ModelSourceConformanceError("blob identity is invalid") from exc
        if blob.relative_path == marker_blob_relative:
            marker_records.append(blob)
    if len(marker_records) != 1 or marker_records[0].sha256 != packet.marker_sha256:
        raise ModelSourceConformanceError("marker is not an exact blob identity")


def _blob_records(packet: ModelSourceConformancePacket) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": blob.relative_path,
            "sha256": blob.sha256,
            "size_bytes": blob.size_bytes,
        }
        for blob in sorted(packet.blobs)
    ]


def _packet_contract(packet: ModelSourceConformancePacket) -> dict[str, Any]:
    return {
        "schema": PACKET_SCHEMA,
        "kernel_id": packet.kernel_id,
        "title": packet.title,
        "source_kernel": packet.source_kernel,
        "attempt_id": packet.attempt_id,
        "requested_accelerator": None,
        "marker": packet.marker,
        "marker_sha256": packet.marker_sha256,
        "mounted_blob_bytes": packet.mounted_blob_bytes,
        "blobs": _blob_records(packet),
        "code_file": packet.code_file,
        "receipt_name": packet.receipt_name,
    }


def _packet_manifest(packet: ModelSourceConformancePacket) -> dict[str, Any]:
    manifest = _packet_contract(packet)
    manifest["packet_contract_sha256"] = packet.packet_contract_sha256
    return manifest


def _kernel_metadata(packet: ModelSourceConformancePacket) -> dict[str, Any]:
    return {
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def render_runner(packet: ModelSourceConformancePacket) -> str:
    """Render the exact runner bytes for an authenticated packet."""

    _validate_packet(packet)
    config = {
        "schema": SCHEMA,
        "requested_source_kernel": packet.source_kernel,
        "requested_accelerator": None,
        "attempt_id": packet.attempt_id,
        "packet_contract_sha256": packet.packet_contract_sha256,
        "marker": packet.marker,
        "marker_sha256": packet.marker_sha256,
        "mounted_blob_bytes": packet.mounted_blob_bytes,
        "blobs": _blob_records(packet),
        "receipt_name": packet.receipt_name,
    }
    encoded = repr(json.dumps(config, sort_keys=True))
    return _RUNNER_TEMPLATE.replace("__CONFIG_JSON__", encoded)


_RUNNER_TEMPLATE = r'''#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


CONFIG = json.loads(__CONFIG_JSON__)
INPUT_ROOT = Path(os.environ.get("TRELLIS2MLX_KAGGLE_INPUT_ROOT", "/kaggle/input"))
WORKING_ROOT = Path(os.environ.get("TRELLIS2MLX_KAGGLE_WORKING_ROOT", "/kaggle/working"))
RECEIPT = WORKING_ROOT / CONFIG["receipt_name"]
TEMP_RECEIPT = WORKING_ROOT / (CONFIG["receipt_name"] + ".tmp")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_snapshot(root: Path) -> list[dict]:
    records = []
    if not root.exists():
        return records
    excluded = {RECEIPT.resolve(strict=False), TEMP_RECEIPT.resolve(strict=False)}
    for path in sorted(root.rglob("*")):
        if path.resolve(strict=False) in excluded or not (path.is_file() or path.is_symlink()):
            continue
        stat = path.lstat()
        records.append({
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": stat.st_size,
            "is_symlink": path.is_symlink(),
        })
    return records


def writable_model_matches(root: Path) -> list[dict]:
    expected_by_size = {}
    for record in CONFIG["blobs"]:
        expected_by_size.setdefault(record["size_bytes"], set()).add(record["sha256"])
    matches = []
    excluded = {RECEIPT.resolve(strict=False), TEMP_RECEIPT.resolve(strict=False)}
    if not root.exists():
        return matches
    for path in sorted(root.rglob("*")):
        if (
            path.resolve(strict=False) in excluded
            or path.is_symlink()
            or not path.is_file()
        ):
            continue
        size_bytes = path.stat().st_size
        expected_digests = expected_by_size.get(size_bytes)
        if not expected_digests:
            continue
        digest = sha256_file(path)
        if digest in expected_digests:
            matches.append({
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "size_bytes": size_bytes,
            })
    return matches


def write_receipt(payload: dict) -> None:
    WORKING_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_RECEIPT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(TEMP_RECEIPT, RECEIPT)


def main() -> int:
    phase = "preflight"
    context = {
        "schema": CONFIG["schema"],
        "status": "running",
        "failure_phase": None,
        "requested_source_kernel": CONFIG["requested_source_kernel"],
        "requested_accelerator": CONFIG["requested_accelerator"],
        "attempt_id": CONFIG["attempt_id"],
        "packet_contract_sha256": CONFIG["packet_contract_sha256"],
        "effective_input_root": str(INPUT_ROOT),
        "effective_working_root": str(WORKING_ROOT),
        "marker": CONFIG["marker"],
        "marker_sha256": None,
        "candidate_count": None,
        "effective_mount_root": None,
        "effective_blob_root": None,
        "mounted_blob_bytes": 0,
        "writable_model_bytes": None,
        "writable_model_matches": None,
        "working_tree_before": None,
        "working_tree_after": None,
        "nvidia_device_nodes": sorted(str(path) for path in Path("/dev").glob("nvidia*")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "blobs": [],
    }
    try:
        WORKING_ROOT.mkdir(parents=True, exist_ok=True)
        phase = "source_mount"
        marker = Path(CONFIG["marker"])
        candidates = []
        if INPUT_ROOT.is_dir():
            candidates = [
                mount
                for mount in sorted(INPUT_ROOT.iterdir())
                if mount.is_dir() and (mount / marker).is_file()
            ]
        context["candidate_count"] = len(candidates)
        if len(candidates) != 1:
            raise RuntimeError(
                "mounted source is missing or ambiguous: "
                f"candidates={[str(path) for path in candidates]}"
            )
        mount = candidates[0]
        blob_root = mount / "runtime" / "huggingface"
        context["effective_mount_root"] = str(mount)
        context["effective_blob_root"] = str(blob_root)
        marker_sha256 = sha256_file(mount / marker)
        context["marker_sha256"] = marker_sha256
        if marker_sha256 != CONFIG["marker_sha256"]:
            raise RuntimeError("mounted source marker digest mismatch")

        phase = "writable_model_payload"
        before = tree_snapshot(WORKING_ROOT)
        context["working_tree_before"] = before
        before_matches = writable_model_matches(WORKING_ROOT)
        context["writable_model_matches"] = before_matches
        context["writable_model_bytes"] = sum(
            record["size_bytes"] for record in before_matches
        )
        if before_matches:
            raise RuntimeError("exact model payload already exists in writable storage")

        phase = "blob_identity"
        records = []
        mounted_bytes = 0
        for expected in CONFIG["blobs"]:
            source = blob_root / expected["relative_path"]
            if not source.is_file():
                raise RuntimeError(f"mounted blob is missing: {source}")
            size_bytes = source.stat().st_size
            actual_sha256 = sha256_file(source)
            record = {
                "relative_path": expected["relative_path"],
                "sha256": actual_sha256,
                "size_bytes": size_bytes,
            }
            records.append(record)
            if record != expected:
                raise RuntimeError(
                    "mounted blob identity mismatch: "
                    f"expected={expected}, actual={record}"
                )
            mounted_bytes += size_bytes
        context["blobs"] = records
        context["mounted_blob_bytes"] = mounted_bytes
        if mounted_bytes != CONFIG["mounted_blob_bytes"]:
            raise RuntimeError("mounted blob byte total mismatch")

        phase = "writable_model_payload"
        after = tree_snapshot(WORKING_ROOT)
        context["working_tree_after"] = after
        after_matches = writable_model_matches(WORKING_ROOT)
        context["writable_model_matches"] = after_matches
        introduced_bytes = sum(
            record["size_bytes"] for record in after if record not in before
        )
        context["writable_model_bytes"] = sum(
            record["size_bytes"] for record in after_matches
        )
        if before != after or introduced_bytes != 0 or after_matches:
            raise RuntimeError("working tree changed while mounted blobs were hashed")
        if context["nvidia_device_nodes"]:
            phase = "cpu_route"
            raise RuntimeError("CPU witness unexpectedly exposes Nvidia device nodes")

        context.update({"status": "completed", "failure_phase": None})
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
