"""Kaggle CUDA witness packet helpers.

The packet uses a private Kaggle dataset for immutable witness inputs and a
private Kaggle script kernel for the CUDA run. Keeping the two surfaces separate
lets agents update data and execution independently while preserving route
identity in metadata and receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Sequence

import numpy as np


class WitnessPacketError(ValueError):
    """Raised when a witness packet would be incomplete or misleading."""


Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class KaggleCudaWitnessPacket:
    capsule_dir: Path
    output_dir: Path
    dataset_id: str
    kernel_id: str
    title: str
    entrypoint: str
    inputs: tuple[str, ...]
    entrypoint_args: tuple[str, ...] = ()
    accelerator: str = "NvidiaTeslaT4"
    enable_internet: bool = False
    output_json: str = "cuda_result.json"
    output_npz: str | None = "cuda_result.npz"
    output_ply: str | None = None
    output_mesh_state: str | None = None
    output_shape_slat: str | None = None
    output_shape_flow_step: str | None = None
    expected_outputs: tuple[str, ...] = ()
    shape_flow_noise_sample: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "capsule_dir", Path(self.capsule_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "entrypoint_args", tuple(self.entrypoint_args))
        object.__setattr__(self, "expected_outputs", tuple(self.expected_outputs))

    @property
    def dataset_dir(self) -> Path:
        return self.output_dir / "dataset"

    @property
    def kernel_dir(self) -> Path:
        return self.output_dir / "kernel"

    @property
    def dataset_slug(self) -> str:
        return _slug_from_ref(self.dataset_id)

    @property
    def outputs(self) -> tuple[str, ...]:
        outputs = [self.output_json]
        if self.output_npz:
            outputs.append(self.output_npz)
        if self.output_ply:
            outputs.append(self.output_ply)
        if self.output_mesh_state:
            outputs.append(self.output_mesh_state)
        if self.output_shape_slat:
            outputs.append(self.output_shape_slat)
        if self.output_shape_flow_step:
            outputs.append(self.output_shape_flow_step)
        outputs.extend(self.expected_outputs)
        return tuple(outputs)


def prepare_packet(packet: KaggleCudaWitnessPacket) -> KaggleCudaWitnessPacket:
    """Create a Kaggle dataset/kernel packet for a CUDA witness capsule."""

    _validate_refs(packet)
    file_sources = _validate_inputs(packet)
    if packet.output_dir.exists():
        shutil.rmtree(packet.output_dir)
    packet.dataset_dir.mkdir(parents=True)
    packet.kernel_dir.mkdir(parents=True)

    file_records: dict[str, dict[str, str | int]] = {}
    for relative_name, source in file_sources.items():
        destination = packet.dataset_dir / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        file_records[relative_name] = {
            "sha256": sha256_file(destination),
            "size_bytes": destination.stat().st_size,
        }

    manifest = {
        "schema": "trellis2mlx.kaggle_cuda_witness.inputs.v1",
        "dataset_id": packet.dataset_id,
        "kernel_id": packet.kernel_id,
        "title": packet.title,
        "entrypoint": packet.entrypoint,
        "entrypoint_args": list(packet.entrypoint_args),
        "input_roles": {
            "shape_flow_noise_sample": packet.shape_flow_noise_sample,
        },
        "accelerator": packet.accelerator,
        "enable_internet": packet.enable_internet,
        "outputs": list(packet.outputs),
        "output_roles": {
            "json": packet.output_json,
            "npz": packet.output_npz,
            "ply": packet.output_ply,
            "mesh_state": packet.output_mesh_state,
            "shape_slat": packet.output_shape_slat,
            "shape_flow_step": packet.output_shape_flow_step,
            "expected": list(packet.expected_outputs),
        },
        "files": file_records,
    }
    manifest_path = packet.dataset_dir / "witness-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    file_records["witness-manifest.json"] = {
        "sha256": sha256_file(manifest_path),
        "size_bytes": manifest_path.stat().st_size,
    }

    _write_json(packet.dataset_dir / "dataset-metadata.json", _dataset_metadata(packet, file_records))
    _write_json(packet.kernel_dir / "kernel-metadata.json", _kernel_metadata(packet))
    (packet.kernel_dir / "run_kaggle_cuda_witness.py").write_text(_runner_script(packet))
    return packet


def load_prepared_packet(output_dir: Path) -> KaggleCudaWitnessPacket:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "dataset" / "witness-manifest.json"
    kernel_metadata_path = output_dir / "kernel" / "kernel-metadata.json"
    if not manifest_path.is_file():
        raise WitnessPacketError(f"missing prepared manifest: {manifest_path}")
    if not kernel_metadata_path.is_file():
        raise WitnessPacketError(f"missing kernel metadata: {kernel_metadata_path}")
    manifest = json.loads(manifest_path.read_text())
    kernel_metadata = json.loads(kernel_metadata_path.read_text())
    inputs = tuple(path for path in manifest["files"] if path != "witness-manifest.json")
    outputs = tuple(manifest.get("outputs", ("cuda_result.json", "cuda_result.npz")))
    output_roles = manifest.get("output_roles") or {}
    input_roles = manifest.get("input_roles") or {}
    if not outputs:
        raise WitnessPacketError("expected at least one output in manifest")
    output_json = output_roles.get("json") or outputs[0]
    output_npz = (
        output_roles["npz"]
        if "npz" in output_roles
        else _legacy_output_by_suffix(outputs[1:], ".npz")
    )
    output_ply = (
        output_roles["ply"]
        if "ply" in output_roles
        else _legacy_output_by_suffix(outputs[1:], ".ply")
    )
    output_mesh_state = (
        output_roles["mesh_state"]
        if "mesh_state" in output_roles
        else _legacy_output_by_suffix(outputs[1:], "_mesh_state.npz")
    )
    output_shape_slat = (
        output_roles["shape_slat"]
        if "shape_slat" in output_roles
        else _legacy_output_by_suffix(outputs[1:], "_shape_slat.npz")
    )
    output_shape_flow_step = (
        output_roles["shape_flow_step"]
        if "shape_flow_step" in output_roles
        else _legacy_output_by_suffix(outputs[1:], "_shape_flow_step.npz")
    )
    claimed_outputs = {
        output
        for output in (
            output_json,
            output_npz,
            output_ply,
            output_mesh_state,
            output_shape_slat,
            output_shape_flow_step,
        )
        if output is not None
    }
    expected_outputs = tuple(
        output_roles.get("expected")
        or (output for output in outputs if output not in claimed_outputs)
    )
    packet = KaggleCudaWitnessPacket(
        capsule_dir=output_dir / "dataset",
        output_dir=output_dir,
        dataset_id=manifest["dataset_id"],
        kernel_id=kernel_metadata["id"],
        title=kernel_metadata["title"],
        entrypoint=manifest["entrypoint"],
        inputs=inputs,
        entrypoint_args=tuple(manifest.get("entrypoint_args", ())),
        accelerator=kernel_metadata.get("machine_shape", manifest.get("accelerator", "NvidiaTeslaT4")),
        enable_internet=_bool_metadata(
            kernel_metadata.get("enable_internet", manifest.get("enable_internet", False))
        ),
        output_json=output_json,
        output_npz=output_npz,
        output_ply=output_ply,
        output_mesh_state=output_mesh_state,
        output_shape_slat=output_shape_slat,
        output_shape_flow_step=output_shape_flow_step,
        expected_outputs=expected_outputs,
        shape_flow_noise_sample=input_roles.get("shape_flow_noise_sample"),
    )
    _validate_refs(packet)
    return packet


def build_dataset_command(packet: KaggleCudaWitnessPacket, *, version: bool = False) -> list[str]:
    if version:
        return [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(packet.dataset_dir),
            "-m",
            "update CUDA witness inputs",
            "-q",
            "-t",
            "-r",
            "skip",
        ]
    return [
        "kaggle",
        "datasets",
        "create",
        "-p",
        str(packet.dataset_dir),
        "-q",
        "-t",
        "-r",
        "skip",
    ]


def build_dataset_status_command(packet: KaggleCudaWitnessPacket) -> list[str]:
    return [
        "kaggle",
        "datasets",
        "status",
        packet.dataset_id,
        "--format",
        "json",
    ]


def build_dataset_manifest_download_command(
    packet: KaggleCudaWitnessPacket,
    output_dir: Path,
) -> list[str]:
    return [
        "kaggle",
        "datasets",
        "download",
        packet.dataset_id,
        "-f",
        "witness-manifest.json",
        "-p",
        str(output_dir),
        "-o",
        "-q",
    ]


def wait_for_published_dataset_manifest(
    packet: KaggleCudaWitnessPacket,
    *,
    report_path: Path,
    runner: Runner = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
    poll_seconds: float = 2.0,
    scratch_root: Path | None = None,
    max_attempts: int | None = None,
) -> dict[str, object]:
    if max_attempts is not None and max_attempts <= 0:
        raise ValueError("max_attempts must be positive when provided")
    expected_path = packet.dataset_dir / "witness-manifest.json"
    expected_sha256 = sha256_file(expected_path)
    stale_observations = 0
    attempts = 0
    last_observation: dict[str, object] = {}
    while True:
        attempts += 1
        status_command = build_dataset_status_command(packet)
        try:
            status_completed = runner(
                status_command,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            status_completed = subprocess.CompletedProcess(
                status_command,
                1,
                stdout="",
                stderr=f"{type(exc).__name__}: {exc}",
            )
        last_observation = {
            "status_command": status_command,
            "status_exit_code": status_completed.returncode,
            "status_stdout": status_completed.stdout,
            "status_stderr": status_completed.stderr,
        }
        dataset_status: dict[str, object] | None = None
        failure_phase = "dataset_status"
        if status_completed.returncode == 0:
            try:
                decoded_status = json.loads(status_completed.stdout)
                if isinstance(decoded_status, dict):
                    dataset_status = decoded_status
                else:
                    last_observation["status_parse_error"] = (
                        "dataset status JSON is not an object"
                    )
            except json.JSONDecodeError as exc:
                last_observation["status_parse_error"] = str(exc)
        if dataset_status and dataset_status.get("status") == "ready":
            scratch_parent = (
                None if scratch_root is None else str(Path(scratch_root))
            )
            with tempfile.TemporaryDirectory(
                prefix="kaggle-dataset-publication-",
                dir=scratch_parent,
            ) as temporary:
                download_dir = Path(temporary)
                download_command = build_dataset_manifest_download_command(
                    packet,
                    download_dir,
                )
                try:
                    download_completed = runner(
                        download_command,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                except OSError as exc:
                    download_completed = subprocess.CompletedProcess(
                        download_command,
                        1,
                        stdout="",
                        stderr=f"{type(exc).__name__}: {exc}",
                    )
                remote_manifest = download_dir / "witness-manifest.json"
                failure_phase = "manifest_download"
                last_observation.update(
                    {
                        "download_command": download_command,
                        "download_exit_code": download_completed.returncode,
                        "download_stdout": download_completed.stdout,
                        "download_stderr": download_completed.stderr,
                        "dataset_status": dataset_status,
                    }
                )
                if (
                    download_completed.returncode == 0
                    and remote_manifest.is_file()
                ):
                    remote_sha256 = sha256_file(remote_manifest)
                    last_observation["remote_manifest_sha256"] = remote_sha256
                    if remote_sha256 == expected_sha256:
                        report = {
                            "schema": (
                                "trellis2mlx.kaggle_cuda_witness."
                                "dataset_publication.v1"
                            ),
                            "status": "done",
                            "failure_phase": None,
                            "dataset_id": packet.dataset_id,
                            "dataset_status": dataset_status,
                            "attempts": attempts,
                            "stale_observations": stale_observations,
                            "expected_manifest_sha256": expected_sha256,
                            "remote_manifest_sha256": remote_sha256,
                        }
                        _write_json(report_path, report)
                        return report
                    stale_observations += 1
                    failure_phase = "stale_manifest"
        observation_report = {
            "schema": (
                "trellis2mlx.kaggle_cuda_witness."
                "dataset_publication.v1"
            ),
            "status": "running",
            "failure_phase": None,
            "current_phase": failure_phase,
            "last_trustworthy_phase": "publication_observed",
            "dataset_id": packet.dataset_id,
            "attempts": attempts,
            "max_attempts": max_attempts,
            "stale_observations": stale_observations,
            "expected_manifest_sha256": expected_sha256,
            "last_observation": last_observation,
        }
        if max_attempts is not None and attempts >= max_attempts:
            observation_report.update(
                {
                    "status": "failed",
                    "failure_phase": failure_phase,
                }
            )
            _write_json(report_path, observation_report)
            return observation_report
        _write_json(report_path, observation_report)
        sleeper(poll_seconds)


def build_kernel_push_command(packet: KaggleCudaWitnessPacket, *, timeout_seconds: int | None = None) -> list[str]:
    cmd = [
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(packet.kernel_dir),
        "--accelerator",
        packet.accelerator,
    ]
    if timeout_seconds is not None:
        cmd += ["--timeout", str(timeout_seconds)]
    return cmd


def build_kernel_status_command(packet: KaggleCudaWitnessPacket) -> list[str]:
    return ["kaggle", "kernels", "status", packet.kernel_id]


def build_kernel_output_command(packet: KaggleCudaWitnessPacket, output_dir: Path) -> list[str]:
    output_names = (*packet.outputs, "kaggle_cuda_witness_receipt.json")
    file_pattern = r"\A(?:" + "|".join(re.escape(name) for name in output_names) + r")\Z"
    return [
        "kaggle",
        "kernels",
        "output",
        packet.kernel_id,
        "-p",
        str(output_dir),
        "-o",
        "--file-pattern",
        file_pattern,
        "--page-size",
        "100",
    ]


def run_command(
    cmd: Sequence[str],
    *,
    phase: str,
    report_path: Path,
    runner: Runner = subprocess.run,
) -> dict[str, object]:
    try:
        completed = runner(list(cmd), capture_output=True, text=True, check=False)
    except OSError as exc:
        report = {
            "schema": "trellis2mlx.kaggle_cuda_witness.command_report.v1",
            "phase": phase,
            "command": list(cmd),
            "status": "failed",
            "failure_phase": f"{phase}_launch",
            "exit_code": 127,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report
    textual_error = _kaggle_textual_error(completed.stdout) or _kaggle_textual_error(completed.stderr)
    failed = completed.returncode != 0 or textual_error
    report = {
        "schema": "trellis2mlx.kaggle_cuda_witness.command_report.v1",
        "phase": phase,
        "command": list(cmd),
        "status": "failed" if failed else "done",
        "failure_phase": phase if failed else None,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_downloaded_outputs(
    packet: KaggleCudaWitnessPacket,
    output_dir: Path,
) -> dict[str, dict[str, str | int]]:
    output_dir = Path(output_dir)
    resolved_output_dir = output_dir.resolve()
    records: dict[str, dict[str, str | int]] = {}
    receipt_name = "kaggle_cuda_witness_receipt.json"
    receipt_path = output_dir / receipt_name
    if not receipt_path.is_file():
        raise WitnessPacketError(f"missing downloaded output: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WitnessPacketError(
            f"invalid JSON downloaded output {receipt_path}: {exc}"
        ) from exc
    if receipt.get("schema") != "trellis2mlx.kaggle_cuda_witness.receipt.v1":
        raise WitnessPacketError("downloaded receipt schema is invalid")
    if receipt.get("status") != "done":
        raise WitnessPacketError(
            f"downloaded receipt status is not done: {receipt.get('status')!r}"
        )
    if receipt.get("failure_phase") is not None:
        raise WitnessPacketError("downloaded done receipt has a failure phase")
    expected_source_identity = {
        "dataset_sources": [packet.dataset_id],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    expected_receipt_identity = {
        "requested_dataset_id": packet.dataset_id,
        "requested_kernel_id": packet.kernel_id,
        "requested_accelerator": packet.accelerator,
        "source_identity": expected_source_identity,
    }
    for field, expected in expected_receipt_identity.items():
        if receipt.get(field) != expected:
            raise WitnessPacketError(
                f"downloaded receipt {field} mismatch: "
                f"expected {expected!r}, got {receipt.get(field)!r}"
            )
    manifest_path = packet.dataset_dir / "witness-manifest.json"
    if not manifest_path.is_file():
        raise WitnessPacketError(
            f"prepared packet manifest is missing: {manifest_path}"
        )
    expected_manifest_record = {
        "sha256": sha256_file(manifest_path),
        "size_bytes": manifest_path.stat().st_size,
    }
    if receipt.get("input_manifest") != expected_manifest_record:
        raise WitnessPacketError(
            "downloaded receipt manifest digest or size mismatch: "
            f"expected {expected_manifest_record!r}, "
            f"got {receipt.get('input_manifest')!r}"
        )
    if receipt.get("cuda_available") is not True:
        raise WitnessPacketError(
            "downloaded receipt does not prove an effective CUDA route"
        )
    cuda_device = receipt.get("cuda_device")
    if not isinstance(cuda_device, str) or not cuda_device.strip():
        raise WitnessPacketError(
            "downloaded receipt CUDA device is missing or blank"
        )
    receipt_outputs = receipt.get("outputs")
    if not isinstance(receipt_outputs, dict):
        raise WitnessPacketError("downloaded receipt is missing its output snapshot")
    if set(receipt_outputs) != set(packet.outputs):
        raise WitnessPacketError(
            "downloaded receipt output set mismatch: "
            f"expected {sorted(packet.outputs)}, got {sorted(receipt_outputs)}"
        )

    for name in packet.outputs:
        canonical_name = _canonical_output_name(name)
        path = output_dir / canonical_name
        if not path.resolve().is_relative_to(resolved_output_dir):
            raise WitnessPacketError(
                f"downloaded output escapes output directory: {path}"
            )
        if not path.is_file():
            raise WitnessPacketError(f"missing downloaded output: {path}")
        size = path.stat().st_size
        if size == 0:
            raise WitnessPacketError(f"blank downloaded output: {path}")
        if path.suffix == ".json":
            try:
                json.loads(path.read_text())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise WitnessPacketError(f"invalid JSON downloaded output {path}: {exc}") from exc
        if path.suffix == ".npz":
            try:
                with np.load(path, allow_pickle=False) as archive:
                    if not archive.files:
                        raise ValueError("archive contains no arrays")
                    members = archive.zip.namelist()
                    if (
                        len(members) != len(archive.files)
                        or any(
                            member != f"{array_name}.npy"
                            for member, array_name in zip(
                                members, archive.files, strict=True
                            )
                        )
                    ):
                        raise ValueError(
                            "archive contains members that are not canonical NPY arrays"
                        )
                    for array_name in archive.files:
                        array = archive[array_name]
                        if not isinstance(array, np.ndarray):
                            raise ValueError(
                                f"member {array_name!r} is not an ndarray"
                            )
                        np.asarray(array)
            except (OSError, ValueError, EOFError) as exc:
                raise WitnessPacketError(
                    f"invalid NPZ downloaded output {path}: {exc}"
                ) from exc
        if path.suffix == ".ply":
            _validate_ply_output(path)
        record = {
            "size_bytes": size,
            "sha256": sha256_file(path),
        }
        receipt_record = receipt_outputs.get(name)
        if not isinstance(receipt_record, dict) or receipt_record.get("exists") is not True:
            raise WitnessPacketError(
                f"downloaded receipt does not prove output existence: {name}"
            )
        if receipt_record.get("size_bytes") != record["size_bytes"]:
            raise WitnessPacketError(
                f"downloaded output size differs from receipt for {name}"
            )
        if receipt_record.get("sha256") != record["sha256"]:
            raise WitnessPacketError(
                f"downloaded output digest differs from receipt for {name}"
            )
        records[name] = record
    receipt_size = receipt_path.stat().st_size
    if receipt_size == 0:
        raise WitnessPacketError(f"blank downloaded output: {receipt_path}")
    records[receipt_name] = {
        "size_bytes": receipt_size,
        "sha256": sha256_file(receipt_path),
    }
    return records


def _validate_ply_output(path: Path) -> None:
    try:
        import trimesh

        mesh = trimesh.load(path, file_type="ply", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(
                f"expected one triangular mesh, got {type(mesh).__name__}"
            )
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        if vertices.ndim != 2 or vertices.shape[0] == 0 or vertices.shape[1] != 3:
            raise ValueError(f"invalid vertex array shape {vertices.shape}")
        if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] != 3:
            raise ValueError(f"invalid face array shape {faces.shape}")
        if not np.isfinite(vertices).all():
            raise ValueError("vertices contain non-finite values")
        if faces.min() < 0 or faces.max() >= vertices.shape[0]:
            raise ValueError("face indices escape the vertex array")
    except Exception as exc:
        if isinstance(exc, WitnessPacketError):
            raise
        raise WitnessPacketError(
            f"invalid PLY downloaded output {path}: {exc}"
        ) from exc


def wait_for_downloaded_outputs(
    packet: KaggleCudaWitnessPacket,
    output_dir: Path,
    *,
    max_wait_seconds: float = 30.0,
    poll_seconds: float = 0.25,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, dict[str, str | int]]:
    deadline = time.monotonic() + max_wait_seconds
    last_error: WitnessPacketError | None = None
    while True:
        try:
            return validate_downloaded_outputs(packet, output_dir)
        except WitnessPacketError as exc:
            last_error = exc
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WitnessPacketError(
                f"downloaded outputs did not stabilize within {max_wait_seconds}s: {last_error}"
            ) from last_error
        sleeper(min(poll_seconds, remaining))


def _kaggle_textual_error(output: str) -> bool:
    for line in output.splitlines():
        lower = line.strip().lower()
        if lower.endswith(" error") or " error:" in lower or lower.startswith("error:"):
            return True
        if "not valid dataset sources" in lower and "could not be added to the kernel" in lower:
            return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare and drive Kaggle CUDA witness packets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--capsule-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--dataset-id", required=True)
    prepare.add_argument("--kernel-id", required=True)
    prepare.add_argument("--title", required=True)
    prepare.add_argument("--entrypoint", required=True)
    prepare.add_argument("--entrypoint-arg", action="append", dest="entrypoint_args", default=[])
    prepare.add_argument("--input", action="append", dest="inputs", required=True)
    prepare.add_argument("--accelerator", default="NvidiaTeslaT4")
    prepare.add_argument("--enable-internet", action="store_true")
    prepare.add_argument("--output-json", default="cuda_result.json")
    prepare.add_argument("--output-npz", default="cuda_result.npz")
    prepare.add_argument("--no-output-npz", action="store_true")
    prepare.add_argument("--output-ply")
    prepare.add_argument("--output-mesh-state")
    prepare.add_argument("--output-shape-slat")
    prepare.add_argument("--output-shape-flow-step")
    prepare.add_argument(
        "--expected-output",
        action="append",
        dest="expected_outputs",
        default=[],
    )
    prepare.add_argument("--shape-flow-noise-sample")

    for name in ("dataset-create", "dataset-version", "kernel-push", "kernel-status", "kernel-output", "print-commands"):
        drive = subparsers.add_parser(name)
        drive.add_argument("--packet-dir", type=Path, required=True)
        drive.add_argument("--report-dir", type=Path)
        if name == "kernel-push":
            drive.add_argument("--timeout-seconds", type=int)
        if name == "dataset-version":
            drive.add_argument("--publication-max-attempts", type=int)
        if name == "kernel-output":
            drive.add_argument("--output-dir", type=Path)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        packet = prepare_packet(
            KaggleCudaWitnessPacket(
                capsule_dir=args.capsule_dir,
                output_dir=args.output_dir,
                dataset_id=args.dataset_id,
                kernel_id=args.kernel_id,
                title=args.title,
                entrypoint=args.entrypoint,
                inputs=tuple(args.inputs),
                entrypoint_args=tuple(args.entrypoint_args),
                accelerator=args.accelerator,
                enable_internet=args.enable_internet,
                output_json=args.output_json,
                output_npz=None if args.no_output_npz else args.output_npz,
                output_ply=args.output_ply,
                output_mesh_state=args.output_mesh_state,
                output_shape_slat=args.output_shape_slat,
                output_shape_flow_step=args.output_shape_flow_step,
                expected_outputs=tuple(args.expected_outputs),
                shape_flow_noise_sample=args.shape_flow_noise_sample,
            )
        )
        print(json.dumps(_prepared_summary(packet), indent=2, sort_keys=True))
        return 0
    if args.command == "print-commands":
        packet = load_prepared_packet(args.packet_dir)
        print(json.dumps(_prepared_summary(packet), indent=2, sort_keys=True))
        return 0
    if args.command == "dataset-create":
        packet = load_prepared_packet(args.packet_dir)
        return _run_cli_command(build_dataset_command(packet), "dataset_create", args)
    if args.command == "dataset-version":
        packet = load_prepared_packet(args.packet_dir)
        report_dir = args.report_dir or Path(args.packet_dir) / "reports"
        report_path = report_dir / "dataset_version.json"
        report = run_command(
            build_dataset_command(packet, version=True),
            phase="dataset_version",
            report_path=report_path,
        )
        if report["status"] == "done":
            publication = wait_for_published_dataset_manifest(
                packet,
                report_path=report_dir / "dataset_publication.json",
                max_attempts=args.publication_max_attempts,
            )
            report["publication"] = publication
            if publication["status"] != "done":
                report.update(
                    {
                        "status": "failed",
                        "failure_phase": "dataset_publication",
                        "exit_code": 1,
                    }
                )
            _write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "done" else int(
            report["exit_code"] or 1
        )
    if args.command == "kernel-push":
        packet = load_prepared_packet(args.packet_dir)
        return _run_cli_command(
            build_kernel_push_command(packet, timeout_seconds=args.timeout_seconds),
            "kernel_push",
            args,
        )
    if args.command == "kernel-status":
        packet = load_prepared_packet(args.packet_dir)
        return _run_cli_command(build_kernel_status_command(packet), "kernel_status", args)
    if args.command == "kernel-output":
        packet = load_prepared_packet(args.packet_dir)
        output_dir = args.output_dir or packet.output_dir / "outputs"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        report_dir = args.report_dir or Path(args.packet_dir) / "reports"
        report_path = report_dir / "kernel_output.json"
        report = run_command(
            build_kernel_output_command(packet, output_dir),
            phase="kernel_output",
            report_path=report_path,
        )
        if report["status"] == "done":
            try:
                report["downloaded_outputs"] = wait_for_downloaded_outputs(packet, output_dir)
            except WitnessPacketError as exc:
                report.update(
                    {
                        "status": "failed",
                        "failure_phase": "kernel_output_validation",
                        "validation_error": str(exc),
                    }
                )
        _write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "done" else 1
    raise AssertionError(f"unhandled command {args.command}")


def _run_cli_command(cmd: Sequence[str], phase: str, args: argparse.Namespace) -> int:
    report_dir = args.report_dir or Path(args.packet_dir) / "reports"
    report = run_command(cmd, phase=phase, report_path=report_dir / f"{phase}.json")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "done" else int(report["exit_code"] or 1)


def _prepared_summary(packet: KaggleCudaWitnessPacket) -> dict[str, object]:
    return {
        "schema": "trellis2mlx.kaggle_cuda_witness.prepared.v1",
        "dataset_id": packet.dataset_id,
        "kernel_id": packet.kernel_id,
        "accelerator": packet.accelerator,
        "enable_internet": packet.enable_internet,
        "entrypoint_args": list(packet.entrypoint_args),
        "outputs": list(packet.outputs),
        "dataset_dir": str(packet.dataset_dir),
        "kernel_dir": str(packet.kernel_dir),
        "commands": {
            "dataset_create": build_dataset_command(packet),
            "dataset_version": build_dataset_command(packet, version=True),
            "kernel_push": build_kernel_push_command(packet),
            "kernel_status": build_kernel_status_command(packet),
            "kernel_output": build_kernel_output_command(packet, packet.output_dir / "outputs"),
        },
    }


def _validate_refs(packet: KaggleCudaWitnessPacket) -> None:
    for field, value in (("dataset_id", packet.dataset_id), ("kernel_id", packet.kernel_id)):
        if len(value.split("/")) != 2 or not all(value.split("/")):
            raise WitnessPacketError(f"{field} must be a Kaggle ref like owner/slug")
    kernel_slug = _slug_from_ref(packet.kernel_id)
    title_slug = _kaggle_slug(packet.title)
    if kernel_slug != title_slug:
        raise WitnessPacketError(
            f"kernel_id slug {kernel_slug!r} must match Kaggle title slug {title_slug!r}; "
            "otherwise Kaggle creates a different route than the packet records"
        )
    if packet.accelerator not in {"NvidiaTeslaT4", "NvidiaTeslaT4Highmem", "NvidiaTeslaP100"}:
        raise WitnessPacketError(f"unsupported accelerator for this witness bridge: {packet.accelerator}")
    if packet.entrypoint not in packet.inputs:
        raise WitnessPacketError("entrypoint must be one of the staged inputs")
    if packet.shape_flow_noise_sample is not None and packet.shape_flow_noise_sample not in packet.inputs:
        raise WitnessPacketError("shape_flow_noise_sample must be one of the staged inputs")
    canonical_outputs = tuple(_canonical_output_name(output) for output in packet.outputs)
    if len(set(canonical_outputs)) != len(canonical_outputs):
        raise WitnessPacketError("declared outputs must be canonically unique")
    if "kaggle_cuda_witness_receipt.json" in canonical_outputs:
        raise WitnessPacketError("declared output collides with the witness receipt")


def _validate_inputs(packet: KaggleCudaWitnessPacket) -> dict[str, Path]:
    if not packet.capsule_dir.is_dir():
        raise WitnessPacketError(f"capsule_dir does not exist: {packet.capsule_dir}")
    if not packet.inputs:
        raise WitnessPacketError("at least one input is required")

    sources: dict[str, Path] = {}
    for relative_name in packet.inputs:
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise WitnessPacketError(f"input must be relative inside capsule_dir: {relative_name}")
        source = packet.capsule_dir / relative_path
        if source.is_dir():
            children = [path for path in sorted(source.rglob("*")) if path.is_file()]
            if not children:
                raise WitnessPacketError(f"empty input directory: {relative_name}")
            for child in children:
                sources[child.relative_to(packet.capsule_dir).as_posix()] = child
            continue
        if not source.is_file():
            raise WitnessPacketError(f"missing input: {relative_name}")
        sources[relative_name] = source
    return sources


def _dataset_metadata(
    packet: KaggleCudaWitnessPacket,
    file_records: dict[str, dict[str, str | int]],
) -> dict[str, object]:
    return {
        "title": packet.title[:50],
        "id": packet.dataset_id,
        "licenses": [{"name": "unknown"}],
        "resources": [
            {"path": path, "description": f"CUDA witness input, sha256={record['sha256']}"}
            for path, record in sorted(file_records.items())
        ],
    }


def _kernel_metadata(packet: KaggleCudaWitnessPacket) -> dict[str, object]:
    return {
        "id": packet.kernel_id,
        "title": packet.title,
        "code_file": "run_kaggle_cuda_witness.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true" if packet.enable_internet else "false",
        "machine_shape": packet.accelerator,
        "dataset_sources": [packet.dataset_id],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def _runner_script(packet: KaggleCudaWitnessPacket) -> str:
    manifest_path = packet.dataset_dir / "witness-manifest.json"
    config = {
        "schema": "trellis2mlx.kaggle_cuda_witness.runner_config.v1",
        "dataset_id": packet.dataset_id,
        "dataset_slug": packet.dataset_slug,
        "kernel_id": packet.kernel_id,
        "accelerator": packet.accelerator,
        "entrypoint": packet.entrypoint,
        "entrypoint_args": list(packet.entrypoint_args),
        "outputs": list(packet.outputs),
        "output_json": packet.output_json,
        "output_npz": packet.output_npz,
        "output_ply": packet.output_ply,
        "output_mesh_state": packet.output_mesh_state,
        "output_shape_slat": packet.output_shape_slat,
        "output_shape_flow_step": packet.output_shape_flow_step,
        "shape_flow_noise_sample": packet.shape_flow_noise_sample,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_identity": _manifest_identity(packet),
        "source_identity": {
            "dataset_sources": [packet.dataset_id],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
    }
    return f"""#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONFIG = json.loads({json.dumps(json.dumps(config, sort_keys=True))})
RECEIPT = Path("kaggle_cuda_witness_receipt.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cuda_snapshot() -> dict:
    available = False
    device = None
    torch_version = None
    try:
        import torch

        torch_version = torch.__version__
        available = bool(torch.cuda.is_available())
        if available:
            device = torch.cuda.get_device_name(0)
    except Exception as exc:
        torch_version = f"unavailable: {{type(exc).__name__}}: {{exc}}"
    return {{
        "cuda_available": available,
        "cuda_device": device,
        "torch": torch_version,
    }}


CUDA = cuda_snapshot()


def write_receipt(status: str, *, phase: str, message: str | None, extra: dict | None = None) -> None:
    payload = {{
        "schema": "trellis2mlx.kaggle_cuda_witness.receipt.v1",
        "status": status,
        "failure_phase": None if status == "done" else phase,
        "message": message,
        "requested_dataset_id": CONFIG["dataset_id"],
        "requested_kernel_id": CONFIG["kernel_id"],
        "requested_accelerator": CONFIG["accelerator"],
        "source_identity": CONFIG["source_identity"],
        **CUDA,
        "timestamp": time.time(),
    }}
    if extra:
        payload.update(extra)
    RECEIPT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")


def mounted_input_snapshot() -> dict:
    root = Path("/kaggle/input")
    if not root.exists():
        return {{"mounted_input_root_exists": False, "mounted_input_dirs": [], "mounted_input_files": []}}
    dirs = [str(path) for path in sorted(root.rglob("*")) if path.is_dir()]
    files = [str(path) for path in sorted(root.rglob("*")) if path.is_file()]
    return {{"mounted_input_root_exists": True, "mounted_input_dirs": dirs, "mounted_input_files": files}}


def find_manifest() -> Path | None:
    preferred = Path("/kaggle/input") / CONFIG["dataset_slug"] / "witness-manifest.json"
    if preferred.exists():
        return preferred
    candidates = sorted(Path("/kaggle/input").rglob("witness-manifest.json"))
    if candidates:
        return candidates[0]
    return None


def output_snapshot() -> dict:
    records = {{}}
    for name in CONFIG["outputs"]:
        path = Path(name)
        if not path.is_file():
            records[name] = {{"exists": False, "sha256": None, "size_bytes": None}}
            continue
        records[name] = {{
            "exists": True,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }}
    return records


def main() -> int:
    manifest_path = find_manifest()
    if manifest_path is None:
        expected = Path("/kaggle/input") / CONFIG["dataset_slug"] / "witness-manifest.json"
        write_receipt("failed", phase="input_mount", message=f"missing {{expected}}", extra=mounted_input_snapshot())
        return 2
    dataset_dir = manifest_path.parent
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != CONFIG["manifest_sha256"]:
        write_receipt(
            "failed",
            phase="input_manifest_digest",
            message="mounted witness manifest digest mismatch",
            extra={{
                "expected_manifest_sha256": CONFIG["manifest_sha256"],
                "actual_manifest_sha256": actual_manifest_sha256,
                "manifest_path": str(manifest_path),
                "mounted_input_snapshot": mounted_input_snapshot(),
            }},
        )
        return 3
    manifest = json.loads(manifest_path.read_text())
    actual_manifest_identity = {{
        field: manifest.get(field)
        for field in CONFIG["manifest_identity"]
    }}
    if actual_manifest_identity != CONFIG["manifest_identity"]:
        write_receipt(
            "failed",
            phase="input_manifest_identity",
            message="mounted witness manifest identity mismatch",
            extra={{
                "expected_manifest_identity": CONFIG["manifest_identity"],
                "actual_manifest_identity": actual_manifest_identity,
                "input_manifest": {{
                    "sha256": actual_manifest_sha256,
                    "size_bytes": manifest_path.stat().st_size,
                }},
                "mounted_input_snapshot": mounted_input_snapshot(),
            }},
        )
        return 3
    if (
        CUDA["cuda_available"] is not True
        or not isinstance(CUDA["cuda_device"], str)
        or not CUDA["cuda_device"].strip()
    ):
        write_receipt(
            "failed",
            phase="cuda_route",
            message="CUDA route is unavailable",
            extra={{
                "input_manifest": {{
                    "sha256": actual_manifest_sha256,
                    "size_bytes": manifest_path.stat().st_size,
                }},
                "mounted_input_snapshot": mounted_input_snapshot(),
            }},
        )
        return 6
    copied = {{}}
    for relative_name, record in manifest["files"].items():
        if relative_name == "witness-manifest.json":
            continue
        source = dataset_dir / relative_name
        if not source.exists():
            extra = {{"manifest": manifest, "missing_input": str(source)}}
            extra.update(mounted_input_snapshot())
            write_receipt("failed", phase="input_mount", message=f"missing input {{source}}", extra=extra)
            return 3
        actual_sha = sha256_file(source)
        if actual_sha != record["sha256"]:
            write_receipt(
                "failed",
                phase="input_digest",
                message=f"sha256 mismatch for {{relative_name}}",
                extra={{"expected_sha256": record["sha256"], "actual_sha256": actual_sha, "manifest": manifest}},
            )
            return 4
        destination = Path(relative_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied[relative_name] = {{"sha256": actual_sha, "size_bytes": destination.stat().st_size}}

    command = [sys.executable, CONFIG["entrypoint"], "--output-json", CONFIG["output_json"]]
    if CONFIG["output_npz"]:
        command += ["--output-npz", CONFIG["output_npz"]]
    command += CONFIG.get("entrypoint_args", [])
    if CONFIG["output_ply"]:
        command += ["--output-ply", CONFIG["output_ply"]]
    if CONFIG["output_mesh_state"]:
        command += ["--output-mesh-state", CONFIG["output_mesh_state"]]
    if CONFIG["output_shape_slat"]:
        command += ["--output-shape-slat", CONFIG["output_shape_slat"]]
    if CONFIG["output_shape_flow_step"]:
        command += ["--output-shape-flow-step", CONFIG["output_shape_flow_step"]]
    if CONFIG["shape_flow_noise_sample"]:
        command += ["--shape-flow-noise-sample", CONFIG["shape_flow_noise_sample"]]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    extra = {{
        "effective_dataset_dir": str(dataset_dir),
        "effective_command": command,
        "input_manifest": {{
            "sha256": sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        }},
        "inputs": copied,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "exit_code": completed.returncode,
        "outputs": output_snapshot(),
        "mounted_input_snapshot": mounted_input_snapshot(),
    }}
    if completed.returncode != 0:
        write_receipt("failed", phase="execution", message="probe exited non-zero", extra=extra)
        return completed.returncode
    missing = [name for name, record in extra["outputs"].items() if not record["exists"]]
    if missing:
        write_receipt("failed", phase="output", message=f"missing outputs: {{missing}}", extra=extra)
        return 5
    write_receipt("done", phase="done", message=None, extra=extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _bool_metadata(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _canonical_output_name(output: str) -> str:
    if not isinstance(output, str) or not output:
        raise WitnessPacketError("output name must be a nonempty string")
    output_path = PurePosixPath(output)
    canonical = output_path.as_posix()
    if output_path.is_absolute() or ".." in output_path.parts:
        raise WitnessPacketError(
            f"output must be relative inside the kernel output directory: {output!r}"
        )
    if canonical in {"", "."} or canonical != output:
        raise WitnessPacketError(
            f"output name must use canonical relative POSIX syntax: {output!r}"
        )
    return canonical


def _manifest_identity(packet: KaggleCudaWitnessPacket) -> dict[str, object]:
    return {
        "schema": "trellis2mlx.kaggle_cuda_witness.inputs.v1",
        "dataset_id": packet.dataset_id,
        "kernel_id": packet.kernel_id,
        "title": packet.title,
        "entrypoint": packet.entrypoint,
        "entrypoint_args": list(packet.entrypoint_args),
        "accelerator": packet.accelerator,
        "enable_internet": packet.enable_internet,
        "outputs": list(packet.outputs),
        "output_roles": {
            "json": packet.output_json,
            "npz": packet.output_npz,
            "ply": packet.output_ply,
            "mesh_state": packet.output_mesh_state,
            "shape_slat": packet.output_shape_slat,
            "shape_flow_step": packet.output_shape_flow_step,
            "expected": list(packet.expected_outputs),
        },
    }


def _legacy_output_by_suffix(outputs: Sequence[str], suffix: str) -> str | None:
    for output in outputs:
        if output.endswith(suffix):
            return output
    return None


def _slug_from_ref(kaggle_ref: str) -> str:
    return kaggle_ref.split("/", 1)[1]


def _kaggle_slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
