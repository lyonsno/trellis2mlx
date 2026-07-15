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
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


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
    output_npz: str = "cuda_result.npz"
    output_ply: str | None = None
    output_mesh_state: str | None = None
    output_shape_slat: str | None = None
    output_shape_flow_step: str | None = None
    shape_flow_noise_sample: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "capsule_dir", Path(self.capsule_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "entrypoint_args", tuple(self.entrypoint_args))

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
        outputs = [self.output_json, self.output_npz]
        if self.output_ply:
            outputs.append(self.output_ply)
        if self.output_mesh_state:
            outputs.append(self.output_mesh_state)
        if self.output_shape_slat:
            outputs.append(self.output_shape_slat)
        if self.output_shape_flow_step:
            outputs.append(self.output_shape_flow_step)
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
    if len(outputs) < 2:
        raise WitnessPacketError(f"expected at least two outputs in manifest, got {outputs}")
    return KaggleCudaWitnessPacket(
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
        output_json=output_roles.get("json") or outputs[0],
        output_npz=output_roles.get("npz") or outputs[1],
        output_ply=output_roles.get("ply") or _legacy_output_by_suffix(outputs[2:], ".ply"),
        output_mesh_state=output_roles.get("mesh_state") or _legacy_output_by_suffix(outputs[2:], "_mesh_state.npz"),
        output_shape_slat=output_roles.get("shape_slat") or _legacy_output_by_suffix(outputs[2:], "_shape_slat.npz"),
        output_shape_flow_step=(
            output_roles.get("shape_flow_step")
            or _legacy_output_by_suffix(outputs[2:], "_shape_flow_step.npz")
        ),
        shape_flow_noise_sample=input_roles.get("shape_flow_noise_sample"),
    )


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
    return [
        "kaggle",
        "kernels",
        "output",
        packet.kernel_id,
        "-p",
        str(output_dir),
        "-o",
        "--file-pattern",
        ".*(cuda_result|kaggle_cuda_witness_receipt).*",
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
    records: dict[str, dict[str, str | int]] = {}
    names = tuple(dict.fromkeys((*packet.outputs, "kaggle_cuda_witness_receipt.json")))
    for name in names:
        path = output_dir / name
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
        if path.suffix == ".npz" and not zipfile.is_zipfile(path):
            raise WitnessPacketError(f"invalid NPZ downloaded output: {path}")
        records[name] = {
            "size_bytes": size,
            "sha256": sha256_file(path),
        }
    return records


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
    prepare.add_argument("--output-ply")
    prepare.add_argument("--output-mesh-state")
    prepare.add_argument("--output-shape-slat")
    prepare.add_argument("--output-shape-flow-step")
    prepare.add_argument("--shape-flow-noise-sample")

    for name in ("dataset-create", "dataset-version", "kernel-push", "kernel-status", "kernel-output", "print-commands"):
        drive = subparsers.add_parser(name)
        drive.add_argument("--packet-dir", type=Path, required=True)
        drive.add_argument("--report-dir", type=Path)
        if name == "kernel-push":
            drive.add_argument("--timeout-seconds", type=int)
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
                output_npz=args.output_npz,
                output_ply=args.output_ply,
                output_mesh_state=args.output_mesh_state,
                output_shape_slat=args.output_shape_slat,
                output_shape_flow_step=args.output_shape_flow_step,
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
        return _run_cli_command(build_dataset_command(packet, version=True), "dataset_version", args)
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
    config = {
        "schema": "trellis2mlx.kaggle_cuda_witness.runner_config.v1",
        "dataset_id": packet.dataset_id,
        "dataset_slug": packet.dataset_slug,
        "kernel_id": packet.kernel_id,
        "accelerator": packet.accelerator,
        "entrypoint": packet.entrypoint,
        "entrypoint_args": list(packet.entrypoint_args),
        "outputs": list(packet.outputs),
        "output_ply": packet.output_ply,
        "output_mesh_state": packet.output_mesh_state,
        "output_shape_slat": packet.output_shape_slat,
        "output_shape_flow_step": packet.output_shape_flow_step,
        "shape_flow_noise_sample": packet.shape_flow_noise_sample,
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


def write_receipt(status: str, *, phase: str, message: str | None, extra: dict | None = None) -> None:
    cuda_available = None
    cuda_device = None
    torch_version = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_device = torch.cuda.get_device_name(0)
    except Exception as exc:
        torch_version = f"unavailable: {{type(exc).__name__}}: {{exc}}"
    payload = {{
        "schema": "trellis2mlx.kaggle_cuda_witness.receipt.v1",
        "status": status,
        "failure_phase": None if status == "done" else phase,
        "message": message,
        "requested_dataset_id": CONFIG["dataset_id"],
        "requested_kernel_id": CONFIG["kernel_id"],
        "requested_accelerator": CONFIG["accelerator"],
        "source_identity": CONFIG["source_identity"],
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "torch": torch_version,
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
    manifest = json.loads(manifest_path.read_text())
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

    command = [sys.executable, CONFIG["entrypoint"], "--output-json", CONFIG["outputs"][0], "--output-npz", CONFIG["outputs"][1]]
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
