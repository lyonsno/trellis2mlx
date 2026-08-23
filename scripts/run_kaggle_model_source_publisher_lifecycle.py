#!/usr/bin/env python3
"""Drive one canonical model-source publisher packet to admission."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from trellmlx.kaggle_cuda_witness import run_command
from trellmlx.kaggle_model_source_publisher import (
    ModelSourcePublisherError,
    build_kernel_output_command,
    build_kernel_push_command,
    build_kernel_status_command,
    load_publisher_packet,
    render_publisher_runner,
    sha256_file,
    validate_downloaded_publisher_receipt,
)
try:
    from scripts.prepare_kaggle_model_source_publisher import (
        validate_r11_publisher_authority,
    )
    from scripts.run_kaggle_model_source_conformance_lifecycle import (
        STATUS_PATTERN,
        TERMINAL_STATUSES,
        RemoteKernelTerminalError,
        begin_phase,
        finish_phase,
        normalize_kernel_status,
        prepare_download_dir,
        select_failure_identity,
        snapshot_recovered_files,
        utc_now,
        verify_local_custody,
        write_json,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from prepare_kaggle_model_source_publisher import (
        validate_r11_publisher_authority,
    )
    from run_kaggle_model_source_conformance_lifecycle import (
        STATUS_PATTERN,
        TERMINAL_STATUSES,
        RemoteKernelTerminalError,
        begin_phase,
        finish_phase,
        normalize_kernel_status,
        prepare_download_dir,
        select_failure_identity,
        snapshot_recovered_files,
        utc_now,
        verify_local_custody,
        write_json,
    )


def verify_generator_custody(object_root: Path) -> dict[str, str]:
    object_root = Path(object_root).resolve(strict=True)
    functions = {
        "load_publisher_packet": load_publisher_packet,
        "render_publisher_runner": render_publisher_runner,
        "validate_r11_publisher_authority": validate_r11_publisher_authority,
        "validate_downloaded_publisher_receipt": (
            validate_downloaded_publisher_receipt
        ),
        "lifecycle_helpers": begin_phase,
    }
    paths = {}
    for name, function in functions.items():
        source_file = inspect.getsourcefile(function)
        if source_file is None:
            raise RuntimeError(f"cannot resolve effective generator source: {name}")
        source_path = Path(source_file).resolve(strict=True)
        if not source_path.is_relative_to(object_root):
            raise RuntimeError(
                "effective generator source is outside object root: "
                f"{name}={source_path}, root={object_root}"
            )
        paths[name] = str(source_path)
    lifecycle_path = Path(__file__).resolve(strict=True)
    if not lifecycle_path.is_relative_to(object_root):
        raise RuntimeError(
            "effective lifecycle source is outside object root: "
            f"path={lifecycle_path}, root={object_root}"
        )
    paths["lifecycle"] = str(lifecycle_path)
    return paths


def validate_prepared_runner(packet) -> dict[str, object]:
    runner_path = packet.kernel_dir / packet.code_file
    expected = render_publisher_runner(packet).encode("utf-8")
    try:
        actual = runner_path.read_bytes()
    except OSError as exc:
        raise ModelSourcePublisherError(
            f"cannot read prepared runner: {runner_path}"
        ) from exc
    if actual != expected:
        raise ModelSourcePublisherError(
            "prepared runner diverges from the verified object generator"
        )
    return {"path": str(runner_path), "sha256": hashlib.sha256(actual).hexdigest()}


def claim_prepared_attempt(
    registry_root: Path,
    *,
    packet,
    packet_manifest_sha256: str,
    runner_sha256: str,
    object_commit: str,
    lifecycle_report: Path,
) -> dict[str, object]:
    owner, slug = packet.kernel_id.split("/", 1)
    packet_dir = Path(packet.output_dir).resolve(strict=True)
    claim_path = (
        Path(registry_root) / owner / slug / f"{packet.attempt_id}.json"
    ).resolve(strict=False)
    if claim_path == packet_dir or claim_path.is_relative_to(packet_dir):
        raise ModelSourcePublisherError(
            "attempt claim must resolve outside the prepared packet: "
            f"claim={claim_path}, packet={packet_dir}"
        )
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim = {
        "schema": "trellis2mlx.kaggle_model_source_publisher.attempt_claim.v1",
        "kernel_id": packet.kernel_id,
        "attempt_id": packet.attempt_id,
        "packet_contract_sha256": packet.packet_contract_sha256,
        "packet_manifest_sha256": packet_manifest_sha256,
        "runner_sha256": runner_sha256,
        "object_commit": object_commit,
        "lifecycle_report": str(Path(lifecycle_report).resolve()),
        "claimed_at": utc_now(),
    }
    encoded = (json.dumps(claim, indent=2, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(
            claim_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise ModelSourcePublisherError(
            "prepared attempt has already been consumed: "
            f"kernel={packet.kernel_id}, attempt={packet.attempt_id}"
        ) from exc
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)
    return {**claim, "path": str(claim_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--lifecycle-report", type=Path, required=True)
    parser.add_argument("--failure-report", type=Path, required=True)
    parser.add_argument("--attempt-registry-root", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--object-commit", required=True)
    parser.add_argument("--expected-kernel-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packet_dir = args.packet_dir.resolve()
    download_dir = args.download_dir.resolve()
    lifecycle_path = args.lifecycle_report.resolve()
    failure_path = args.failure_report.resolve()
    object_root = args.object_root.resolve()
    registry_root = args.attempt_registry_root.resolve()
    started_epoch = time.time()
    lifecycle: dict[str, object] = {
        "schema": "trellis2mlx.kaggle_model_source_publisher.lifecycle.v1",
        "status": "running",
        "current_phase": "preflight",
        "failure_phase": None,
        "last_trustworthy_phase": None,
        "started_at": utc_now(),
        "started_epoch": started_epoch,
        "terminal_at": None,
        "total_elapsed_seconds": None,
        "packet_dir": str(packet_dir),
        "download_dir": str(download_dir),
        "object_root": str(object_root),
        "object_commit": args.object_commit,
        "attempt_registry_root": str(registry_root),
        "kaggle_token_present": bool(os.environ.get("KAGGLE_API_TOKEN")),
        "phase_events": [],
        "kernel_status_observations": [],
    }
    write_json(lifecycle_path, lifecycle)
    active_event: dict[str, object] | None = None
    try:
        active_event = begin_phase(lifecycle, lifecycle_path, "preflight")
        if not lifecycle["kaggle_token_present"]:
            raise RuntimeError("KAGGLE_API_TOKEN is absent")
        lifecycle["local_custody"] = verify_local_custody(
            object_root, args.object_commit
        )
        lifecycle["generator_custody"] = verify_generator_custody(object_root)
        packet = load_publisher_packet(packet_dir)
        if packet.kernel_id != args.expected_kernel_id:
            raise RuntimeError(
                "packet kernel mismatch: "
                f"expected={args.expected_kernel_id}, actual={packet.kernel_id}"
            )
        validate_r11_publisher_authority(packet)
        runner_identity = validate_prepared_runner(packet)
        packet_manifest_sha256 = sha256_file(packet_dir / "packet.json")
        lifecycle["admitted_runner"] = runner_identity
        lifecycle["packet_manifest_sha256"] = packet_manifest_sha256
        lifecycle["attempt_claim"] = claim_prepared_attempt(
            registry_root,
            packet=packet,
            packet_manifest_sha256=packet_manifest_sha256,
            runner_sha256=str(runner_identity["sha256"]),
            object_commit=args.object_commit,
            lifecycle_report=lifecycle_path,
        )
        lifecycle.update(
            {
                "kernel_id": packet.kernel_id,
                "attempt_id": packet.attempt_id,
                "packet_contract_sha256": packet.packet_contract_sha256,
                "requested_accelerator": None,
                "repository_count": len(packet.repositories),
                "blob_count": packet.blob_count,
                "published_blob_bytes": packet.published_blob_bytes,
                "last_trustworthy_phase": (
                    "packet_reloaded_and_private_cpu_route_validated"
                ),
            }
        )
        finish_phase(lifecycle, lifecycle_path, active_event)
        active_event = None

        active_event = begin_phase(lifecycle, lifecycle_path, "kernel_push")
        push = run_command(
            build_kernel_push_command(packet),
            phase="kernel_push",
            report_path=lifecycle_path.parent / "kernel-push.json",
        )
        lifecycle["kernel_push"] = push
        if push.get("status") != "done":
            raise RuntimeError(f"kernel push failed: {push}")
        lifecycle["last_trustworthy_phase"] = "private_cpu_kernel_push_accepted"
        finish_phase(lifecycle, lifecycle_path, active_event)
        active_event = None

        active_event = begin_phase(lifecycle, lifecycle_path, "kernel_wait")
        wait_started = float(active_event["started_epoch"])
        first_running_epoch: float | None = None
        terminal_epoch: float | None = None
        terminal_status: str | None = None
        while terminal_status is None:
            command = build_kernel_status_command(packet)
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            observed_epoch = time.time()
            observation: dict[str, object] = {
                "observed_at": utc_now(),
                "observed_epoch": observed_epoch,
                "command": command,
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
            match = STATUS_PATTERN.search(completed.stdout)
            if completed.returncode != 0 or match is None:
                lifecycle["kernel_status_observations"].append(observation)
                write_json(lifecycle_path, lifecycle)
                raise RuntimeError(f"kernel status observation failed: {observation}")
            raw_status = match.group(1)
            status = normalize_kernel_status(raw_status)
            observation["raw_status"] = raw_status
            observation["normalized_status"] = status
            lifecycle["kernel_status_observations"].append(observation)
            lifecycle["last_kernel_status"] = status
            lifecycle["last_trustworthy_phase"] = "kernel_status_observed"
            if status == "running" and first_running_epoch is None:
                first_running_epoch = observed_epoch
                lifecycle["first_running_at"] = observation["observed_at"]
            if status in TERMINAL_STATUSES:
                terminal_status = status
                terminal_epoch = observed_epoch
            write_json(lifecycle_path, lifecycle)
            if terminal_status is None:
                time.sleep(10.0)
        assert terminal_epoch is not None
        lifecycle["terminal_kernel_status"] = terminal_status
        lifecycle["kernel_timing"] = {
            "wait_total_seconds": terminal_epoch - wait_started,
            "queue_seconds_to_first_running": (
                first_running_epoch - wait_started
                if first_running_epoch is not None
                else None
            ),
            "execution_seconds_after_first_running": (
                terminal_epoch - first_running_epoch
                if first_running_epoch is not None
                else None
            ),
            "first_running_observed": first_running_epoch is not None,
        }
        if terminal_status != "complete":
            lifecycle["primary_failure"] = {
                "failure_phase": "kernel_terminal",
                "error_type": "RemoteKernelTerminalError",
                "error_message": (
                    f"kernel terminal status is not complete: {terminal_status}"
                ),
            }
        finish_phase(lifecycle, lifecycle_path, active_event)
        active_event = None

        active_event = begin_phase(
            lifecycle, lifecycle_path, "kernel_output_download"
        )
        prepare_download_dir(download_dir)
        download = run_command(
            build_kernel_output_command(packet, download_dir),
            phase="kernel_output_download",
            report_path=lifecycle_path.parent / "kernel-output-download.json",
        )
        lifecycle["kernel_output_download"] = download
        lifecycle["recovered_files"] = snapshot_recovered_files(download_dir)
        if download.get("status") != "done":
            raise RuntimeError(f"kernel output download failed: {download}")
        lifecycle["last_trustworthy_phase"] = "kernel_receipt_downloaded"
        finish_phase(lifecycle, lifecycle_path, active_event)
        active_event = None

        if terminal_status != "complete":
            raise RemoteKernelTerminalError(
                f"kernel terminal status is not complete: {terminal_status}"
            )

        active_event = begin_phase(lifecycle, lifecycle_path, "semantic_validation")
        receipt = validate_downloaded_publisher_receipt(packet, download_dir)
        lifecycle.update(
            {
                "last_trustworthy_phase": (
                    "live_model_source_publication_admitted"
                ),
                "effective_cache_root": receipt["effective_cache_root"],
                "admitted_blob_count": receipt["blob_count"],
                "admitted_blob_bytes": receipt["published_blob_bytes"],
            }
        )
        finish_phase(lifecycle, lifecycle_path, active_event)
        active_event = None

        ended_epoch = time.time()
        lifecycle.update(
            {
                "status": "completed",
                "current_phase": "completed",
                "terminal_at": utc_now(),
                "total_elapsed_seconds": ended_epoch - started_epoch,
            }
        )
        failure_path.unlink(missing_ok=True)
        write_json(lifecycle_path, lifecycle)
        return 0
    except BaseException as exc:
        if active_event is not None and active_event.get("status") == "running":
            finish_phase(lifecycle, lifecycle_path, active_event, status="failed")
        ended_epoch = time.time()
        failure_phase, error_type, error_message = select_failure_identity(
            lifecycle, exc
        )
        lifecycle.update(
            {
                "status": "failed",
                "current_phase": failure_phase,
                "failure_phase": failure_phase,
                "terminal_at": utc_now(),
                "total_elapsed_seconds": ended_epoch - started_epoch,
                "error_type": error_type,
                "error_message": error_message,
                "traceback": traceback.format_exc(),
            }
        )
        write_json(lifecycle_path, lifecycle)
        write_json(failure_path, lifecycle)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
