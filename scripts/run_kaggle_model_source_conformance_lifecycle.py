#!/usr/bin/env python3
"""Drive one prepared CPU model-source conformance packet to admission."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

from trellmlx.kaggle_cuda_witness import run_command
from trellmlx.kaggle_model_source_conformance import (
    ModelSourceConformanceError,
    build_kernel_output_command,
    build_kernel_push_command,
    build_kernel_status_command,
    load_prepared_packet,
    validate_downloaded_receipt,
)
try:
    from scripts.prepare_kaggle_model_source_conformance import validate_r11_authority
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from prepare_kaggle_model_source_conformance import validate_r11_authority


STATUS_PATTERN = re.compile(r'has status "([^"]+)"', re.IGNORECASE)
TERMINAL_STATUSES = {"complete", "error", "cancelled", "failed"}


class RemoteKernelTerminalError(RuntimeError):
    """The remote kernel failed before optional local output recovery."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def normalize_kernel_status(value: str) -> str:
    return value.strip().lower().rsplit(".", 1)[-1]


def prepare_download_dir(download_dir: Path) -> Path:
    download_dir = Path(download_dir)
    if download_dir.exists():
        if not download_dir.is_dir():
            raise ModelSourceConformanceError(
                f"kernel output download path is not a directory: {download_dir}"
            )
        if any(download_dir.iterdir()):
            raise ModelSourceConformanceError(
                f"kernel output download directory is not empty: {download_dir}"
            )
    else:
        download_dir.mkdir(parents=True)
    return download_dir


def select_failure_identity(
    lifecycle: dict[str, object],
    exc: BaseException,
) -> tuple[str, str, str]:
    primary_failure = lifecycle.get("primary_failure")
    if isinstance(primary_failure, dict):
        return (
            str(primary_failure["failure_phase"]),
            str(primary_failure["error_type"]),
            str(primary_failure["error_message"]),
        )
    return (
        str(lifecycle.get("current_phase") or "unknown"),
        type(exc).__name__,
        str(exc),
    )


def begin_phase(
    lifecycle: dict[str, object],
    path: Path,
    phase: str,
) -> dict[str, object]:
    event: dict[str, object] = {
        "phase": phase,
        "started_at": utc_now(),
        "started_epoch": time.time(),
        "status": "running",
    }
    lifecycle["current_phase"] = phase
    lifecycle["phase_events"].append(event)
    write_json(path, lifecycle)
    return event


def finish_phase(
    lifecycle: dict[str, object],
    path: Path,
    event: dict[str, object],
    *,
    status: str = "completed",
) -> None:
    ended_epoch = time.time()
    event.update(
        {
            "status": status,
            "ended_at": utc_now(),
            "ended_epoch": ended_epoch,
            "elapsed_seconds": ended_epoch - float(event["started_epoch"]),
        }
    )
    write_json(path, lifecycle)


def verify_local_custody(object_root: Path, expected_commit: str) -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=object_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=object_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if head != expected_commit or status:
        raise RuntimeError(
            f"object custody drift: expected clean {expected_commit}, "
            f"got {head}, status={status!r}"
        )
    return {"object_commit": head, "object_status": "clean"}


def snapshot_recovered_files(download_dir: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(Path(download_dir).rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append(
            {
                "path": path.relative_to(download_dir).as_posix(),
                "sha256": digest.hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def claim_prepared_attempt(packet_dir: Path, attempt_id: str) -> dict[str, object]:
    claim_path = Path(packet_dir) / "attempt-consumed.json"
    claim = {
        "schema": "trellis2mlx.kaggle_model_source_conformance.attempt_claim.v1",
        "attempt_id": attempt_id,
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
        raise ModelSourceConformanceError(
            f"prepared attempt has already been consumed: {attempt_id}"
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
    started_epoch = time.time()
    lifecycle: dict[str, object] = {
        "schema": "trellis2mlx.kaggle_model_source_conformance.lifecycle.v1",
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
            object_root,
            args.object_commit,
        )
        packet = load_prepared_packet(packet_dir)
        if packet.kernel_id != args.expected_kernel_id:
            raise RuntimeError(
                "packet kernel mismatch: "
                f"expected={args.expected_kernel_id}, actual={packet.kernel_id}"
            )
        validate_r11_authority(packet)
        lifecycle["attempt_claim"] = claim_prepared_attempt(
            packet_dir,
            packet.attempt_id,
        )
        lifecycle.update(
            {
                "kernel_id": packet.kernel_id,
                "source_kernel": packet.source_kernel,
                "attempt_id": packet.attempt_id,
                "packet_contract_sha256": packet.packet_contract_sha256,
                "requested_accelerator": None,
                "blob_count": len(packet.blobs),
                "mounted_blob_bytes": packet.mounted_blob_bytes,
                "last_trustworthy_phase": "packet_reloaded_and_cpu_route_validated",
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
                command,
                capture_output=True,
                text=True,
                check=False,
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
                    "kernel terminal status is not complete: " f"{terminal_status}"
                ),
            }
        finish_phase(lifecycle, lifecycle_path, active_event)
        active_event = None

        active_event = begin_phase(lifecycle, lifecycle_path, "kernel_output_download")
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
        lifecycle["last_trustworthy_phase"] = "kernel_output_downloaded"
        finish_phase(lifecycle, lifecycle_path, active_event)
        active_event = None

        if terminal_status != "complete":
            raise RemoteKernelTerminalError(
                f"kernel terminal status is not complete: {terminal_status}"
            )

        active_event = begin_phase(lifecycle, lifecycle_path, "semantic_validation")
        receipt = validate_downloaded_receipt(packet, download_dir)
        lifecycle.update(
            {
                "last_trustworthy_phase": "live_model_source_conformance_admitted",
                "effective_mount_root": receipt["effective_mount_root"],
                "effective_blob_root": receipt["effective_blob_root"],
                "admitted_blob_count": len(receipt["blobs"]),
                "admitted_blob_bytes": receipt["mounted_blob_bytes"],
                "writable_model_bytes": receipt["writable_model_bytes"],
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
            lifecycle,
            exc,
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
