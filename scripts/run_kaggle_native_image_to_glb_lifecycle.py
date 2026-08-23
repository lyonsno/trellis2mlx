#!/usr/bin/env python3
"""Drive one prepared native image-to-GLB packet to semantic admission."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

from trellmlx.kaggle_cuda_witness import (
    admit_kernel_output_download_dir,
    build_dataset_command,
    build_kernel_output_command,
    build_kernel_push_command,
    build_kernel_status_command,
    load_prepared_packet,
    run_command,
    wait_for_downloaded_outputs,
    wait_for_published_dataset_manifest,
)


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


def require_done(report: dict[str, object], phase: str) -> None:
    if report.get("status") != "done":
        raise RuntimeError(f"{phase} failed: {report}")


def prepare_kernel_output_download(download_dir: Path) -> Path:
    return admit_kernel_output_download_dir(download_dir)


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


def snapshot_recovered_files(download_dir: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(download_dir.rglob("*")):
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


def begin_phase(lifecycle: dict[str, object], path: Path, phase: str) -> dict[str, object]:
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


def recover_kernel_output(
    *,
    packet: object,
    download_dir: Path,
    lifecycle: dict[str, object],
    lifecycle_path: Path,
    terminal_status: str,
) -> dict[str, object] | None:
    remote_failed = terminal_status != "complete"
    if remote_failed:
        lifecycle["primary_failure"] = {
            "failure_phase": "kernel_terminal",
            "terminal_kernel_status": terminal_status,
            "error_type": "RemoteKernelTerminalError",
            "error_message": f"kernel terminal status is not complete: {terminal_status}",
        }
        lifecycle["last_trustworthy_phase"] = "kernel_terminal_failure_observed"
        write_json(lifecycle_path, lifecycle)

    event = begin_phase(lifecycle, lifecycle_path, "kernel_output_download")
    try:
        prepare_kernel_output_download(download_dir)
        output_report = run_command(
            build_kernel_output_command(packet, download_dir),
            phase="kernel_output",
            report_path=lifecycle_path.parent / "kernel-output.json",
        )
        lifecycle["kernel_output"] = output_report
        require_done(output_report, "kernel_output")
        if remote_failed:
            lifecycle["output_recovery"] = {
                "status": "recovered",
                "files": snapshot_recovered_files(download_dir),
            }
            downloaded = None
        else:
            downloaded = wait_for_downloaded_outputs(packet, download_dir)
            lifecycle["downloaded_outputs"] = downloaded
            lifecycle["output_recovery"] = {"status": "completed"}
        finish_phase(lifecycle, lifecycle_path, event)
        return downloaded
    except BaseException as exc:
        finish_phase(lifecycle, lifecycle_path, event, status="failed")
        if not remote_failed:
            raise
        lifecycle["output_recovery"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(lifecycle_path, lifecycle)
        return None


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


def load_native_module(packet_dir: Path):
    path = packet_dir / "dataset" / "source_cuda_native_image_to_glb_witness.py"
    spec = importlib.util.spec_from_file_location("attempt_native_image_to_glb", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load packet native-image module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--lifecycle-report", type=Path, required=True)
    parser.add_argument("--failure-report", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--object-commit", required=True)
    parser.add_argument("--expected-run-id", required=True)
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
        "schema": "trellis2mlx.native_image_to_glb_kaggle_lifecycle.v2",
        "status": "running",
        "current_phase": "preflight",
        "failure_phase": None,
        "last_trustworthy_phase": None,
        "started_at": utc_now(),
        "started_epoch": started_epoch,
        "terminal_at": None,
        "total_elapsed_seconds": None,
        "run_id": args.expected_run_id,
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
        lifecycle["local_custody"] = verify_local_custody(object_root, args.object_commit)
        packet = load_prepared_packet(packet_dir)
        if packet.run_id != args.expected_run_id:
            raise RuntimeError(
                f"packet run mismatch: expected {args.expected_run_id}, got {packet.run_id}"
            )
        if packet.enable_internet is not True:
            raise RuntimeError("packet internet route is disabled")
        if packet.accelerator != "NvidiaTeslaT4":
            raise RuntimeError(f"packet accelerator is not exact T4: {packet.accelerator}")
        lifecycle.update(
            {
                "dataset_id": packet.dataset_id,
                "kernel_id": packet.kernel_id,
                "entrypoint": packet.entrypoint,
                "entrypoint_args": list(packet.entrypoint_args),
                "expected_outputs": list(packet.outputs),
                "effective_route": {
                    "accelerator": packet.accelerator,
                    "enable_internet": packet.enable_internet,
                    "dataset_id": packet.dataset_id,
                    "kernel_id": packet.kernel_id,
                    "kernel_sources": list(packet.kernel_sources),
                },
                "last_trustworthy_phase": "packet_reloaded_and_route_validated",
            }
        )
        finish_phase(lifecycle, lifecycle_path, active_event)
        active_event = None

        active_event = begin_phase(lifecycle, lifecycle_path, "dataset_create")
        dataset_create = run_command(
            build_dataset_command(packet),
            phase="dataset_create",
            report_path=lifecycle_path.parent / "dataset-create.json",
        )
        lifecycle["dataset_create"] = dataset_create
        require_done(dataset_create, "dataset_create")
        lifecycle["last_trustworthy_phase"] = "dataset_create_accepted"
        finish_phase(lifecycle, lifecycle_path, active_event)
        active_event = None

        active_event = begin_phase(lifecycle, lifecycle_path, "dataset_publication")
        publication = wait_for_published_dataset_manifest(
            packet,
            report_path=lifecycle_path.parent / "dataset-publication.json",
            poll_seconds=2.0,
            max_attempts=None,
        )
        lifecycle["dataset_publication"] = publication
        require_done(publication, "dataset_publication")
        lifecycle["last_trustworthy_phase"] = "published_manifest_digest_matched"
        finish_phase(lifecycle, lifecycle_path, active_event)
        active_event = None

        active_event = begin_phase(lifecycle, lifecycle_path, "kernel_push")
        kernel_push = run_command(
            build_kernel_push_command(packet, timeout_seconds=None),
            phase="kernel_push",
            report_path=lifecycle_path.parent / "kernel-push.json",
        )
        lifecycle["kernel_push"] = kernel_push
        require_done(kernel_push, "kernel_push")
        lifecycle["last_trustworthy_phase"] = "private_kernel_push_accepted"
        finish_phase(lifecycle, lifecycle_path, active_event)
        active_event = None

        active_event = begin_phase(lifecycle, lifecycle_path, "kernel_wait")
        wait_started = float(active_event["started_epoch"])
        first_running_epoch: float | None = None
        terminal_epoch: float | None = None
        terminal_status: str | None = None
        while terminal_status is None:
            command = build_kernel_status_command(packet)
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
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
        lifecycle["terminal_kernel_status"] = terminal_status
        lifecycle["kernel_timing"] = {
            "wait_total_seconds": terminal_epoch - wait_started,
            "queue_seconds_to_first_running": (
                first_running_epoch - wait_started if first_running_epoch is not None else None
            ),
            "execution_seconds_after_first_running": (
                terminal_epoch - first_running_epoch if first_running_epoch is not None else None
            ),
            "first_running_observed": first_running_epoch is not None,
        }
        finish_phase(lifecycle, lifecycle_path, active_event)
        active_event = None

        downloaded = recover_kernel_output(
            packet=packet,
            download_dir=download_dir,
            lifecycle=lifecycle,
            lifecycle_path=lifecycle_path,
            terminal_status=terminal_status,
        )
        if terminal_status != "complete":
            raise RemoteKernelTerminalError(
                f"kernel terminal status is not complete: {terminal_status}"
            )
        assert downloaded is not None
        lifecycle["last_trustworthy_phase"] = "downloaded_outputs_receipt_validated"

        active_event = begin_phase(lifecycle, lifecycle_path, "semantic_validation")
        native = load_native_module(packet_dir)
        admitted = native.validate_downloaded_native_image_to_glb_outputs(packet, download_dir)
        report = admitted["report"]
        lifecycle.update(
            {
                "last_trustworthy_phase": "native_pixal9_semantically_admitted",
                "admitted_run_id": report["run_id"],
                "admitted_report_status": report["status"],
                "admitted_image_sha256": report["effective_inputs"]["image"]["sha256"],
                "remote_elapsed_seconds": report["elapsed_seconds"],
                "remote_phase_timings": report["phase_timings"],
                "orientation_observer": report["orientation_observer"],
                "consumer_glb": str(download_dir / "12-consumer_glb.glb"),
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
