"""Wait for the private saved-SLat Kaggle decoder assay and retain its terminal result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trellmlx.kaggle_cuda_witness import (  # noqa: E402
    build_kernel_status_command, load_prepared_packet, main as packet_main,
)


def classify_kernel_status(output: str) -> str:
    matches = re.findall(r'KernelWorkerStatus\.([A-Z_]+)', output)
    if len(matches) != 1:
        raise ValueError(f"unrecognized Kaggle kernel status: {output!r}")
    state = matches[0]
    if state in {"QUEUED", "RUNNING", "PENDING", "PREPARING"}:
        return "pending"
    if state == "COMPLETE":
        return "complete"
    if state in {"ERROR", "CANCELLED", "CANCELED", "FAILED"}:
        return "failed"
    raise ValueError(f"unrecognized Kaggle kernel status: {state}")


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    report = {
        "schema": "trellis2mlx.kaggle_saved_slat_watch.v1", "status": "running",
        "phase": "load_packet", "failure_phase": None,
        "packet_dir": str(args.packet_dir), "output_dir": str(args.output_dir),
        "poll_seconds": args.poll_seconds, "last_trustworthy_phase": "request_received",
    }
    _write_report(args.report, report)
    try:
        if args.poll_seconds <= 0:
            raise ValueError("poll interval must be positive")
        packet = load_prepared_packet(args.packet_dir)
        report.update(kernel_id=packet.kernel_id, dataset_id=packet.dataset_id,
                      last_trustworthy_phase="packet_loaded", phase="kernel_status")
        _write_report(args.report, report)
        while True:
            observed = subprocess.run(build_kernel_status_command(packet),
                                      capture_output=True, text=True, check=False)
            if observed.returncode != 0:
                raise RuntimeError(f"Kaggle status failed: exit={observed.returncode} "
                                   f"stdout={observed.stdout!r} stderr={observed.stderr!r}")
            state = classify_kernel_status(observed.stdout)
            report.update(kernel_status=observed.stdout.strip(), terminal_state=state,
                          last_trustworthy_phase="kernel_status_observed")
            _write_report(args.report, report)
            if state != "pending":
                break
            time.sleep(args.poll_seconds)

        report["phase"] = "download_terminal_outputs"
        _write_report(args.report, report)
        output_exit = packet_main([
            "kernel-output", "--packet-dir", str(args.packet_dir),
            "--report-dir", str(args.report.parent),
            "--output-dir", str(args.output_dir),
        ])
        report["output_download_exit_code"] = output_exit
        report["last_trustworthy_phase"] = "kernel_terminal_observed"
        if state != "complete" or output_exit != 0:
            raise RuntimeError(f"terminal kernel state={state}, output download exit={output_exit}")
        child = json.loads((args.output_dir / packet.output_json).read_text())
        receipt = json.loads((args.output_dir / "kaggle_cuda_witness_receipt.json").read_text())
        expected_sha = packet.entrypoint_args[packet.entrypoint_args.index("--shape-slat-sha256") + 1]
        if child.get("status") != "done" or receipt.get("status") != "done":
            raise RuntimeError("kernel completed without successful child and outer receipts")
        if child.get("input_identity", {}).get("sha256") != expected_sha:
            raise RuntimeError("child report lost the exact saved-SLat input identity")
        report.update(status="done", phase="complete", child_report=str(args.output_dir / packet.output_json),
                      receipt=str(args.output_dir / "kaggle_cuda_witness_receipt.json"),
                      last_trustworthy_phase="validated_decoder_state")
        _write_report(args.report, report)
        return 0
    except BaseException as exc:
        report.update(status="failed", failure_phase=report["phase"],
                      error_type=type(exc).__name__, error_message=str(exc),
                      traceback=traceback.format_exc())
        _write_report(args.report, report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
