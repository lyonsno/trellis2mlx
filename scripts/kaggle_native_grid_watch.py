"""Retain the terminal Kaggle native-grid result, including failed runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import traceback


KERNEL_ID = "noahboo/t2mlx-kiln-1024-native-cuda-grid-control"
TERMINAL = {"COMPLETE", "ERROR", "FAILED", "CANCELLED", "CANCELED"}
PENDING = {"QUEUED", "RUNNING", "PENDING", "PREPARING"}


def status_from_text(output: str) -> str:
    matches = re.findall(r"KernelWorkerStatus\.([A-Z_]+)", output)
    if len(matches) != 1 or matches[0] not in TERMINAL | PENDING:
        raise ValueError(f"unrecognized Kaggle status: {output!r}")
    return matches[0]


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    report = {"schema": "trellis2mlx.kaggle_native_grid_watch.v1", "status": "running",
              "phase": "status", "kernel_id": KERNEL_ID, "output_dir": str(args.output_dir)}
    write_report(args.report, report)
    try:
        if args.poll_seconds <= 0:
            raise ValueError("poll interval must be positive")
        while True:
            observed = subprocess.run(["kaggle", "kernels", "status", KERNEL_ID],
                                      capture_output=True, text=True, check=False)
            if observed.returncode:
                raise RuntimeError(f"Kaggle status exited {observed.returncode}: {observed.stderr}")
            state = status_from_text(observed.stdout)
            report.update(kernel_status=state, last_trustworthy_phase="remote_status_observed")
            write_report(args.report, report)
            if state in TERMINAL:
                break
            time.sleep(args.poll_seconds)
        report["phase"] = "download"
        write_report(args.report, report)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        download = subprocess.run(["kaggle", "kernels", "output", KERNEL_ID, "--path", str(args.output_dir), "--force"],
                                  capture_output=True, text=True, check=False)
        report.update(download_returncode=download.returncode, download_stdout=download.stdout,
                      download_stderr=download.stderr)
        write_report(args.report, report)
        receipt_path = args.output_dir / "native_grid_kernel_receipt.json"
        if not receipt_path.exists():
            raise RuntimeError("kernel terminated without its durable receipt")
        receipt = json.loads(receipt_path.read_text())
        report["child_status"] = receipt.get("status")
        report["child_report"] = receipt.get("report")
        if state != "COMPLETE" or download.returncode or receipt.get("status") != "done":
            raise RuntimeError(f"kernel={state}, download={download.returncode}, child={receipt.get('status')}")
        native_report = args.output_dir / "native_grid_report.json"
        if not native_report.exists() or json.loads(native_report.read_text()).get("status") != "done":
            raise RuntimeError("successful receipt lacks successful native report")
        report.update(status="done", phase="complete", last_trustworthy_phase="native_report_verified")
        write_report(args.report, report)
        return 0
    except BaseException as exc:
        report.update(status="failed", failure_phase=report["phase"], error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        write_report(args.report, report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
