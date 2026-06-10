"""GPU Greenroom submit and receipt contracts for TRELLIS2MLX smokes.

This module deliberately talks to Greenroom through its CLI/filesystem
contract. The model and generation code stay unaware of queue custody; this
adapter only builds explicit branch-accurate requests and validates receipts.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA = "trellis2mlx.greenroom_report.v1"

Runner = Callable[..., subprocess.CompletedProcess]


class GreenroomReceiptError(ValueError):
    """Raised when a Greenroom receipt does not prove the requested route."""


@dataclass(frozen=True)
class GreenroomRequest:
    """Concrete TRELLIS2MLX Greenroom job request."""

    input_path: Path | str
    output_dir: Path | str
    repo_root: Path | str
    seed: int = 42
    resolution: int = 512
    target_faces: int = 200_000
    texture_size: int = 1024
    job_type: str = "trellis2mlx"

    def __post_init__(self) -> None:
        if not self.job_type:
            raise ValueError("job_type is required")
        if self.resolution < 1:
            raise ValueError("resolution must be >= 1")
        if self.target_faces < 1:
            raise ValueError("target_faces must be >= 1")
        if self.texture_size < 1:
            raise ValueError("texture_size must be >= 1")

    @property
    def expected_output_path(self) -> Path:
        return Path(self.output_dir) / f"seed-{self.seed}.glb"

    def params(self) -> dict[str, str]:
        return {
            "seed": str(self.seed),
            "resolution": str(self.resolution),
            "target_faces": str(self.target_faces),
            "texture_size": str(self.texture_size),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_type": self.job_type,
            "input_path": str(self.input_path),
            "output_dir": str(self.output_dir),
            "repo_root": str(self.repo_root),
            "seed": self.seed,
            "resolution": self.resolution,
            "target_faces": self.target_faces,
            "texture_size": self.texture_size,
            "expected_output_path": str(self.expected_output_path),
        }


@dataclass(frozen=True)
class GreenroomSubmitResult:
    """Observed result from `gpu-greenroom submit`."""

    job_id: str
    job_dir: Path | None
    command: tuple[str, ...]
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_dir": str(self.job_dir) if self.job_dir is not None else None,
            "command": list(self.command),
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


def build_submit_command(
    request: GreenroomRequest,
    *,
    queue_dir: Path | str | None = None,
    executable: str = "gpu-greenroom",
) -> list[str]:
    """Build a branch-accurate `gpu-greenroom submit` command."""

    cmd = [executable]
    if queue_dir is not None:
        cmd.extend(["--queue-dir", str(queue_dir)])
    cmd.extend(
        [
            "submit",
            request.job_type,
            str(request.input_path),
            str(request.output_dir),
            "--cwd",
            str(request.repo_root),
            "-p",
        ]
    )
    cmd.extend(f"{key}={value}" for key, value in request.params().items())
    return cmd


def submit_greenroom_request(
    request: GreenroomRequest,
    *,
    queue_dir: Path | str | None = None,
    executable: str = "gpu-greenroom",
    runner: Runner = subprocess.run,
) -> GreenroomSubmitResult:
    """Submit a Greenroom job and parse the CLI receipt pointer.

    This function only queues work. It does not run the Greenroom worker and it
    does not wait for GPU execution.
    """

    cmd = build_submit_command(request, queue_dir=queue_dir, executable=executable)
    completed = runner(cmd, capture_output=True, text=True, check=True)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    job_id = _parse_job_id(stdout)
    return GreenroomSubmitResult(
        job_id=job_id,
        job_dir=_parse_job_dir(stdout),
        command=tuple(cmd),
        stdout=stdout,
        stderr=stderr,
    )


def validate_receipt(
    receipt: Mapping[str, Any],
    request: GreenroomRequest,
    *,
    require_output: bool = True,
) -> Mapping[str, Any]:
    """Validate that a Greenroom receipt proves the requested route."""

    _expect_equal(receipt, "job_type", request.job_type)
    _expect_equal(receipt, "input_path", str(request.input_path))
    _expect_equal(receipt, "output_dir", str(request.output_dir))
    _expect_equal(receipt, "effective_cwd", str(request.repo_root))

    status = receipt.get("status")
    exit_code = receipt.get("exit_code")
    if status != "done" or exit_code != 0:
        raise GreenroomReceiptError(
            "Greenroom job failed or did not complete cleanly: "
            f"status={status!r}, exit_code={exit_code!r}, "
            f"failure_phase={receipt.get('failure_phase')!r}, "
            f"error_message={receipt.get('error_message')!r}"
        )

    ignored_params = receipt.get("ignored_params")
    if ignored_params not in (None, {}, []):
        raise GreenroomReceiptError(f"Greenroom ignored_params must be empty, got {ignored_params!r}")

    effective_env = receipt.get("effective_env")
    if not isinstance(effective_env, Mapping):
        raise GreenroomReceiptError("effective_env must be present in Greenroom receipt")
    if effective_env.get("PYTHONPATH") != ".":
        raise GreenroomReceiptError(
            f"effective_env PYTHONPATH must be '.', got {effective_env.get('PYTHONPATH')!r}"
        )

    effective_route = receipt.get("effective_route")
    if not isinstance(effective_route, str) or not effective_route:
        raise GreenroomReceiptError("effective_route must be present in Greenroom receipt")

    expected_output = request.expected_output_path
    route_tokens = _effective_route_tokens(effective_route)
    _expect_route_flag(route_tokens, "--output", str(expected_output))
    _expect_route_flag(route_tokens, "--seed", str(request.seed))
    _expect_route_flag(route_tokens, "--resolution", str(request.resolution))
    _expect_route_flag(route_tokens, "--target-faces", str(request.target_faces))
    _expect_route_flag(route_tokens, "--texture-size", str(request.texture_size))

    if require_output and not expected_output.exists():
        raise GreenroomReceiptError(f"expected output does not exist: {expected_output}")

    return receipt


def load_receipt(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_greenroom_report(
    *,
    request: GreenroomRequest,
    receipt: Mapping[str, Any],
    receipt_path: Path | str,
    report_path: Path | str,
) -> dict[str, Any]:
    """Validate a receipt and write compact evidence for a smoke run."""

    validate_receipt(receipt, request)
    output_path = request.expected_output_path
    report = {
        "schema": SCHEMA,
        "job_id": receipt["job_id"],
        "job_type": receipt["job_type"],
        "request": request.to_dict(),
        "receipt_path": str(receipt_path),
        "status": receipt.get("status"),
        "exit_code": receipt.get("exit_code"),
        "failure_phase": receipt.get("failure_phase"),
        "error_message": receipt.get("error_message"),
        "effective_route": receipt.get("effective_route"),
        "effective_cwd": receipt.get("effective_cwd"),
        "effective_env": dict(receipt.get("effective_env") or {}),
        "effective_defaults": dict(receipt.get("effective_defaults") or {}),
        "effective_timeout": receipt.get("effective_timeout"),
        "ignored_params": receipt.get("ignored_params"),
        "output_path": str(output_path),
        "output_exists": output_path.exists(),
        "output_size_bytes": output_path.stat().st_size if output_path.exists() else None,
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _expect_equal(receipt: Mapping[str, Any], key: str, expected: str) -> None:
    actual = receipt.get(key)
    if actual != expected:
        raise GreenroomReceiptError(f"{key} mismatch: expected {expected!r}, got {actual!r}")


def _effective_route_tokens(effective_route: str) -> list[str]:
    try:
        return shlex.split(effective_route)
    except ValueError as exc:
        raise GreenroomReceiptError(f"effective_route is not shell-tokenizable: {exc}") from exc


def _expect_route_flag(route_tokens: Sequence[str], flag: str, expected: str) -> None:
    try:
        index = route_tokens.index(flag)
    except ValueError as exc:
        raise GreenroomReceiptError(f"effective_route missing {flag}") from exc
    value_index = index + 1
    if value_index >= len(route_tokens):
        raise GreenroomReceiptError(f"effective_route missing value for {flag}")
    actual = route_tokens[value_index]
    if actual != expected:
        raise GreenroomReceiptError(
            f"effective_route {flag} mismatch: expected {expected!r}, got {actual!r}"
        )


def _parse_job_id(stdout: str) -> str:
    match = re.search(r"^Submitted job (?P<job_id>\S+)\s*$", stdout, flags=re.MULTILINE)
    if match is None:
        raise GreenroomReceiptError("could not parse Greenroom job id from submit output")
    return match.group("job_id")


def _parse_job_dir(stdout: str) -> Path | None:
    match = re.search(r"^\s*Dir:\s*(?P<job_dir>.+?)\s*$", stdout, flags=re.MULTILINE)
    if match is None:
        return None
    return Path(match.group("job_dir"))


def _request_from_args(args: argparse.Namespace) -> GreenroomRequest:
    return GreenroomRequest(
        input_path=args.image,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        seed=args.seed,
        resolution=args.resolution,
        target_faces=args.target_faces,
        texture_size=args.texture_size,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit and validate TRELLIS2MLX Greenroom jobs.")
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit", help="Submit a TRELLIS2MLX job to GPU Greenroom.")
    submit.add_argument("--image", required=True, type=Path)
    submit.add_argument("--output-dir", required=True, type=Path)
    submit.add_argument("--repo-root", required=True, type=Path)
    submit.add_argument("--queue-dir", type=Path)
    submit.add_argument("--greenroom", default="gpu-greenroom")
    submit.add_argument("--seed", type=int, default=42)
    submit.add_argument("--resolution", type=int, default=512)
    submit.add_argument("--target-faces", type=int, default=200_000)
    submit.add_argument("--texture-size", type=int, default=1024)
    submit.add_argument("--dry-run", action="store_true")

    report = sub.add_parser("report", help="Validate a completed receipt and write smoke evidence.")
    report.add_argument("--image", required=True, type=Path)
    report.add_argument("--output-dir", required=True, type=Path)
    report.add_argument("--repo-root", required=True, type=Path)
    report.add_argument("--receipt", required=True, type=Path)
    report.add_argument("--report", required=True, type=Path)
    report.add_argument("--seed", type=int, default=42)
    report.add_argument("--resolution", type=int, default=512)
    report.add_argument("--target-faces", type=int, default=200_000)
    report.add_argument("--texture-size", type=int, default=1024)

    raw_argv = list(argv) if argv is not None else None
    args = parser.parse_args(raw_argv)
    request = _request_from_args(args)

    if args.command == "submit":
        cmd = build_submit_command(
            request,
            queue_dir=args.queue_dir,
            executable=args.greenroom,
        )
        if args.dry_run:
            print(json.dumps({"command": cmd, "request": request.to_dict()}, indent=2))
            return 0
        result = submit_greenroom_request(
            request,
            queue_dir=args.queue_dir,
            executable=args.greenroom,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    receipt = load_receipt(args.receipt)
    written = write_greenroom_report(
        request=request,
        receipt=receipt,
        receipt_path=args.receipt,
        report_path=args.report,
    )
    print(json.dumps(written, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
