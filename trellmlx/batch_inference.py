"""Batch inference scheduler for trellis2mlx.

The first batch boundary is intentionally outside the model internals: one
subprocess per generation job, explicit concurrency, and a durable report of
the effective route. That keeps the contract portable for the MLX Swift port
while avoiding the known Metal scheduler risk of hidden process fanout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence


SCHEMA = "trellis2mlx.batch_report.v1"
METAL_RISK_DIAGNOSTIC = "known_metal_deadlock_risk:max_concurrent>2"


@dataclass(frozen=True)
class BatchRequest:
    """High-level request expanded into concrete generation jobs."""

    images: tuple[str, ...]
    seeds: tuple[int, ...]
    output_dir: Path | str
    output_prefix: str = "trellis"
    outputs: tuple[Path | str, ...] | None = None
    resolution: int = 1024
    max_tokens: int = 49152
    target_faces: int = 200_000
    compile: bool = False
    quantize: int = 0
    no_rembg: bool = False
    no_cleanup: bool = False


@dataclass(frozen=True)
class BatchJob:
    """Concrete single-generation job."""

    images: tuple[str, ...]
    seed: int
    output_path: Path
    resolution: int = 1024
    max_tokens: int = 49152
    target_faces: int = 200_000
    compile: bool = False
    quantize: int = 0
    no_rembg: bool = False
    no_cleanup: bool = False


@dataclass(frozen=True)
class BatchRunOptions:
    """Runtime options for executing a batch."""

    max_concurrent: int
    repo_root: Path | str = Path(".")
    report_path: Path | str | None = None
    python_executable: str = sys.executable


@dataclass(frozen=True)
class BatchJobResult:
    """Observed result for one batch job."""

    seed: int
    images: tuple[str, ...]
    output_path: str
    command: tuple[str, ...]
    returncode: int
    elapsed_seconds: float
    output_exists: bool
    output_size_bytes: int | None
    stdout: str
    stderr: str
    failure_phase: str | None = None


@dataclass(frozen=True)
class BatchRunReport:
    """Durable report for one batch invocation."""

    schema: str
    requested_concurrency: int
    effective_concurrency: int
    diagnostics: list[str]
    repo_root: str
    results: list[BatchJobResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(result.returncode == 0 for result in self.results)

    def to_dict(self) -> dict:
        return asdict(self)


Runner = Callable[..., subprocess.CompletedProcess]


def build_batch_jobs(request: BatchRequest) -> list[BatchJob]:
    """Expand a request into concrete jobs without touching runtime state."""

    if not request.images:
        raise ValueError("at least one image is required")
    if not request.seeds:
        raise ValueError("at least one seed is required")

    output_dir = Path(request.output_dir)
    if request.outputs is not None and len(request.outputs) != len(request.seeds):
        raise ValueError(
            f"outputs count ({len(request.outputs)}) must match seeds count ({len(request.seeds)})"
        )

    jobs: list[BatchJob] = []
    for index, seed in enumerate(request.seeds):
        if request.outputs is None:
            output_path = output_dir / f"{request.output_prefix}-seed-{seed}.glb"
        else:
            output_path = Path(request.outputs[index])

        jobs.append(
            BatchJob(
                images=tuple(request.images),
                seed=seed,
                output_path=output_path,
                resolution=request.resolution,
                max_tokens=request.max_tokens,
                target_faces=request.target_faces,
                compile=request.compile,
                quantize=request.quantize,
                no_rembg=request.no_rembg,
                no_cleanup=request.no_cleanup,
            )
        )
    return jobs


def build_generate_command(job: BatchJob, python_executable: str = sys.executable) -> list[str]:
    """Build the `generate.py` command for one job."""

    cmd = [
        python_executable,
        "generate.py",
        "--image",
        *job.images,
        "--seed",
        str(job.seed),
        "--output",
        str(job.output_path),
        "--resolution",
        str(job.resolution),
        "--max-tokens",
        str(job.max_tokens),
        "--target-faces",
        str(job.target_faces),
    ]
    if job.compile:
        cmd.append("--compile")
    if job.quantize:
        cmd.extend(["--quantize", str(job.quantize)])
    if job.no_rembg:
        cmd.append("--no-rembg")
    if job.no_cleanup:
        cmd.append("--no-cleanup")
    return cmd


def run_batch(
    jobs: Sequence[BatchJob],
    options: BatchRunOptions,
    *,
    runner: Runner = subprocess.run,
) -> BatchRunReport:
    """Run batch jobs and write a durable report when requested.

    `max_concurrent` is reported exactly as requested and is not silently
    capped. A diagnostic is emitted when the request exceeds the locally known
    Metal-safe process width from prior bench evidence.
    """

    if options.max_concurrent < 1:
        raise ValueError("max_concurrent must be >= 1")

    repo_root = Path(options.repo_root)
    effective_concurrency = min(options.max_concurrent, len(jobs)) if jobs else 0
    diagnostics: list[str] = []
    if options.max_concurrent > 2:
        diagnostics.append(METAL_RISK_DIAGNOSTIC)

    results_by_index: dict[int, BatchJobResult] = {}
    if jobs:
        with ThreadPoolExecutor(max_workers=options.max_concurrent) as executor:
            futures = {
                executor.submit(_run_one_job, index, job, options, repo_root, runner): index
                for index, job in enumerate(jobs)
            }
            for future in as_completed(futures):
                index, result = future.result()
                results_by_index[index] = result

    report = BatchRunReport(
        schema=SCHEMA,
        requested_concurrency=options.max_concurrent,
        effective_concurrency=effective_concurrency,
        diagnostics=diagnostics,
        repo_root=str(repo_root),
        results=[results_by_index[index] for index in sorted(results_by_index)],
    )

    if options.report_path is not None:
        report_path = Path(options.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")

    return report


def _run_one_job(
    index: int,
    job: BatchJob,
    options: BatchRunOptions,
    repo_root: Path,
    runner: Runner,
) -> tuple[int, BatchJobResult]:
    cmd = build_generate_command(job, options.python_executable)
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    job.output_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    stdout = ""
    stderr = ""
    failure_phase = None
    try:
        completed = runner(
            cmd,
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )
        returncode = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if returncode != 0:
            failure_phase = "subprocess"
    except Exception as exc:  # pragma: no cover - exercised by callers in production.
        returncode = 1
        stderr = f"{type(exc).__name__}: {exc}"
        failure_phase = "runner_exception"

    elapsed = time.perf_counter() - start
    output_exists = job.output_path.exists()
    output_size = job.output_path.stat().st_size if output_exists else None

    return index, BatchJobResult(
        seed=job.seed,
        images=job.images,
        output_path=str(job.output_path),
        command=tuple(cmd),
        returncode=returncode,
        elapsed_seconds=elapsed,
        output_exists=output_exists,
        output_size_bytes=output_size,
        stdout=stdout,
        stderr=stderr,
        failure_phase=failure_phase,
    )


def _parse_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a trellis2mlx generation batch.")
    parser.add_argument("--image", nargs="+", required=True,
                        help="Input image(s). Multiple values are treated as multi-view conditioning for every job.")
    parser.add_argument("--seeds", type=_parse_ints, default=(42, 123),
                        help="Comma-separated seeds to generate.")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/trellis2mlx-batch"))
    parser.add_argument("--output-prefix", default="trellis")
    parser.add_argument("--outputs", nargs="*", type=Path,
                        help="Optional explicit output GLB paths, one per seed.")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--report", type=Path, default=Path("/tmp/trellis2mlx-batch/report.json"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=49152)
    parser.add_argument("--target-faces", type=int, default=200_000)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--quantize", type=int, default=0, choices=[0, 4, 8])
    parser.add_argument("--no-rembg", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    request = BatchRequest(
        images=tuple(args.image),
        seeds=args.seeds,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        outputs=tuple(args.outputs) if args.outputs else None,
        resolution=args.resolution,
        max_tokens=args.max_tokens,
        target_faces=args.target_faces,
        compile=args.compile,
        quantize=args.quantize,
        no_rembg=args.no_rembg,
        no_cleanup=args.no_cleanup,
    )
    jobs = build_batch_jobs(request)
    report = run_batch(
        jobs,
        BatchRunOptions(
            max_concurrent=args.max_concurrent,
            repo_root=args.repo_root,
            report_path=args.report,
            python_executable=args.python,
        ),
    )

    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
