"""Report-bearing command gates for witness implementation episodes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Callable, Sequence
import sys
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class GateCommand:
    name: str
    argv: tuple[str, ...]
    expected_returncodes: tuple[int, ...] = (0,)
    required_output_substrings: tuple[str, ...] = ()
    expected_pytest_failure_node: str | None = None
    required_failure_substrings: tuple[str, ...] = ()


GateRunner = Callable[..., object]


def native_image_to_glb_source_paths(root: Path) -> tuple[Path, ...]:
    root = Path(root).resolve()
    return tuple(
        root / relative
        for relative in (
            "pyproject.toml",
            "scripts/source_cuda_native_image_to_glb_witness.py",
            "scripts/prepare_native_image_to_glb_attempt.py",
            "scripts/run_native_image_to_glb_gate.py",
            "trellmlx/kaggle_cuda_witness.py",
            "trellmlx/witness_authority.py",
            "trellmlx/witness_gate.py",
            "trellmlx/native_image_to_glb_attempt.py",
            "tests/test_witness_authority.py",
            "tests/test_witness_gate.py",
            "tests/test_native_image_to_glb_attempt.py",
            "tests/test_prepare_native_image_to_glb_attempt.py",
            "tests/test_source_cuda_native_image_to_glb_witness.py",
            "tests/test_kaggle_cuda_witness.py",
            "tests/test_source_cuda_native_mechanics_smoke.py",
        )
    )


def _file_identity(paths: Sequence[Path]) -> dict[str, dict[str, object]]:
    identity: dict[str, dict[str, object]] = {}
    for source_path in paths:
        source = Path(source_path).resolve()
        record: dict[str, object] = {
            "exists": source.is_file(),
            "size_bytes": None,
            "sha256": None,
        }
        if source.is_file():
            data = source.read_bytes()
            record.update(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        identity[str(source)] = record
    return identity


def validate_gate_report(report_path: Path) -> dict[str, object]:
    try:
        report = json.loads(Path(report_path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("gate report is missing or invalid") from exc
    if report.get("schema") != "trellis2mlx.witness_gate.v1":
        raise ValueError("gate report schema is invalid")
    source_identity = report.get("source_identity")
    final_identity = report.get("final_source_identity")
    if not isinstance(source_identity, dict) or source_identity != final_identity:
        raise ValueError("gate source identity is missing or changed")
    if report.get("status") != "passed" or report.get("failure_phase") is not None:
        raise ValueError("gate report is not passed")
    current = _file_identity(tuple(Path(path) for path in source_identity))
    if current != source_identity:
        raise ValueError("gate source identity changed after the receipt was written")
    return report


def native_image_to_glb_gate_commands(
    profile: str,
    *,
    python: str | None = None,
) -> tuple[GateCommand, ...]:
    executable = python or sys.executable
    source_files = (
        "scripts/source_cuda_native_image_to_glb_witness.py",
        "scripts/prepare_native_image_to_glb_attempt.py",
        "scripts/run_native_image_to_glb_gate.py",
        "trellmlx/witness_authority.py",
        "trellmlx/witness_gate.py",
        "trellmlx/native_image_to_glb_attempt.py",
        "trellmlx/kaggle_cuda_witness.py",
    )
    affected_tests = (
        "tests/test_witness_authority.py",
        "tests/test_witness_gate.py",
        "tests/test_native_image_to_glb_attempt.py",
        "tests/test_prepare_native_image_to_glb_attempt.py",
        "tests/test_source_cuda_native_image_to_glb_witness.py",
        "tests/test_kaggle_cuda_witness.py",
        "tests/test_source_cuda_native_mechanics_smoke.py",
    )
    static = GateCommand(
        "static",
        (executable, "-m", "py_compile", *source_files),
    )
    affected = GateCommand(
        "affected-tests",
        (executable, "-m", "pytest", "-q", *affected_tests),
    )
    if profile == "focused":
        return (
            static,
            GateCommand("focused-tests", affected.argv),
        )
    if profile != "final":
        raise ValueError(f"unknown native image-to-GLB gate profile: {profile!r}")
    known_node = (
        "tests/test_source_cuda_perturbed_sparse_full_decode.py::"
        "test_sparse_flow_decode_coords_reorder_matches_source_pipeline"
    )
    return (
        static,
        affected,
        GateCommand(
            "repository-suite",
            (
                executable,
                "-m",
                "pytest",
                "-q",
                f"--deselect={known_node}",
            ),
        ),
        GateCommand(
            "known-host-limitation",
            (executable, "-m", "pytest", "-q", known_node),
            expected_returncodes=(1,),
            required_output_substrings=(known_node, "No module named 'torch'"),
            expected_pytest_failure_node=known_node,
            required_failure_substrings=("No module named 'torch'",),
        ),
    )


def run_gate(
    commands: Sequence[GateCommand],
    *,
    cwd: Path,
    report_path: Path,
    source_paths: Sequence[Path] = (),
    runner: GateRunner | None = None,
) -> dict[str, object]:
    cwd = Path(cwd).resolve()
    report_path = Path(report_path).resolve()
    run = runner or subprocess.run

    def persist(payload: dict[str, object]) -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=report_path.parent,
            prefix=f".{report_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(report_path)

    started = time.perf_counter()
    initial_identity = _file_identity(source_paths)
    report: dict[str, object] = {
        "schema": "trellis2mlx.witness_gate.v1",
        "status": "running",
        "failure_phase": None,
        "cwd": str(cwd),
        "source_identity": initial_identity,
        "final_source_identity": initial_identity,
        "commands": [],
        "elapsed_seconds": 0.0,
    }
    persist(report)
    command_records = report["commands"]
    assert isinstance(command_records, list)

    for command in commands:
        command_started = time.perf_counter()
        junit_path: Path | None = None
        try:
            argv = list(command.argv)
            if command.expected_pytest_failure_node is not None:
                junit_path = report_path.parent / f".{report_path.name}.{command.name}.junit.xml"
                junit_path.unlink(missing_ok=True)
                argv.append(f"--junitxml={junit_path}")
            completed = run(
                argv,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
            )
            returncode = int(completed.returncode)
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            combined_output = f"{stdout}\n{stderr}"
            structured_failure = None
            if command.expected_pytest_failure_node is not None:
                structured_failure = _admit_expected_pytest_failure(
                    junit_path,
                    command.expected_pytest_failure_node,
                    command.required_failure_substrings,
                )
            admitted = (
                returncode in command.expected_returncodes
                and all(
                    required in combined_output
                    for required in command.required_output_substrings
                )
                and structured_failure is not False
            )
            record = {
                "name": command.name,
                "argv": argv,
                "expected_returncodes": list(command.expected_returncodes),
                "required_output_substrings": list(
                    command.required_output_substrings
                ),
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
                "admitted": admitted,
                "structured_pytest_failure_admitted": structured_failure,
                "required_failure_substrings": list(
                    command.required_failure_substrings
                ),
                "elapsed_seconds": time.perf_counter() - command_started,
            }
        except BaseException as exc:
            admitted = False
            record = {
                "name": command.name,
                "argv": list(command.argv),
                "expected_returncodes": list(command.expected_returncodes),
                "required_output_substrings": list(
                    command.required_output_substrings
                ),
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "admitted": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.perf_counter() - command_started,
            }
        finally:
            if junit_path is not None:
                junit_path.unlink(missing_ok=True)
        command_records.append(record)
        final_identity = _file_identity(source_paths)
        report["final_source_identity"] = final_identity
        if final_identity != initial_identity:
            report["status"] = "failed"
            report["failure_phase"] = "source_identity_changed"
        elif not admitted:
            report["status"] = "failed"
            report["failure_phase"] = command.name
        report["elapsed_seconds"] = time.perf_counter() - started
        persist(report)
        if report["status"] == "failed":
            return report

    report["status"] = "passed"
    report["failure_phase"] = None
    report["elapsed_seconds"] = time.perf_counter() - started
    report["final_source_identity"] = _file_identity(source_paths)
    if report["final_source_identity"] != initial_identity:
        report["status"] = "failed"
        report["failure_phase"] = "source_identity_changed"
    persist(report)
    return report


def _admit_expected_pytest_failure(
    junit_path: Path | None,
    node: str,
    required_failure_substrings: Sequence[str],
) -> bool:
    if junit_path is None or not junit_path.is_file():
        return False
    try:
        root = ET.parse(junit_path).getroot()
    except (ET.ParseError, OSError):
        return False
    tests = root.findall(".//testcase")
    failures = root.findall(".//failure")
    errors = root.findall(".//error")
    skipped = root.findall(".//skipped")
    if len(tests) != 1 or len(failures) != 1 or errors or skipped:
        return False
    expected_path, expected_name = node.split("::", 1)
    testcase = tests[0]
    failure = failures[0]
    failure_text = " ".join(
        (failure.get("message", ""), "".join(failure.itertext()))
    )
    if any(required not in failure_text for required in required_failure_substrings):
        return False
    classname = testcase.get("classname", "")
    module = Path(expected_path).with_suffix("").as_posix().replace("/", ".")
    return testcase.get("name") == expected_name and (
        classname == module or classname.endswith(f".{Path(expected_path).stem}")
    )
