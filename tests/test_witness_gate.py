import json
import subprocess
from pathlib import Path

import pytest

from trellmlx.witness_gate import (
    GateCommand,
    native_image_to_glb_gate_commands,
    native_image_to_glb_source_paths,
    run_gate,
    validate_gate_report,
)


def test_gate_writes_terminal_report_with_frozen_source_identity(tmp_path):
    source = tmp_path / "witness.py"
    source.write_text("print('witness')\n")
    report_path = tmp_path / "reports" / "gate.json"
    calls = []

    def runner(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        return subprocess.CompletedProcess(argv, 0, "passed\n", "")

    report = run_gate(
        (
            GateCommand("static", ("python", "-m", "py_compile", str(source))),
            GateCommand("focused", ("pytest", "-q", "tests/test_witness.py")),
        ),
        cwd=tmp_path,
        report_path=report_path,
        source_paths=(source,),
        runner=runner,
    )

    persisted = json.loads(report_path.read_text())
    assert report == persisted
    assert persisted["status"] == "passed"
    assert persisted["failure_phase"] is None
    assert [command["name"] for command in persisted["commands"]] == [
        "static",
        "focused",
    ]
    assert persisted["source_identity"][str(source)]["sha256"]
    assert len(calls) == 2


def test_gate_writes_failure_before_returning_nonpassing_result(tmp_path):
    report_path = tmp_path / "gate.json"

    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 3, "", "specific failure\n")

    report = run_gate(
        (GateCommand("focused", ("pytest", "-q")),),
        cwd=tmp_path,
        report_path=report_path,
        runner=runner,
    )

    assert report["status"] == "failed"
    assert report["failure_phase"] == "focused"
    assert report["commands"][0]["returncode"] == 3
    assert json.loads(report_path.read_text()) == report


def test_gate_admits_one_explicit_expected_failure_without_hiding_other_failures(
    tmp_path,
):
    report_path = tmp_path / "gate.json"

    def runner(argv, **kwargs):
        junit = Path(next(value.split("=", 1)[1] for value in argv if value.startswith("--junitxml=")))
        junit.write_text(
            '<testsuites tests="1" failures="1" errors="0" skipped="0">'
            '<testsuite tests="1" failures="1" errors="0" skipped="0">'
            '<testcase classname="test_known" name="test_missing_torch">'
            '<failure>No module named \'torch\'</failure></testcase>'
            '</testsuite></testsuites>'
        )
        return subprocess.CompletedProcess(
            argv,
            1,
            "test_known.py::test_missing_torch FAILED\nNo module named 'torch'\n",
            "",
        )

    report = run_gate(
        (
            GateCommand(
                "known-host-limitation",
                ("pytest", "-q", "test_known.py::test_missing_torch"),
                expected_returncodes=(1,),
                required_output_substrings=(
                    "test_known.py::test_missing_torch",
                    "No module named 'torch'",
                ),
                expected_pytest_failure_node="test_known.py::test_missing_torch",
                required_failure_substrings=("No module named 'torch'",),
            ),
        ),
        cwd=tmp_path,
        report_path=report_path,
        runner=runner,
    )

    assert report["status"] == "passed"
    assert report["commands"][0]["admitted"] is True
    assert report["commands"][0]["expected_returncodes"] == [1]


def test_gate_invalidates_receipt_if_source_changes_during_gate(tmp_path):
    source = tmp_path / "witness.py"
    source.write_text("before\n")
    report_path = tmp_path / "gate.json"

    def runner(argv, **kwargs):
        source.write_text("after\n")
        return subprocess.CompletedProcess(argv, 0, "passed\n", "")

    report = run_gate(
        (GateCommand("focused", ("pytest", "-q")),),
        cwd=tmp_path,
        report_path=report_path,
        source_paths=(source,),
        runner=runner,
    )

    assert report["status"] == "failed"
    assert report["failure_phase"] == "source_identity_changed"
    assert report["source_identity"] != report["final_source_identity"]


def test_native_image_gate_profiles_batch_focused_and_final_work():
    focused = native_image_to_glb_gate_commands("focused", python="python-test")
    final = native_image_to_glb_gate_commands("final", python="python-test")

    assert [command.name for command in focused] == ["static", "focused-tests"]
    assert focused[0].argv[:3] == ("python-test", "-m", "py_compile")
    assert "tests/test_source_cuda_native_image_to_glb_witness.py" in focused[1].argv
    assert [command.name for command in final] == [
        "static",
        "affected-tests",
        "repository-suite",
        "known-host-limitation",
    ]
    known = final[-1]
    assert known.expected_returncodes == (1,)
    assert "No module named 'torch'" in known.required_output_substrings


def test_native_image_gate_rejects_unknown_profile():
    with pytest.raises(ValueError, match="profile"):
        native_image_to_glb_gate_commands("everything")


@pytest.mark.parametrize(
    ("failure_body", "external_output", "extra_xml"),
    (
        (
            "AssertionError: unrelated selected-test failure",
            "warning: No module named 'torch'",
            "",
        ),
        (
            "AssertionError: unrelated selected-test failure",
            "plugin message: No module named 'torch'",
            "",
        ),
        (
            "No module named 'torch'",
            "",
            '<testcase classname="another_surface" name="test_teardown">'
            '<error>plugin teardown failed</error></testcase>',
        ),
    ),
)
def test_expected_pytest_failure_rejects_unrelated_errors(
    tmp_path,
    failure_body,
    external_output,
    extra_xml,
):
    node = "test_known.py::test_missing_torch"

    def runner(argv, **kwargs):
        junit = Path(
            next(value.split("=", 1)[1] for value in argv if value.startswith("--junitxml="))
        )
        junit.write_text(
            '<testsuites><testsuite>'
            '<testcase classname="test_known" name="test_missing_torch">'
            f'<failure>{failure_body}</failure></testcase>'
            f'{extra_xml}</testsuite></testsuites>'
        )
        return subprocess.CompletedProcess(
            argv,
            1,
            f"{node} FAILED\nNo module named 'torch'\n{external_output}\n",
            "",
        )

    report = run_gate(
        (
            GateCommand(
                "known-host-limitation",
                ("pytest", "-q", node),
                expected_returncodes=(1,),
                required_output_substrings=(node, "No module named 'torch'"),
                expected_pytest_failure_node=node,
                required_failure_substrings=("No module named 'torch'",),
            ),
        ),
        cwd=tmp_path,
        report_path=tmp_path / "gate.json",
        runner=runner,
    )

    assert report["status"] == "failed"
    assert report["failure_phase"] == "known-host-limitation"


def test_native_gate_identity_includes_and_revalidates_non_diff_consumer(tmp_path):
    root = tmp_path / "repo"
    (root / "trellmlx").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "tests").mkdir()
    for path in native_image_to_glb_source_paths(root):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)
    consumer = root / "trellmlx/kaggle_cuda_witness.py"

    def runner(argv, **kwargs):
        consumer.write_text("mutated consumer")
        return subprocess.CompletedProcess(argv, 0, "passed\n", "")

    report_path = tmp_path / "gate.json"
    report = run_gate(
        (GateCommand("focused", ("pytest", "-q")),),
        cwd=root,
        report_path=report_path,
        source_paths=native_image_to_glb_source_paths(root),
        runner=runner,
    )

    assert report["failure_phase"] == "source_identity_changed"
    with pytest.raises(ValueError, match="source identity"):
        validate_gate_report(report_path)
