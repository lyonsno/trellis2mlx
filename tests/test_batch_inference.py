"""Batch inference scheduling contracts."""

import concurrent.futures
from datetime import datetime
import json
import os
import subprocess
import sys
import time
from uuid import UUID

import pytest


def test_build_batch_jobs_expands_seeds_and_outputs(tmp_path):
    from trellmlx.batch_inference import BatchRequest, build_batch_jobs

    request = BatchRequest(
        images=("assets/shoe_input.png",),
        seeds=(11, 12),
        output_dir=tmp_path,
        output_prefix="shoe",
        target_faces=123_456,
        no_cleanup=True,
    )

    jobs = build_batch_jobs(request)

    assert [job.seed for job in jobs] == [11, 12]
    assert [job.output_path.name for job in jobs] == [
        "shoe-seed-11.glb",
        "shoe-seed-12.glb",
    ]
    assert all(job.images == ("assets/shoe_input.png",) for job in jobs)
    assert all(job.target_faces == 123_456 for job in jobs)
    assert all(job.no_cleanup is True for job in jobs)


def test_build_batch_jobs_requires_output_count_to_match_seed_count(tmp_path):
    from trellmlx.batch_inference import BatchRequest, build_batch_jobs

    request = BatchRequest(
        images=("assets/shoe_input.png",),
        seeds=(1, 2),
        output_dir=tmp_path,
        outputs=(tmp_path / "one.glb",),
    )

    with pytest.raises(ValueError, match="outputs count"):
        build_batch_jobs(request)


def test_run_interleaved_batch_records_stage_major_report(tmp_path):
    from trellmlx.batch_inference import BatchJob, BatchRunOptions, run_interleaved_batch
    from trellmlx.interleaved_generation import GenerationStageResult, StageRunnerOutput

    calls = []

    def image_conditioning(invocation, state, context):
        calls.append((invocation.stage, invocation.job_id, invocation.stage_seed, state.next_stage_index))
        return StageRunnerOutput(
            result=GenerationStageResult(
                invocation.stage,
                elapsed_seconds=0.01,
                output_counts={"images": len(invocation.images)},
            ),
            artifacts={"conditioning_key": f"conditioning://{invocation.job_id}"},
        )

    def sparse_structure(invocation, state, context):
        calls.append((invocation.stage, invocation.job_id, invocation.stage_seed, state.next_stage_index))
        assert state.artifacts["conditioning_key"] == f"conditioning://{invocation.job_id}"
        return StageRunnerOutput(
            result=GenerationStageResult(
                invocation.stage,
                elapsed_seconds=0.02,
                output_counts={"sparse_tokens": 17},
            ),
            artifacts={"sparse_tokens": 17},
        )

    report_path = tmp_path / "reports" / "interleaved.json"
    jobs = [
        BatchJob(images=("subject.png",), seed=101, output_path=tmp_path / "seed-101.glb", resolution=512),
        BatchJob(images=("subject.png",), seed=202, output_path=tmp_path / "seed-202.glb", resolution=512),
    ]

    report = run_interleaved_batch(
        jobs,
        BatchRunOptions(
            max_concurrent=1,
            repo_root=tmp_path,
            report_path=report_path,
            command_line=("python", "-m", "trellmlx.batch_inference", "--mode", "interleaved"),
        ),
        stages=("image_conditioning", "sparse_structure"),
        handlers={
            "image_conditioning": image_conditioning,
            "sparse_structure": sparse_structure,
        },
        run_id="fixture-interleaved",
    )

    assert report.schema == "trellis2mlx.interleaved_batch_report.v1"
    assert report.requested_concurrency == 1
    assert report.effective_concurrency == 1
    assert report.diagnostics == []
    assert report.context_run_id == "fixture-interleaved"
    assert report.context_closed is True
    assert report.ok is True
    assert [(call[0], call[1], call[3]) for call in calls] == [
        ("image_conditioning", "seed-101", 0),
        ("image_conditioning", "seed-202", 0),
        ("sparse_structure", "seed-101", 1),
        ("sparse_structure", "seed-202", 1),
    ]
    assert calls[0][2] != calls[1][2]
    assert [result.job_id for result in report.results] == ["seed-101", "seed-202"]
    assert report.results[0].stage_results[0]["stage"] == "image_conditioning"
    assert report.results[0].stage_results[1]["output_counts"] == {"sparse_tokens": 17}
    assert report.results[0].artifacts == {
        "conditioning_key": "conditioning://seed-101",
        "sparse_tokens": 17,
    }
    assert report.results[0].failure_phase is None

    persisted = json.loads(report_path.read_text())
    assert persisted["schema"] == "trellis2mlx.interleaved_batch_report.v1"
    assert persisted["execution_mode"] == "interleaved"
    assert persisted["stages"] == ["image_conditioning", "sparse_structure"]
    assert persisted["results"][1]["job_id"] == "seed-202"
    assert persisted["results"][1]["config"]["resolution"] == 512
    assert persisted["identity"]["command_line"] == [
        "python",
        "-m",
        "trellmlx.batch_inference",
        "--mode",
        "interleaved",
    ]


def test_run_interleaved_batch_reports_context_load_failure(tmp_path):
    from trellmlx.batch_inference import BatchJob, BatchRunOptions, run_interleaved_batch
    from trellmlx.interleaved_generation import GenerationStageResult, StageHandleSpec, build_stage_context_factory

    calls = []

    def fail_load(runtime):
        raise RuntimeError("weights missing")

    context_factory, context_closer = build_stage_context_factory(
        [
            StageHandleSpec(
                "dinov3",
                "fixture",
                fail_load,
                metadata={"role": "dinov3"},
            )
        ],
        run_id="load-failure",
    )
    report_path = tmp_path / "reports" / "interleaved-load-failure.json"

    report = run_interleaved_batch(
        [BatchJob(images=("subject.png",), seed=101, output_path=tmp_path / "seed-101.glb")],
        BatchRunOptions(max_concurrent=1, repo_root=tmp_path, report_path=report_path),
        stages=("image_conditioning",),
        handlers={
            "image_conditioning": lambda invocation, state, context: calls.append(invocation.job_id)
            or GenerationStageResult(invocation.stage, elapsed_seconds=0.0)
        },
        context_factory=context_factory,
        context_closer=context_closer,
        run_id="load-failure",
    )

    assert calls == []
    assert report.ok is False
    assert report.failure_phase == "context_load"
    assert report.context_closed is False
    assert report.load_reports[0]["handle_id"] == "dinov3"
    assert report.load_reports[0]["load_phase"] == "load_error"
    assert report.load_reports[0]["elapsed_seconds"] >= 0.0
    assert "weights missing" in report.load_reports[0]["error"]
    assert report.close_reports == []
    assert report.results[0].job_id == "seed-101"
    assert report.results[0].failure_phase == "context_load"
    assert report.results[0].stage_results == []

    persisted = json.loads(report_path.read_text())
    assert persisted["failure_phase"] == "context_load"
    assert persisted["load_reports"][0]["elapsed_seconds"] >= 0.0
    assert persisted["results"][0]["failure_phase"] == "context_load"


def test_run_interleaved_batch_preserves_stage_evidence_on_close_failure(tmp_path):
    from trellmlx.batch_inference import BatchJob, BatchRunOptions, run_interleaved_batch
    from trellmlx.interleaved_generation import (
        GenerationStageResult,
        StageHandleSpec,
        StageRunnerOutput,
        build_stage_context_factory,
    )

    handle = object()

    def close_handle(runtime, loaded_handle):
        assert loaded_handle is handle
        raise RuntimeError("close exploded")

    context_factory, context_closer = build_stage_context_factory(
        [
            StageHandleSpec(
                "shape_flow",
                "fixture",
                lambda runtime: handle,
                close=close_handle,
                metadata={"role": "shape_flow"},
            )
        ],
        run_id="close-failure-context",
    )

    def stage(invocation, state, context):
        assert context.require_handle("shape_flow") is handle
        return StageRunnerOutput(
            result=GenerationStageResult(
                invocation.stage,
                elapsed_seconds=0.01,
                output_counts={"tokens": 19},
            ),
            artifacts={"shape_key": f"shape://{invocation.job_id}"},
        )

    report_path = tmp_path / "reports" / "close-failure.json"

    report = run_interleaved_batch(
        [BatchJob(images=("subject.png",), seed=101, output_path=tmp_path / "seed-101.glb")],
        BatchRunOptions(max_concurrent=1, repo_root=tmp_path, report_path=report_path),
        stages=("shape",),
        handlers={"shape": stage},
        context_factory=context_factory,
        context_closer=context_closer,
    )

    assert report.ok is False
    assert report.failure_phase == "context_close"
    assert report.context_run_id == "close-failure-context"
    assert report.context_closed is False
    assert report.load_reports[0]["handle_id"] == "shape_flow"
    assert report.load_reports[0]["load_phase"] == "loaded"
    assert report.load_reports[0]["elapsed_seconds"] >= 0.0
    assert report.close_reports[0]["handle_id"] == "shape_flow"
    assert report.close_reports[0]["close_phase"] == "close_error"
    assert report.close_reports[0]["elapsed_seconds"] >= 0.0
    assert "close exploded" in report.close_reports[0]["error"]
    assert report.results[0].failure_phase is None
    assert report.results[0].stage_results[0]["stage"] == "shape"
    assert report.results[0].stage_results[0]["output_counts"] == {"tokens": 19}
    assert report.results[0].artifacts == {"shape_key": "shape://seed-101"}

    persisted = json.loads(report_path.read_text())
    assert persisted["failure_phase"] == "context_close"
    assert persisted["load_reports"][0]["elapsed_seconds"] >= 0.0
    assert persisted["close_reports"][0]["close_phase"] == "close_error"
    assert persisted["close_reports"][0]["elapsed_seconds"] >= 0.0
    assert persisted["results"][0]["stage_results"][0]["output_counts"] == {"tokens": 19}


def test_run_interleaved_batch_writes_setup_failure_report(tmp_path):
    from trellmlx.batch_inference import BatchJob, BatchRunOptions, run_interleaved_batch
    from trellmlx.interleaved_generation import GenerationStageResult

    report_path = tmp_path / "reports" / "setup-failure.json"

    report = run_interleaved_batch(
        [
            BatchJob(images=("subject.png",), seed=101, output_path=tmp_path / "seed-101-a.glb"),
            BatchJob(images=("subject.png",), seed=101, output_path=tmp_path / "seed-101-b.glb"),
        ],
        BatchRunOptions(max_concurrent=1, repo_root=tmp_path, report_path=report_path),
        stages=("shape",),
        handlers={"shape": lambda invocation, state, context: GenerationStageResult(invocation.stage, 0.0)},
    )

    assert report.ok is False
    assert report.failure_phase == "setup"
    assert "duplicate job_id: seed-101" in report.error_message
    assert [result.job_id for result in report.results] == ["seed-101", "seed-101"]
    assert [result.failure_phase for result in report.results] == ["setup", "setup"]

    persisted = json.loads(report_path.read_text())
    assert persisted["failure_phase"] == "setup"
    assert "duplicate job_id: seed-101" in persisted["error_message"]
    assert len(persisted["results"]) == 2


@pytest.mark.parametrize(
    ("jobs", "stages", "handlers", "expected_error", "expected_result_count"),
    [
        (
            [
                ("subject.png", 101, "seed-101-a.glb"),
                ("subject.png", 101, "seed-101-b.glb"),
            ],
            ("shape",),
            {"shape": lambda invocation, state, context: None},
            "duplicate job_id: seed-101",
            2,
        ),
        (
            [("subject.png", 101, "seed-101.glb")],
            ("shape", "texture"),
            {"shape": lambda invocation, state, context: None},
            "missing stage handlers: texture",
            1,
        ),
    ],
)
def test_run_interleaved_batch_setup_failures_report_zero_effective_concurrency(
    tmp_path,
    jobs,
    stages,
    handlers,
    expected_error,
    expected_result_count,
):
    from trellmlx.batch_inference import BatchJob, BatchRunOptions, run_interleaved_batch

    report_path = tmp_path / "reports" / "setup-zero-concurrency.json"
    batch_jobs = [
        BatchJob(images=(image,), seed=seed, output_path=tmp_path / output_name)
        for image, seed, output_name in jobs
    ]

    report = run_interleaved_batch(
        batch_jobs,
        BatchRunOptions(max_concurrent=4, repo_root=tmp_path, report_path=report_path),
        stages=stages,
        handlers=handlers,
    )

    assert report.ok is False
    assert report.failure_phase == "setup"
    assert expected_error in report.error_message
    assert report.requested_concurrency == 4
    assert report.effective_concurrency == 0
    assert len(report.results) == expected_result_count

    persisted = json.loads(report_path.read_text())
    assert persisted["failure_phase"] == "setup"
    assert expected_error in persisted["error_message"]
    assert persisted["requested_concurrency"] == 4
    assert persisted["effective_concurrency"] == 0
    assert len(persisted["results"]) == expected_result_count


@pytest.mark.parametrize(
    ("job_count", "options", "expected_error", "expected_concurrency", "expected_result_count"),
    [
        (
            1,
            {"max_concurrent": 0},
            "max_concurrent must be >= 1",
            0,
            1,
        ),
        (
            0,
            {"max_concurrent": 1},
            "run_interleaved_batch requires at least one job",
            0,
            0,
        ),
    ],
)
def test_run_interleaved_batch_writes_early_setup_failure_report(
    tmp_path,
    job_count,
    options,
    expected_error,
    expected_concurrency,
    expected_result_count,
):
    from trellmlx.batch_inference import BatchJob, BatchRunOptions, run_interleaved_batch

    report_path = tmp_path / "reports" / "early-setup-failure.json"
    jobs = [
        BatchJob(
            images=("subject.png",),
            seed=101 + index,
            output_path=tmp_path / f"seed-{101 + index}.glb",
        )
        for index in range(job_count)
    ]

    report = run_interleaved_batch(
        jobs,
        BatchRunOptions(
            max_concurrent=options["max_concurrent"],
            repo_root=tmp_path,
            report_path=report_path,
        ),
        stages=("shape",),
        handlers={"shape": lambda invocation, state, context: None},
    )

    assert report.ok is False
    assert report.failure_phase == "setup"
    assert expected_error in report.error_message
    assert report.effective_concurrency == expected_concurrency
    assert len(report.results) == expected_result_count
    assert [result.failure_phase for result in report.results] == ["setup"] * expected_result_count

    persisted = json.loads(report_path.read_text())
    assert persisted["failure_phase"] == "setup"
    assert expected_error in persisted["error_message"]
    assert persisted["effective_concurrency"] == expected_concurrency
    assert len(persisted["results"]) == expected_result_count


def test_run_interleaved_batch_load_failure_uses_factory_run_id(tmp_path):
    from trellmlx.batch_inference import BatchJob, BatchRunOptions, run_interleaved_batch
    from trellmlx.interleaved_generation import GenerationStageResult, StageHandleSpec, build_stage_context_factory

    def fail_load(runtime):
        assert runtime.run_id == "actual-context-run"
        raise RuntimeError("weights missing")

    context_factory, context_closer = build_stage_context_factory(
        [StageHandleSpec("dinov3", "fixture", fail_load)],
        run_id="actual-context-run",
    )

    report = run_interleaved_batch(
        [BatchJob(images=("subject.png",), seed=101, output_path=tmp_path / "seed-101.glb")],
        BatchRunOptions(max_concurrent=1, repo_root=tmp_path, report_path=tmp_path / "report.json"),
        stages=("image_conditioning",),
        handlers={
            "image_conditioning": lambda invocation, state, context: GenerationStageResult(
                invocation.stage,
                elapsed_seconds=0.0,
            )
        },
        context_factory=context_factory,
        context_closer=context_closer,
    )

    assert report.failure_phase == "context_load"
    assert report.context_run_id == "actual-context-run"
    assert json.loads((tmp_path / "report.json").read_text())["context_run_id"] == "actual-context-run"


def test_batch_cli_rejects_interleaved_mode_until_handlers_are_wired(tmp_path, capsys):
    from trellmlx.batch_inference import main

    with pytest.raises(SystemExit) as exc_info:
        main([
            "--mode",
            "interleaved",
            "--image",
            "subject.png",
            "--seeds",
            "101",
            "--output-dir",
            str(tmp_path / "out"),
            "--report",
            str(tmp_path / "report.json"),
        ])

    assert exc_info.value.code == 2
    assert "interleaved CLI mode requires production stage handlers" in capsys.readouterr().err
    assert not (tmp_path / "report.json").exists()


def test_run_batch_records_effective_concurrency_and_mohel_indicator(tmp_path, monkeypatch):
    from trellmlx.batch_inference import BatchJob, BatchRunOptions, run_batch

    monkeypatch.delenv("PYTHONPATH", raising=False)
    calls = []

    def runner(cmd, cwd, env, capture_output, text):
        calls.append((cmd, cwd, env, capture_output, text))
        output_path = cmd[cmd.index("--output") + 1]
        if "--seed" in cmd and cmd[cmd.index("--seed") + 1] == "2":
            return subprocess.CompletedProcess(
                cmd,
                7,
                stdout="started\n",
                stderr="model exploded\n",
            )
        with open(output_path, "wb") as handle:
            handle.write(b"glb")
        return subprocess.CompletedProcess(cmd, 0, stdout="saved\n", stderr="")

    jobs = [
        BatchJob(images=("a.png",), seed=1, output_path=tmp_path / "nested" / "one.glb"),
        BatchJob(images=("a.png",), seed=2, output_path=tmp_path / "nested" / "two.glb"),
        BatchJob(images=("a.png",), seed=3, output_path=tmp_path / "nested" / "three.glb"),
    ]
    report_path = tmp_path / "report.json"

    report = run_batch(
        jobs,
        BatchRunOptions(
            max_concurrent=3,
            repo_root=tmp_path,
            report_path=report_path,
            python_executable="python-test",
        ),
        runner=runner,
    )

    assert report.requested_concurrency == 3
    assert report.effective_concurrency == 3
    assert report.diagnostics == ["known_metal_deadlock_risk:max_concurrent>2"]
    assert [result.seed for result in report.results] == [1, 2, 3]
    assert [result.returncode for result in report.results] == [0, 7, 0]
    assert report.results[0].output_exists is True
    assert report.results[0].output_size_bytes == 3
    assert report.results[1].output_exists is False
    assert report.results[1].failure_phase == "subprocess"
    assert all(call[0][:2] == ["python-test", "generate.py"] for call in calls)
    assert all(call[2]["PYTHONPATH"] == str(tmp_path) for call in calls)

    persisted = json.loads(report_path.read_text())
    assert persisted["schema"] == "trellis2mlx.batch_report.v1"
    assert persisted["requested_concurrency"] == 3
    assert persisted["effective_concurrency"] == 3
    assert persisted["diagnostics"] == ["known_metal_deadlock_risk:max_concurrent>2"]
    assert persisted["results"][1]["stderr"] == "model exploded\n"
    assert persisted["results"][1]["stdout_log_path"].endswith("job-001-seed-2.stdout.log")
    assert persisted["results"][1]["stderr_log_path"].endswith("job-001-seed-2.stderr.log")
    assert "started\n" in (tmp_path / "logs" / "job-001-seed-2.stdout.log").read_text()
    assert "model exploded\n" in (tmp_path / "logs" / "job-001-seed-2.stderr.log").read_text()


def test_run_batch_writes_live_job_logs_before_final_report(tmp_path, monkeypatch):
    from trellmlx.batch_inference import BatchJob, BatchRunOptions, run_batch

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sentinel = tmp_path / "first-line-seen"
    generate_py = repo_root / "generate.py"
    generate_py.write_text(
        """
import argparse
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--image", nargs="+")
parser.add_argument("--seed")
parser.add_argument("--output")
parser.add_argument("--resolution")
parser.add_argument("--max-tokens")
parser.add_argument("--target-faces")
args = parser.parse_args()

print("stdout-live", flush=True)
print("stderr-live", file=sys.stderr, flush=True)
with open(os.environ["TRELLIS2MLX_TEST_SENTINEL"], "w") as handle:
    handle.write("seen")
time.sleep(0.5)
with open(args.output, "wb") as handle:
    handle.write(b"glb")
print("stdout-done", flush=True)
""".lstrip()
    )
    monkeypatch.setenv("TRELLIS2MLX_TEST_SENTINEL", str(sentinel))

    report_path = tmp_path / "report.json"
    log_dir = tmp_path / "job-logs"
    stdout_log = log_dir / "job-000-seed-7.stdout.log"
    stderr_log = log_dir / "job-000-seed-7.stderr.log"
    job = BatchJob(images=("a.png",), seed=7, output_path=tmp_path / "out" / "live.glb")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            run_batch,
            [job],
            BatchRunOptions(
                max_concurrent=1,
                repo_root=repo_root,
                report_path=report_path,
                log_dir=log_dir,
                python_executable=sys.executable,
            ),
        )
        deadline = time.monotonic() + 3
        while not sentinel.exists() and time.monotonic() < deadline:
            time.sleep(0.02)

        assert sentinel.exists()
        while time.monotonic() < deadline:
            stdout_seen = stdout_log.exists() and "stdout-live" in stdout_log.read_text()
            stderr_seen = stderr_log.exists() and "stderr-live" in stderr_log.read_text()
            if stdout_seen and stderr_seen:
                break
            time.sleep(0.02)

        assert stdout_log.exists()
        assert stderr_log.exists()
        assert "stdout-live" in stdout_log.read_text()
        assert "stderr-live" in stderr_log.read_text()
        assert not report_path.exists()

        report = future.result(timeout=5)

    assert report.ok
    assert report.results[0].stdout_log_path == str(stdout_log)
    assert report.results[0].stderr_log_path == str(stderr_log)
    assert "stdout-done" in stdout_log.read_text()
    assert json.loads(report_path.read_text())["results"][0]["stdout_log_path"] == str(stdout_log)


def test_run_batch_persists_route_identity_for_measurement(tmp_path):
    from trellmlx.batch_inference import BatchJob, BatchRunOptions, run_batch

    def runner(cmd, cwd, env, capture_output, text):
        output_path = cmd[cmd.index("--output") + 1]
        with open(output_path, "wb") as handle:
            handle.write(b"glb")
        return subprocess.CompletedProcess(cmd, 0, stdout="saved\n", stderr="")

    report_path = tmp_path / "report.json"

    run_batch(
        [BatchJob(images=("a.png",), seed=1, output_path=tmp_path / "out" / "one.glb")],
        BatchRunOptions(
            max_concurrent=1,
            repo_root=tmp_path,
            report_path=report_path,
            python_executable="python-test",
            command_line=("python", "-m", "trellmlx.batch_inference", "--seeds", "1"),
        ),
        runner=runner,
    )

    persisted = json.loads(report_path.read_text())
    assert UUID(persisted["batch_run_id"])
    started_at = datetime.fromisoformat(persisted["started_at"].replace("Z", "+00:00"))
    finished_at = datetime.fromisoformat(persisted["finished_at"].replace("Z", "+00:00"))
    assert finished_at >= started_at
    assert persisted["identity"]["command_line"] == [
        "python",
        "-m",
        "trellmlx.batch_inference",
        "--seeds",
        "1",
    ]
    assert persisted["identity"]["checkout"]["repo_root"] == str(tmp_path)
    assert set(persisted["identity"]["checkout"]) >= {
        "repo_root",
        "git_commit",
        "git_branch",
        "git_dirty",
    }
    assert persisted["identity"]["host"]["hostname"]
    assert persisted["identity"]["host"]["os_system"]
    assert "memory_bytes" in persisted["identity"]["host"]
    assert persisted["identity"]["runtime"]["python_executable"] == "python-test"
    assert persisted["identity"]["runtime"]["python_version"] == sys.version
    assert "mlx_version" in persisted["identity"]["runtime"]


def test_run_batch_prepends_repo_root_to_existing_pythonpath(tmp_path, monkeypatch):
    from trellmlx.batch_inference import BatchJob, BatchRunOptions, run_batch

    monkeypatch.setenv("PYTHONPATH", "/opt/extra")
    seen_pythonpaths = []

    def runner(cmd, cwd, env, capture_output, text):
        seen_pythonpaths.append(env["PYTHONPATH"])
        output_path = cmd[cmd.index("--output") + 1]
        with open(output_path, "wb") as handle:
            handle.write(b"glb")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    run_batch(
        [BatchJob(images=("a.png",), seed=1, output_path=tmp_path / "out" / "one.glb")],
        BatchRunOptions(
            max_concurrent=1,
            repo_root=tmp_path,
            python_executable="python-test",
        ),
        runner=runner,
    )

    assert seen_pythonpaths == [f"{tmp_path}{os.pathsep}/opt/extra"]
