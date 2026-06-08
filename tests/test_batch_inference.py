"""Batch inference scheduling contracts."""

import json
import os
import subprocess

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
