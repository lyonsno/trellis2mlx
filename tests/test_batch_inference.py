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
