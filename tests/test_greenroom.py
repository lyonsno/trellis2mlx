"""GPU Greenroom request and receipt contracts."""

import json
import subprocess

import pytest


def test_build_submit_command_preserves_branch_cwd_and_params(tmp_path):
    from trellmlx.greenroom import GreenroomRequest, build_submit_command

    request = GreenroomRequest(
        input_path=tmp_path / "subject.png",
        output_dir=tmp_path / "out",
        repo_root=tmp_path / "branch-worktree",
        seed=101,
        resolution=512,
        target_faces=200_000,
        texture_size=1024,
    )

    cmd = build_submit_command(
        request,
        queue_dir=tmp_path / "queue",
        executable="gpu-greenroom-test",
    )

    assert cmd == [
        "gpu-greenroom-test",
        "--queue-dir",
        str(tmp_path / "queue"),
        "submit",
        "trellis2mlx",
        str(tmp_path / "subject.png"),
        str(tmp_path / "out"),
        "--cwd",
        str(tmp_path / "branch-worktree"),
        "-p",
        "seed=101",
        "resolution=512",
        "target_faces=200000",
        "texture_size=1024",
    ]


def test_submit_greenroom_request_parses_job_id_and_dir(tmp_path):
    from trellmlx.greenroom import GreenroomRequest, submit_greenroom_request

    calls = []

    def runner(cmd, capture_output, text, check):
        calls.append((cmd, capture_output, text, check))
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=(
                "Submitted job f5c2aa710a90\n"
                "  Type: trellis2mlx\n"
                f"  Dir: {tmp_path / 'queue' / 'pending' / 'f5c2aa710a90'}\n"
            ),
            stderr="",
        )

    result = submit_greenroom_request(
        GreenroomRequest(
            input_path=tmp_path / "subject.png",
            output_dir=tmp_path / "out",
            repo_root=tmp_path / "repo",
        ),
        queue_dir=tmp_path / "queue",
        executable="gpu-greenroom-test",
        runner=runner,
    )

    assert result.job_id == "f5c2aa710a90"
    assert result.job_dir == tmp_path / "queue" / "pending" / "f5c2aa710a90"
    assert calls[0][0][0] == "gpu-greenroom-test"
    assert calls[0][1:] == (True, True, True)


def test_validate_receipt_rejects_effective_cwd_mismatch(tmp_path):
    from trellmlx.greenroom import GreenroomReceiptError, GreenroomRequest, validate_receipt

    request = GreenroomRequest(
        input_path=tmp_path / "subject.png",
        output_dir=tmp_path / "out",
        repo_root=tmp_path / "branch-worktree",
    )
    receipt = _done_receipt(request) | {"effective_cwd": str(tmp_path / "main-checkout")}

    with pytest.raises(GreenroomReceiptError, match="effective_cwd"):
        validate_receipt(receipt, request)


def test_validate_receipt_rejects_ignored_params(tmp_path):
    from trellmlx.greenroom import GreenroomReceiptError, GreenroomRequest, validate_receipt

    request = GreenroomRequest(
        input_path=tmp_path / "subject.png",
        output_dir=tmp_path / "out",
        repo_root=tmp_path / "branch-worktree",
    )
    receipt = _done_receipt(request) | {"ignored_params": {"cwd": "silently ignored"}}

    with pytest.raises(GreenroomReceiptError, match="ignored_params"):
        validate_receipt(receipt, request)


def test_validate_receipt_rejects_failed_job(tmp_path):
    from trellmlx.greenroom import GreenroomReceiptError, GreenroomRequest, validate_receipt

    request = GreenroomRequest(
        input_path=tmp_path / "subject.png",
        output_dir=tmp_path / "out",
        repo_root=tmp_path / "branch-worktree",
    )
    receipt = _done_receipt(request) | {
        "status": "failed",
        "exit_code": 1,
        "failure_phase": "execution",
        "error_message": "model exploded",
    }

    with pytest.raises(GreenroomReceiptError, match="failed"):
        validate_receipt(receipt, request)


def test_validate_receipt_rejects_missing_output(tmp_path):
    from trellmlx.greenroom import GreenroomReceiptError, GreenroomRequest, validate_receipt

    request = GreenroomRequest(
        input_path=tmp_path / "subject.png",
        output_dir=tmp_path / "out",
        repo_root=tmp_path / "branch-worktree",
    )

    with pytest.raises(GreenroomReceiptError, match="expected output"):
        validate_receipt(_done_receipt(request), request)


def test_validate_receipt_rejects_wrong_effective_generation_params(tmp_path):
    from trellmlx.greenroom import GreenroomReceiptError, GreenroomRequest, validate_receipt

    request = GreenroomRequest(
        input_path=tmp_path / "subject.png",
        output_dir=tmp_path / "out",
        repo_root=tmp_path / "branch-worktree",
        seed=101,
        resolution=768,
        target_faces=333_333,
        texture_size=2048,
    )
    expected_output = request.expected_output_path
    expected_output.parent.mkdir(parents=True)
    expected_output.write_bytes(b"glb")
    receipt = _done_receipt(request) | {
        "effective_route": (
            f"python -u generate.py --image {request.input_path} "
            f"--output {expected_output} --seed 42 --resolution 512 "
            "--target-faces 200000 --texture-size 1024"
        ),
    }

    with pytest.raises(GreenroomReceiptError, match="effective_route"):
        validate_receipt(receipt, request)


def test_validate_receipt_rejects_duplicate_effective_generation_params(tmp_path):
    from trellmlx.greenroom import GreenroomReceiptError, GreenroomRequest, validate_receipt

    request = GreenroomRequest(
        input_path=tmp_path / "subject.png",
        output_dir=tmp_path / "out",
        repo_root=tmp_path / "branch-worktree",
        seed=101,
        resolution=768,
        target_faces=333_333,
        texture_size=2048,
    )
    expected_output = request.expected_output_path
    expected_output.parent.mkdir(parents=True)
    expected_output.write_bytes(b"glb")
    receipt = _done_receipt(request) | {
        "effective_route": (
            f"python -u generate.py --image {request.input_path} "
            f"--output {expected_output} --seed 101 --seed 42 "
            "--resolution 768 --resolution 512 "
            "--target-faces 333333 --target-faces 200000 "
            "--texture-size 2048 --texture-size 1024"
        ),
    }

    with pytest.raises(GreenroomReceiptError, match="duplicate --seed"):
        validate_receipt(receipt, request)


def test_write_greenroom_report_records_effective_evidence(tmp_path):
    from trellmlx.greenroom import GreenroomRequest, write_greenroom_report

    request = GreenroomRequest(
        input_path=tmp_path / "subject.png",
        output_dir=tmp_path / "out",
        repo_root=tmp_path / "branch-worktree",
        seed=101,
    )
    expected_output = request.expected_output_path
    expected_output.parent.mkdir(parents=True)
    expected_output.write_bytes(b"glb")
    receipt_path = tmp_path / "queue" / "done" / "f5c2aa710a90" / "receipt.json"
    receipt_path.parent.mkdir(parents=True)
    receipt = _done_receipt(request)
    receipt_path.write_text(json.dumps(receipt))
    report_path = tmp_path / "report.json"

    report = write_greenroom_report(
        request=request,
        receipt=receipt,
        receipt_path=receipt_path,
        report_path=report_path,
    )

    assert report["schema"] == "trellis2mlx.greenroom_report.v1"
    assert report["job_id"] == "f5c2aa710a90"
    assert report["job_type"] == "trellis2mlx"
    assert report["request"]["repo_root"] == str(tmp_path / "branch-worktree")
    assert report["receipt_path"] == str(receipt_path)
    assert report["effective_cwd"] == str(tmp_path / "branch-worktree")
    assert report["effective_env"] == {"PYTHONPATH": "."}
    assert report["output_path"] == str(expected_output)
    assert report["output_exists"] is True
    assert json.loads(report_path.read_text()) == report


def _done_receipt(request):
    return {
        "job_id": "f5c2aa710a90",
        "job_type": request.job_type,
        "status": "done",
        "input_path": str(request.input_path),
        "output_dir": str(request.output_dir),
        "effective_route": (
            f"python -u generate.py --image {request.input_path} "
            f"--output {request.expected_output_path} --seed {request.seed} "
            f"--resolution {request.resolution} --target-faces {request.target_faces} "
            f"--texture-size {request.texture_size}"
        ),
        "effective_cwd": str(request.repo_root),
        "effective_env": {"PYTHONPATH": "."},
        "effective_defaults": {
            "seed": "42",
            "resolution": "512",
            "target_faces": "200000",
            "texture_size": "1024",
        },
        "effective_timeout": None,
        "ignored_params": None,
        "started_at": 1718000000.0,
        "finished_at": 1718001200.0,
        "exit_code": 0,
        "failure_phase": None,
        "error_message": None,
    }
