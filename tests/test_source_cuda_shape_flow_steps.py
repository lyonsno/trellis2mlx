import hashlib
import json

import numpy as np


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _args(tmp_path, inputs):
    return [
        "--output-json",
        str(tmp_path / "source-steps.json"),
        "--output-npz",
        str(tmp_path / "source-steps.npz"),
        "--mlx-shape-flow-steps",
        str(inputs["steps"]),
        "--mlx-shape-flow-steps-sha256",
        _sha256(inputs["steps"]),
        "--mlx-run-report",
        str(inputs["report"]),
        "--mlx-run-report-sha256",
        _sha256(inputs["report"]),
        "--conditioning",
        str(inputs["conditioning"]),
        "--conditioning-sha256",
        _sha256(inputs["conditioning"]),
        "--source-tar",
        str(inputs["source_tar"]),
        "--source-tar-sha256",
        _sha256(inputs["source_tar"]),
        "--no-download",
    ]


def _inputs(tmp_path):
    paths = {
        "steps": tmp_path / "shape_flow_steps.npz",
        "report": tmp_path / "run_report.json",
        "conditioning": tmp_path / "conditioning.npz",
        "source_tar": tmp_path / "source.tar",
    }
    for index, path in enumerate(paths.values()):
        path.write_bytes(f"input-{index}".encode())
    return paths


def test_parser_requires_only_direct_source_recurrence_inputs(tmp_path):
    from scripts.source_cuda_shape_flow_steps import build_parser

    inputs = _inputs(tmp_path)
    args = build_parser().parse_args(_args(tmp_path, inputs))

    assert args.output_npz == tmp_path / "source-steps.npz"
    assert args.source_tar == inputs["source_tar"]
    assert args.no_download is True


def test_no_download_writes_failure_report_and_invalidates_stale_primary(
    tmp_path, monkeypatch
):
    from scripts import source_cuda_shape_flow_steps as witness

    inputs = _inputs(tmp_path)
    stale = tmp_path / "source-steps.npz"
    stale.write_bytes(b"stale-primary")
    fake_trajectory = {
        "coords": np.zeros((2, 4), dtype=np.int32),
        "noise": np.zeros((2, 32), dtype=np.float32),
    }
    monkeypatch.setattr(
        witness,
        "load_mlx_trajectory",
        lambda *args, **kwargs: (fake_trajectory, {"capture_sha256": "1" * 64}),
    )
    monkeypatch.setattr(
        witness,
        "_load_conditioning",
        lambda path: (
            np.zeros((1, 2, 3), dtype=np.float32),
            np.zeros((1, 2, 3), dtype=np.float32),
        ),
    )

    exit_code = witness.main(_args(tmp_path, inputs))

    assert exit_code == 1
    assert not stale.exists()
    report = json.loads((tmp_path / "source-steps.json").read_text())
    assert report["status"] == "failed"
    assert report["primary_output_status"] == "missing"
    assert report["failure_phase"] == "input_validation"
    assert report["last_trustworthy_phase"] == "input_validation"
    assert "--no-download" in report["error"]


def test_digest_mismatch_fails_before_primary_without_erasing_unowned_file(
    tmp_path,
):
    from scripts import source_cuda_shape_flow_steps as witness

    inputs = _inputs(tmp_path)
    stale = tmp_path / "source-steps.npz"
    stale.write_bytes(b"untrusted-but-preexisting")
    args = _args(tmp_path, inputs)
    digest_index = args.index("--source-tar-sha256") + 1
    args[digest_index] = "0" * 64

    exit_code = witness.main(args)

    assert exit_code == 1
    assert stale.read_bytes() == b"untrusted-but-preexisting"
    report = json.loads((tmp_path / "source-steps.json").read_text())
    assert report["primary_output_status"] == "preexisting_untrusted_preserved"
    assert report["failure_phase"] == "request_validation"
    assert "source tar SHA256 mismatch" in report["error"]
