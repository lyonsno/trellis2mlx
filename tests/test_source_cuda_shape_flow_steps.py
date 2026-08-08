import hashlib
import json

import numpy as np


LUT_SHA256 = "1" * 64
MODULATION_REPORT_SHA256 = "2" * 64
SOURCE_CHECKPOINT_SHA256 = "3" * 64


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
        "--mlx-timestep-modulation-route",
        "default",
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


def _use_source_cuda_lut(args):
    route_index = args.index("--mlx-timestep-modulation-route") + 1
    args[route_index] = "source-cuda-lut"
    no_download_index = args.index("--no-download")
    args[no_download_index:no_download_index] = [
        "--expected-modulation-lut-sha256",
        LUT_SHA256,
        "--expected-modulation-report-sha256",
        MODULATION_REPORT_SHA256,
        "--expected-modulation-source-checkpoint-sha256",
        SOURCE_CHECKPOINT_SHA256,
    ]
    return args


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


def test_source_cuda_lut_identity_reaches_trajectory_validation(tmp_path, monkeypatch):
    from scripts import source_cuda_shape_flow_steps as witness

    inputs = _inputs(tmp_path)
    captured = {}
    fake_trajectory = {
        "coords": np.zeros((2, 4), dtype=np.int32),
        "noise": np.zeros((2, 32), dtype=np.float32),
    }

    def fake_load(*args, expected_modulation_identity):
        captured["identity"] = expected_modulation_identity
        return fake_trajectory, {"capture_sha256": "4" * 64}

    monkeypatch.setattr(witness, "load_mlx_trajectory", fake_load)
    monkeypatch.setattr(
        witness,
        "_load_conditioning",
        lambda path: (
            np.zeros((1, 2, 3), dtype=np.float32),
            np.zeros((1, 2, 3), dtype=np.float32),
        ),
    )

    exit_code = witness.main(_use_source_cuda_lut(_args(tmp_path, inputs)))

    assert exit_code == 1
    assert captured["identity"] == {
        "npz_sha256_effective": LUT_SHA256,
        "report_sha256_effective": MODULATION_REPORT_SHA256,
        "source_checkpoint_sha256_effective": SOURCE_CHECKPOINT_SHA256,
    }
    report = json.loads((tmp_path / "source-steps.json").read_text())
    assert report["failure_phase"] == "input_validation"
    assert report["requested_route"]["mlx_timestep_modulation_route"] == (
        "source-cuda-lut"
    )
    assert report["requested_route"]["expected_modulation_identity"] == (
        captured["identity"]
    )


def test_default_route_rejects_modulation_hashes_before_input_loading(
    tmp_path, monkeypatch
):
    from scripts import source_cuda_shape_flow_steps as witness

    inputs = _inputs(tmp_path)
    args = _args(tmp_path, inputs)
    no_download_index = args.index("--no-download")
    args[no_download_index:no_download_index] = [
        "--expected-modulation-lut-sha256",
        LUT_SHA256,
    ]
    monkeypatch.setattr(
        witness,
        "load_mlx_trajectory",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("trajectory loader must not run")
        ),
    )

    exit_code = witness.main(args)

    assert exit_code == 1
    report = json.loads((tmp_path / "source-steps.json").read_text())
    assert report["failure_phase"] == "request_validation"
    assert "require source-cuda-lut mode" in report["error"]


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
