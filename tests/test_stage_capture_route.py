from pathlib import Path


GENERATE_SOURCE = Path(__file__).resolve().parents[1] / "generate.py"


def test_generate_exposes_stage_capture_cli_contracts():
    source = GENERATE_SOURCE.read_text()

    assert "--stop-after-stage" in source
    assert "--shared-noise" in source
    assert "--save-checkpoints is required when --stop-after-stage is set" in source
    for stage in (
        "conditioning",
        "sparse_coords",
        "sparse_flow_step",
        "sparse_flow_steps",
        "sparse_flow_block_trace",
        "sparse_internals",
        "shape_flow_step",
        "shape_flow_steps",
        "shape_flow_block_trace",
        "shape_slat",
        "decoder_output",
    ):
        assert f'"{stage}"' in source


def test_generate_stage_capture_saves_requested_sparse_and_shape_artifacts():
    source = GENERATE_SOURCE.read_text()

    conditioning = source.index('save_checkpoint(\n            args.save_checkpoints,\n            "conditioning"')
    sparse_internals = source.index('save_checkpoint(\n            args.save_checkpoints,\n            "sparse_internals"', conditioning)
    sparse_coords = source.index('save_checkpoint(\n            args.save_checkpoints,\n            "sparse_coords"', sparse_internals)
    shape_slat = source.index('save_checkpoint(\n            args.save_checkpoints,\n            "shape_slat"', sparse_coords)
    decoder_output = source.index('save_checkpoint(\n            args.save_checkpoints,\n            "decoder_output"', shape_slat)

    assert 'args.stop_after_stage == "conditioning"' in source[conditioning:sparse_internals]
    assert "Stop after stage: sparse_internals" in source[sparse_internals:sparse_coords]
    assert 'args.stop_after_stage == "sparse_coords"' in source[sparse_coords:shape_slat]
    assert 'args.stop_after_stage == "shape_slat"' in source[shape_slat:decoder_output]
    assert 'args.stop_after_stage == "decoder_output"' in source[decoder_output:]


def test_sparse_coords_stop_happens_before_shape_weight_load():
    source = GENERATE_SOURCE.read_text()

    sparse_coords = source.index('save_checkpoint(\n            args.save_checkpoints,\n            "sparse_coords"')
    shape_model_import = source.index("from trellmlx.models.slat_flow import SLatFlowModel")
    shape_weight_load = source.index("slat_flow_img2shape_dit_1_3B_512_bf16.safetensors")

    assert sparse_coords < shape_model_import
    assert sparse_coords < shape_weight_load


def test_stage_capture_wrapper_builds_generate_stop_route(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "shape_slat",
            "--seed",
            "42",
            "--steps",
            "8",
            "--resolution",
            "512",
            "--no-cascade",
            "--no-rembg",
            "--shared-noise",
            "noise.npz",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")

    assert "--stop-after-stage" in command
    assert command[command.index("--stop-after-stage") + 1] == "shape_slat"
    assert "--save-checkpoints" in command
    assert command[command.index("--save-checkpoints") + 1] == str(tmp_path / "checkpoints")
    assert "--shared-noise" in command
    assert command[command.index("--shared-noise") + 1] == "noise.npz"
    assert "--no-cascade" in command
    assert "--no-rembg" in command


def test_stage_capture_parser_accepts_expected_repo_commit(tmp_path):
    from scripts.run_mlx_stage_capture import build_parser

    expected_commit = "a" * 40
    args, unknown = build_parser().parse_known_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "shape_flow_step",
            "--expected-repo-commit",
            expected_commit,
        ]
    )

    assert unknown == []
    assert args.expected_repo_commit == expected_commit


def test_stage_capture_route_records_requested_and_effective_repo_identity(tmp_path):
    from scripts.run_mlx_stage_capture import (
        _build_generate_command,
        build_parser,
        build_route_identity,
    )

    expected_commit = "d" * 40
    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "shape_flow_step",
            "--expected-repo-commit",
            expected_commit,
        ]
    )
    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(
        args,
        command,
        repo_identity={
            "commit_requested": expected_commit,
            "commit_effective": expected_commit,
            "dirty": False,
            "status_porcelain": "",
        },
    )

    assert route_identity["route"]["repo_commit_requested"] == expected_commit
    assert route_identity["route"]["repo_commit_effective"] == expected_commit
    assert route_identity["route"]["repo_dirty"] is False
    assert route_identity["route"]["repo_status_porcelain"] == ""


def test_stage_capture_reports_repo_identity_read_failure_before_generate(
    tmp_path,
    monkeypatch,
):
    import json
    import subprocess

    from scripts.run_mlx_stage_capture import main

    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    output_dir = tmp_path / "output"
    expected_commit = "e" * 40

    def unreadable_repo(_requested):
        raise subprocess.CalledProcessError(
            128,
            ["git", "rev-parse", "HEAD"],
            stderr="fatal: not a git repository",
        )

    monkeypatch.setattr(
        "scripts.run_mlx_stage_capture._read_repo_identity",
        unreadable_repo,
    )

    def unexpected_generate(*args, **kwargs):
        raise AssertionError("generation must not start without repo identity")

    monkeypatch.setattr(
        "scripts.run_mlx_stage_capture.subprocess.run",
        unexpected_generate,
    )

    result = main(
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--stop-after-stage",
            "shape_flow_step",
            "--expected-repo-commit",
            expected_commit,
        ]
    )

    assert result == 2
    report = json.loads((output_dir / "run_report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "preflight_repo_identity"
    assert report["primary_output_status"] == "not_started"
    assert report["repo_identity"] is None
    assert report["exit_code"] == 2


def test_stage_capture_rejects_substituted_repo_commit_before_generate(
    tmp_path,
    monkeypatch,
):
    import json

    from scripts.run_mlx_stage_capture import main

    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    output_dir = tmp_path / "output"
    expected_commit = "a" * 40
    effective_commit = "b" * 40
    monkeypatch.setattr(
        "scripts.run_mlx_stage_capture._read_repo_identity",
        lambda requested: {
            "commit_requested": requested,
            "commit_effective": effective_commit,
            "dirty": False,
            "status_porcelain": "",
        },
    )

    def unexpected_generate(*args, **kwargs):
        raise AssertionError("generation must not start under a substituted commit")

    monkeypatch.setattr(
        "scripts.run_mlx_stage_capture.subprocess.run",
        unexpected_generate,
    )

    result = main(
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--stop-after-stage",
            "shape_flow_step",
            "--expected-repo-commit",
            expected_commit,
        ]
    )

    assert result == 2
    report = json.loads((output_dir / "run_report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "preflight_repo_identity"
    assert report["primary_output_status"] == "not_started"
    assert report["repo_identity"] == {
        "commit_requested": expected_commit,
        "commit_effective": effective_commit,
        "dirty": False,
        "status_porcelain": "",
    }


def test_stage_capture_rejects_dirty_expected_repo_before_generate(
    tmp_path,
    monkeypatch,
):
    import json

    from scripts.run_mlx_stage_capture import main

    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    output_dir = tmp_path / "output"
    expected_commit = "c" * 40
    dirty_status = " M trellmlx/models/slat_flow.py\n"
    monkeypatch.setattr(
        "scripts.run_mlx_stage_capture._read_repo_identity",
        lambda requested: {
            "commit_requested": requested,
            "commit_effective": expected_commit,
            "dirty": True,
            "status_porcelain": dirty_status,
        },
    )

    def unexpected_generate(*args, **kwargs):
        raise AssertionError("generation must not start from a dirty checkout")

    monkeypatch.setattr(
        "scripts.run_mlx_stage_capture.subprocess.run",
        unexpected_generate,
    )

    result = main(
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--stop-after-stage",
            "shape_flow_step",
            "--expected-repo-commit",
            expected_commit,
        ]
    )

    assert result == 2
    report = json.loads((output_dir / "run_report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "preflight_repo_identity"
    assert report["primary_output_status"] == "not_started"
    assert report["repo_identity"] == {
        "commit_requested": expected_commit,
        "commit_effective": expected_commit,
        "dirty": True,
        "status_porcelain": dirty_status,
    }


def test_stage_capture_rejects_repo_mutation_during_successful_generate(
    tmp_path,
    monkeypatch,
):
    import json
    import subprocess

    import numpy as np

    from scripts.run_mlx_stage_capture import main

    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    output_dir = tmp_path / "output"
    expected_commit = "f" * 40
    clean = {
        "commit_requested": expected_commit,
        "commit_effective": expected_commit,
        "dirty": False,
        "status_porcelain": "",
    }
    dirty = {
        "commit_requested": expected_commit,
        "commit_effective": expected_commit,
        "dirty": True,
        "status_porcelain": " M generate.py\n",
    }
    identities = iter((clean, dirty))
    monkeypatch.setattr(
        "scripts.run_mlx_stage_capture._read_repo_identity",
        lambda _requested: next(identities),
    )

    def successful_generate(command, **_kwargs):
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        np.savez(checkpoint_dir / "conditioning.npz", cond=np.zeros((1, 1)))
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(
        "scripts.run_mlx_stage_capture.subprocess.run",
        successful_generate,
    )

    result = main(
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--stop-after-stage",
            "conditioning",
            "--expected-repo-commit",
            expected_commit,
        ]
    )

    assert result == 2
    report = json.loads((output_dir / "run_report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "postflight_repo_identity"
    assert report["primary_output_status"] == "invalid"
    assert report["repo_identity_preflight"] == clean
    assert report["repo_identity_postflight"] == dirty


def test_stage_capture_rejects_unpinned_repo_movement_during_generate(
    tmp_path,
    monkeypatch,
):
    import json
    import subprocess

    import numpy as np

    from scripts.run_mlx_stage_capture import main

    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    output_dir = tmp_path / "output"
    clean_a = {
        "commit_requested": None,
        "commit_effective": "a" * 40,
        "dirty": False,
        "status_porcelain": "",
    }
    clean_b = {
        "commit_requested": None,
        "commit_effective": "b" * 40,
        "dirty": False,
        "status_porcelain": "",
    }
    identities = iter((clean_a, clean_b))
    monkeypatch.setattr(
        "scripts.run_mlx_stage_capture._read_repo_identity",
        lambda _requested: next(identities),
    )

    def successful_generate(command, **_kwargs):
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        np.savez(checkpoint_dir / "conditioning.npz", cond=np.zeros((1, 1)))
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(
        "scripts.run_mlx_stage_capture.subprocess.run",
        successful_generate,
    )

    result = main(
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--stop-after-stage",
            "conditioning",
        ]
    )

    assert result == 2
    report = json.loads((output_dir / "run_report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "postflight_repo_identity"
    assert report["primary_output_status"] == "invalid"
    assert report["repo_identity_preflight"] == clean_a
    assert report["repo_identity_postflight"] == clean_b


def test_stage_capture_rejects_unreadable_unpinned_postflight_identity(
    tmp_path,
    monkeypatch,
):
    import json
    import subprocess

    import numpy as np

    from scripts.run_mlx_stage_capture import main

    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    output_dir = tmp_path / "output"
    clean = {
        "commit_requested": None,
        "commit_effective": "a" * 40,
        "dirty": False,
        "status_porcelain": "",
    }
    calls = 0

    def read_identity(_requested):
        nonlocal calls
        calls += 1
        if calls == 1:
            return clean
        raise subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(
        "scripts.run_mlx_stage_capture._read_repo_identity",
        read_identity,
    )

    def successful_generate(command, **_kwargs):
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        np.savez(checkpoint_dir / "conditioning.npz", cond=np.zeros((1, 1)))
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(
        "scripts.run_mlx_stage_capture.subprocess.run",
        successful_generate,
    )

    result = main(
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--stop-after-stage",
            "conditioning",
        ]
    )

    assert result == 2
    report = json.loads((output_dir / "run_report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "postflight_repo_identity"
    assert report["primary_output_status"] == "invalid"
    assert report["repo_identity_preflight"] == clean
    assert report["repo_identity_postflight"] is None
    assert "rev-parse" in report["repo_identity_postflight_error"]
    assert "exit status 128" in report["repo_identity_postflight_error"]


def test_stage_capture_rejects_missing_manifest_before_generate_and_reports(
    tmp_path, monkeypatch
):
    import json

    from scripts.run_mlx_stage_capture import main

    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    missing_manifest = tmp_path / "missing-manifest.json"
    output_dir = tmp_path / "output"

    def unexpected_generate(*args, **kwargs):
        raise AssertionError("generation must not start with a missing manifest")

    monkeypatch.setattr("scripts.run_mlx_stage_capture.subprocess.run", unexpected_generate)

    result = main(
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--stop-after-stage",
            "shape_flow_block_trace",
            "--shape-flow-block-injection-manifest",
            str(missing_manifest),
        ]
    )

    assert result == 2
    report = json.loads((output_dir / "run_report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "preflight_inputs"
    assert report["primary_output_status"] == "not_started"
    assert report["last_trustworthy_phase"] == "requested_route_parsed"
    assert report["requested_inputs"]["shape_flow_block_injection_manifest"] == str(
        missing_manifest
    )
    assert report["invalid_inputs"] == [
        {
            "field": "shape_flow_block_injection_manifest",
            "path": str(missing_manifest),
            "reason": "missing",
        }
    ]
    assert not (output_dir / "route_identity.json").exists()
    assert not (output_dir / "stdout.log").exists()


def test_stage_capture_wrapper_records_sparse_only_shared_noise(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "shape_slat",
            "--no-cascade",
            "--shared-noise",
            "noise.npz",
            "--shared-noise-sparse-only",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert "--shared-noise" in command
    assert command[command.index("--shared-noise") + 1] == "noise.npz"
    assert "--shared-noise-sparse-only" in command
    assert route_identity["route"]["shared_noise_sparse_only"] is True


def test_generate_exposes_sparse_only_shared_noise_contract():
    source = GENERATE_SOURCE.read_text()

    assert "--shared-noise-sparse-only" in source
    assert "--shared-noise-sparse-only requires --shared-noise" in source
    assert "Ignoring shared slat_noise_pool" in source


def test_stage_capture_source_quality_profile_sets_measured_500k_budget(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "mesh_uv",
            "--smoke-profile",
            "source-quality",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert command[command.index("--target-faces") + 1] == "500000"
    assert route_identity["route"]["target_faces"] == 500_000
    assert route_identity["route"]["smoke_profile"] == "source-quality"


def test_stage_capture_source_quality_profile_preserves_explicit_target_override(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "mesh_uv",
            "--smoke-profile",
            "source-quality",
            "--target-faces",
            "650000",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert command[command.index("--target-faces") + 1] == "650000"
    assert route_identity["route"]["target_faces"] == 650_000
    assert route_identity["route"]["smoke_profile"] == "source-quality"


def test_stage_capture_route_identity_records_attention_backend(tmp_path, monkeypatch):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    monkeypatch.setenv("TRELLIS2MLX_ATTENTION_BACKEND", "manual")
    monkeypatch.setenv(
        "TRELLIS2MLX_ATTENTION_SOFTMAX_BACKEND",
        "source-cuda-turing",
    )
    monkeypatch.setenv(
        "TRELLIS2MLX_ATTENTION_VALUE_BACKEND",
        "source-cuda-sequential",
    )
    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "sparse_flow_step",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert route_identity["route"]["attention_backend"] == "manual"
    assert route_identity["route"]["attention_softmax_backend_requested"] == (
        "source-cuda-turing"
    )
    assert route_identity["route"]["attention_softmax_backend_effective"] == (
        "source-cuda-turing"
    )
    assert route_identity["route"]["attention_value_backend_requested"] == (
        "source-cuda-sequential"
    )
    assert route_identity["route"]["attention_value_backend_effective"] == (
        "source-cuda-sequential"
    )
    assert route_identity["env"]["TRELLIS2MLX_ATTENTION_BACKEND"] == "manual"
    assert route_identity["env"][
        "TRELLIS2MLX_ATTENTION_SOFTMAX_BACKEND"
    ] == "source-cuda-turing"
    assert route_identity["env"][
        "TRELLIS2MLX_ATTENTION_VALUE_BACKEND"
    ] == "source-cuda-sequential"


def test_stage_capture_route_identity_marks_manual_selectors_ignored_by_fast_backend(
    tmp_path,
    monkeypatch,
):
    from scripts.run_mlx_stage_capture import (
        _build_generate_command,
        build_parser,
        build_route_identity,
    )

    monkeypatch.setenv("TRELLIS2MLX_ATTENTION_BACKEND", "fast")
    monkeypatch.setenv(
        "TRELLIS2MLX_ATTENTION_SOFTMAX_BACKEND",
        "source-cuda-turing",
    )
    monkeypatch.setenv(
        "TRELLIS2MLX_ATTENTION_VALUE_BACKEND",
        "source-cuda-sequential",
    )
    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "sparse_flow_step",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert route_identity["route"]["attention_backend"] == "fast"
    assert route_identity["route"]["attention_softmax_backend_requested"] == (
        "source-cuda-turing"
    )
    assert route_identity["route"]["attention_softmax_backend_effective"] == (
        "fused-fast-attention"
    )
    assert route_identity["route"]["attention_value_backend_requested"] == (
        "source-cuda-sequential"
    )
    assert route_identity["route"]["attention_value_backend_effective"] == (
        "fused-fast-attention"
    )


def test_stage_capture_route_identity_normalizes_mlx_manual_alias(
    tmp_path,
    monkeypatch,
):
    from scripts.run_mlx_stage_capture import (
        _build_generate_command,
        build_parser,
        build_route_identity,
    )

    monkeypatch.setenv("TRELLIS2MLX_ATTENTION_BACKEND", "mlx-manual")
    monkeypatch.setenv(
        "TRELLIS2MLX_ATTENTION_SOFTMAX_BACKEND",
        "source-cuda-turing",
    )
    monkeypatch.setenv(
        "TRELLIS2MLX_ATTENTION_VALUE_BACKEND",
        "source-cuda-sequential",
    )
    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "sparse_flow_step",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert route_identity["route"]["attention_backend_requested"] == "mlx-manual"
    assert route_identity["route"]["attention_backend"] == "manual"
    assert route_identity["route"]["attention_softmax_backend_effective"] == (
        "source-cuda-turing"
    )
    assert route_identity["route"]["attention_value_backend_effective"] == (
        "source-cuda-sequential"
    )


def test_stage_capture_forwards_and_records_qk_norm_backend(tmp_path):
    from scripts.run_mlx_stage_capture import (
        _build_generate_command,
        build_parser,
        build_route_identity,
    )

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "shape_flow_block_trace",
            "--qk-norm-backend",
            "mlx-sum",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    option = command.index("--qk-norm-backend")
    assert command[option + 1] == "mlx-sum"
    assert route_identity["route"]["qk_norm_backend_requested"] == "mlx-sum"


def test_stage_capture_wrapper_exposes_sparse_flow_step_route(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "sparse_flow_step",
            "--seed",
            "42",
            "--steps",
            "8",
            "--resolution",
            "512",
            "--shared-noise",
            "noise.npz",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")

    assert command[command.index("--stop-after-stage") + 1] == "sparse_flow_step"


def test_generate_exposes_sparse_flow_steps_capture():
    source = GENERATE_SOURCE.read_text()

    assert '"sparse_flow_steps"' in source
    assert "capture_steps=step_captures" in source
    assert 'save_checkpoint(\n            args.save_checkpoints,\n            "sparse_flow_steps"' in source


def test_stage_capture_wrapper_exposes_sparse_flow_steps_route(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "sparse_flow_steps",
            "--seed",
            "42",
            "--steps",
            "8",
            "--resolution",
            "512",
            "--shared-noise",
            "noise.npz",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")

    assert command[command.index("--stop-after-stage") + 1] == "sparse_flow_steps"


def test_generate_shared_noise_feeds_shape_slat_noise_pool():
    source = GENERATE_SOURCE.read_text()

    sparse_noise = source.index('shared_noise["ss_noise"]')
    shape_noise = source.index('shared_noise["slat_noise_pool"]')
    broadcast = source.index("np.broadcast_to", shape_noise)
    lr_shape_noise = source.index("Shared shape SLat noise", shape_noise)

    assert sparse_noise < shape_noise
    assert shape_noise < broadcast
    assert broadcast < lr_shape_noise


def test_generate_exposes_shape_flow_step_capture():
    source = GENERATE_SOURCE.read_text()

    assert '"shape_flow_step"' in source
    assert 'save_checkpoint(\n            args.save_checkpoints,\n            "shape_flow_step"' in source
    assert 'args.stop_after_stage == "shape_flow_step"' in source


def test_stage_capture_wrapper_exposes_shape_flow_step_route(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "shape_flow_step",
            "--seed",
            "42",
            "--steps",
            "8",
            "--resolution",
            "512",
            "--shared-noise",
            "noise.npz",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")

    assert command[command.index("--stop-after-stage") + 1] == "shape_flow_step"


def test_generate_exposes_shape_flow_steps_capture():
    source = GENERATE_SOURCE.read_text()

    assert '"shape_flow_steps"' in source
    assert "capture_steps=shape_step_captures" in source
    assert 'save_checkpoint(\n            args.save_checkpoints,\n            "shape_flow_steps"' in source
    for field in (
        "sample_in",
        "pred_pos",
        "pred_neg",
        "pred_cfg",
        "x0_pos",
        "x0_cfg",
        "std_pos",
        "std_cfg",
        "ratio_raw",
        "std_ratio",
        "ratio_effective",
        "x0_rescaled",
        "x0_after_rescale",
        "pred_final",
        "sample_next",
    ):
        assert f'shape_stack_step("{field}")' in source


def test_stage_capture_wrapper_exposes_shape_flow_steps_route(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "shape_flow_steps",
            "--seed",
            "42",
            "--steps",
            "8",
            "--resolution",
            "512",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")

    assert command[command.index("--stop-after-stage") + 1] == "shape_flow_steps"


def test_stage_capture_forwards_and_records_shape_flow_layernorm_backend(tmp_path):
    from scripts.run_mlx_stage_capture import (
        _build_generate_command,
        build_parser,
        build_route_identity,
    )

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "shape_flow_steps",
            "--shape-flow-layernorm-backend",
            "cuda-welford-metal",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    option = command.index("--shape-flow-layernorm-backend")
    assert command[option + 1] == "cuda-welford-metal"
    assert (
        route_identity["route"]["shape_flow_layernorm_backend_requested"]
        == "cuda-welford-metal"
    )


def test_stage_capture_forwards_and_records_turing_rsqrt_lut(tmp_path):
    import hashlib

    import numpy as np

    from scripts.run_mlx_stage_capture import (
        _build_generate_command,
        build_parser,
        build_route_identity,
    )

    lut = tmp_path / "turing-rsqrt.npz"
    np.savez_compressed(
        lut,
        normalized_delta=np.zeros((1 << 24,), dtype=np.int8),
    )
    digest = hashlib.sha256(lut.read_bytes()).hexdigest()
    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path / "output"),
            "--stop-after-stage",
            "shape_flow_steps",
            "--shape-flow-layernorm-backend",
            "cuda-welford-turing-t4",
            "--turing-rsqrt-lut",
            str(lut),
            "--expected-turing-rsqrt-lut-sha256",
            digest,
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert command[command.index("--turing-rsqrt-lut") + 1] == str(lut)
    assert (
        command[
            command.index("--expected-turing-rsqrt-lut-sha256") + 1
        ]
        == digest
    )
    assert (
        route_identity["route"]["turing_rsqrt_lut_sha256_effective"]
        == digest
    )
    assert (
        route_identity["route"]["turing_rsqrt_lut_sha256_requested"]
        == digest
    )


def test_stage_capture_rejects_turing_rsqrt_lut_hash_mismatch(tmp_path):
    import pytest

    from scripts.run_mlx_stage_capture import (
        _validate_turing_rsqrt_route_args,
        build_parser,
    )

    lut = tmp_path / "turing-rsqrt.npz"
    lut.write_bytes(b"substituted-lut")
    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path / "output"),
            "--stop-after-stage",
            "shape_flow_steps",
            "--shape-flow-layernorm-backend",
            "cuda-welford-turing-t4",
            "--turing-rsqrt-lut",
            str(lut),
            "--expected-turing-rsqrt-lut-sha256",
            "a" * 64,
        ]
    )

    with pytest.raises(ValueError, match="Turing rsqrt LUT SHA256 mismatch"):
        _validate_turing_rsqrt_route_args(args)


def test_stage_capture_reports_turing_lut_substitution_before_generate(
    tmp_path, monkeypatch
):
    import hashlib
    import json

    from scripts.run_mlx_stage_capture import main

    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    lut = tmp_path / "turing-rsqrt.npz"
    lut.write_bytes(b"substituted-lut")
    effective_digest = hashlib.sha256(lut.read_bytes()).hexdigest()
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        "scripts.run_mlx_stage_capture._read_repo_identity",
        lambda requested: {
            "commit_requested": requested,
            "commit_effective": "a" * 40,
            "dirty": False,
            "status_porcelain": "",
        },
    )

    def unexpected_generate(*_args, **_kwargs):
        raise AssertionError("generation must not start with a substituted LUT")

    monkeypatch.setattr(
        "scripts.run_mlx_stage_capture.subprocess.run",
        unexpected_generate,
    )

    result = main(
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--stop-after-stage",
            "shape_flow_step",
            "--shape-flow-layernorm-backend",
            "cuda-welford-turing-t4",
            "--turing-rsqrt-lut",
            str(lut),
            "--expected-turing-rsqrt-lut-sha256",
            "a" * 64,
        ]
    )

    assert result == 2
    report = json.loads((output_dir / "run_report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "preflight_turing_rsqrt_route"
    assert report["last_trustworthy_phase"] == "requested_route_parsed"
    assert report["primary_output_status"] == "not_started"
    assert report["turing_rsqrt_lut_identity"] == {
        "path": str(lut),
        "sha256_requested": "a" * 64,
        "sha256_effective": effective_digest,
    }
    assert report["requested_inputs"]["turing_rsqrt_lut"] == str(lut)
    assert "Turing rsqrt LUT SHA256 mismatch" in report["error"]
    assert not (output_dir / "route_identity.json").exists()
    assert not (output_dir / "stdout.log").exists()


def test_stage_capture_reports_unreadable_turing_lut_before_generate(
    tmp_path, monkeypatch
):
    import json

    from scripts import run_mlx_stage_capture as capture

    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    lut = tmp_path / "turing-rsqrt.npz"
    lut.write_bytes(b"unreadable-lut")
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        capture,
        "_read_repo_identity",
        lambda requested: {
            "commit_requested": requested,
            "commit_effective": "a" * 40,
            "dirty": False,
            "status_porcelain": "",
        },
    )
    original_sha256_file = capture._sha256_file

    def unreadable_lut(path):
        if Path(path) == lut:
            raise PermissionError(f"permission denied: {lut}")
        return original_sha256_file(path)

    monkeypatch.setattr(capture, "_sha256_file", unreadable_lut)

    def unexpected_generate(*_args, **_kwargs):
        raise AssertionError("generation must not start with an unreadable LUT")

    monkeypatch.setattr(capture.subprocess, "run", unexpected_generate)

    result = capture.main(
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--stop-after-stage",
            "shape_flow_step",
            "--shape-flow-layernorm-backend",
            "cuda-welford-turing-t4",
            "--turing-rsqrt-lut",
            str(lut),
            "--expected-turing-rsqrt-lut-sha256",
            "a" * 64,
        ]
    )

    assert result == 2
    report = json.loads((output_dir / "run_report.json").read_text())
    assert report["failure_phase"] == "preflight_turing_rsqrt_route"
    assert report["primary_output_status"] == "not_started"
    assert report["turing_rsqrt_lut_identity"]["path"] == str(lut)
    assert report["turing_rsqrt_lut_identity"]["sha256_effective"] is None
    assert "permission denied" in report["error"]
    assert not (output_dir / "route_identity.json").exists()
    assert not (output_dir / "stdout.log").exists()


def test_stage_capture_rejects_hash_matched_malformed_turing_lut_before_generate(
    tmp_path, monkeypatch
):
    import hashlib
    import json

    import numpy as np

    from scripts import run_mlx_stage_capture as capture

    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    lut = tmp_path / "turing-rsqrt.npz"
    np.savez(lut, wrong_key=np.zeros((16,), dtype=np.int8))
    digest = hashlib.sha256(lut.read_bytes()).hexdigest()
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        capture,
        "_read_repo_identity",
        lambda requested: {
            "commit_requested": requested,
            "commit_effective": "a" * 40,
            "dirty": False,
            "status_porcelain": "",
        },
    )

    def unexpected_generate(*_args, **_kwargs):
        raise AssertionError("generation must not start with a malformed LUT")

    monkeypatch.setattr(capture.subprocess, "run", unexpected_generate)

    result = capture.main(
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--stop-after-stage",
            "shape_flow_step",
            "--shape-flow-layernorm-backend",
            "cuda-welford-turing-t4",
            "--turing-rsqrt-lut",
            str(lut),
            "--expected-turing-rsqrt-lut-sha256",
            digest,
        ]
    )

    assert result == 2
    report = json.loads((output_dir / "run_report.json").read_text())
    assert report["failure_phase"] == "preflight_turing_rsqrt_route"
    assert report["primary_output_status"] == "not_started"
    assert report["turing_rsqrt_lut_identity"] == {
        "path": str(lut),
        "sha256_requested": digest,
        "sha256_effective": digest,
    }
    assert "omits normalized_delta" in report["error"]
    assert not (output_dir / "route_identity.json").exists()
    assert not (output_dir / "stdout.log").exists()


def test_generate_turing_rsqrt_lut_loader_rejects_substitution(tmp_path):
    import generate
    import pytest

    lut = tmp_path / "turing-rsqrt.npz"
    lut.write_bytes(b"substituted-lut")

    with pytest.raises(ValueError, match="Turing rsqrt LUT SHA256 mismatch"):
        generate._load_turing_rsqrt_lut(lut, "a" * 64)


def test_generate_turing_rsqrt_lut_loader_rejects_malformed_payload(tmp_path):
    import hashlib

    import generate
    import numpy as np
    import pytest

    lut = tmp_path / "turing-rsqrt.npz"
    np.savez(lut, wrong_key=np.zeros((16,), dtype=np.int8))
    digest = hashlib.sha256(lut.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="omits normalized_delta"):
        generate._load_turing_rsqrt_lut(lut, digest)

    np.savez(lut, normalized_delta=np.zeros((16,), dtype=np.int8))
    digest = hashlib.sha256(lut.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=r"must be int8\[16777216\]"):
        generate._load_turing_rsqrt_lut(lut, digest)


def test_stage_capture_forwards_and_records_turing_rope_phase_lut(tmp_path):
    import hashlib

    import numpy as np

    from scripts.run_mlx_stage_capture import (
        _build_generate_command,
        build_parser,
        build_route_identity,
    )

    lut = tmp_path / "turing-rope.npz"
    phases = np.zeros((64, 21, 2), dtype=np.float32)
    phases[..., 0] = 1.0
    np.savez(lut, phase_pairs=phases)
    digest = hashlib.sha256(lut.read_bytes()).hexdigest()
    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path / "output"),
            "--stop-after-stage",
            "shape_flow_block_trace",
            "--no-cascade",
            "--rope-backend",
            "cuda-polar-turing-t4",
            "--turing-rope-phase-lut",
            str(lut),
            "--expected-turing-rope-phase-lut-sha256",
            digest,
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert command[command.index("--rope-backend") + 1] == "cuda-polar-turing-t4"
    assert command[command.index("--turing-rope-phase-lut") + 1] == str(lut)
    assert (
        command[
            command.index("--expected-turing-rope-phase-lut-sha256") + 1
        ]
        == digest
    )
    assert route_identity["route"]["rope_backend_requested"] == (
        "cuda-polar-turing-t4"
    )
    assert route_identity["route"]["turing_rope_phase_lut_path"] == str(lut)
    assert (
        route_identity["route"]["turing_rope_phase_lut_sha256_requested"]
        == digest
    )
    assert (
        route_identity["route"]["turing_rope_phase_lut_sha256_effective"]
        == digest
    )


def test_stage_capture_rejects_hash_matched_malformed_turing_rope_lut(tmp_path):
    import hashlib

    import numpy as np
    import pytest

    from scripts.run_mlx_stage_capture import (
        _validate_turing_rope_route_args,
        build_parser,
    )

    lut = tmp_path / "turing-rope.npz"
    np.savez(lut, phase_pairs=np.zeros((64, 20, 2), dtype=np.float32))
    digest = hashlib.sha256(lut.read_bytes()).hexdigest()
    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path / "output"),
            "--stop-after-stage",
            "shape_flow_block_trace",
            "--rope-backend",
            "cuda-polar-turing-t4",
            "--turing-rope-phase-lut",
            str(lut),
            "--expected-turing-rope-phase-lut-sha256",
            digest,
        ]
    )

    with pytest.raises(
        ValueError, match=r"Turing RoPE phase LUT must be float32\[64,21,2\]"
    ):
        _validate_turing_rope_route_args(args)


def test_generate_turing_rope_phase_lut_loader_rejects_substitution_and_malformed(
    tmp_path,
):
    import hashlib

    import generate
    import numpy as np
    import pytest

    lut = tmp_path / "turing-rope.npz"
    lut.write_bytes(b"substituted-lut")
    with pytest.raises(ValueError, match="Turing RoPE phase LUT SHA256 mismatch"):
        generate._load_turing_rope_phase_lut(lut, "a" * 64)

    np.savez(lut, wrong_key=np.zeros((64, 21, 2), dtype=np.float32))
    digest = hashlib.sha256(lut.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="omits phase_pairs"):
        generate._load_turing_rope_phase_lut(lut, digest)

    np.savez(lut, phase_pairs=np.zeros((64, 21, 2), dtype=np.float16))
    digest = hashlib.sha256(lut.read_bytes()).hexdigest()
    with pytest.raises(
        ValueError, match=r"must be float32\[64,21,2\]"
    ):
        generate._load_turing_rope_phase_lut(lut, digest)


def test_shape_flow_checkpoint_rope_fallback_cannot_impersonate_requested_route(
    tmp_path,
):
    import numpy as np
    import pytest

    from scripts.run_mlx_stage_capture import _bind_effective_rope_backend

    checkpoint = tmp_path / "shape_flow_block_trace.npz"
    np.savez(
        checkpoint,
        rope_backend=np.array("mlx-real"),
        shape_flow_turing_rope_phase_lut_sha256=np.array(""),
    )
    route_identity = {
        "route": {
            "rope_backend_requested": "cuda-polar-turing-t4",
            "turing_rope_phase_lut_sha256_effective": "a" * 64,
        }
    }

    with pytest.raises(
        ValueError,
        match=(
            "shape-flow effective RoPE backend 'mlx-real' does not match "
            "requested 'cuda-polar-turing-t4'"
        ),
    ):
        _bind_effective_rope_backend(route_identity, checkpoint)


def test_shape_flow_turing_rope_checkpoint_requires_effective_lut_hash(tmp_path):
    import numpy as np
    import pytest

    from scripts.run_mlx_stage_capture import _bind_effective_rope_backend

    checkpoint = tmp_path / "shape_flow_block_trace.npz"
    np.savez(
        checkpoint,
        rope_backend=np.array("cuda-polar-turing-t4"),
    )
    route_identity = {
        "route": {
            "rope_backend_requested": "cuda-polar-turing-t4",
            "turing_rope_phase_lut_sha256_effective": "a" * 64,
        }
    }

    with pytest.raises(
        ValueError, match="omits effective Turing RoPE phase LUT SHA256"
    ):
        _bind_effective_rope_backend(route_identity, checkpoint)


def _write_valid_shape_flow_steps_checkpoint(path, *, steps=3, tokens=2, channels=4):
    import numpy as np

    rescale_t = np.float32(3.0)
    schedule = np.linspace(1, 0, steps + 1, dtype=np.float64)
    schedule = rescale_t * schedule / (1 + (rescale_t - 1) * schedule)
    t = schedule[:-1].astype(np.float32)
    t_prev = schedule[1:].astype(np.float32)
    pred_final = np.ones((steps, tokens, channels), dtype=np.float32)
    sample_in = np.empty_like(pred_final)
    sample_next = np.empty_like(pred_final)
    sample_in[0] = 0.0
    for index in range(steps):
        sample_next[index] = (
            sample_in[index]
            - np.float32(t[index] - t_prev[index]) * pred_final[index]
        )
        if index + 1 < steps:
            sample_in[index + 1] = sample_next[index]
    stepped = np.zeros_like(sample_in)
    scalar_steps = np.ones((steps,), dtype=np.float32)
    coords_3d = np.arange(tokens * 3, dtype=np.int32).reshape(tokens, 3)
    coords = np.column_stack([np.zeros(tokens, dtype=np.int32), coords_3d])
    np.savez(
        path,
        noise=sample_in[0],
        sample_feats=sample_in[0],
        coords=coords,
        coords_3d=coords_3d,
        sample_in=sample_in,
        pred_pos=stepped,
        pred_neg=stepped,
        pred_cfg=stepped,
        x0_pos=stepped,
        x0_cfg=stepped,
        std_pos=scalar_steps,
        std_cfg=scalar_steps,
        ratio_raw=scalar_steps,
        std_ratio=scalar_steps,
        ratio_effective=scalar_steps,
        x0_rescaled=stepped,
        x0_after_rescale=stepped,
        pred_final=pred_final,
        pred_v_feats=pred_final,
        sample_next=sample_next,
        t=t,
        t_prev=t_prev,
        steps=np.array(steps, dtype=np.int32),
        guidance_strength=np.array(7.5, dtype=np.float32),
        guidance_rescale=np.array(0.5, dtype=np.float32),
        guidance_interval=np.array([0.6, 1.0], dtype=np.float32),
        rescale_t=rescale_t,
        sigma_min=np.array(1e-5, dtype=np.float32),
        shape_flow_block_injection_json=np.array(""),
        shape_flow_layernorm_backend=np.array("mlx-two-pass"),
        qk_norm_backend=np.array("source-cuda-warp32"),
        rope_backend=np.array("mlx-real"),
        shape_flow_turing_rope_phase_lut_sha256=np.array(""),
    )


def _rewrite_npz_array(path, name, value):
    import numpy as np

    with np.load(path, allow_pickle=False) as checkpoint:
        payload = {key: np.asarray(checkpoint[key]) for key in checkpoint.files}
    payload[name] = value
    np.savez(path, **payload)


def test_shape_flow_steps_missing_effective_layernorm_backend_fails_loud(tmp_path):
    import numpy as np
    import pytest

    from scripts.run_mlx_stage_capture import _validate_shape_flow_steps_checkpoint

    checkpoint = tmp_path / "shape_flow_steps.npz"
    _write_valid_shape_flow_steps_checkpoint(checkpoint)
    with np.load(checkpoint, allow_pickle=False) as loaded:
        payload = {
            key: np.asarray(loaded[key])
            for key in loaded.files
            if key != "shape_flow_layernorm_backend"
        }
    np.savez(checkpoint, **payload)

    with pytest.raises(ValueError, match="missing required arrays.*shape_flow_layernorm_backend"):
        _validate_shape_flow_steps_checkpoint(
            checkpoint,
            expected_steps=3,
            expected_route={
                "shape_flow_layernorm_backend_requested": "cuda-welford-metal",
            },
        )


def test_shape_flow_steps_missing_effective_qk_norm_backend_fails_loud(tmp_path):
    import numpy as np
    import pytest

    from scripts.run_mlx_stage_capture import _validate_shape_flow_steps_checkpoint

    checkpoint = tmp_path / "shape_flow_steps.npz"
    _write_valid_shape_flow_steps_checkpoint(checkpoint)
    with np.load(checkpoint, allow_pickle=False) as loaded:
        payload = {
            key: np.asarray(loaded[key])
            for key in loaded.files
            if key != "qk_norm_backend"
        }
    np.savez(checkpoint, **payload)

    with pytest.raises(
        ValueError, match="missing required arrays.*qk_norm_backend"
    ):
        _validate_shape_flow_steps_checkpoint(
            checkpoint,
            expected_steps=3,
            expected_route={
                "shape_flow_layernorm_backend_requested": "mlx-two-pass",
                "qk_norm_backend_requested": "source-cuda-warp32",
            },
        )


def test_shape_flow_steps_missing_effective_rope_backend_fails_loud(tmp_path):
    import numpy as np
    import pytest

    from scripts.run_mlx_stage_capture import _validate_shape_flow_steps_checkpoint

    checkpoint = tmp_path / "shape_flow_steps.npz"
    _write_valid_shape_flow_steps_checkpoint(checkpoint)
    with np.load(checkpoint, allow_pickle=False) as loaded:
        payload = {
            key: np.asarray(loaded[key])
            for key in loaded.files
            if key != "rope_backend"
        }
    np.savez(checkpoint, **payload)

    with pytest.raises(
        ValueError, match="missing required arrays.*rope_backend"
    ):
        _validate_shape_flow_steps_checkpoint(
            checkpoint,
            expected_steps=3,
            expected_route={
                "shape_flow_layernorm_backend_requested": "mlx-two-pass",
                "qk_norm_backend_requested": "source-cuda-warp32",
                "rope_backend_requested": "mlx-real",
            },
        )


def test_shape_flow_steps_rope_fallback_cannot_impersonate_requested_route(
    tmp_path,
):
    import pytest

    from scripts.run_mlx_stage_capture import _validate_shape_flow_steps_checkpoint

    checkpoint = tmp_path / "shape_flow_steps.npz"
    _write_valid_shape_flow_steps_checkpoint(checkpoint)

    with pytest.raises(
        ValueError,
        match=(
            "shape_flow_steps effective RoPE backend 'mlx-real' does not "
            "match requested 'source-complex'"
        ),
    ):
        _validate_shape_flow_steps_checkpoint(
            checkpoint,
            expected_steps=3,
            expected_route={
                "shape_flow_layernorm_backend_requested": "mlx-two-pass",
                "qk_norm_backend_requested": "source-cuda-warp32",
                "rope_backend_requested": "source-complex",
            },
        )


def test_shape_flow_steps_qk_norm_fallback_cannot_impersonate_requested_route(
    tmp_path,
):
    import numpy as np
    import pytest

    from scripts.run_mlx_stage_capture import _validate_shape_flow_steps_checkpoint

    checkpoint = tmp_path / "shape_flow_steps.npz"
    _write_valid_shape_flow_steps_checkpoint(checkpoint)
    _rewrite_npz_array(
        checkpoint,
        "qk_norm_backend",
        np.array("source-cuda-warp32"),
    )

    with pytest.raises(
        ValueError,
        match=(
            "shape_flow_steps effective Q/K norm backend "
            "'source-cuda-warp32' does not match requested 'mlx-sum'"
        ),
    ):
        _validate_shape_flow_steps_checkpoint(
            checkpoint,
            expected_steps=3,
            expected_route={
                "shape_flow_layernorm_backend_requested": "mlx-two-pass",
                "qk_norm_backend_requested": "mlx-sum",
            },
        )


def test_shape_flow_steps_layernorm_backend_fallback_cannot_impersonate_requested_route(
    tmp_path,
):
    import pytest

    from scripts.run_mlx_stage_capture import _validate_shape_flow_steps_checkpoint

    checkpoint = tmp_path / "shape_flow_steps.npz"
    _write_valid_shape_flow_steps_checkpoint(checkpoint)

    with pytest.raises(
        ValueError,
        match=(
            "shape_flow_steps effective LayerNorm backend 'mlx-two-pass' "
            "does not match requested 'cuda-welford-metal'"
        ),
    ):
        _validate_shape_flow_steps_checkpoint(
            checkpoint,
            expected_steps=3,
            expected_route={
                "shape_flow_layernorm_backend_requested": "cuda-welford-metal",
            },
        )


def test_shape_flow_steps_turing_backend_requires_effective_lut_hash(tmp_path):
    import numpy as np
    import pytest

    from scripts.run_mlx_stage_capture import (
        _validate_shape_flow_steps_checkpoint,
    )

    checkpoint = tmp_path / "shape_flow_steps.npz"
    _write_valid_shape_flow_steps_checkpoint(checkpoint)
    _rewrite_npz_array(
        checkpoint,
        "shape_flow_layernorm_backend",
        np.array("cuda-welford-turing-t4"),
    )

    with pytest.raises(
        ValueError, match="omits effective Turing rsqrt LUT SHA256"
    ):
        _validate_shape_flow_steps_checkpoint(
            checkpoint,
            expected_steps=3,
            expected_route={
                "shape_flow_layernorm_backend_requested": (
                    "cuda-welford-turing-t4"
                ),
                "turing_rsqrt_lut_sha256_effective": "a" * 64,
            },
        )


def test_shape_flow_steps_turing_backend_rejects_lut_hash_mismatch(tmp_path):
    import numpy as np
    import pytest

    from scripts.run_mlx_stage_capture import (
        _validate_shape_flow_steps_checkpoint,
    )

    checkpoint = tmp_path / "shape_flow_steps.npz"
    _write_valid_shape_flow_steps_checkpoint(checkpoint)
    _rewrite_npz_array(
        checkpoint,
        "shape_flow_layernorm_backend",
        np.array("cuda-welford-turing-t4"),
    )
    _rewrite_npz_array(
        checkpoint,
        "shape_flow_turing_rsqrt_lut_sha256",
        np.array("b" * 64),
    )

    with pytest.raises(ValueError, match="effective Turing rsqrt LUT SHA256"):
        _validate_shape_flow_steps_checkpoint(
            checkpoint,
            expected_steps=3,
            expected_route={
                "shape_flow_layernorm_backend_requested": (
                    "cuda-welford-turing-t4"
                ),
                "turing_rsqrt_lut_sha256_effective": "a" * 64,
            },
        )


def test_shape_flow_steps_partial_checkpoint_fails_loud(tmp_path, monkeypatch):
    import json
    import subprocess

    import numpy as np

    from scripts.run_mlx_stage_capture import main

    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    output_dir = tmp_path / "output"

    def write_partial(command, **kwargs):
        checkpoint = output_dir / "checkpoints" / "shape_flow_steps.npz"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez(checkpoint, sample_in=np.zeros((8, 2, 32), dtype=np.float32))
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("scripts.run_mlx_stage_capture.subprocess.run", write_partial)

    result = main(
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--stop-after-stage",
            "shape_flow_steps",
            "--steps",
            "8",
        ]
    )

    assert result == 2
    report = json.loads((output_dir / "run_report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "validate_primary_output"
    assert report["primary_output_status"] == "invalid"
    assert "missing required arrays" in report["error"]


def test_shape_flow_steps_stale_checkpoint_cannot_impersonate_current_run(tmp_path, monkeypatch):
    import json
    import subprocess

    import numpy as np

    from scripts.run_mlx_stage_capture import main

    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    output_dir = tmp_path / "output"
    stale = output_dir / "checkpoints" / "shape_flow_steps.npz"
    stale.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stale, sample_in=np.zeros((8, 2, 32), dtype=np.float32))

    def write_nothing(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("scripts.run_mlx_stage_capture.subprocess.run", write_nothing)

    result = main(
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--stop-after-stage",
            "shape_flow_steps",
            "--steps",
            "8",
        ]
    )

    assert result == 2
    assert not stale.exists()
    report = json.loads((output_dir / "run_report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "missing_primary_output"
    assert report["primary_output_status"] == "missing"


def test_shape_flow_steps_primary_cannot_alias_requested_input(tmp_path, monkeypatch):
    import json

    from scripts.run_mlx_stage_capture import main

    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    output_dir = tmp_path / "output"
    protected = output_dir / "checkpoints" / "shape_flow_steps.npz"
    protected.parent.mkdir(parents=True, exist_ok=True)
    protected.write_bytes(b"protected shape flow input")

    def unexpected_generate(*args, **kwargs):
        raise AssertionError("generation must not start with a primary/input collision")

    monkeypatch.setattr("scripts.run_mlx_stage_capture.subprocess.run", unexpected_generate)

    result = main(
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--stop-after-stage",
            "shape_flow_steps",
            "--steps",
            "8",
            "--shape-flow-noise-sample",
            str(protected),
        ]
    )

    assert result == 2
    assert protected.read_bytes() == b"protected shape flow input"
    report = json.loads((output_dir / "run_report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "preflight_output_collision"
    assert report["primary_output_status"] == "not_started"
    assert report["collisions"] == [
        {"field": "shape_flow_noise_sample", "path": str(protected)}
    ]


def test_shape_flow_steps_complete_checkpoint_is_route_bound(tmp_path, monkeypatch):
    import json
    import subprocess

    from scripts.run_mlx_stage_capture import main

    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    output_dir = tmp_path / "output"

    def write_complete(command, **kwargs):
        checkpoint = output_dir / "checkpoints" / "shape_flow_steps.npz"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        _write_valid_shape_flow_steps_checkpoint(checkpoint)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("scripts.run_mlx_stage_capture.subprocess.run", write_complete)

    result = main(
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--stop-after-stage",
            "shape_flow_steps",
            "--steps",
            "3",
        ]
    )

    assert result == 0
    report = json.loads((output_dir / "run_report.json").read_text())
    route = json.loads((output_dir / "route_identity.json").read_text())
    assert report["status"] == "done"
    assert report["last_trustworthy_phase"] == "shape_flow_steps_validated"
    assert report["primary_output_status"] == "written"
    assert report["primary_output_validation"]["step_count"] == 3
    assert report["primary_output_validation"]["token_count"] == 2
    assert route["route"]["shape_flow_steps_output"] == report["primary_output_validation"]


def test_shape_flow_steps_rejects_sampler_inconsistent_final_transition(tmp_path):
    import numpy as np
    import pytest

    from scripts.run_mlx_stage_capture import _validate_shape_flow_steps_checkpoint

    checkpoint = tmp_path / "shape_flow_steps.npz"
    _write_valid_shape_flow_steps_checkpoint(checkpoint)
    with np.load(checkpoint, allow_pickle=False) as artifact:
        sample_next = np.asarray(artifact["sample_next"]).copy()
    sample_next[-1] += np.float32(0.25)
    _rewrite_npz_array(checkpoint, "sample_next", sample_next)

    with pytest.raises(ValueError, match="Euler transition"):
        _validate_shape_flow_steps_checkpoint(checkpoint, expected_steps=3)


def test_shape_flow_steps_rejects_wrong_contiguous_schedule(tmp_path):
    import numpy as np
    import pytest

    from scripts.run_mlx_stage_capture import _validate_shape_flow_steps_checkpoint

    checkpoint = tmp_path / "shape_flow_steps.npz"
    _write_valid_shape_flow_steps_checkpoint(checkpoint)
    _rewrite_npz_array(
        checkpoint,
        "t",
        np.array([1.0, 0.75, 0.25], dtype=np.float32),
    )
    _rewrite_npz_array(
        checkpoint,
        "t_prev",
        np.array([0.75, 0.25, 0.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="rescaled route schedule"):
        _validate_shape_flow_steps_checkpoint(checkpoint, expected_steps=3)


def test_shape_flow_steps_rejects_nonfinite_sampler_metadata(tmp_path):
    import numpy as np
    import pytest

    from scripts.run_mlx_stage_capture import _validate_shape_flow_steps_checkpoint

    checkpoint = tmp_path / "shape_flow_steps.npz"
    _write_valid_shape_flow_steps_checkpoint(checkpoint)
    _rewrite_npz_array(
        checkpoint,
        "guidance_strength",
        np.array(np.nan, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="guidance_strength"):
        _validate_shape_flow_steps_checkpoint(checkpoint, expected_steps=3)


def test_shape_flow_steps_rejects_wrong_tensor_dtype(tmp_path):
    import numpy as np
    import pytest

    from scripts.run_mlx_stage_capture import _validate_shape_flow_steps_checkpoint

    checkpoint = tmp_path / "shape_flow_steps.npz"
    _write_valid_shape_flow_steps_checkpoint(checkpoint)
    with np.load(checkpoint, allow_pickle=False) as artifact:
        sample_in = np.asarray(artifact["sample_in"], dtype=np.float64)
    _rewrite_npz_array(checkpoint, "sample_in", sample_in)

    with pytest.raises(ValueError, match="sample_in dtype"):
        _validate_shape_flow_steps_checkpoint(checkpoint, expected_steps=3)


def test_shape_flow_steps_rejects_invalid_injection_identity(tmp_path):
    import numpy as np
    import pytest

    from scripts.run_mlx_stage_capture import _validate_shape_flow_steps_checkpoint

    checkpoint = tmp_path / "shape_flow_steps.npz"
    _write_valid_shape_flow_steps_checkpoint(checkpoint)
    _rewrite_npz_array(
        checkpoint,
        "shape_flow_block_injection_json",
        np.array("not-json"),
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        _validate_shape_flow_steps_checkpoint(checkpoint, expected_steps=3)


def test_shape_flow_steps_rejects_wrong_step_count_dtype(tmp_path):
    import numpy as np
    import pytest

    from scripts.run_mlx_stage_capture import _validate_shape_flow_steps_checkpoint

    checkpoint = tmp_path / "shape_flow_steps.npz"
    _write_valid_shape_flow_steps_checkpoint(checkpoint)
    _rewrite_npz_array(checkpoint, "steps", np.array(3.0, dtype=np.float32))

    with pytest.raises(ValueError, match="steps must be an int32 scalar"):
        _validate_shape_flow_steps_checkpoint(checkpoint, expected_steps=3)


def test_shape_flow_steps_rejects_empty_identity_when_injection_requested(
    tmp_path, monkeypatch
):
    import json
    import subprocess

    from scripts.run_mlx_stage_capture import main

    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    trace = tmp_path / "injection-trace.npz"
    trace.write_bytes(b"trace")
    output_dir = tmp_path / "output"

    def write_complete(command, **kwargs):
        checkpoint = output_dir / "checkpoints" / "shape_flow_steps.npz"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        _write_valid_shape_flow_steps_checkpoint(checkpoint)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("scripts.run_mlx_stage_capture.subprocess.run", write_complete)

    result = main(
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--stop-after-stage",
            "shape_flow_steps",
            "--steps",
            "3",
            "--shape-flow-block-injection-trace",
            str(trace),
            "--shape-flow-block-injection-step-index",
            "1",
            "--shape-flow-block-injection-block-index",
            "7",
            "--shape-flow-block-injection-branch",
            "pos",
            "--shape-flow-block-injection-stage",
            "after_self",
        ]
    )

    assert result == 2
    report = json.loads((output_dir / "run_report.json").read_text())
    assert report["failure_phase"] == "validate_primary_output"
    assert "requested injection but checkpoint identity is empty" in report["error"]


def test_shape_flow_steps_rejects_mismatched_injection_identity(tmp_path):
    import hashlib
    import json
    import numpy as np
    import pytest

    from scripts.run_mlx_stage_capture import _validate_shape_flow_steps_checkpoint

    trace = tmp_path / "injection-trace.npz"
    trace.write_bytes(b"trace")
    checkpoint = tmp_path / "shape_flow_steps.npz"
    _write_valid_shape_flow_steps_checkpoint(checkpoint)
    identity = {
        "trace_path": str(trace),
        "trace_sha256": "wrong",
        "array_key": "pos_block7_after_self",
        "branch": "pos",
        "step_index": 1,
        "block_index": 7,
        "stage": "after_self",
        "source_delta_scale": 1.0,
        "route_identity_evidence": True,
    }
    _rewrite_npz_array(
        checkpoint,
        "shape_flow_block_injection_json",
        np.array(json.dumps(identity, sort_keys=True)),
    )
    expected_route = {
        "shape_flow_block_injection_trace_path": str(trace),
        "shape_flow_block_injection_trace_sha256": hashlib.sha256(b"trace").hexdigest(),
        "shape_flow_block_injection_manifest_path": None,
        "shape_flow_block_injection_manifest_sha256": None,
        "shape_flow_block_injection_step_index": 1,
        "shape_flow_block_injection_block_index": 7,
        "shape_flow_block_injection_branch": "pos",
        "shape_flow_block_injection_stage": "after_self",
        "shape_flow_block_injection_array_key": None,
        "shape_flow_block_injection_scale": 1.0,
    }

    with pytest.raises(ValueError, match="trace_sha256"):
        _validate_shape_flow_steps_checkpoint(
            checkpoint,
            expected_steps=3,
            expected_route=expected_route,
        )


def test_shape_flow_steps_binds_matching_trace_injection_identity(tmp_path):
    import hashlib
    import json
    import numpy as np

    from scripts.run_mlx_stage_capture import _validate_shape_flow_steps_checkpoint

    trace = tmp_path / "injection-trace.npz"
    trace.write_bytes(b"trace")
    trace_sha = hashlib.sha256(b"trace").hexdigest()
    checkpoint = tmp_path / "shape_flow_steps.npz"
    _write_valid_shape_flow_steps_checkpoint(checkpoint)
    identity = {
        "trace_path": str(trace),
        "trace_sha256": trace_sha,
        "array_key": "pos_block7_after_self",
        "branch": "pos",
        "step_index": 1,
        "block_index": 7,
        "stage": "after_self",
        "source_delta_scale": 1.0,
        "route_identity_evidence": True,
    }
    _rewrite_npz_array(
        checkpoint,
        "shape_flow_block_injection_json",
        np.array(json.dumps(identity, sort_keys=True)),
    )
    expected_route = {
        "shape_flow_block_injection_trace_path": str(trace),
        "shape_flow_block_injection_trace_sha256": trace_sha,
        "shape_flow_block_injection_manifest_path": None,
        "shape_flow_block_injection_manifest_sha256": None,
        "shape_flow_block_injection_step_index": 1,
        "shape_flow_block_injection_block_index": 7,
        "shape_flow_block_injection_branch": "pos",
        "shape_flow_block_injection_stage": "after_self",
        "shape_flow_block_injection_array_key": None,
        "shape_flow_block_injection_scale": 1.0,
    }

    validation = _validate_shape_flow_steps_checkpoint(
        checkpoint,
        expected_steps=3,
        expected_route=expected_route,
    )

    assert validation["sampler"]["shape_flow_block_injection_route"] == {
        "mode": "trace",
        "route_identity_match": True,
        "trace_path": str(trace),
        "trace_sha256": trace_sha,
        "array_key": "pos_block7_after_self",
        "branch": "pos",
        "step_index": 1,
        "block_index": 7,
        "stage": "after_self",
        "source_delta_scale": 1.0,
    }


def test_shape_flow_steps_binds_matching_manifest_injection_identity(tmp_path):
    import hashlib
    import json
    import numpy as np

    from scripts.run_mlx_stage_capture import _validate_shape_flow_steps_checkpoint

    from trellmlx.shape_block_injection import load_shape_block_injection_manifest

    trace = tmp_path / "trace.npz"
    np.savez(
        trace,
        pos_block7_after_self=np.zeros((1, 2, 4), dtype=np.float32),
        trace_block_index=np.array(7, dtype=np.int32),
        shape_flow_trace_step_index=np.array(1, dtype=np.int32),
        route_identity_json=np.array(
            json.dumps(
                {
                    "effective_route": "official-source-cuda",
                    "effective_device_type": "cuda",
                }
            )
        ),
    )
    manifest = tmp_path / "injection-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "test-manifest",
                "sites": [
                    {
                        "trace_path": trace.name,
                        "branch": "pos",
                        "step_index": 1,
                        "block_index": 7,
                        "stage": "after_self",
                    }
                ],
            }
        )
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    checkpoint = tmp_path / "shape_flow_steps.npz"
    _write_valid_shape_flow_steps_checkpoint(checkpoint)
    identity = load_shape_block_injection_manifest(manifest).report_identity()
    _rewrite_npz_array(
        checkpoint,
        "shape_flow_block_injection_json",
        np.array(json.dumps(identity, sort_keys=True)),
    )
    expected_route = {
        "shape_flow_block_injection_trace_path": None,
        "shape_flow_block_injection_trace_sha256": None,
        "shape_flow_block_injection_manifest_path": str(manifest),
        "shape_flow_block_injection_manifest_sha256": manifest_sha,
    }

    validation = _validate_shape_flow_steps_checkpoint(
        checkpoint,
        expected_steps=3,
        expected_route=expected_route,
    )

    assert validation["sampler"]["shape_flow_block_injection_route"] == {
        "mode": "manifest",
        "route_identity_match": True,
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha,
        "site_count": 1,
    }


def test_shape_flow_steps_rejects_truncated_manifest_site_identity(tmp_path):
    import hashlib
    import json
    import numpy as np
    import pytest

    from scripts.run_mlx_stage_capture import _validate_shape_flow_steps_checkpoint

    trace = tmp_path / "trace.npz"
    np.savez(
        trace,
        pos_block7_after_self=np.zeros((1, 2, 4), dtype=np.float32),
        trace_block_index=np.array(7, dtype=np.int32),
        shape_flow_trace_step_index=np.array(1, dtype=np.int32),
        route_identity_json=np.array(
            json.dumps(
                {
                    "effective_route": "official-source-cuda",
                    "effective_device_type": "cuda",
                }
            )
        ),
    )
    manifest = tmp_path / "injection-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sites": [
                    {
                        "trace_path": trace.name,
                        "branch": "pos",
                        "step_index": 1,
                        "block_index": 7,
                        "stage": "after_self",
                    }
                ]
            }
        )
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    checkpoint = tmp_path / "shape_flow_steps.npz"
    _write_valid_shape_flow_steps_checkpoint(checkpoint)
    truncated_identity = {
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha,
        "route_identity_evidence": True,
        "sites": [{"route_identity_evidence": True}],
    }
    _rewrite_npz_array(
        checkpoint,
        "shape_flow_block_injection_json",
        np.array(json.dumps(truncated_identity, sort_keys=True)),
    )
    expected_route = {
        "shape_flow_block_injection_trace_path": None,
        "shape_flow_block_injection_trace_sha256": None,
        "shape_flow_block_injection_manifest_path": str(manifest),
        "shape_flow_block_injection_manifest_sha256": manifest_sha,
    }

    with pytest.raises(ValueError, match="effective identity does not match requested manifest"):
        _validate_shape_flow_steps_checkpoint(
            checkpoint,
            expected_steps=3,
            expected_route=expected_route,
        )


def test_stage_capture_wrapper_exposes_shape_flow_block_trace_route(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser

    support_path = tmp_path / "reference_shape_slat.npz"
    support_path.write_bytes(b"shape-slat-support")
    noise_path = tmp_path / "shape_flow_step.npz"
    noise_path.write_bytes(b"shape-flow-noise")
    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "shape_flow_block_trace",
            "--no-cascade",
            "--shape-slat-support-sample",
            str(support_path),
            "--shape-flow-noise-sample",
            str(noise_path),
            "--shape-flow-trace-block-index",
            "7",
            "--shape-flow-trace-step-index",
            "3",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")

    assert command[command.index("--stop-after-stage") + 1] == "shape_flow_block_trace"
    assert "--shape-flow-trace-block-index" in command
    assert command[command.index("--shape-flow-trace-block-index") + 1] == "7"
    assert "--shape-flow-trace-step-index" in command
    assert command[command.index("--shape-flow-trace-step-index") + 1] == "3"
    assert "--shape-flow-noise-sample" in command
    assert command[command.index("--shape-flow-noise-sample") + 1] == str(noise_path)


def test_stage_capture_route_identity_records_shape_flow_trace_indices(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    support_path = tmp_path / "reference_shape_slat.npz"
    support_path.write_bytes(b"shape-slat-support")
    noise_path = tmp_path / "shape_flow_step.npz"
    noise_path.write_bytes(b"shape-flow-noise")
    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "shape_flow_block_trace",
            "--no-cascade",
            "--shape-slat-support-sample",
            str(support_path),
            "--shape-flow-noise-sample",
            str(noise_path),
            "--shape-flow-trace-block-index",
            "11",
            "--shape-flow-trace-step-index",
            "4",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert route_identity["route"]["shape_flow_trace_block_index"] == 11
    assert route_identity["route"]["shape_flow_trace_step_index"] == 4
    assert route_identity["route"]["shape_flow_noise_sample_path"] == str(noise_path)
    assert route_identity["route"]["shape_flow_noise_sample_sha256"] is not None


def test_stage_capture_wrapper_exposes_sparse_flow_block_trace_route(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "sparse_flow_block_trace",
            "--seed",
            "42",
            "--steps",
            "8",
            "--resolution",
            "512",
            "--shared-noise",
            "noise.npz",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")

    assert command[command.index("--stop-after-stage") + 1] == "sparse_flow_block_trace"


def test_stage_capture_wrapper_forwards_sparse_flow_trace_block_index(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "sparse_flow_block_trace",
            "--sparse-flow-trace-block-index",
            "5",
            "--seed",
            "42",
            "--steps",
            "8",
            "--resolution",
            "512",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert "--sparse-flow-trace-block-index" in command
    assert command[command.index("--sparse-flow-trace-block-index") + 1] == "5"
    assert route_identity["route"]["sparse_flow_trace_block_index"] == 5


def test_stage_capture_wrapper_forwards_sparse_flow_trace_step_index(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "sparse_flow_block_trace",
            "--seed",
            "42",
            "--steps",
            "8",
            "--resolution",
            "512",
            "--shared-noise",
            "noise.npz",
            "--sparse-flow-trace-step-index",
            "5",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert "--sparse-flow-trace-step-index" in command
    assert command[command.index("--sparse-flow-trace-step-index") + 1] == "5"
    assert route_identity["route"]["sparse_flow_trace_step_index"] == 5


def test_stage_capture_wrapper_forwards_sparse_flow_trace_sample(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    sample_path = tmp_path / "reference_sparse_flow_steps.npz"
    sample_path.write_bytes(b"sample")

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "sparse_flow_block_trace",
            "--seed",
            "42",
            "--steps",
            "8",
            "--resolution",
            "512",
            "--sparse-flow-trace-step-index",
            "5",
            "--sparse-flow-trace-sample",
            str(sample_path),
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert "--sparse-flow-trace-sample" in command
    assert command[command.index("--sparse-flow-trace-sample") + 1] == str(sample_path)
    assert route_identity["route"]["sparse_flow_trace_sample_path"] == str(sample_path)
    assert route_identity["route"]["sparse_flow_trace_sample_sha256"] is not None


def test_stage_capture_wrapper_forwards_sparse_flow_trace_sample_for_step_capture(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    sample_path = tmp_path / "reference_sparse_flow_steps.npz"
    sample_path.write_bytes(b"sample")

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "sparse_flow_step",
            "--seed",
            "42",
            "--steps",
            "8",
            "--resolution",
            "512",
            "--sparse-flow-trace-step-index",
            "5",
            "--sparse-flow-trace-sample",
            str(sample_path),
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert "--sparse-flow-trace-sample" in command
    assert command[command.index("--sparse-flow-trace-sample") + 1] == str(sample_path)
    assert command[command.index("--sparse-flow-trace-step-index") + 1] == "5"
    assert route_identity["route"]["sparse_flow_trace_sample_path"] == str(sample_path)


def test_stage_capture_wrapper_forwards_sparse_flow_start_sample_for_continuation(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    sample_path = tmp_path / "reference_sparse_flow_steps.npz"
    sample_path.write_bytes(b"sample")

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "sparse_flow_steps",
            "--seed",
            "42",
            "--steps",
            "8",
            "--resolution",
            "512",
            "--sparse-flow-start-step-index",
            "5",
            "--sparse-flow-start-sample",
            str(sample_path),
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert "--sparse-flow-start-sample" in command
    assert command[command.index("--sparse-flow-start-sample") + 1] == str(sample_path)
    assert command[command.index("--sparse-flow-start-step-index") + 1] == "5"
    assert route_identity["route"]["sparse_flow_start_sample_path"] == str(sample_path)
    assert route_identity["route"]["sparse_flow_start_sample_sha256"] is not None
    assert route_identity["route"]["sparse_flow_start_step_index"] == 5


def test_generate_sparse_flow_step_can_load_trace_sample():
    source = GENERATE_SOURCE.read_text()

    assert 'args.stop_after_stage == "sparse_flow_step" and args.sparse_flow_trace_sample' in source
    assert "step_sample_npz = np.load(args.sparse_flow_trace_sample)" in source
    assert "start_step_index=sparse_flow_start_step_index" in source
    assert 'sample_in=np.array(step_capture["sample_in"])' in source


def test_generate_sparse_flow_can_continue_from_start_sample():
    source = GENERATE_SOURCE.read_text()

    assert "--sparse-flow-start-sample" in source
    assert "--sparse-flow-start-step-index" in source
    assert "start_sample_npz = np.load(args.sparse_flow_start_sample)" in source
    assert "sparse_flow_start_step_index = args.sparse_flow_start_step_index" in source
    assert "start_step_index=sparse_flow_start_step_index" in source
    assert "sparse_flow_start_sample_path=np.array(sparse_flow_start_sample_path)" in source


def test_stage_capture_wrapper_forwards_sparse_flow_trace_block_input_sample(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    block_input_path = tmp_path / "reference_block8_trace.npz"
    block_input_path.write_bytes(b"block-input")

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "sparse_flow_block_trace",
            "--seed",
            "42",
            "--steps",
            "8",
            "--resolution",
            "512",
            "--sparse-flow-trace-step-index",
            "5",
            "--sparse-flow-trace-block-index",
            "8",
            "--sparse-flow-trace-block-input-sample",
            str(block_input_path),
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert "--sparse-flow-trace-block-input-sample" in command
    assert command[command.index("--sparse-flow-trace-block-input-sample") + 1] == str(block_input_path)
    assert route_identity["route"]["sparse_flow_trace_block_input_sample_path"] == str(block_input_path)
    assert route_identity["route"]["sparse_flow_trace_block_input_sample_sha256"] is not None


def test_stage_capture_wrapper_forwards_conditioning_sample(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    conditioning_path = tmp_path / "reference_conditioning.npz"
    conditioning_path.write_bytes(b"conditioning")

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "sparse_flow_steps",
            "--seed",
            "42",
            "--steps",
            "8",
            "--resolution",
            "512",
            "--conditioning-sample",
            str(conditioning_path),
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert "--conditioning-sample" in command
    assert command[command.index("--conditioning-sample") + 1] == str(conditioning_path)
    assert route_identity["route"]["conditioning_sample_path"] == str(conditioning_path)
    assert route_identity["route"]["conditioning_sample_sha256"] is not None


def test_stage_capture_wrapper_forwards_shape_slat_sample(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    shape_slat_path = tmp_path / "reference_shape_slat.npz"
    shape_slat_path.write_bytes(b"shape-slat")

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "decoder_output",
            "--shape-slat-sample",
            str(shape_slat_path),
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert "--shape-slat-sample" in command
    assert command[command.index("--shape-slat-sample") + 1] == str(shape_slat_path)
    assert route_identity["route"]["shape_slat_sample_path"] == str(shape_slat_path)
    assert route_identity["route"]["shape_slat_sample_sha256"] is not None


def test_stage_capture_wrapper_forwards_shape_slat_support_sample(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    support_path = tmp_path / "reference_shape_slat.npz"
    support_path.write_bytes(b"shape-slat-support")

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "shape_slat",
            "--no-cascade",
            "--shape-slat-support-sample",
            str(support_path),
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert "--shape-slat-support-sample" in command
    assert command[command.index("--shape-slat-support-sample") + 1] == str(support_path)
    assert route_identity["route"]["shape_slat_support_sample_path"] == str(support_path)
    assert route_identity["route"]["shape_slat_support_sample_sha256"] is not None


def test_generate_exposes_shape_slat_sample_decoder_replay():
    source = GENERATE_SOURCE.read_text()

    assert "--shape-slat-sample" in source
    assert "shape_slat_sample_npz = np.load(args.shape_slat_sample)" in source
    assert "--shape-slat-sample requires --stop-after-stage decoder_output" in source
    assert "Shape SLat replay" in source


def test_generate_exposes_shape_slat_support_replay():
    source = GENERATE_SOURCE.read_text()

    assert "--shape-slat-support-sample" in source
    assert "shape_slat_support_sample_npz = np.load(args.shape_slat_support_sample)" in source
    assert "--shape-slat-support-sample requires --no-cascade" in source
    assert "Shape SLat support replay" in source
    assert "lr_coords_4d = support_coords.astype(np.int32, copy=False)" in source


def test_generate_exposes_shape_flow_block_trace():
    source = GENERATE_SOURCE.read_text()

    assert '"shape_flow_block_trace"' in source
    assert "--shape-flow-trace-block-index" in source
    assert "--shape-flow-trace-step-index" in source
    assert "--shape-flow-noise-sample" in source
    assert "shape_flow_noise_sample_npz = np.load(args.shape_flow_noise_sample)" in source
    assert "shape flow noise sample coords do not exactly match" in source
    assert "lr_slat_flow.trace_block" in source
    assert 'save_checkpoint(\n            args.save_checkpoints,\n            "shape_flow_block_trace"' in source


def test_generate_shape_flow_block_trace_filters_selected_payload_keys():
    import generate
    import numpy as np
    import pytest

    payload = {
        "pos_block29_after_self": np.ones((1, 2, 3), dtype=np.float32),
        "neg_block29_after_self": np.zeros((1, 2, 3), dtype=np.float32),
    }

    selected = generate._parse_shape_flow_trace_keys(
        "pos_block29_after_self,neg_block29_after_self,pos_block29_after_self"
    )
    filtered = generate._filter_shape_flow_trace_payload(payload, selected)

    assert selected == ["pos_block29_after_self", "neg_block29_after_self"]
    assert list(filtered) == selected
    with pytest.raises(ValueError, match="shape-flow-trace-keys.*missing key.*not_present"):
        generate._filter_shape_flow_trace_payload(payload, ["not_present"])


def test_generate_shape_flow_full_trace_materializes_effective_key_order():
    import generate
    import numpy as np

    payload = {
        "pos_block29_after_self": np.ones((1, 2, 3), dtype=np.float32),
        "pos_final_output": np.zeros((1, 2, 3), dtype=np.float32),
        "neg_final_output": np.full((1, 2, 3), 2, dtype=np.float32),
    }

    filtered, effective = generate._select_shape_flow_trace_payload(payload, [])

    assert filtered is payload
    assert effective == list(payload)


def test_generate_sparse_flow_block_trace_can_target_sampler_step():
    source = GENERATE_SOURCE.read_text()

    assert "--sparse-flow-trace-step-index" in source
    assert "sparse_flow_trace_step_index=np.array(trace_step_index" in source
    assert "trace_step_index = args.sparse_flow_trace_step_index" in source


def test_generate_sparse_flow_block_trace_can_load_trace_sample():
    source = GENERATE_SOURCE.read_text()

    assert "--sparse-flow-trace-sample" in source
    assert "trace_sample_npz = np.load(args.sparse_flow_trace_sample)" in source
    assert "trace_sample = mx.array(trace_sample_np)" in source


def test_generate_sparse_flow_block_trace_can_replay_projected_block_input():
    source = GENERATE_SOURCE.read_text()

    assert "--sparse-flow-trace-block-input-sample" in source
    assert "block_input_npz = np.load(args.sparse_flow_trace_block_input_sample)" in source
    assert "trace_projected_block_input" in source
    assert 'f"pos_block{trace_block_index}_input"' in source
    assert 'f"neg_block{trace_block_index}_input"' in source


def test_generate_sparse_flow_block_trace_can_save_selected_trace_keys():
    source = GENERATE_SOURCE.read_text()

    assert "--sparse-flow-trace-keys" in source
    assert "selected_trace_keys = _parse_sparse_flow_trace_keys" in source
    assert "trace_payload = _filter_sparse_flow_trace_payload(" in source
    assert "sparse_flow_trace_selected_keys=np.array(selected_trace_keys, dtype=str)" in source


def test_generate_sparse_flow_block_trace_forwards_active_sparse_block_injection():
    source = GENERATE_SOURCE.read_text()
    trace_start = source.index('if args.stop_after_stage == "sparse_flow_block_trace":')
    trace_end = source.index('step_capture = {} if args.stop_after_stage == "sparse_flow_step"', trace_start)
    trace_source = source[trace_start:trace_end]

    assert "def active_trace_injection(branch):" in trace_source
    assert "active_for_step_branch" in trace_source
    assert 'sparse_block_injection=active_trace_injection("pos")' in trace_source
    assert 'sparse_block_injection_branch="pos"' in trace_source
    assert 'sparse_block_injection=active_trace_injection("neg")' in trace_source
    assert 'sparse_block_injection_branch="neg"' in trace_source


def test_generate_can_load_conditioning_sample():
    source = GENERATE_SOURCE.read_text()

    assert "--conditioning-sample" in source
    assert "conditioning_sample_npz = np.load(args.conditioning_sample)" in source
    assert 'cond = mx.array(conditioning_sample_npz["cond"])' in source
    assert 'neg_cond = mx.array(conditioning_sample_npz["neg_cond"])' in source


def test_generate_default_image_conditioning_binds_negative_conditioning():
    source = GENERATE_SOURCE.read_text()
    image_branch_start = source.index("elif args.image:")

    image_branch = source[
        image_branch_start:source.index('    else:\n        print("No image', image_branch_start)
    ]

    assert "neg_cond = mx.zeros_like(cond)" in image_branch


def test_stage_capture_wrapper_forwards_sparse_flow_trace_no_kv_cache(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "sparse_flow_block_trace",
            "--sparse-flow-trace-no-kv-cache",
            "--seed",
            "42",
            "--steps",
            "8",
            "--resolution",
            "512",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert "--sparse-flow-trace-no-kv-cache" in command
    assert route_identity["route"]["sparse_flow_trace_uses_kv_cache"] is False


def test_stage_capture_wrapper_forwards_sparse_flow_trace_keys(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "sparse_flow_block_trace",
            "--sparse-flow-trace-keys",
            "pos_block5_input,pos_block5_after_self,pos_block5_after_mlp",
            "--seed",
            "42",
            "--steps",
            "8",
            "--resolution",
            "512",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert "--sparse-flow-trace-keys" in command
    assert (
        command[command.index("--sparse-flow-trace-keys") + 1]
        == "pos_block5_input,pos_block5_after_self,pos_block5_after_mlp"
    )
    assert route_identity["route"]["sparse_flow_trace_keys"] == [
        "pos_block5_input",
        "pos_block5_after_self",
        "pos_block5_after_mlp",
    ]


def test_stage_capture_wrapper_forwards_shape_flow_trace_keys(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "shape_flow_block_trace",
            "--shape-flow-trace-keys",
            "pos_block29_after_self,neg_block29_after_self,pos_final_output,neg_final_output",
            "--seed",
            "42",
            "--steps",
            "8",
            "--resolution",
            "512",
            "--no-cascade",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert command[command.index("--shape-flow-trace-keys") + 1] == (
        "pos_block29_after_self,neg_block29_after_self,pos_final_output,neg_final_output"
    )
    assert route_identity["route"]["shape_flow_trace_keys"] == [
        "pos_block29_after_self",
        "neg_block29_after_self",
        "pos_final_output",
        "neg_final_output",
    ]
    assert route_identity["route"]["shape_flow_trace_key_selection"] == "explicit"
    assert route_identity["route"]["shape_flow_trace_requested_keys"] == route_identity["route"][
        "shape_flow_trace_keys"
    ]


def test_stage_capture_binds_omitted_full_shape_trace_keys_from_primary_output(tmp_path):
    import numpy as np
    import pytest

    from scripts.run_mlx_stage_capture import (
        _bind_effective_shape_flow_trace_keys,
        _build_generate_command,
        build_parser,
        build_route_identity,
    )

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "shape_flow_block_trace",
        ]
    )
    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)
    checkpoint = tmp_path / "checkpoints" / "shape_flow_block_trace.npz"
    checkpoint.parent.mkdir()
    effective = ["pos_block29_after_self", "pos_final_output", "neg_final_output"]
    np.savez(
        checkpoint,
        shape_flow_trace_selected_keys=np.asarray(effective),
        pos_block29_after_self=np.zeros((1, 2, 3), dtype=np.float32),
        pos_final_output=np.zeros((1, 2, 3), dtype=np.float32),
        neg_final_output=np.zeros((1, 2, 3), dtype=np.float32),
    )

    assert route_identity["route"]["shape_flow_trace_key_selection"] == "full"
    assert route_identity["route"]["shape_flow_trace_requested_keys"] == []
    assert route_identity["route"]["shape_flow_trace_keys"] is None

    _bind_effective_shape_flow_trace_keys(route_identity, checkpoint)

    assert route_identity["route"]["shape_flow_trace_keys"] == effective

    route_identity["route"]["shape_flow_trace_key_selection"] = "stale-default"
    with pytest.raises(ValueError, match="unsupported shape-flow trace key selection"):
        _bind_effective_shape_flow_trace_keys(route_identity, checkpoint)


def test_stage_capture_wrapper_forwards_sparse_block_injection_route(tmp_path):
    import numpy as np

    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    trace = tmp_path / "source_block0_trace.npz"
    np.savez(trace, pos_block0_norm1=np.zeros((1, 2, 1), dtype=np.float32))

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "sparse_flow_steps",
            "--sparse-flow-block-injection-trace",
            str(trace),
            "--sparse-flow-block-injection-step-index",
            "2",
            "--sparse-flow-block-injection-block-index",
            "0",
            "--sparse-flow-block-injection-branch",
            "both",
            "--sparse-flow-block-injection-stage",
            "norm1",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert "--sparse-flow-block-injection-trace" in command
    assert command[command.index("--sparse-flow-block-injection-trace") + 1] == str(trace)
    assert "--sparse-flow-block-injection-stage" in command
    assert command[command.index("--sparse-flow-block-injection-stage") + 1] == "norm1"
    assert route_identity["route"]["sparse_flow_block_injection_trace_path"] == str(trace)
    assert route_identity["route"]["sparse_flow_block_injection_trace_sha256"] is not None
    assert route_identity["route"]["sparse_flow_block_injection_step_index"] == 2
    assert route_identity["route"]["sparse_flow_block_injection_block_index"] == 0
    assert route_identity["route"]["sparse_flow_block_injection_branch"] == "both"
    assert route_identity["route"]["sparse_flow_block_injection_stage"] == "norm1"


def test_stage_capture_wrapper_forwards_layernorm_correction_route(tmp_path):
    from scripts.run_mlx_stage_capture import _build_generate_command, build_parser, build_route_identity

    report = tmp_path / "block0_norm1_boundary_probe.json"
    report.write_text('{"schema":"trellis2mlx.noaffine_layernorm_boundary_probe.v1"}')

    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path),
            "--stop-after-stage",
            "sparse_flow_steps",
            "--sparse-flow-layernorm-correction-report",
            str(report),
            "--sparse-flow-layernorm-correction-step-index",
            "2",
            "--sparse-flow-layernorm-correction-block-index",
            "0",
            "--sparse-flow-layernorm-correction-branch",
            "pos",
            "--sparse-flow-layernorm-correction-mode",
            "scale",
            "--sparse-flow-layernorm-correction-include",
            "improved",
        ]
    )

    command = _build_generate_command(args, tmp_path / "checkpoints")
    route_identity = build_route_identity(args, command)

    assert "--sparse-flow-layernorm-correction-report" in command
    assert command[command.index("--sparse-flow-layernorm-correction-report") + 1] == str(report)
    assert "--sparse-flow-layernorm-correction-mode" in command
    assert command[command.index("--sparse-flow-layernorm-correction-mode") + 1] == "scale"
    assert route_identity["route"]["sparse_flow_layernorm_correction_report_path"] == str(report)
    assert route_identity["route"]["sparse_flow_layernorm_correction_report_sha256"] is not None
    assert route_identity["route"]["sparse_flow_layernorm_correction_step_index"] == 2
    assert route_identity["route"]["sparse_flow_layernorm_correction_block_index"] == 0
    assert route_identity["route"]["sparse_flow_layernorm_correction_branch"] == "pos"
    assert route_identity["route"]["sparse_flow_layernorm_correction_mode"] == "scale"
    assert route_identity["route"]["sparse_flow_layernorm_correction_include"] == "improved"


def test_stage_capture_artifact_status_binds_path_size_and_digest(tmp_path):
    import hashlib

    from scripts.run_mlx_stage_capture import _artifact_status

    checkpoint = tmp_path / "shape_flow_step.npz"
    checkpoint.write_bytes(b"bound evidence")

    status = _artifact_status(tmp_path, "shape_flow_step")

    assert status == {
        "shape_flow_step.npz": {
            "path": str(checkpoint),
            "size_bytes": len(b"bound evidence"),
            "sha256": hashlib.sha256(b"bound evidence").hexdigest(),
        }
    }
