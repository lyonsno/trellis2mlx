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
