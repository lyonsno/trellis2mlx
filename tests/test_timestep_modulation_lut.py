import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


TIMESTEP_BITS = np.asarray(
    [
        0x447A0000,
        0x446EA2E9,
        0x44610000,
        0x44505555,
        0x443B8000,
        0x4420B6DB,
        0x43FA0000,
        0x43960000,
    ],
    dtype=np.uint32,
)
SOURCE_CHECKPOINT_SHA256 = "e" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_pair(tmp_path: Path) -> tuple[Path, Path, str, str]:
    npz_path = tmp_path / "cuda_result.npz"
    bits = np.zeros((8, 9216), dtype=np.uint16)
    bits[5, 450] = np.uint16(0x3E9F)
    bits[5, 5072] = np.uint16(0x3C04)
    bits[5, 5160] = np.uint16(0x399D)
    bits[5, 5392] = np.uint16(0xBB42)
    np.savez(
        npz_path,
        step_indices=np.arange(8, dtype=np.int32),
        timestep_float32=TIMESTEP_BITS.view(np.float32),
        source_modulation_bfloat16_bits=bits,
    )
    npz_sha256 = _sha256(npz_path)
    report_path = tmp_path / "cuda_result.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.cuda_timestep_modulation_witness.v1",
                "status": "done",
                "effective_route": {
                    "device_type": "cuda",
                    "cuda_device": "Tesla T4",
                    "torch": "2.10.0+cu128",
                },
                "inputs": {
                    "source_checkpoint_sha256_effective": (
                        SOURCE_CHECKPOINT_SHA256
                    ),
                },
                "primary_output": {
                    "path": "cuda_result.npz",
                    "sha256": npz_sha256,
                    "size_bytes": npz_path.stat().st_size,
                },
                "schedule_identity": {
                    "step_indices_effective": list(range(8)),
                    "step_indices_expected": list(range(8)),
                    "timestep_float32_bits_effective": [
                        f"0x{int(value):08x}" for value in TIMESTEP_BITS
                    ],
                    "timestep_float32_bits_expected": [
                        f"0x{int(value):08x}" for value in TIMESTEP_BITS
                    ],
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    return npz_path, report_path, npz_sha256, _sha256(report_path)


def test_source_cuda_timestep_modulation_lut_loads_authenticated_canonical_table(
    tmp_path,
):
    from trellmlx.timestep_modulation_lut import (
        load_source_cuda_timestep_modulation_lut,
    )

    npz_path, report_path, npz_sha256, report_sha256 = _write_source_pair(
        tmp_path
    )
    lut = load_source_cuda_timestep_modulation_lut(
        npz_path=npz_path,
        report_path=report_path,
        expected_npz_sha256=npz_sha256,
        expected_report_sha256=report_sha256,
        expected_source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
    )

    step5 = lut.lookup_numpy(
        np.asarray([TIMESTEP_BITS.view(np.float32)[5]], dtype=np.float32)
    )
    assert step5.shape == (1, 9216)
    assert step5.dtype == np.float32
    assert step5[0, 450].view(np.uint32) == np.uint32(0x3E9F0000)
    assert step5[0, 5072].view(np.uint32) == np.uint32(0x3C040000)
    identity = lut.report_identity()
    assert identity["route_identity_evidence"] is True
    assert identity["npz_sha256_effective"] == npz_sha256
    assert identity["report_sha256_effective"] == report_sha256
    assert (
        identity["source_checkpoint_sha256_effective"]
        == SOURCE_CHECKPOINT_SHA256
    )


def test_source_cuda_timestep_modulation_lut_rejects_substituted_primary(
    tmp_path,
):
    from trellmlx.timestep_modulation_lut import (
        load_source_cuda_timestep_modulation_lut,
    )

    npz_path, report_path, npz_sha256, report_sha256 = _write_source_pair(
        tmp_path
    )
    with np.load(npz_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    arrays["source_modulation_bfloat16_bits"] = arrays[
        "source_modulation_bfloat16_bits"
    ].copy()
    arrays["source_modulation_bfloat16_bits"][5, 450] ^= np.uint16(1)
    np.savez(npz_path, **arrays)

    with pytest.raises(ValueError, match="NPZ SHA256 mismatch"):
        load_source_cuda_timestep_modulation_lut(
            npz_path=npz_path,
            report_path=report_path,
            expected_npz_sha256=npz_sha256,
            expected_report_sha256=report_sha256,
            expected_source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
        )


def test_source_cuda_timestep_modulation_lut_rejects_noncanonical_schedule(
    tmp_path,
):
    from trellmlx.timestep_modulation_lut import (
        load_source_cuda_timestep_modulation_lut,
    )

    npz_path, report_path, _, _ = _write_source_pair(tmp_path)
    with np.load(npz_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    arrays["timestep_float32"] = arrays["timestep_float32"].copy()
    arrays["timestep_float32"][5] = np.nextafter(
        arrays["timestep_float32"][5],
        np.float32(np.inf),
        dtype=np.float32,
    )
    np.savez(npz_path, **arrays)
    npz_sha256 = _sha256(npz_path)
    report = json.loads(report_path.read_text())
    report["primary_output"]["sha256"] = npz_sha256
    report["primary_output"]["size_bytes"] = npz_path.stat().st_size
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="canonical eight-step schedule"):
        load_source_cuda_timestep_modulation_lut(
            npz_path=npz_path,
            report_path=report_path,
            expected_npz_sha256=npz_sha256,
            expected_report_sha256=_sha256(report_path),
            expected_source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
        )


def test_shape_shared_modulation_uses_lut_without_falling_back(tmp_path):
    import mlx.core as mx

    from trellmlx.models.slat_flow import _shape_shared_modulation
    from trellmlx.timestep_modulation_lut import (
        load_source_cuda_timestep_modulation_lut,
    )

    npz_path, report_path, npz_sha256, report_sha256 = _write_source_pair(
        tmp_path
    )
    lut = load_source_cuda_timestep_modulation_lut(
        npz_path=npz_path,
        report_path=report_path,
        expected_npz_sha256=npz_sha256,
        expected_report_sha256=report_sha256,
        expected_source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,
    )
    timestep = mx.array(
        [float(TIMESTEP_BITS.view(np.float32)[5])],
        dtype=mx.float32,
    )

    modulation = _shape_shared_modulation(
        timestep,
        t_embedder=None,
        adaLN_modulation=None,
        compute_dtype=mx.bfloat16,
        shape_timestep_modulation_lut=lut,
    )
    mx.eval(modulation)
    modulation_bits = (
        np.asarray(modulation.astype(mx.float32))
        .view(np.uint32)
        .reshape(1, 9216)
        >> np.uint32(16)
    ).astype(np.uint16)

    assert modulation.dtype == mx.bfloat16
    assert modulation_bits[0, 450] == np.uint16(0x3E9F)
    assert modulation_bits[0, 5072] == np.uint16(0x3C04)


def test_stage_capture_threads_authenticated_timestep_modulation_route(
    tmp_path,
):
    from scripts.run_mlx_stage_capture import (
        _build_generate_command,
        build_parser,
        build_route_identity,
    )

    npz_path, report_path, npz_sha256, report_sha256 = _write_source_pair(
        tmp_path
    )
    args = build_parser().parse_args(
        [
            "--image",
            "input.png",
            "--output-dir",
            str(tmp_path / "output"),
            "--stop-after-stage",
            "shape_flow_steps",
            "--no-cascade",
            "--shape-timestep-modulation-lut",
            str(npz_path),
            "--shape-timestep-modulation-report",
            str(report_path),
            "--expected-shape-timestep-modulation-lut-sha256",
            npz_sha256,
            "--expected-shape-timestep-modulation-report-sha256",
            report_sha256,
            "--expected-shape-timestep-modulation-source-checkpoint-sha256",
            SOURCE_CHECKPOINT_SHA256,
        ]
    )
    command = _build_generate_command(args, tmp_path / "checkpoints")
    route = build_route_identity(
        args,
        command,
        repo_identity={
            "commit_requested": None,
            "commit_effective": "a" * 40,
            "dirty": True,
            "status_porcelain": " M generate.py\n",
        },
    )["route"]

    for flag, expected in (
        ("--shape-timestep-modulation-lut", str(npz_path)),
        ("--shape-timestep-modulation-report", str(report_path)),
        (
            "--expected-shape-timestep-modulation-lut-sha256",
            npz_sha256,
        ),
        (
            "--expected-shape-timestep-modulation-report-sha256",
            report_sha256,
        ),
        (
            "--expected-shape-timestep-modulation-source-checkpoint-sha256",
            SOURCE_CHECKPOINT_SHA256,
        ),
    ):
        assert command[command.index(flag) + 1] == expected
    assert route["shape_timestep_modulation_lut_path"] == str(npz_path)
    assert route["shape_timestep_modulation_lut_sha256_effective"] == npz_sha256
    assert route["shape_timestep_modulation_report_path"] == str(report_path)
    assert (
        route["shape_timestep_modulation_report_sha256_effective"]
        == report_sha256
    )


def test_stage_capture_binds_effective_timestep_modulation_identity(
    tmp_path,
):
    from scripts.run_mlx_stage_capture import (
        _bind_effective_shape_timestep_modulation_identity,
    )

    npz_path, report_path, npz_sha256, report_sha256 = _write_source_pair(
        tmp_path
    )
    identity = {
        "schema": "trellis2mlx.source_cuda_timestep_modulation_lut.v1",
        "route_identity_evidence": True,
        "route": "source-cuda-t4-canonical-shared-adaln-lut",
        "npz_path": str(npz_path),
        "npz_sha256_effective": npz_sha256,
        "report_path": str(report_path),
        "report_sha256_effective": report_sha256,
        "source_checkpoint_sha256_effective": SOURCE_CHECKPOINT_SHA256,
        "step_indices": list(range(8)),
        "timestep_float32_bits": [
            f"0x{int(value):08x}" for value in TIMESTEP_BITS
        ],
        "modulation_shape": [8, 9216],
    }
    route_identity = {
        "route": {"shape_timestep_modulation_identity": identity}
    }
    checkpoint = tmp_path / "shape_flow_step.npz"
    np.savez(
        checkpoint,
        shape_timestep_modulation_lut_json=np.asarray(
            json.dumps(identity, sort_keys=True)
        ),
    )

    binding = _bind_effective_shape_timestep_modulation_identity(
        route_identity,
        checkpoint,
    )

    assert binding["shape_timestep_modulation_route"] == identity
    assert binding["sha256"] == _sha256(checkpoint)
    assert (
        route_identity["route"][
            "shape_timestep_modulation_identity_effective"
        ]
        == identity
    )


@pytest.mark.parametrize("checkpoint_value", [None, "", '{"counterfeit":true}'])
def test_stage_capture_rejects_missing_or_substituted_effective_modulation(
    tmp_path,
    checkpoint_value,
):
    from scripts.run_mlx_stage_capture import (
        _bind_effective_shape_timestep_modulation_identity,
    )

    identity = {
        "route_identity_evidence": True,
        "npz_sha256_effective": "a" * 64,
        "report_sha256_effective": "b" * 64,
        "source_checkpoint_sha256_effective": "c" * 64,
    }
    route_identity = {
        "route": {"shape_timestep_modulation_identity": identity}
    }
    checkpoint = tmp_path / "shape_flow_block_trace.npz"
    payload = {}
    if checkpoint_value is not None:
        payload["shape_timestep_modulation_lut_json"] = np.asarray(
            checkpoint_value
        )
    np.savez(checkpoint, **payload)

    with pytest.raises(
        ValueError,
        match="timestep modulation identity",
    ):
        _bind_effective_shape_timestep_modulation_identity(
            route_identity,
            checkpoint,
        )


@pytest.mark.parametrize(
    ("stop_after_stage", "no_cascade", "message"),
    [
        ("shape_flow_steps", False, "requires --no-cascade"),
        ("shape_slat", True, "only valid for shape-flow diagnostic stops"),
    ],
)
def test_stage_capture_rejects_timestep_modulation_outside_bounded_route(
    tmp_path,
    stop_after_stage,
    no_cascade,
    message,
):
    from scripts.run_mlx_stage_capture import (
        _validate_shape_timestep_modulation_route_args,
        build_parser,
    )

    npz_path, report_path, npz_sha256, report_sha256 = _write_source_pair(
        tmp_path
    )
    argv = [
        "--image",
        "input.png",
        "--output-dir",
        str(tmp_path / "output"),
        "--stop-after-stage",
        stop_after_stage,
        "--shape-timestep-modulation-lut",
        str(npz_path),
        "--shape-timestep-modulation-report",
        str(report_path),
        "--expected-shape-timestep-modulation-lut-sha256",
        npz_sha256,
        "--expected-shape-timestep-modulation-report-sha256",
        report_sha256,
        "--expected-shape-timestep-modulation-source-checkpoint-sha256",
        SOURCE_CHECKPOINT_SHA256,
    ]
    if no_cascade:
        argv.append("--no-cascade")
    args = build_parser().parse_args(argv)

    with pytest.raises(ValueError, match=message):
        _validate_shape_timestep_modulation_route_args(args)
