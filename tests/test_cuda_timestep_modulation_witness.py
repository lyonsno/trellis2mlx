import hashlib
import json
import os
import sys

import numpy as np
import pytest


STAGES = (
    "embedding",
    "linear0",
    "silu0",
    "linear1",
    "silu1",
    "modulation_float32",
    "modulation_bfloat16_bits",
)
PROJECTION_BATCH_MODE = "independent-singletons"


def _arrays(*, steps=3, width=4):
    values = {}
    for index, name in enumerate(STAGES):
        dtype = np.uint16 if name.endswith("_bits") else np.float32
        values[name] = np.full((steps, width), index, dtype=dtype)
    return values


def test_modulation_analysis_finds_first_stage_and_step_without_reordering():
    from scripts.cuda_timestep_modulation_witness import analyze_modulation

    candidate = _arrays()
    source = {name: value.copy() for name, value in candidate.items()}
    source["linear1"][1, 2] = np.float32(9)
    source["modulation_bfloat16_bits"][2, 0] = np.uint16(7)

    report = analyze_modulation(
        step_indices=np.asarray([0, 1, 2], dtype=np.int32),
        candidate_arrays=candidate,
        source_arrays=source,
    )

    assert report["first_float32_divergence"] == {
        "stage": "linear1",
        "step_index": 1,
    }
    assert report["first_bfloat16_modulation_divergence"] == {
        "stage": "modulation_bfloat16_bits",
        "step_index": 2,
    }
    assert report["stages"]["linear0"]["exact"] is True
    assert report["stages"]["linear1"]["per_step"][1]["nonzero"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"torch_version": "2.9.0+cu126"}, "Torch route"),
        ({"cuda_device": "NVIDIA A100-SXM4-40GB"}, "Tesla T4"),
    ],
)
def test_modulation_route_rejects_backend_substitution(mutation, message):
    from scripts.cuda_timestep_modulation_witness import validate_route

    with pytest.raises(ValueError, match=message):
        validate_route(
            torch_version=mutation.get("torch_version", "2.10.0+cu128"),
            cuda_device=mutation.get("cuda_device", "Tesla T4"),
        )


def test_modulation_analysis_rejects_nonfinite_or_malformed_arrays():
    from scripts.cuda_timestep_modulation_witness import analyze_modulation

    candidate = _arrays()
    source = {name: value.copy() for name, value in candidate.items()}
    source["silu0"][0, 0] = np.nan
    with pytest.raises(ValueError, match="silu0 contains non-finite"):
        analyze_modulation(
            step_indices=np.asarray([0, 1, 2], dtype=np.int32),
            candidate_arrays=candidate,
            source_arrays=source,
        )

    source = {name: value.copy() for name, value in candidate.items()}
    source["linear0"] = source["linear0"][:, :-1]
    with pytest.raises(ValueError, match="linear0 shape mismatch"):
        analyze_modulation(
            step_indices=np.asarray([0, 1, 2], dtype=np.int32),
            candidate_arrays=candidate,
            source_arrays=source,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            {"source_checkpoint_sha256": "f" * 64},
            "source checkpoint sha256 mismatch",
        ),
        (
            {"candidate_route": "trellis2mlx/substituted-route"},
            "candidate route mismatch",
        ),
        (
            {"candidate_projection_batch_mode": "batched-eight"},
            "projection batch mode mismatch",
        ),
    ],
)
def test_modulation_provenance_rejects_checkpoint_route_or_batch_substitution(
    mutation, message
):
    from scripts.cuda_timestep_modulation_witness import validate_provenance

    expected_checkpoint = "a" * 64
    expected_route = "trellis2mlx/source-shared-modulation"
    with pytest.raises(ValueError, match=message):
        validate_provenance(
            source_checkpoint_sha256=mutation.get(
                "source_checkpoint_sha256", expected_checkpoint
            ),
            expected_source_checkpoint_sha256=expected_checkpoint,
            candidate_route=mutation.get("candidate_route", expected_route),
            expected_candidate_route=expected_route,
            candidate_projection_batch_mode=mutation.get(
                "candidate_projection_batch_mode", PROJECTION_BATCH_MODE
            ),
            requested_projection_batch_mode=PROJECTION_BATCH_MODE,
        )


@pytest.mark.parametrize(
    ("mode", "expected_shapes"),
    [
        ("batched-eight", [(8,)]),
        ("independent-singletons", [(1,)] * 8),
    ],
)
def test_modulation_projection_batches_preserve_requested_gemm_shape(
    mode, expected_shapes
):
    from scripts.cuda_timestep_modulation_witness import _projection_batches

    timesteps = np.arange(8, dtype=np.float32)

    assert [
        batch.shape for batch in _projection_batches(timesteps, mode=mode)
    ] == expected_shapes


def test_modulation_candidate_requires_projection_batch_identity(tmp_path):
    from scripts.cuda_timestep_modulation_witness import _load_candidate

    candidate = tmp_path / "candidate.npz"
    np.savez(
        candidate,
        step_indices=np.arange(8, dtype=np.int32),
        timestep_float32=np.asarray(
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
        ).view(np.float32),
        candidate_route=np.asarray("trellis2mlx/source-shared-modulation"),
        **_arrays(steps=8),
    )

    with pytest.raises(
        ValueError, match="candidate missing required arrays.*projection_batch_mode"
    ):
        _load_candidate(candidate)


def test_modulation_primary_validation_rejects_partial_or_corrupted_output(
    tmp_path,
):
    from scripts.cuda_timestep_modulation_witness import (
        validate_written_primary,
    )

    step_indices = np.asarray([0, 1], dtype=np.int32)
    timesteps = np.asarray([1.0, 0.5], dtype=np.float32)
    source_arrays = _arrays(steps=2)
    output = tmp_path / "cuda_result.npz"
    np.savez(
        output,
        step_indices=step_indices,
        timestep_float32=timesteps,
        **{
            f"source_{name}": value
            for name, value in source_arrays.items()
            if name != "linear1"
        },
    )

    with pytest.raises(ValueError, match="primary missing required arrays"):
        validate_written_primary(
            output,
            step_indices=step_indices,
            timesteps=timesteps,
            source_arrays=source_arrays,
            projection_batch_mode=PROJECTION_BATCH_MODE,
        )


def test_modulation_primary_validation_rejects_batch_mode_substitution(tmp_path):
    from scripts.cuda_timestep_modulation_witness import (
        validate_written_primary,
    )

    step_indices = np.asarray([0, 1], dtype=np.int32)
    timesteps = np.asarray([1.0, 0.5], dtype=np.float32)
    source_arrays = _arrays(steps=2)
    output = tmp_path / "cuda_result.npz"
    np.savez(
        output,
        step_indices=step_indices,
        timestep_float32=timesteps,
        projection_batch_mode=np.asarray("batched-eight"),
        **{
            f"source_{name}": value
            for name, value in source_arrays.items()
        },
    )

    with pytest.raises(
        ValueError,
        match="primary array projection_batch_mode differs from authenticated memory",
    ):
        validate_written_primary(
            output,
            step_indices=step_indices,
            timesteps=timesteps,
            source_arrays=source_arrays,
            projection_batch_mode=PROJECTION_BATCH_MODE,
        )


@pytest.mark.parametrize("case", ["partial", "nine_rows", "wrong_timestep"])
def test_modulation_schedule_requires_exact_eight_canonical_coordinates(case):
    from scripts.cuda_timestep_modulation_witness import validate_schedule

    step_indices = np.arange(8, dtype=np.int32)
    timestep_bits = np.asarray(
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
    timesteps = timestep_bits.view(np.float32)
    if case == "partial":
        step_indices = step_indices[:3]
        timesteps = timesteps[:3]
    elif case == "nine_rows":
        step_indices = np.arange(9, dtype=np.int32)
        timesteps = np.concatenate(
            [timesteps, np.asarray([0.0], dtype=np.float32)]
        )
    else:
        timesteps = timesteps.copy()
        timesteps[5] = np.nextafter(
            timesteps[5], np.float32(np.inf), dtype=np.float32
        )

    with pytest.raises(ValueError, match="canonical eight-step schedule"):
        validate_schedule(step_indices=step_indices, timesteps=timesteps)


def test_modulation_main_preserves_input_when_primary_aliases_it(
    monkeypatch, tmp_path
):
    from scripts import cuda_timestep_modulation_witness as witness

    weights = tmp_path / "weights.npz"
    candidate = tmp_path / "candidate.npz"
    weights.write_bytes(b"protected-weights")
    candidate.write_bytes(b"protected-candidate")
    output_json = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuda_timestep_modulation_witness.py",
            "--weights",
            str(weights),
            "--expected-weights-sha256",
            hashlib.sha256(weights.read_bytes()).hexdigest(),
            "--expected-source-checkpoint-sha256",
            "a" * 64,
            "--candidate",
            str(candidate),
            "--expected-candidate-sha256",
            hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "--expected-candidate-route",
            "trellis2mlx/source-shared-modulation",
            "--projection-batch-mode",
            PROJECTION_BATCH_MODE,
            "--output-json",
            str(output_json),
            "--output-npz",
            str(weights),
        ],
    )

    assert witness.main() == 1
    report = json.loads(output_json.read_text())
    assert report["failure_phase"] == "request_validation"
    assert report["primary_output_status"] == "protected_input"
    assert "output NPZ aliases protected input" in report["error"]
    assert weights.read_bytes() == b"protected-weights"
    assert candidate.read_bytes() == b"protected-candidate"


def test_modulation_main_routes_hardlinked_failure_report_away_from_input(
    monkeypatch, tmp_path
):
    from scripts import cuda_timestep_modulation_witness as witness

    weights = tmp_path / "weights.npz"
    candidate = tmp_path / "candidate.npz"
    weights.write_bytes(b"protected-weights")
    candidate.write_bytes(b"protected-candidate")
    requested_json = tmp_path / "result.json"
    os.link(candidate, requested_json)
    output_npz = tmp_path / "result.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuda_timestep_modulation_witness.py",
            "--weights",
            str(weights),
            "--expected-weights-sha256",
            hashlib.sha256(weights.read_bytes()).hexdigest(),
            "--expected-source-checkpoint-sha256",
            "a" * 64,
            "--candidate",
            str(candidate),
            "--expected-candidate-sha256",
            hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "--expected-candidate-route",
            "trellis2mlx/source-shared-modulation",
            "--projection-batch-mode",
            PROJECTION_BATCH_MODE,
            "--output-json",
            str(requested_json),
            "--output-npz",
            str(output_npz),
        ],
    )

    assert witness.main() == 1
    fallback = tmp_path / "weights.npz.timestep-modulation-failure.json"
    report = json.loads(fallback.read_text())
    assert report["failure_phase"] == "request_validation"
    assert report["output_json_requested"] == str(requested_json)
    assert report["output_json_effective"] == str(fallback)
    assert "output JSON aliases protected input" in report["error"]
    assert candidate.read_bytes() == b"protected-candidate"
    assert requested_json.read_bytes() == b"protected-candidate"


def test_modulation_main_rejects_partial_schedule_before_torch_and_clears_primary(
    monkeypatch, tmp_path
):
    from scripts import cuda_timestep_modulation_witness as witness

    weights = tmp_path / "weights.npz"
    candidate = tmp_path / "candidate.npz"
    weights.write_bytes(b"digest-bound-weights")
    candidate_arrays = _arrays(steps=3)
    np.savez(
        candidate,
        step_indices=np.arange(3, dtype=np.int32),
        timestep_float32=np.asarray(
            [1000.0, 954.5455, 900.0], dtype=np.float32
        ),
        candidate_route=np.asarray(
            "trellis2mlx/source-shared-modulation"
        ),
        projection_batch_mode=np.asarray(PROJECTION_BATCH_MODE),
        **candidate_arrays,
    )
    monkeypatch.setattr(
        witness,
        "_load_weights",
        lambda path: ({}, "a" * 64),
    )
    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    output_npz.write_bytes(b"stale")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuda_timestep_modulation_witness.py",
            "--weights",
            str(weights),
            "--expected-weights-sha256",
            hashlib.sha256(weights.read_bytes()).hexdigest(),
            "--expected-source-checkpoint-sha256",
            "a" * 64,
            "--candidate",
            str(candidate),
            "--expected-candidate-sha256",
            hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "--expected-candidate-route",
            "trellis2mlx/source-shared-modulation",
            "--projection-batch-mode",
            PROJECTION_BATCH_MODE,
            "--output-json",
            str(output_json),
            "--output-npz",
            str(output_npz),
        ],
    )

    assert witness.main() == 1
    report = json.loads(output_json.read_text())
    assert report["failure_phase"] == "input_validation"
    assert report["last_trustworthy_phase"] == "request_validated"
    assert report["primary_output_status"] == "missing"
    assert "canonical eight-step schedule" in report["error"]
    assert "torch" not in sys.modules
    assert not output_npz.exists()


def test_modulation_main_rejects_substituted_input_and_removes_stale_primary(
    monkeypatch, tmp_path
):
    from scripts import cuda_timestep_modulation_witness as witness

    weights = tmp_path / "weights.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(weights, marker=np.asarray(1, dtype=np.int32))
    np.savez(candidate, marker=np.asarray(2, dtype=np.int32))
    output_json = tmp_path / "cuda_result.json"
    output_npz = tmp_path / "cuda_result.npz"
    output_npz.write_bytes(b"stale")
    candidate_digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuda_timestep_modulation_witness.py",
            "--weights",
            str(weights),
            "--expected-weights-sha256",
            "0" * 64,
            "--expected-source-checkpoint-sha256",
            "a" * 64,
            "--candidate",
            str(candidate),
            "--expected-candidate-sha256",
            candidate_digest,
            "--expected-candidate-route",
            "trellis2mlx/source-shared-modulation",
            "--projection-batch-mode",
            PROJECTION_BATCH_MODE,
            "--output-json",
            str(output_json),
            "--output-npz",
            str(output_npz),
        ],
    )

    assert witness.main() == 1
    report = json.loads(output_json.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "input_validation"
    assert report["primary_output_status"] == "missing"
    assert "weights sha256 mismatch" in report["error"]
    assert not output_npz.exists()
