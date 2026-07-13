import json
import hashlib
from pathlib import Path

import numpy as np
import pytest


def _guided_pred(sample: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    sigma_min = 1e-5
    t = 1.0
    cfg = 7.5 * pos + (1 - 7.5) * neg
    x0_pos = (1 - sigma_min) * sample - (sigma_min + (1 - sigma_min) * t) * pos
    x0_cfg = (1 - sigma_min) * sample - (sigma_min + (1 - sigma_min) * t) * cfg
    ratio = np.std(x0_pos) / np.std(x0_cfg)
    x0 = 0.5 * (x0_cfg * ratio) + 0.5 * x0_cfg
    return ((1 - sigma_min) * sample - x0) / (sigma_min + (1 - sigma_min) * t)


def _write_trace(
    path: Path,
    *,
    pos: np.ndarray,
    neg: np.ndarray,
    coords: np.ndarray,
    manifest_class: str | None,
    t: float = 1000.0,
    block_input: np.ndarray | None = None,
    stage: str = "after_self",
    injection_mode: str = "valid",
    steps: int = 4,
    rescale_t: float = 1.0,
    block29_trace_sha256: str = "c" * 64,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    route_vector = {
        "effective_device_type": "cuda",
        "effective_route": "official-trellis2-source-cuda-shape-flow-block-trace",
        "conditioning_sha256": "1" * 64,
        "shape_flow_noise_sample_sha256": "2" * 64,
        "shape_slat_support_sample_sha256": "3" * 64,
        "source_tar_sha256": "4" * 64,
        "steps": steps,
    }
    prefix_site = {
        "route_identity_evidence": True,
        "comparison_class": "mlx_shape_flow_with_source_cuda_block_stage_injection",
        "block_index": 28,
        "step_index": 0,
        "stage": "after_mlp",
        "branch": "both",
        "source_delta_scale": 1.0,
        "trace_sha256": "a" * 64,
        "trace_identity": route_vector,
    }
    injection = prefix_site
    if manifest_class is not None:
        injection = {
            "route_identity_evidence": True,
            "manifest_identity": {
                "schema": "trellis2mlx.shape_block_injection_manifest.v1",
                "comparison_class": manifest_class,
            },
            "manifest_sha256": "b" * 64,
            "sites": [
                prefix_site,
                {
                    "route_identity_evidence": True,
                    "block_index": 29,
                    "step_index": 0,
                    "stage": stage,
                    "branch": "both",
                    "source_delta_scale": 1.0,
                    "trace_sha256": block29_trace_sha256,
                    "trace_identity": route_vector,
                },
            ],
        }
    if injection_mode == "none":
        injection = None
    elif injection_mode == "wrong_site" and isinstance(injection, dict):
        injection["sites"][-1].update(block_index=7, step_index=3, stage="after_mlp", branch="neg")
    if block_input is None:
        block_input = np.zeros_like(pos)
    arrays = {
        "coords": coords,
        "pos_block29_input": block_input[None],
        "neg_block29_input": block_input[None],
        "pos_final_output": pos[None],
        "neg_final_output": neg[None],
        "trace_block_index": np.asarray(29, dtype=np.int32),
        "shape_flow_trace_step_index": np.asarray(0, dtype=np.int32),
        "guidance_strength": np.asarray(7.5, dtype=np.float32),
        "guidance_rescale": np.asarray(0.5, dtype=np.float32),
        "t": np.asarray(t, dtype=np.float32),
        "t_prev": np.asarray(0.75, dtype=np.float32),
        "steps": np.asarray(steps, dtype=np.int32),
        "rescale_t": np.asarray(rescale_t, dtype=np.float32),
        "guidance_interval": np.asarray([0.6, 1.0], dtype=np.float32),
    }
    if injection is not None:
        arrays["shape_flow_block_injection_json"] = np.asarray(json.dumps(injection))
    np.savez(path, **arrays)
    return path


def test_summary_reconstructs_guided_endpoint_and_preserves_manifest_identity(tmp_path: Path) -> None:
    from scripts.summarize_shape_block_operation_replays import CandidateSpec, summarize_replays

    coords = np.asarray([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    sample = np.asarray([[0.4, -0.2], [0.1, 0.7]], dtype=np.float32)
    source_pos = np.asarray([[0.2, -0.1], [0.3, 0.4]], dtype=np.float32)
    source_neg = np.asarray([[0.1, -0.3], [0.2, 0.5]], dtype=np.float32)
    source_pred = _guided_pred(sample, source_pos, source_neg)

    source_trace = _write_trace(
        tmp_path / "source.npz",
        pos=source_pos,
        neg=source_neg,
        coords=coords,
        manifest_class=None,
    )
    source_step = tmp_path / "source-step.npz"
    np.savez(
        source_step,
        coords=coords,
        sample_feats=sample,
        pred_final=source_pred,
        sample_next=sample - 0.25 * source_pred,
        t=np.asarray(1.0, dtype=np.float32),
        t_prev=np.asarray(0.75, dtype=np.float32),
        steps=np.asarray(4, dtype=np.int32),
        rescale_t=np.asarray(1.0, dtype=np.float32),
        guidance_interval=np.asarray([0.6, 1.0], dtype=np.float32),
    )
    natural = _write_trace(
        tmp_path / "natural.npz",
        pos=source_pos + 0.02,
        neg=source_neg - 0.01,
        coords=coords,
        manifest_class=None,
    )
    replay = _write_trace(
        tmp_path / "after-self.npz",
        pos=source_pos + 0.001,
        neg=source_neg,
        coords=coords,
        manifest_class="exact_source_cuda_prefix28_plus_block29_after_self",
        block29_trace_sha256=hashlib.sha256(source_trace.read_bytes()).hexdigest(),
    )

    report = summarize_replays(
        source_trace_path=source_trace,
        source_step_path=source_step,
        candidates=[
            CandidateSpec("natural", natural, None),
            CandidateSpec(
                "after_self",
                replay,
                "exact_source_cuda_prefix28_plus_block29_after_self",
            ),
        ],
    )

    assert report["schema"] == "trellis2mlx.shape_block_operation_replays.v1"
    assert report["source_reconstruction"]["pred_final_max_abs"] < 1e-6
    assert [row["name"] for row in report["replay_rows"]] == ["natural", "after_self", "source"]
    assert report["replay_rows"][1]["manifest_identity"]["comparison_class"] == (
        "exact_source_cuda_prefix28_plus_block29_after_self"
    )
    assert report["replay_rows"][1]["pred_final_source_mean_abs"] < (
        report["replay_rows"][0]["pred_final_source_mean_abs"]
    )
    assert report["replay_rows"][-1]["pred_final_source_mean_abs"] == 0.0


def test_source_route_vector_accepts_byte_encoded_json(tmp_path: Path) -> None:
    from scripts.summarize_shape_block_operation_replays import _source_route_vector

    route_vector = {
        "effective_device_type": "cuda",
        "effective_route": "official-trellis2-source-cuda-shape-flow-block-trace",
        "conditioning_sha256": "1" * 64,
        "shape_flow_noise_sample_sha256": "2" * 64,
        "shape_slat_support_sample_sha256": "3" * 64,
        "source_tar_sha256": "4" * 64,
        "steps": 4,
    }
    trace_path = tmp_path / "byte-route.npz"
    np.savez(trace_path, route_identity_json=np.asarray(json.dumps(route_vector).encode("utf-8")))

    with np.load(trace_path, allow_pickle=False) as trace:
        assert _source_route_vector(trace) == route_vector


def test_summary_rejects_wrong_coords_and_manifest_fallback(tmp_path: Path) -> None:
    from scripts.summarize_shape_block_operation_replays import (
        CandidateSpec,
        ReplayContractError,
        summarize_replays,
    )

    coords = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    values = np.asarray([[0.2, 0.4]], dtype=np.float32)
    source = _write_trace(
        tmp_path / "source.npz",
        pos=values,
        neg=values * 0.5,
        coords=coords,
        manifest_class=None,
    )
    step = tmp_path / "step.npz"
    sample = np.asarray([[0.1, 0.3]], dtype=np.float32)
    np.savez(
        step,
        coords=coords,
        sample_feats=sample,
        pred_final=_guided_pred(sample, values, values * 0.5),
        sample_next=sample - 0.25 * _guided_pred(sample, values, values * 0.5),
        t=np.asarray(1.0, dtype=np.float32),
        t_prev=np.asarray(0.75, dtype=np.float32),
        steps=np.asarray(4, dtype=np.int32),
        rescale_t=np.asarray(1.0, dtype=np.float32),
        guidance_interval=np.asarray([0.6, 1.0], dtype=np.float32),
    )
    candidate = _write_trace(
        tmp_path / "candidate.npz",
        pos=values,
        neg=values * 0.5,
        coords=np.asarray([[0, 9, 9, 9]], dtype=np.int32),
        manifest_class=None,
    )

    with pytest.raises(ReplayContractError, match="coords"):
        summarize_replays(
            source_trace_path=source,
            source_step_path=step,
            candidates=[
                CandidateSpec(
                    "after_self",
                    candidate,
                    "exact_source_prefix28_plus_block29_after_self",
                )
            ],
        )


def test_direct_site_intervention_identity_is_preserved() -> None:
    from scripts.summarize_shape_block_operation_replays import _manifest_identity

    evidence = {
        "route_identity_evidence": True,
        "comparison_class": "mlx_shape_flow_with_source_cuda_block_stage_injection",
        "block_index": 28,
        "step_index": 0,
        "stage": "after_mlp",
        "branch": "both",
        "trace_sha256": "a" * 64,
    }
    identity = _manifest_identity(
        {"shape_flow_block_injection_json": np.asarray(json.dumps(evidence))}
    )

    assert identity == {
        "schema": "trellis2mlx.shape_block_direct_injection.v1",
        "comparison_class": "mlx_shape_flow_with_source_cuda_block_stage_injection",
        "block_index": 28,
        "step_index": 0,
        "stage": "after_mlp",
        "branch": "both",
        "trace_sha256": "a" * 64,
    }


def _write_source_pair(tmp_path: Path) -> tuple[Path, Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    sample = np.asarray([[0.4, -0.2], [0.1, 0.7]], dtype=np.float32)
    source_pos = np.asarray([[0.2, -0.1], [0.3, 0.4]], dtype=np.float32)
    source_neg = np.asarray([[0.1, -0.3], [0.2, 0.5]], dtype=np.float32)
    source_trace = _write_trace(
        tmp_path / "source.npz", pos=source_pos, neg=source_neg, coords=coords,
        manifest_class=None, block_input=sample,
    )
    source_pred = _guided_pred(sample, source_pos, source_neg)
    source_step = tmp_path / "source-step.npz"
    np.savez(
        source_step, coords=coords, sample_feats=sample, pred_final=source_pred,
        sample_next=sample - 0.25 * source_pred, t=np.asarray(1.0), t_prev=np.asarray(0.75),
        steps=np.asarray(4), rescale_t=np.asarray(1.0),
        guidance_interval=np.asarray([0.6, 1.0]),
    )
    return source_trace, source_step, coords, sample, source_pos, source_neg


def test_summary_rejects_missing_or_wrong_intervention_sites_and_common_input(tmp_path: Path) -> None:
    from scripts.summarize_shape_block_operation_replays import CandidateSpec, ReplayContractError, summarize_replays

    source, step, coords, sample, pos, neg = _write_source_pair(tmp_path)
    cases = [
        (
            "missing intervention evidence",
            _write_trace(tmp_path / "missing.npz", pos=pos, neg=neg, coords=coords,
                         manifest_class=None, block_input=sample, injection_mode="none"),
            None,
        ),
        (
            "block29 intervention site",
            _write_trace(tmp_path / "wrong-site.npz", pos=pos, neg=neg, coords=coords,
                         manifest_class="exact_source_cuda_prefix28_plus_block29_after_self",
                         block_input=sample, injection_mode="wrong_site"),
            "exact_source_cuda_prefix28_plus_block29_after_self",
        ),
        (
            "does not match source trace",
            _write_trace(tmp_path / "wrong-trace.npz", pos=pos, neg=neg, coords=coords,
                         manifest_class="exact_source_cuda_prefix28_plus_block29_after_self",
                         block_input=sample),
            "exact_source_cuda_prefix28_plus_block29_after_self",
        ),
        (
            "block29 input",
            _write_trace(tmp_path / "wrong-input.npz", pos=pos, neg=neg, coords=coords,
                         manifest_class=None, block_input=sample + 1),
            None,
        ),
    ]
    for expected_error, path, manifest_class in cases:
        with pytest.raises(ReplayContractError, match=expected_error):
            summarize_replays(
                source_trace_path=source,
                source_step_path=step,
                candidates=[CandidateSpec("candidate", path, manifest_class)],
            )


def test_summary_derives_intervention_depth_from_validated_stage_not_argument_order(tmp_path: Path) -> None:
    from scripts.summarize_shape_block_operation_replays import CandidateSpec, summarize_replays

    source, step, coords, sample, pos, neg = _write_source_pair(tmp_path)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    after_cross = _write_trace(
        tmp_path / "after-cross.npz", pos=pos, neg=neg, coords=coords,
        manifest_class="exact_source_cuda_prefix28_plus_block29_after_cross",
        block_input=sample, stage="after_cross",
        block29_trace_sha256=source_sha,
    )
    after_self = _write_trace(
        tmp_path / "after-self-depth.npz", pos=pos, neg=neg, coords=coords,
        manifest_class="exact_source_cuda_prefix28_plus_block29_after_self",
        block_input=sample, stage="after_self",
        block29_trace_sha256=source_sha,
    )
    report = summarize_replays(
        source_trace_path=source,
        source_step_path=step,
        candidates=[
            CandidateSpec("after_cross", after_cross, "exact_source_cuda_prefix28_plus_block29_after_cross"),
            CandidateSpec("after_self", after_self, "exact_source_cuda_prefix28_plus_block29_after_self"),
        ],
    )
    assert [(row["name"], row["intervention_depth"]) for row in report["replay_rows"]] == [
        ("after_self", 2), ("after_cross", 3), ("source", 5),
    ]


def test_summary_rejects_direct_model_time_schedule_mismatch_and_bad_source_euler(tmp_path: Path) -> None:
    from scripts.summarize_shape_block_operation_replays import CandidateSpec, ReplayContractError, summarize_replays

    source, step, coords, sample, pos, neg = _write_source_pair(tmp_path)
    wrong_source_t = _write_trace(
        tmp_path / "wrong-source-t.npz", pos=pos, neg=neg, coords=coords,
        manifest_class=None, block_input=sample, t=900.0,
    )
    source_time_candidate = _write_trace(
        tmp_path / "source-time-candidate.npz", pos=pos, neg=neg, coords=coords,
        manifest_class=None, block_input=sample,
    )
    with pytest.raises(ReplayContractError, match="source trace.*1000x"):
        summarize_replays(source_trace_path=wrong_source_t, source_step_path=step,
                          candidates=[CandidateSpec("candidate", source_time_candidate, None)])

    direct_t = _write_trace(tmp_path / "direct-t.npz", pos=pos, neg=neg, coords=coords,
                            manifest_class=None, block_input=sample, t=1.0)
    with pytest.raises(ReplayContractError, match="1000x"):
        summarize_replays(source_trace_path=source, source_step_path=step,
                          candidates=[CandidateSpec("direct_t", direct_t, None)])

    wrong_schedule = _write_trace(tmp_path / "wrong-schedule.npz", pos=pos, neg=neg, coords=coords,
                                  manifest_class=None, block_input=sample, steps=8)
    with pytest.raises(ReplayContractError, match="schedule"):
        summarize_replays(source_trace_path=source, source_step_path=step,
                          candidates=[CandidateSpec("wrong_schedule", wrong_schedule, None)])

    pred = _guided_pred(sample, pos, neg)
    np.savez(step, coords=coords, sample_feats=sample, pred_final=pred, sample_next=sample,
             t=np.asarray(1.0), t_prev=np.asarray(.75), steps=np.asarray(4),
             rescale_t=np.asarray(1.0), guidance_interval=np.asarray([.6, 1.0]))
    valid = _write_trace(tmp_path / "valid.npz", pos=pos, neg=neg, coords=coords,
                         manifest_class=None, block_input=sample)
    with pytest.raises(ReplayContractError, match="source sample_next"):
        summarize_replays(source_trace_path=source, source_step_path=step,
                          candidates=[CandidateSpec("valid", valid, None)])
