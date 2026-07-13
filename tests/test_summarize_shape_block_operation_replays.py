import json
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
    t: float = 1.0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    injection = None
    if manifest_class is not None:
        injection = {
            "route_identity_evidence": True,
            "manifest_identity": {
                "schema": "trellis2mlx.shape_block_injection_manifest.v1",
                "comparison_class": manifest_class,
            },
            "sites": [{"block_index": 29, "stage": "after_self"}],
        }
    arrays = {
        "coords": coords,
        "pos_final_output": pos[None],
        "neg_final_output": neg[None],
        "trace_block_index": np.asarray(29, dtype=np.int32),
        "shape_flow_trace_step_index": np.asarray(0, dtype=np.int32),
        "guidance_strength": np.asarray(7.5, dtype=np.float32),
        "guidance_rescale": np.asarray(0.5, dtype=np.float32),
        "t": np.asarray(t, dtype=np.float32),
        "t_prev": np.asarray(0.75, dtype=np.float32),
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
        manifest_class="exact_source_prefix28_plus_block29_after_self",
        t=1000.0,
    )

    report = summarize_replays(
        source_trace_path=source_trace,
        source_step_path=source_step,
        candidates=[
            CandidateSpec("natural", natural, None),
            CandidateSpec(
                "after_self",
                replay,
                "exact_source_prefix28_plus_block29_after_self",
            ),
        ],
    )

    assert report["schema"] == "trellis2mlx.shape_block_operation_replays.v1"
    assert report["source_reconstruction"]["pred_final_max_abs"] < 1e-6
    assert [row["name"] for row in report["replay_rows"]] == ["natural", "after_self", "source"]
    assert report["replay_rows"][1]["manifest_identity"]["comparison_class"] == (
        "exact_source_prefix28_plus_block29_after_self"
    )
    assert report["replay_rows"][1]["pred_final_source_mean_abs"] < (
        report["replay_rows"][0]["pred_final_source_mean_abs"]
    )
    assert report["replay_rows"][-1]["pred_final_source_mean_abs"] == 0.0


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
        sample_next=sample,
        t=np.asarray(1.0, dtype=np.float32),
        t_prev=np.asarray(0.75, dtype=np.float32),
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
