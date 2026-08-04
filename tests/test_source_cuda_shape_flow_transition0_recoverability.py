import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parser_accepts_separate_source_recurrence_output(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import build_parser

    trace_path = tmp_path / "source-recurrence.npz"
    args = build_parser().parse_args(
        [
            "--output-json",
            str(tmp_path / "report.json"),
            "--output-npz",
            str(tmp_path / "matrix.npz"),
            "--mlx-shape-flow-steps",
            str(tmp_path / "mlx-steps.npz"),
            "--mlx-shape-flow-steps-sha256",
            "1" * 64,
            "--mlx-run-report",
            str(tmp_path / "mlx-report.json"),
            "--mlx-run-report-sha256",
            "2" * 64,
            "--conditioning",
            str(tmp_path / "conditioning.npz"),
            "--conditioning-sha256",
            "3" * 64,
            "--accepted-source-baseline",
            str(tmp_path / "source.npz"),
            "--accepted-source-baseline-sha256",
            "4" * 64,
            "--accepted-source-report",
            str(tmp_path / "source.json"),
            "--accepted-source-report-sha256",
            "5" * 64,
            "--accepted-suffix-result",
            str(tmp_path / "suffix.npz"),
            "--accepted-suffix-result-sha256",
            "6" * 64,
            "--accepted-suffix-report",
            str(tmp_path / "suffix.json"),
            "--accepted-suffix-report-sha256",
            "7" * 64,
            "--source-tar",
            str(tmp_path / "source.tar"),
            "--source-tar-sha256",
            "8" * 64,
            "--source-recurrence-output",
            str(trace_path),
        ]
    )

    assert args.source_recurrence_output == trace_path


def _source_recurrence_fixture(*, std_reduction_scale=np.float32(1.0)):
    from scripts.source_cuda_shape_flow_suffix_ladder import _schedule_pairs
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        build_source_recurrence_arrays,
    )

    coords = np.asarray([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=np.int32)
    shape = (coords.shape[0], 32)
    schedule = _schedule_pairs(8, 3.0)
    noise = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100.0
    pred_pos = (
        np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 250.0 + 0.1
    )
    pred_neg = (
        np.flip(pred_pos, axis=1).copy() * np.float32(-0.35)
    )

    def source_step(sample_in, t, t_prev):
        active = 0.6 <= t <= 1.0
        pred_cfg = (
            np.float32(7.5) * pred_pos
            + np.float32(1.0 - 7.5) * pred_neg
            if active
            else pred_pos
        )
        captured = {
            "guidance_active": np.asarray(active, dtype=np.bool_),
            "pred_cfg": pred_cfg,
        }
        if active:
            sigma_min = np.float32(1e-5)
            one_minus_sigma = np.float32(1.0 - sigma_min)
            coefficient = np.float32(
                sigma_min
                + np.float32(1.0 - sigma_min) * np.float32(t)
            )
            x0_pos = one_minus_sigma * sample_in - coefficient * pred_pos
            x0_cfg = one_minus_sigma * sample_in - coefficient * pred_cfg

            def sparse_std(value):
                row_mean = value.mean(axis=1, keepdims=True, dtype=np.float32)
                row_mean2 = np.square(value).mean(
                    axis=1, keepdims=True, dtype=np.float32
                )
                mean = row_mean.mean(axis=0, keepdims=True, dtype=np.float32)
                mean2 = row_mean2.mean(axis=0, keepdims=True, dtype=np.float32)
                return np.sqrt(mean2 - np.square(mean), dtype=np.float32)

            std_pos = sparse_std(x0_pos) * std_reduction_scale
            std_cfg = sparse_std(x0_cfg) * std_reduction_scale
            std_ratio = std_pos / std_cfg
            x0_rescaled = x0_cfg * std_ratio
            x0_after_rescale = (
                np.float32(0.5) * x0_rescaled + np.float32(0.5) * x0_cfg
            )
            pred_final = (
                one_minus_sigma * sample_in - x0_after_rescale
            ) / coefficient
            captured.update(
                {
                    "x0_pos": x0_pos,
                    "x0_cfg": x0_cfg,
                    "std_pos": std_pos,
                    "std_cfg": std_cfg,
                    "std_ratio": std_ratio,
                    "x0_rescaled": x0_rescaled,
                    "x0_after_rescale": x0_after_rescale,
                }
            )
        else:
            pred_final = pred_cfg
        sample_next = (
            sample_in - np.float32(t - t_prev) * pred_final
        ).astype(np.float32)
        return captured, pred_final.astype(np.float32), sample_next

    guidance, pred_final, sample_next = source_step(noise, *schedule[0])
    suffix_steps = []
    sample_in = sample_next
    for t, t_prev in schedule[1:]:
        step_guidance, step_pred_final, next_sample = source_step(
            sample_in, t, t_prev
        )
        suffix_steps.append(
            {
                "sample_in": sample_in,
                "pred_pos": pred_pos,
                "pred_neg": pred_neg,
                **step_guidance,
                "pred_final": step_pred_final,
                "sample_next": next_sample,
                "t": np.asarray(t, dtype=np.float32),
                "t_prev": np.asarray(t_prev, dtype=np.float32),
            }
        )
        sample_in = next_sample
    arrays = build_source_recurrence_arrays(
        coords=coords,
        noise=noise,
        transition0_t=np.asarray(schedule[0][0], dtype=np.float32),
        transition0_t_prev=np.asarray(schedule[0][1], dtype=np.float32),
        source_transition0_pred_pos=pred_pos,
        source_transition0_pred_neg=pred_neg,
        source_transition0_guidance=guidance,
        source_transition0_pred_final=pred_final,
        source_transition0_sample_next=sample_next,
        suffix_steps=suffix_steps,
    )
    source_candidate = {
        "name": "source-native-control",
        "source_step_indices": list(range(1, 8)),
        "source_step_count": 7,
    }
    report = {
        "effective_route": {
            "device_type": "cuda",
            "attention_backend": "sdpa",
            "conv_backend": "none",
            "one_model_load": True,
            "steps": 8,
            "rescale_t": 3.0,
            "route": "official-source-cuda-external-transition0-recoverability",
            "comparison_class": "external-transition0-plus-source-cuda-suffix",
            "model_ref": (
                "microsoft/TRELLIS.2-4B/ckpts/"
                "slat_flow_img2shape_dit_1_3B_512_bf16"
            ),
            "cuda_device": "Tesla T4",
            "candidate_names": [
                "source-native-control",
                "external-cuda-welford-metal",
            ],
        },
        "pipeline_config": {
            "sampler_name": "FlowEulerGuidanceIntervalSampler",
            "sampler_args": {"sigma_min": 1e-5},
            "sampler_params": {
                "steps": 8,
                "rescale_t": 3.0,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.5,
                "guidance_interval": [0.6, 1.0],
            }
        },
        "inputs": {
            "expected_digests": {
                name: "1" * 64
                for name in (
                    "MLX run report",
                    "MLX shape-flow steps",
                    "accepted source baseline",
                    "accepted source report",
                    "accepted suffix report",
                    "accepted suffix result",
                    "conditioning",
                    "external transition report",
                    "external transition step",
                    "source tar",
                )
            }
        },
        "candidates": [source_candidate],
    }
    return arrays, report


def test_source_recurrence_artifact_binds_route_arrays_and_exact_linkage(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        validate_source_recurrence_artifact,
        write_source_recurrence_artifact,
    )

    arrays, report = _source_recurrence_fixture()
    output = tmp_path / "source-recurrence.npz"
    receipt = write_source_recurrence_artifact(output, arrays=arrays, report=report)

    assert receipt["validation"]["step_count"] == 8
    assert receipt["validation"]["recurrence_exact"] is True
    assert receipt["sha256"] == _sha256(output)
    expected_active = int(arrays["guidance_active"].sum())
    with np.load(output, allow_pickle=False) as archive:
        assert archive["pred_cfg"].shape[0] == 8
        for name in (
            "x0_pos",
            "x0_cfg",
            "std_pos",
            "std_cfg",
            "std_ratio",
            "x0_rescaled",
            "x0_after_rescale",
        ):
            assert archive[name].shape[0] == expected_active
    assert validate_source_recurrence_artifact(output)["all_arrays_bound"] is True


def test_source_recurrence_artifact_accepts_float32_backend_reduction_drift(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        write_source_recurrence_artifact,
    )

    arrays, report = _source_recurrence_fixture(
        std_reduction_scale=np.float32(1.00008)
    )
    output = tmp_path / "source-recurrence.npz"

    receipt = write_source_recurrence_artifact(
        output, arrays=arrays, report=report
    )

    assert receipt["validation"]["all_arrays_bound"] is True


def test_source_recurrence_artifact_rejects_material_std_reduction_drift(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        write_source_recurrence_artifact,
    )

    arrays, report = _source_recurrence_fixture(
        std_reduction_scale=np.float32(1.001)
    )

    with pytest.raises(ValueError, match="std_pos|std_cfg"):
        write_source_recurrence_artifact(
            tmp_path / "source-recurrence.npz", arrays=arrays, report=report
        )


def test_source_recurrence_artifact_admits_truthful_direct_source_route(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        SOURCE_DIRECT_COMPARISON_CLASS,
        SOURCE_DIRECT_INPUT_DIGESTS,
        SOURCE_DIRECT_ROUTE,
        validate_source_recurrence_artifact,
        write_source_recurrence_artifact,
    )

    arrays, report = _source_recurrence_fixture()
    report["effective_route"].update(
        {
            "route": SOURCE_DIRECT_ROUTE,
            "comparison_class": SOURCE_DIRECT_COMPARISON_CLASS,
            "candidate_names": ["source-native-control"],
        }
    )
    report["inputs"]["expected_digests"] = {
        name: "1" * 64 for name in SOURCE_DIRECT_INPUT_DIGESTS
    }
    report["candidates"] = [
        {
            "name": "source-native-control",
            "source_step_indices": list(range(8)),
            "source_step_count": 8,
        }
    ]

    output = tmp_path / "direct-source-recurrence.npz"
    receipt = write_source_recurrence_artifact(
        output, arrays=arrays, report=report
    )

    assert receipt["validation"]["recurrence_exact"] is True
    assert validate_source_recurrence_artifact(output)["step_count"] == 8


def test_guided_prediction_captures_the_exact_source_postprocessing_chain():
    from scripts.source_cuda_shape_block29_basin_map import _guided_prediction

    class FakeSparse:
        def __init__(self, feats):
            self.feats = np.asarray(feats, dtype=np.float32)

        @property
        def ndim(self):
            return 2

        def std(self, dim=None, keepdim=False):
            assert dim == [1]
            assert keepdim is True
            return np.asarray([[self.feats.std(dtype=np.float32)]], dtype=np.float32)

        def _binary(self, other, op):
            rhs = other.feats if isinstance(other, FakeSparse) else other
            return FakeSparse(op(self.feats, rhs))

        def __add__(self, other):
            return self._binary(other, np.add)

        def __radd__(self, other):
            return self._binary(other, np.add)

        def __sub__(self, other):
            return self._binary(other, np.subtract)

        def __rsub__(self, other):
            return FakeSparse(np.subtract(other, self.feats))

        def __mul__(self, other):
            return self._binary(other, np.multiply)

        def __rmul__(self, other):
            return self._binary(other, np.multiply)

        def __truediv__(self, other):
            return self._binary(other, np.divide)

    class FakeSampler:
        @staticmethod
        def _pred_to_xstart(sample, t, pred):
            return np.float32(0.75) * sample - np.float32(0.25) * pred

        @staticmethod
        def _xstart_to_pred(sample, t, x0):
            return (np.float32(0.75) * sample - x0) / np.float32(0.25)

    sample = FakeSparse([[0.5, -0.25, 1.0], [0.75, -0.5, 0.25]])
    pred_pos = FakeSparse([[0.2, -0.1, 0.4], [0.3, -0.2, 0.1]])
    pred_neg = FakeSparse([[-0.1, 0.05, -0.2], [-0.15, 0.1, -0.05]])
    captured = {}

    pred = _guided_prediction(
        sampler=FakeSampler(),
        sample=sample,
        pred_pos=pred_pos,
        pred_neg=pred_neg,
        t=0.75,
        guidance_strength=2.0,
        guidance_rescale=0.5,
        guidance_interval=(0.6, 1.0),
        capture=captured,
    )

    assert set(captured) == {
        "guidance_active",
        "pred_cfg",
        "x0_pos",
        "x0_cfg",
        "std_pos",
        "std_cfg",
        "std_ratio",
        "x0_rescaled",
        "x0_after_rescale",
    }
    assert captured["x0_after_rescale"] is not pred
    assert np.array_equal(
        pred.feats,
        FakeSampler._xstart_to_pred(
            sample, 0.75, captured["x0_after_rescale"]
        ).feats,
    )


def test_source_recurrence_artifact_rejects_corruption_and_route_substitution(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        build_source_recurrence_metadata,
        validate_source_recurrence_artifact,
    )

    arrays, report = _source_recurrence_fixture()
    output = tmp_path / "source-recurrence.npz"
    metadata = build_source_recurrence_metadata(report, arrays)
    np.savez(
        output,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    corrupted = dict(arrays)
    corrupted["pred_neg"] = arrays["pred_neg"].copy()
    corrupted["pred_neg"][3, 0, 0] += np.float32(1.0)
    np.savez(
        output,
        **corrupted,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    with pytest.raises(ValueError, match="pred_neg digest"):
        validate_source_recurrence_artifact(output)

    substituted_report = json.loads(json.dumps(report))
    substituted_report["effective_route"]["device_type"] = "cpu"
    substituted_metadata = build_source_recurrence_metadata(
        substituted_report, arrays
    )
    np.savez(
        output,
        **arrays,
        metadata_json=np.asarray(json.dumps(substituted_metadata, sort_keys=True)),
    )
    with pytest.raises(ValueError, match="route"):
        validate_source_recurrence_artifact(output)


def test_source_recurrence_rejects_self_consistent_route_and_sampler_substitution(
    tmp_path,
):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        build_source_recurrence_metadata,
        validate_source_recurrence_artifact,
    )

    arrays, report = _source_recurrence_fixture()
    substituted = json.loads(json.dumps(report))
    substituted["effective_route"].update(
        {
            "attention_backend": "route-substituted",
            "conv_backend": "spconv",
            "one_model_load": False,
        }
    )
    substituted["pipeline_config"]["sampler_params"]["guidance_strength"] = 999.0
    metadata = build_source_recurrence_metadata(substituted, arrays)
    output = tmp_path / "source-recurrence.npz"
    np.savez(
        output,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    with pytest.raises(ValueError, match="route|sampler"):
        validate_source_recurrence_artifact(output)


def test_source_recurrence_rejects_rehashed_semantically_unbound_guidance(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        build_source_recurrence_metadata,
        validate_source_recurrence_artifact,
    )

    arrays, report = _source_recurrence_fixture()
    fabricated = dict(arrays)
    fabricated["pred_cfg"] = np.full_like(arrays["pred_cfg"], -123.0)
    fabricated["x0_after_rescale"] = np.full_like(
        arrays["x0_after_rescale"], 456.0
    )
    metadata = build_source_recurrence_metadata(report, fabricated)
    output = tmp_path / "source-recurrence.npz"
    np.savez(
        output,
        **fabricated,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    with pytest.raises(ValueError, match="guidance"):
        validate_source_recurrence_artifact(output)


def test_source_recurrence_artifact_rejects_self_consistent_broken_linkage(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        build_source_recurrence_metadata,
        validate_source_recurrence_artifact,
    )

    arrays, report = _source_recurrence_fixture()
    broken = dict(arrays)
    broken["sample_in"] = arrays["sample_in"].copy()
    broken["sample_in"][4, 0, 0] += np.float32(1.0)
    metadata = build_source_recurrence_metadata(report, broken)
    output = tmp_path / "source-recurrence.npz"
    np.savez(
        output,
        **broken,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    with pytest.raises(ValueError, match="step linkage"):
        validate_source_recurrence_artifact(output)


def test_source_recurrence_rejects_rehashed_guidance_activity_outside_schedule(
    tmp_path,
):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        build_source_recurrence_metadata,
        validate_source_recurrence_artifact,
    )

    arrays, report = _source_recurrence_fixture()
    broken = dict(arrays)
    broken["guidance_active"] = arrays["guidance_active"].copy()
    broken["guidance_active"][-1] = True
    broken["guidance_step_indices"] = np.flatnonzero(
        broken["guidance_active"]
    ).astype(np.int32)
    for name in (
        "x0_pos",
        "x0_cfg",
        "std_pos",
        "std_cfg",
        "std_ratio",
        "x0_rescaled",
        "x0_after_rescale",
    ):
        broken[name] = np.concatenate([arrays[name], arrays[name][-1:]], axis=0)
    metadata = build_source_recurrence_metadata(report, broken)
    output = tmp_path / "source-recurrence.npz"
    np.savez(
        output,
        **broken,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    with pytest.raises(ValueError, match="guidance activity"):
        validate_source_recurrence_artifact(output)


def test_source_recurrence_artifact_rejects_rehashed_prediction_without_euler_change(
    tmp_path,
):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        build_source_recurrence_metadata,
        validate_source_recurrence_artifact,
    )

    arrays, report = _source_recurrence_fixture()
    broken = dict(arrays)
    broken["pred_final"] = arrays["pred_final"].copy()
    broken["pred_final"][4, 0, 0] += np.float32(1.0)
    metadata = build_source_recurrence_metadata(report, broken)
    output = tmp_path / "source-recurrence.npz"
    np.savez(
        output,
        **broken,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    with pytest.raises(ValueError, match="guidance|Euler transition"):
        validate_source_recurrence_artifact(output)


def test_source_recurrence_artifact_removes_partial_write(tmp_path, monkeypatch):
    from scripts import source_cuda_shape_flow_transition0_recoverability as witness

    arrays, report = _source_recurrence_fixture()
    output = tmp_path / "source-recurrence.npz"

    def partial_save(path, **payload):
        path.write_bytes(b"partial archive")
        raise OSError("disk full during recurrence write")

    monkeypatch.setattr(witness.np, "savez", partial_save)
    with pytest.raises(OSError, match="disk full"):
        witness.write_source_recurrence_artifact(
            output, arrays=arrays, report=report
        )

    assert not output.exists()


def _write_transition_capture(tmp_path, *, omit=None):
    shape = (8, 2, 3)
    sample_in = np.zeros(shape, dtype=np.float32)
    arrays = {
        "noise": sample_in[0].copy(),
        "sample_in": sample_in,
        "pred_pos": np.full(shape, 1.0, dtype=np.float32),
        "pred_neg": np.full(shape, 2.0, dtype=np.float32),
        "pred_cfg": np.full(shape, 3.0, dtype=np.float32),
        "x0_pos": np.full(shape, 4.0, dtype=np.float32),
        "x0_cfg": np.full(shape, 5.0, dtype=np.float32),
        "x0_rescaled": np.full(shape, 6.0, dtype=np.float32),
        "x0_after_rescale": np.full(shape, 7.0, dtype=np.float32),
        "pred_final": np.full(shape, 8.0, dtype=np.float32),
        "sample_next": np.full(shape, 9.0, dtype=np.float32),
        "t": np.linspace(1.0, 0.2, 8, dtype=np.float32),
        "t_prev": np.linspace(0.9, 0.1, 8, dtype=np.float32),
        "std_pos": np.ones(8, dtype=np.float32),
        "std_cfg": np.ones(8, dtype=np.float32),
        "ratio_raw": np.ones(8, dtype=np.float32),
        "std_ratio": np.ones(8, dtype=np.float32),
        "ratio_effective": np.ones(8, dtype=np.float32),
    }
    if omit:
        arrays.pop(omit)
    path = tmp_path / "shape_flow_steps.npz"
    np.savez(path, **arrays)
    return path


def _write_external_transition_capture(
    tmp_path,
    *,
    break_euler=False,
    layernorm_backend="cuda-welford-metal",
    turing_lut_sha256=None,
):
    from scripts.source_cuda_shape_flow_suffix_ladder import _schedule_pairs

    shape = (2, 3)
    coords = np.arange(shape[0] * 4, dtype=np.int32).reshape(shape[0], 4)
    noise = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 10.0
    pred_final = np.full(shape, 0.25, dtype=np.float32)
    t, t_prev = np.asarray(_schedule_pairs(8, 3.0)[0], dtype=np.float32)
    sample_next = noise - (t - t_prev) * pred_final
    if break_euler:
        sample_next = sample_next.copy()
        sample_next[0, 0] += np.float32(0.01)
    path = tmp_path / "external-shape-flow-step.npz"
    arrays = dict(
        noise=noise,
        sample_feats=noise,
        coords=coords,
        coords_3d=coords[:, 1:],
        pred_pos=np.full(shape, 0.1, dtype=np.float32),
        pred_neg=np.full(shape, -0.1, dtype=np.float32),
        pred_cfg=np.full(shape, 0.2, dtype=np.float32),
        x0_pos=np.full(shape, 0.3, dtype=np.float32),
        x0_cfg=np.full(shape, 0.4, dtype=np.float32),
        std_pos=np.asarray(1.0, dtype=np.float32),
        std_cfg=np.asarray(2.0, dtype=np.float32),
        ratio_raw=np.asarray(0.5, dtype=np.float32),
        std_ratio=np.asarray(0.5, dtype=np.float32),
        ratio_effective=np.asarray(0.5, dtype=np.float32),
        x0_rescaled=np.full(shape, 0.5, dtype=np.float32),
        x0_after_rescale=np.full(shape, 0.5, dtype=np.float32),
        pred_final=pred_final,
        pred_v_feats=pred_final,
        sample_next=sample_next.astype(np.float32),
        t=t,
        t_prev=t_prev,
        steps=np.asarray(8, dtype=np.int32),
        guidance_strength=np.asarray(7.5, dtype=np.float32),
        guidance_rescale=np.asarray(0.5, dtype=np.float32),
        guidance_interval=np.asarray([0.6, 1.0], dtype=np.float32),
        rescale_t=np.asarray(3.0, dtype=np.float32),
        shape_flow_block_injection_json=np.asarray(""),
        shape_flow_layernorm_backend=np.asarray(layernorm_backend),
    )
    if turing_lut_sha256 is not None:
        arrays["shape_flow_turing_rsqrt_lut_sha256"] = np.asarray(
            turing_lut_sha256
        )
    np.savez(path, **arrays)
    trajectory = {
        "noise": noise,
        "sample_in": np.stack([noise] * 8),
        "coords": coords,
        "t": np.asarray([pair[0] for pair in _schedule_pairs(8, 3.0)], dtype=np.float32),
        "t_prev": np.asarray(
            [pair[1] for pair in _schedule_pairs(8, 3.0)], dtype=np.float32
        ),
    }
    return path, trajectory


def _write_external_transition_report(
    tmp_path,
    capture,
    *,
    expected_commit,
    layernorm_backend="cuda-welford-metal",
    turing_lut_sha256=None,
):
    conditioning_sha = "1" * 64
    noise_sha = "2" * 64
    support_sha = "3" * 64
    repo_identity = {
        "commit_effective": expected_commit,
        "commit_requested": expected_commit,
        "dirty": False,
        "status_porcelain": "",
    }
    route = {
        "family": "trellis2mlx/mlx",
        "backend": "mlx-metal",
        "attention_backend": "fast",
        "steps": 8,
        "cascade": False,
        "conditioning_sample_sha256": conditioning_sha,
        "shape_flow_noise_sample_sha256": noise_sha,
        "shape_slat_support_sample_sha256": support_sha,
        "shape_flow_layernorm_backend_requested": layernorm_backend,
        "shape_flow_layernorm_backend_effective": layernorm_backend,
        "repo_commit_requested": expected_commit,
        "repo_commit_effective": expected_commit,
        "repo_dirty": False,
        "repo_status_porcelain": "",
        "repo_identity_postflight": dict(repo_identity),
        "repo_identity_postflight_error": None,
        "shape_flow_block_injection_trace_path": None,
        "shape_flow_block_injection_manifest_path": None,
    }
    if turing_lut_sha256 is not None:
        route.update(
            {
                "shape_flow_turing_rsqrt_lut_sha256_effective": turing_lut_sha256,
                "turing_rsqrt_lut_sha256_requested": turing_lut_sha256,
                "turing_rsqrt_lut_sha256_effective": turing_lut_sha256,
            }
        )
    report = {
        "schema": "trellis2mlx.mlx_stage_capture_run_report.v1",
        "status": "done",
        "exit_code": 0,
        "failure_phase": None,
        "last_trustworthy_phase": "shape_flow_step_saved",
        "primary_output_status": "written",
        "artifacts": {
            "shape_flow_step.npz": {
                "path": str(capture),
                "sha256": _sha256(capture),
                "size_bytes": capture.stat().st_size,
            }
        },
        "repo_identity_preflight": dict(repo_identity),
        "repo_identity_postflight": dict(repo_identity),
        "repo_identity_postflight_error": None,
        "route_identity": {
            "schema": "trellis2mlx.mlx_stage_capture_route.v1",
            "env": {
                "MLX_METAL_PATH": None,
                "PYTHONPATH": ".",
                "TRELLIS2MLX_ATTENTION_BACKEND": "fast",
            },
            "requested_stop": "shape_flow_step",
            "requested_outputs": {
                "conditioning": False,
                "decoder_output": False,
                "mesh_clean": False,
                "mesh_raw": False,
                "mesh_uv": False,
                "shape_flow_block_trace": False,
                "shape_flow_step": True,
                "shape_flow_steps": False,
                "shape_slat": False,
                "sparse_coords": False,
                "sparse_flow_block_trace": False,
                "sparse_flow_step": False,
                "sparse_flow_steps": False,
                "sparse_internals": False,
            },
            "route": route,
        },
    }
    path = tmp_path / "external-run-report.json"
    path.write_text(json.dumps(report))
    identity = {
        "conditioning_sha256": conditioning_sha,
        "shape_flow_noise_sample_sha256": noise_sha,
        "shape_slat_support_sample_sha256": support_sha,
    }
    return path, identity


def _matrix_artifact_fixture():
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        MLX_COMPONENT_NAMES,
        transition0_candidate_specs,
    )

    shape = (2, 3)
    arrays = {
        "coords": np.zeros((shape[0], 4), dtype=np.int32),
        "source_anchor_shape_slat": np.full(shape, 1.0, dtype=np.float32),
        "mlx_anchor_shape_slat": np.full(shape, 2.0, dtype=np.float32),
        "accepted_switch_1_shape_slat": np.full(shape, 3.0, dtype=np.float32),
        "source_transition0_pred_pos": np.full(shape, 4.0, dtype=np.float32),
        "source_transition0_pred_neg": np.full(shape, 5.0, dtype=np.float32),
    }
    for index, name in enumerate(MLX_COMPONENT_NAMES):
        arrays[f"mlx_transition0_{name}"] = np.full(
            shape, 10.0 + index, dtype=np.float32
        )
    candidates = []
    for index, spec in enumerate(transition0_candidate_specs()):
        start_key = f"candidate_{index}_transition0_sample_next"
        output_key = f"candidate_{index}_shape_slat"
        arrays[start_key] = np.full(shape, 30.0 + index, dtype=np.float32)
        value = np.full(shape, 40.0 + index, dtype=np.float32)
        arrays[output_key] = value
        candidates.append(
            {
                **spec,
                "output_key": output_key,
                "shape": list(shape),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
        )
    return arrays, candidates


def _external_artifact_fixture():
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        EXTERNAL_CANDIDATE_NAMES,
    )

    shape = (2, 3)
    arrays = {
        "coords": np.zeros((shape[0], 4), dtype=np.int32),
        "source_anchor_shape_slat": np.full(shape, 1.0, dtype=np.float32),
        "mlx_anchor_shape_slat": np.full(shape, 2.0, dtype=np.float32),
        "accepted_switch_1_shape_slat": np.full(shape, 3.0, dtype=np.float32),
        "source_transition0_pred_pos": np.full(shape, 4.0, dtype=np.float32),
        "source_transition0_pred_neg": np.full(shape, 5.0, dtype=np.float32),
        "external_transition0_sample_next": np.full(shape, 6.0, dtype=np.float32),
    }
    candidates = []
    for index, name in enumerate(EXTERNAL_CANDIDATE_NAMES):
        start_key = f"candidate_{index}_transition0_sample_next"
        output_key = f"candidate_{index}_shape_slat"
        arrays[start_key] = (
            arrays["external_transition0_sample_next"].copy()
            if index == 1
            else np.full(shape, 10.0, dtype=np.float32)
        )
        value = np.full(shape, 20.0 + index, dtype=np.float32)
        arrays[output_key] = value
        candidates.append(
            {
                "name": name,
                "output_key": output_key,
                "shape": list(shape),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
        )
    return arrays, candidates


def test_transition0_candidate_specs_are_complete_and_nonredundant():
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        transition0_candidate_specs,
    )

    specs = transition0_candidate_specs()
    assert [spec["name"] for spec in specs] == [
        "source-native-control",
        "mlx-pos-source-neg",
        "source-pos-mlx-neg",
        "mlx-both-source-post",
        "mlx-final-source-euler",
    ]
    assert len({json.dumps(spec, sort_keys=True) for spec in specs}) == len(specs)
    assert specs[0] == {
        "name": "source-native-control",
        "positive": "source",
        "negative": "source",
        "post": "source-guidance-rescale-euler",
    }
    assert specs[-1]["post"] == "mlx-final-source-euler"


def test_external_transition0_candidate_specs_are_exact_and_ordered():
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        external_transition0_candidate_specs,
    )

    assert external_transition0_candidate_specs() == [
        {
            "name": "source-native-control",
            "positive": "source",
            "negative": "source",
            "post": "source-guidance-rescale-euler",
        },
        {
            "name": "external-cuda-welford-metal",
            "positive": "external-captured",
            "negative": "external-captured",
            "post": "external-captured-sample-next",
        },
    ]


def test_load_transition0_components_requires_complete_float32_intermediates(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        load_mlx_transition0_components,
    )

    capture = _write_transition_capture(tmp_path)
    components = load_mlx_transition0_components(
        capture,
        expected_sha256=_sha256(capture),
        expected_shape=(2, 3),
    )
    assert set(components) >= {
        "noise",
        "pred_pos",
        "pred_neg",
        "pred_cfg",
        "x0_pos",
        "x0_cfg",
        "x0_rescaled",
        "x0_after_rescale",
        "pred_final",
        "sample_next",
    }
    assert components["pred_pos"].shape == (2, 3)

    incomplete = _write_transition_capture(tmp_path, omit="pred_neg")
    with pytest.raises(ValueError, match="pred_neg"):
        load_mlx_transition0_components(
            incomplete,
            expected_sha256=_sha256(incomplete),
            expected_shape=(2, 3),
        )


def test_compose_candidate_pairs_maps_only_the_declared_prediction_branch():
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        compose_candidate_pairs,
    )

    source_pos = object()
    source_neg = object()
    mlx_pos = object()
    mlx_neg = object()
    pairs = compose_candidate_pairs(
        source_pos=source_pos,
        source_neg=source_neg,
        mlx_pos=mlx_pos,
        mlx_neg=mlx_neg,
    )

    assert pairs["source-native-control"] == (source_pos, source_neg)
    assert pairs["mlx-pos-source-neg"] == (mlx_pos, source_neg)
    assert pairs["source-pos-mlx-neg"] == (source_pos, mlx_neg)
    assert pairs["mlx-both-source-post"] == (mlx_pos, mlx_neg)
    assert "mlx-final-source-euler" not in pairs


def test_validate_result_requires_cuda_route_all_candidates_and_exact_control():
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        transition0_candidate_specs,
        validate_result_manifest,
    )

    candidates = []
    for index, spec in enumerate(transition0_candidate_specs()):
        candidates.append(
            {
                "name": spec["name"],
                "output_key": f"candidate_{index}_shape_slat",
                "source_step_indices": list(range(1, 8)),
                "source_step_count": 7,
                "step_elapsed_seconds": [1.0] * 7,
                "vs_source_anchor": {
                    "exact": index == 0,
                    "mean_abs": 0.0 if index == 0 else 1.0,
                    "max_abs": 0.0 if index == 0 else 1.0,
                    "nonzero": 0 if index == 0 else 1,
                },
                "vs_mlx_anchor": {
                    "exact": False,
                    "mean_abs": 1.0,
                    "max_abs": 1.0,
                    "nonzero": 1,
                },
            }
        )
    payload = {
        "status": "done",
        "effective_route": {
            "device_type": "cuda",
            "cuda_device": "Tesla T4",
            "attention_backend": "sdpa",
            "conv_backend": "none",
            "steps": 8,
            "one_model_load": True,
            "candidate_names": [candidate["name"] for candidate in candidates],
        },
        "candidates": candidates,
        "timing": {
            "source_steps_completed": 35,
            "source_steps_requested": 35,
            "candidates_completed": 5,
            "candidates_requested": 5,
        },
    }

    validate_result_manifest(payload)
    payload["candidates"][0]["vs_source_anchor"]["exact"] = False
    with pytest.raises(ValueError, match="source-native control"):
        validate_result_manifest(payload)


def test_external_request_is_all_or_none():
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        external_transition_request,
    )

    empty = SimpleNamespace(
        external_transition0_step=None,
        external_transition0_step_sha256=None,
        external_transition0_report=None,
        external_transition0_report_sha256=None,
        expected_external_repo_commit=None,
        expected_external_layernorm_backend=None,
        expected_external_turing_rsqrt_lut_sha256=None,
    )
    assert external_transition_request(empty) is None

    partial = SimpleNamespace(**vars(empty))
    partial.external_transition0_step = "step.npz"
    with pytest.raises(ValueError, match="must be supplied together"):
        external_transition_request(partial)


def test_external_request_requires_allowlisted_backend_and_turing_lut():
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        external_transition_request,
    )

    base = {
        "external_transition0_step": "step.npz",
        "external_transition0_step_sha256": "1" * 64,
        "external_transition0_report": "report.json",
        "external_transition0_report_sha256": "2" * 64,
        "expected_external_repo_commit": "a" * 40,
        "expected_external_layernorm_backend": "cuda-welford-turing-t4",
        "expected_external_turing_rsqrt_lut_sha256": None,
    }
    with pytest.raises(ValueError, match="Turing rsqrt LUT"):
        external_transition_request(SimpleNamespace(**base))

    base["expected_external_turing_rsqrt_lut_sha256"] = "3" * 64
    request = external_transition_request(SimpleNamespace(**base))
    assert request["expected_layernorm_backend"] == "cuda-welford-turing-t4"
    assert request["expected_turing_rsqrt_lut_sha256"] == "3" * 64
    assert request["candidate_name"] == "external-cuda-welford-turing-t4"

    base["expected_external_layernorm_backend"] = "default"
    with pytest.raises(ValueError, match="unsupported external LayerNorm backend"):
        external_transition_request(SimpleNamespace(**base))


def test_external_transition_loader_binds_route_repo_and_step_semantics(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        load_external_transition0_start,
    )

    expected_commit = "a" * 40
    capture, trajectory = _write_external_transition_capture(tmp_path)
    report, expected_identity = _write_external_transition_report(
        tmp_path,
        capture,
        expected_commit=expected_commit,
    )
    arrays, identity = load_external_transition0_start(
        capture,
        report,
        expected_step_sha256=_sha256(capture),
        expected_report_sha256=_sha256(report),
        expected_repo_commit=expected_commit,
        trajectory=trajectory,
        expected_mlx_identity=expected_identity,
    )
    assert np.array_equal(arrays["sample_next"], np.load(capture)["sample_next"])
    assert identity["layernorm_backend"] == "cuda-welford-metal"
    assert identity["repo_commit"] == expected_commit

    payload = json.loads(report.read_text())
    payload["route_identity"]["route"]["shape_flow_layernorm_backend_effective"] = (
        "default"
    )
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="layernorm backend"):
        load_external_transition0_start(
            capture,
            report,
            expected_step_sha256=_sha256(capture),
            expected_report_sha256=_sha256(report),
            expected_repo_commit=expected_commit,
            trajectory=trajectory,
            expected_mlx_identity=expected_identity,
        )

    broken, trajectory = _write_external_transition_capture(
        tmp_path, break_euler=True
    )
    report, expected_identity = _write_external_transition_report(
        tmp_path,
        broken,
        expected_commit=expected_commit,
    )
    with pytest.raises(ValueError, match="Euler recurrence"):
        load_external_transition0_start(
            broken,
            report,
            expected_step_sha256=_sha256(broken),
            expected_report_sha256=_sha256(report),
            expected_repo_commit=expected_commit,
            trajectory=trajectory,
            expected_mlx_identity=expected_identity,
        )


def test_external_transition_loader_triple_binds_turing_lut_identity(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        load_external_transition0_start,
    )

    expected_commit = "a" * 40
    lut_sha256 = "4" * 64
    capture, trajectory = _write_external_transition_capture(
        tmp_path,
        layernorm_backend="cuda-welford-turing-t4",
        turing_lut_sha256=lut_sha256,
    )
    report, expected_identity = _write_external_transition_report(
        tmp_path,
        capture,
        expected_commit=expected_commit,
        layernorm_backend="cuda-welford-turing-t4",
        turing_lut_sha256=lut_sha256,
    )
    kwargs = {
        "expected_step_sha256": _sha256(capture),
        "expected_report_sha256": _sha256(report),
        "expected_repo_commit": expected_commit,
        "expected_layernorm_backend": "cuda-welford-turing-t4",
        "expected_turing_rsqrt_lut_sha256": lut_sha256,
        "trajectory": trajectory,
        "expected_mlx_identity": expected_identity,
    }
    _, identity = load_external_transition0_start(capture, report, **kwargs)
    assert identity["layernorm_backend"] == "cuda-welford-turing-t4"
    assert identity["turing_rsqrt_lut_sha256"] == lut_sha256
    assert identity["candidate_name"] == "external-cuda-welford-turing-t4"

    payload = json.loads(report.read_text())
    payload["route_identity"]["route"][
        "turing_rsqrt_lut_sha256_effective"
    ] = "5" * 64
    report.write_text(json.dumps(payload))
    kwargs["expected_report_sha256"] = _sha256(report)
    with pytest.raises(ValueError, match="Turing rsqrt LUT"):
        load_external_transition0_start(capture, report, **kwargs)

    payload["route_identity"]["route"][
        "turing_rsqrt_lut_sha256_effective"
    ] = lut_sha256
    report.write_text(json.dumps(payload))
    with np.load(capture, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["shape_flow_turing_rsqrt_lut_sha256"] = np.asarray("6" * 64)
    np.savez(capture, **arrays)
    report_payload = json.loads(report.read_text())
    report_payload["artifacts"]["shape_flow_step.npz"]["sha256"] = _sha256(capture)
    report_payload["artifacts"]["shape_flow_step.npz"][
        "size_bytes"
    ] = capture.stat().st_size
    report.write_text(json.dumps(report_payload))
    kwargs["expected_step_sha256"] = _sha256(capture)
    kwargs["expected_report_sha256"] = _sha256(report)
    with pytest.raises(ValueError, match="checkpoint Turing rsqrt LUT"):
        load_external_transition0_start(capture, report, **kwargs)


def test_external_transition_loader_rejects_turing_lut_scalar_on_metal(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        load_external_transition0_start,
    )

    expected_commit = "a" * 40
    capture, trajectory = _write_external_transition_capture(
        tmp_path,
        layernorm_backend="cuda-welford-metal",
        turing_lut_sha256="4" * 64,
    )
    report, expected_identity = _write_external_transition_report(
        tmp_path,
        capture,
        expected_commit=expected_commit,
        layernorm_backend="cuda-welford-metal",
    )

    with pytest.raises(
        ValueError,
        match="non-Turing checkpoint unexpectedly carries Turing rsqrt LUT identity",
    ):
        load_external_transition0_start(
            capture,
            report,
            expected_step_sha256=_sha256(capture),
            expected_report_sha256=_sha256(report),
            expected_repo_commit=expected_commit,
            expected_layernorm_backend="cuda-welford-metal",
            expected_turing_rsqrt_lut_sha256=None,
            trajectory=trajectory,
            expected_mlx_identity=expected_identity,
        )


def test_external_result_requires_two_candidates_and_fourteen_source_steps():
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        EXTERNAL_CANDIDATE_NAMES,
        validate_external_result_manifest,
    )

    candidates = []
    for index, name in enumerate(EXTERNAL_CANDIDATE_NAMES):
        candidates.append(
            {
                "name": name,
                "output_key": f"candidate_{index}_shape_slat",
                "transition0_sample_next_sha256": (
                    "1" * 64 if index == 0 else "2" * 64
                ),
                "source_step_indices": list(range(1, 8)),
                "source_step_count": 7,
                "step_elapsed_seconds": [1.0] * 7,
                "vs_source_anchor": {
                    "exact": index == 0,
                    "nonzero": 0 if index == 0 else 1,
                },
            }
        )
    payload = {
        "status": "done",
        "effective_route": {
            "device_type": "cuda",
            "attention_backend": "sdpa",
            "conv_backend": "none",
            "steps": 8,
            "one_model_load": True,
            "candidate_names": list(EXTERNAL_CANDIDATE_NAMES),
            "comparison_class": "external-transition0-plus-source-cuda-suffix",
        },
        "inputs": {
            "external_transition": {
                "layernorm_backend": "cuda-welford-metal",
                "sample_next_sha256": "2" * 64,
            }
        },
        "candidates": candidates,
        "timing": {
            "source_steps_completed": 14,
            "source_steps_requested": 14,
            "candidates_completed": 2,
            "candidates_requested": 2,
        },
    }

    validate_external_result_manifest(payload)
    payload["timing"]["source_steps_completed"] = 7
    with pytest.raises(ValueError, match="source_steps_completed"):
        validate_external_result_manifest(payload)
    payload["timing"]["source_steps_completed"] = 14
    payload["candidates"][1]["transition0_sample_next_sha256"] = "3" * 64
    with pytest.raises(ValueError, match="external transition digest"):
        validate_external_result_manifest(payload)


def test_external_result_propagates_turing_candidate_identity():
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        validate_external_result_manifest,
    )

    names = [
        "source-native-control",
        "external-cuda-welford-turing-t4",
    ]
    candidates = []
    for index, name in enumerate(names):
        candidates.append(
            {
                "name": name,
                "output_key": f"candidate_{index}_shape_slat",
                "transition0_sample_next_sha256": (
                    "1" * 64 if index == 0 else "2" * 64
                ),
                "source_step_indices": list(range(1, 8)),
                "source_step_count": 7,
                "step_elapsed_seconds": [1.0] * 7,
                "vs_source_anchor": {
                    "exact": index == 0,
                    "nonzero": 0 if index == 0 else 1,
                },
            }
        )
    payload = {
        "status": "done",
        "effective_route": {
            "device_type": "cuda",
            "attention_backend": "sdpa",
            "conv_backend": "none",
            "steps": 8,
            "one_model_load": True,
            "candidate_names": names,
            "comparison_class": "external-transition0-plus-source-cuda-suffix",
        },
        "inputs": {
            "external_transition": {
                "candidate_name": names[1],
                "layernorm_backend": "cuda-welford-turing-t4",
                "turing_rsqrt_lut_sha256": "3" * 64,
                "sample_next_sha256": "2" * 64,
            }
        },
        "candidates": candidates,
        "timing": {
            "source_steps_completed": 14,
            "source_steps_requested": 14,
            "candidates_completed": 2,
            "candidates_requested": 2,
        },
    }
    validate_external_result_manifest(payload)
    payload["effective_route"]["candidate_names"][1] = "external-cuda-welford-metal"
    with pytest.raises(ValueError, match="candidate_names"):
        validate_external_result_manifest(payload)
    payload["effective_route"]["candidate_names"] = names
    payload["inputs"]["external_transition"]["candidate_name"] = (
        "external-cuda-welford-metal"
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_external_result_manifest(payload)

def test_saved_artifact_binds_every_candidate_array_and_metadata(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        build_artifact_metadata,
        validate_saved_artifact,
    )

    arrays, candidates = _matrix_artifact_fixture()
    report = {
        "effective_route": {
            "device_type": "cuda",
            "candidate_names": [candidate["name"] for candidate in candidates],
        },
        "inputs": {"expected_digests": {}},
        "candidate_specs": [
            {key: candidate[key] for key in ("name", "positive", "negative", "post")}
            for candidate in candidates
        ],
        "candidates": candidates,
        "anchors": {},
    }
    metadata = build_artifact_metadata(report, arrays)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output = tmp_path / "matrix.npz"
    np.savez(output, **arrays)

    validation = validate_saved_artifact(output, candidates=candidates)
    assert validation["candidate_count"] == 5
    assert validation["all_matrix_arrays_bound"] is True

    with np.load(output, allow_pickle=False) as archive:
        corrupted = {key: np.asarray(archive[key]) for key in archive.files}
    corrupted.pop("mlx_transition0_pred_neg")
    np.savez(output, **corrupted)
    with pytest.raises(ValueError, match="mlx_transition0_pred_neg"):
        validate_saved_artifact(output, candidates=candidates)

    arrays, candidates = _matrix_artifact_fixture()
    metadata = build_artifact_metadata(report, arrays)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    arrays["candidate_3_transition0_sample_next"] = np.full(
        (2, 3), -999.0, dtype=np.float32
    )
    np.savez(output, **arrays)
    with pytest.raises(ValueError, match="candidate_3_transition0_sample_next digest"):
        validate_saved_artifact(output, candidates=candidates)


def test_external_saved_artifact_binds_both_candidates_and_intervention(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        build_artifact_metadata,
        validate_saved_artifact,
    )

    arrays, candidates = _external_artifact_fixture()
    report = {
        "effective_route": {
            "candidate_names": [candidate["name"] for candidate in candidates]
        },
        "inputs": {
            "external_transition": {
                "layernorm_backend": "cuda-welford-metal",
                "sample_next_sha256": hashlib.sha256(
                    arrays["external_transition0_sample_next"].tobytes()
                ).hexdigest()
            }
        },
        "candidate_specs": [{"name": candidate["name"]} for candidate in candidates],
        "candidates": candidates,
        "anchors": {},
    }
    arrays["metadata_json"] = np.asarray(
        json.dumps(build_artifact_metadata(report, arrays), sort_keys=True)
    )
    output = tmp_path / "external-result.npz"
    np.savez(output, **arrays)

    validation = validate_saved_artifact(output, candidates=candidates)
    assert validation["candidate_count"] == 2
    assert validation["all_matrix_arrays_bound"] is True

    arrays.pop("external_transition0_sample_next")
    np.savez(output, **arrays)
    with pytest.raises(ValueError, match="external_transition0_sample_next"):
        validate_saved_artifact(output, candidates=candidates)


def test_external_saved_artifact_round_trips_turing_candidate_identity(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        build_artifact_metadata,
        validate_saved_artifact,
    )

    arrays, candidates = _external_artifact_fixture()
    candidates[1]["name"] = "external-cuda-welford-turing-t4"
    external_sha256 = hashlib.sha256(
        arrays["external_transition0_sample_next"].tobytes()
    ).hexdigest()
    report = {
        "effective_route": {
            "candidate_names": [candidate["name"] for candidate in candidates]
        },
        "inputs": {
            "external_transition": {
                "candidate_name": "external-cuda-welford-turing-t4",
                "layernorm_backend": "cuda-welford-turing-t4",
                "turing_rsqrt_lut_sha256": "3" * 64,
                "sample_next_sha256": external_sha256,
            }
        },
        "candidate_specs": [{"name": candidate["name"]} for candidate in candidates],
        "candidates": candidates,
        "anchors": {},
    }
    arrays["metadata_json"] = np.asarray(
        json.dumps(build_artifact_metadata(report, arrays), sort_keys=True)
    )
    output = tmp_path / "external-turing-result.npz"
    np.savez(output, **arrays)

    validation = validate_saved_artifact(output, candidates=candidates)
    assert validation["candidate_count"] == 2

    report["inputs"]["external_transition"]["layernorm_backend"] = (
        "cuda-welford-metal"
    )
    report["inputs"]["external_transition"]["turing_rsqrt_lut_sha256"] = None
    arrays.pop("metadata_json")
    arrays["metadata_json"] = np.asarray(
        json.dumps(build_artifact_metadata(report, arrays), sort_keys=True)
    )
    np.savez(output, **arrays)
    with pytest.raises(ValueError, match="does not match"):
        validate_saved_artifact(output, candidates=candidates)


def test_accepted_suffix_is_cross_bound_to_current_admitted_inputs(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        _load_accepted_suffix,
    )

    coords = np.zeros((2, 4), dtype=np.int32)
    source_anchor = np.ones((2, 3), dtype=np.float32)
    result_path = tmp_path / "suffix.npz"
    np.savez(
        result_path,
        coords=coords,
        accepted_source_anchor_shape_slat=source_anchor,
        mlx_anchor_shape_slat=np.full((2, 3), 2.0, dtype=np.float32),
        switch_0_shape_slat=source_anchor,
        switch_1_shape_slat=np.full((2, 3), 3.0, dtype=np.float32),
    )
    mlx_identity = {
        "capture_sha256": "1" * 64,
        "run_report_sha256": "2" * 64,
        "conditioning_sha256": "5" * 64,
        "shape_flow_noise_sample_sha256": "8" * 64,
        "shape_slat_support_sample_sha256": "9" * 64,
    }
    source_identity = {
        "baseline_sha256": "3" * 64,
        "report_sha256": "4" * 64,
    }
    report = {
        "status": "done",
        "effective_route": {
            "device_type": "cuda",
            "attention_backend": "sdpa",
            "conv_backend": "none",
            "steps": 8,
            "one_model_load": True,
        },
        "inputs": {
            "mlx": dict(mlx_identity),
            "conditioning_sha256": "5" * 64,
            "source_tar_sha256": "6" * 64,
            "accepted_source": dict(source_identity),
        },
        "points": [{"switch_step": index} for index in range(9)],
    }
    report_path = tmp_path / "suffix.json"
    report_path.write_text(json.dumps(report))
    kwargs = {
        "expected_result_sha256": _sha256(result_path),
        "expected_report_sha256": _sha256(report_path),
        "source_anchor": source_anchor,
        "coords": coords,
        "expected_mlx_identity": mlx_identity,
        "expected_source_identity": source_identity,
        "expected_conditioning_sha256": "5" * 64,
        "expected_source_tar_sha256": "6" * 64,
    }

    _load_accepted_suffix(result_path, report_path, **kwargs)
    report["inputs"]["mlx"]["capture_sha256"] = "7" * 64
    report_path.write_text(json.dumps(report))
    kwargs["expected_report_sha256"] = _sha256(report_path)
    with pytest.raises(ValueError, match="MLX capture"):
        _load_accepted_suffix(result_path, report_path, **kwargs)


def test_source_control_guard_rejects_before_later_candidates():
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        require_exact_source_control,
    )

    require_exact_source_control(
        candidate_index=0, metrics={"exact": True, "nonzero": 0}
    )
    with pytest.raises(ValueError, match="source-native control"):
        require_exact_source_control(
            candidate_index=0, metrics={"exact": False, "nonzero": 1}
        )
    require_exact_source_control(
        candidate_index=1, metrics={"exact": False, "nonzero": 1}
    )


def test_control_gated_coordinator_defers_all_intervention_work():
    from scripts.source_cuda_shape_flow_transition0_recoverability import (
        run_control_gated_candidates,
        transition0_candidate_specs,
    )

    specs = transition0_candidate_specs()
    events = []

    def build_starts(names):
        events.append(("build", tuple(names)))
        return (
            {name: f"start:{name}" for name in names},
            {name: f"first:{name}" for name in names},
        )

    def execute_mismatch(index, spec, start, first_step):
        events.append(("execute", index, spec["name"], start, first_step))
        return {
            "name": spec["name"],
            "vs_source_anchor": {"exact": False, "nonzero": 1},
        }

    with pytest.raises(ValueError, match="source-native control"):
        run_control_gated_candidates(
            specs=specs,
            build_starts=build_starts,
            execute_candidate=execute_mismatch,
        )
    assert events == [
        ("build", ("source-native-control",)),
        (
            "execute",
            0,
            "source-native-control",
            "start:source-native-control",
            "first:source-native-control",
        ),
    ]

    events.clear()

    def execute_exact(index, spec, start, first_step):
        events.append(("execute", index, spec["name"], start, first_step))
        return {
            "name": spec["name"],
            "vs_source_anchor": {
                "exact": index == 0,
                "nonzero": 0 if index == 0 else 1,
            },
        }

    results = run_control_gated_candidates(
        specs=specs,
        build_starts=build_starts,
        execute_candidate=execute_exact,
    )
    assert [result["name"] for result in results] == [
        spec["name"] for spec in specs
    ]
    assert events[0] == ("build", ("source-native-control",))
    assert events[1][0:3] == ("execute", 0, "source-native-control")
    assert events[2] == ("build", tuple(spec["name"] for spec in specs[1:]))
    assert [event[2] for event in events[3:]] == [
        spec["name"] for spec in specs[1:]
    ]


def _matrix_cli_args(tmp_path, *, expected_overrides=None):
    expected_overrides = expected_overrides or {}
    paths = {
        "mlx_steps": tmp_path / "steps.npz",
        "mlx_report": tmp_path / "mlx-report.json",
        "conditioning": tmp_path / "conditioning.npz",
        "source_baseline": tmp_path / "source-baseline.npz",
        "source_report": tmp_path / "source-report.json",
        "suffix_result": tmp_path / "suffix-result.npz",
        "suffix_report": tmp_path / "suffix-report.json",
        "source_tar": tmp_path / "source.tar",
    }
    for path in paths.values():
        path.write_bytes(b"input:" + path.name.encode())
    expected = {name: _sha256(path) for name, path in paths.items()}
    expected.update(expected_overrides)
    args = [
        "--mlx-shape-flow-steps", str(paths["mlx_steps"]),
        "--mlx-shape-flow-steps-sha256", expected["mlx_steps"],
        "--mlx-run-report", str(paths["mlx_report"]),
        "--mlx-run-report-sha256", expected["mlx_report"],
        "--conditioning", str(paths["conditioning"]),
        "--conditioning-sha256", expected["conditioning"],
        "--accepted-source-baseline", str(paths["source_baseline"]),
        "--accepted-source-baseline-sha256", expected["source_baseline"],
        "--accepted-source-report", str(paths["source_report"]),
        "--accepted-source-report-sha256", expected["source_report"],
        "--accepted-suffix-result", str(paths["suffix_result"]),
        "--accepted-suffix-result-sha256", expected["suffix_result"],
        "--accepted-suffix-report", str(paths["suffix_report"]),
        "--accepted-suffix-report-sha256", expected["suffix_report"],
        "--source-tar", str(paths["source_tar"]),
        "--source-tar-sha256", expected["source_tar"],
        "--no-download",
    ]
    return args, paths


def test_cli_missing_inputs_preserves_stale_primary_and_writes_failure_report(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import main

    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    output_npz.write_bytes(b"stale")
    status = main(
        [
            "--output-json",
            str(output_json),
            "--output-npz",
            str(output_npz),
            "--mlx-shape-flow-steps",
            str(tmp_path / "missing-steps.npz"),
            "--mlx-shape-flow-steps-sha256",
            "0" * 64,
            "--mlx-run-report",
            str(tmp_path / "missing-run.json"),
            "--mlx-run-report-sha256",
            "0" * 64,
            "--conditioning",
            str(tmp_path / "missing-conditioning.npz"),
            "--conditioning-sha256",
            "0" * 64,
            "--accepted-source-baseline",
            str(tmp_path / "missing-baseline.npz"),
            "--accepted-source-baseline-sha256",
            "0" * 64,
            "--accepted-source-report",
            str(tmp_path / "missing-source.json"),
            "--accepted-source-report-sha256",
            "0" * 64,
            "--accepted-suffix-result",
            str(tmp_path / "missing-suffix.npz"),
            "--accepted-suffix-result-sha256",
            "0" * 64,
            "--accepted-suffix-report",
            str(tmp_path / "missing-suffix.json"),
            "--accepted-suffix-report-sha256",
            "0" * 64,
            "--source-tar",
            str(tmp_path / "missing-source.tar"),
            "--source-tar-sha256",
            "0" * 64,
            "--no-download",
        ]
    )

    assert status == 1
    report = json.loads(output_json.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "request_validation"
    assert report["last_trustworthy_phase"] == "arguments_parsed"
    assert report["primary_output_status"] == "preexisting_untrusted_preserved"
    assert output_npz.read_bytes() == b"stale"


def test_cli_rejects_substituted_input_digest_before_stale_output_mutation(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import main

    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    output_npz.write_bytes(b"stale")
    args, _ = _matrix_cli_args(
        tmp_path,
        expected_overrides={"suffix_result": "0" * 64},
    )
    status = main(
        [
            "--output-json", str(output_json),
            "--output-npz", str(output_npz),
            *args,
        ]
    )

    assert status == 1
    report = json.loads(output_json.read_text())
    assert report["failure_phase"] == "request_validation"
    assert "accepted suffix result SHA256 mismatch" in report["error"]
    assert output_npz.read_bytes() == b"stale"


def test_cli_partial_external_request_fails_before_stale_output_mutation(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import main

    output_json = tmp_path / "result.json"
    output_npz = tmp_path / "result.npz"
    output_npz.write_bytes(b"stale")
    args, _ = _matrix_cli_args(tmp_path)
    status = main(
        [
            "--output-json",
            str(output_json),
            "--output-npz",
            str(output_npz),
            *args,
            "--external-transition0-step",
            str(tmp_path / "external-step.npz"),
        ]
    )

    assert status == 1
    report = json.loads(output_json.read_text())
    assert report["failure_phase"] == "request_validation"
    assert "must be supplied together" in report["error"]
    assert report["primary_output_status"] == "preexisting_untrusted_preserved"
    assert output_npz.read_bytes() == b"stale"


def test_cli_report_collision_preserves_input_and_writes_safe_failure_report(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import main

    output_npz = tmp_path / "result.npz"
    args, paths = _matrix_cli_args(tmp_path)
    source_report_bytes = paths["source_report"].read_bytes()
    status = main(
        [
            "--output-json", str(paths["source_report"]),
            "--output-npz", str(output_npz),
            *args,
        ]
    )

    fallback = paths["source_report"].with_name(
        paths["source_report"].name
        + ".transition0-recoverability.failure.json"
    )
    assert status == 1
    assert paths["source_report"].read_bytes() == source_report_bytes
    report = json.loads(fallback.read_text())
    assert report["failure_phase"] == "request_validation"
    assert report["requested_output_json"] == str(paths["source_report"])
    assert report["effective_failure_report"] == str(fallback)
    assert "collides" in report["error"]


def test_cli_report_collision_precedes_partial_external_request_error(tmp_path):
    from scripts.source_cuda_shape_flow_transition0_recoverability import main

    output_npz = tmp_path / "result.npz"
    output_npz.write_bytes(b"stale")
    args, paths = _matrix_cli_args(tmp_path)
    source_report_bytes = paths["source_report"].read_bytes()
    status = main(
        [
            "--output-json",
            str(paths["source_report"]),
            "--output-npz",
            str(output_npz),
            *args,
            "--external-transition0-step",
            str(tmp_path / "external-step.npz"),
        ]
    )

    fallback = paths["source_report"].with_name(
        paths["source_report"].name
        + ".transition0-recoverability.failure.json"
    )
    assert status == 1
    assert paths["source_report"].read_bytes() == source_report_bytes
    assert output_npz.read_bytes() == b"stale"
    report = json.loads(fallback.read_text())
    assert report["failure_phase"] == "request_validation"
    assert report["primary_output_status"] == "preexisting_untrusted_preserved"
    assert "collides with protected paths: accepted source report" in report["error"]
