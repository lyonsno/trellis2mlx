"""Fail-first unit contracts for 1024 HR shape-flow first-prediction comparison."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/shape_first_prediction.py"
PREPARE_SCRIPT = ROOT / "scripts/prepare_shape_first_prediction.py"


def load_module(path: Path = SCRIPT, name: str = "shape_first_prediction"):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"cannot load module spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_valid_inputs(tmp_path: Path, rows: int = 2) -> tuple[Path, np.ndarray, np.ndarray]:
    coords = np.array([[0, i + 1, i + 2, i + 3] for i in range(rows)], dtype=np.int32)
    noise = np.linspace(-0.5, 0.5, rows * 32, dtype=np.float32).reshape(rows, 32)
    cond = np.ones((1, 4101, 1024), dtype=np.float32) * np.float32(0.125)
    neg_cond = np.zeros((1, 4101, 1024), dtype=np.float32)
    t = np.array([1000.0], dtype=np.float32)
    t_prev = np.array([np.float32(33.0 / 34.0)], dtype=np.float32)
    inp = tmp_path / "inputs.npz"
    np.savez(
        inp,
        noise=noise,
        cond=cond,
        neg_cond=neg_cond,
        coords=coords,
        t=t,
        t_prev=t_prev,
    )
    return inp, coords, noise


@pytest.mark.parametrize(
    "kind",
    ["partial", "reordered", "wrong_channels", "nonfinite", "empty", "wrong_dtype"],
)
def test_complete_prediction_support_is_required(kind: str):
    mod = load_module()
    coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    pred = np.ones((2, 32), dtype=np.float32)
    got = coords.copy()
    if kind == "partial":
        pred = pred[:1]
        got = got[:1]
    elif kind == "reordered":
        got = got[::-1]
    elif kind == "wrong_channels":
        pred = pred[:, :31]
    elif kind == "nonfinite":
        pred[1, 2] = np.nan
    elif kind == "empty":
        pred = pred[:0]
        got = got[:0]
    elif kind == "wrong_dtype":
        pred = pred.astype(np.float16)
    with pytest.raises(ValueError):
        mod.validate_output(pred, got, coords)


def test_step1_euler_bundle_validates_and_Rejects_branch_corruption():
    mod = load_module()
    coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    sample_in = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(2, 32)
    pred_pos = np.linspace(0.1, 0.9, 64, dtype=np.float32).reshape(2, 32)
    pred_neg = np.linspace(-0.2, 0.4, 64, dtype=np.float32).reshape(2, 32)
    bundle = mod.compute_step1_euler(sample_in, pred_pos, pred_neg, coords)
    assert mod.validate_step1_bundle(bundle, coords, sample_in) is None

    bad_neg = dict(bundle, pred_neg=bundle["pred_neg"].copy())
    bad_neg["pred_neg"][0, 0] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        mod.validate_step1_bundle(bad_neg, coords, sample_in)

    bad_sample = dict(bundle, sample_in=bundle["sample_in"] + np.float32(1e-3))
    with pytest.raises(ValueError, match="sample_in"):
        mod.validate_step1_bundle(bad_sample, coords, sample_in)


@pytest.mark.parametrize(
    "mutation",
    ["cpu", "attention", "source", "precision", "runtime", "resolution"],
)
def test_cuda_route_cannot_silently_fallback(mutation: str):
    mod = load_module()
    route = dict(
        cuda_available=True,
        attention="xformers",
        sparse_container="none",
        trellis=mod.TRELLIS_REVISION,
        torch="2.11.0+cu130",
        xformers="0.0.35",
        torso_dtype="float16",
        hr_resolution=64,
    )
    assert mod.validate_route(route, "cuda") is None
    assert mod.validate_route(dict(route, torch="2.11.0+cu128"), "cuda") is None
    if mutation == "cpu":
        route["cuda_available"] = False
    elif mutation == "attention":
        route["attention"] = "sdpa"
    elif mutation == "source":
        route["trellis"] = "wrong"
    elif mutation == "precision":
        route["torso_dtype"] = "bfloat16"
    elif mutation == "runtime":
        route["torch"] = "2.10.0+cu128"
    elif mutation == "resolution":
        route["hr_resolution"] = 32
    with pytest.raises(ValueError):
        mod.validate_route(route, "cuda")


@pytest.mark.parametrize(
    "profile,dtype,qk,modulation",
    [
        ("fp16-control", "float16", "mlx-sum", "float32"),
        ("production-bf16", "bfloat16", "source-cuda-warp32", "bfloat16"),
        ("bf16-f32-modulation", "bfloat16", "source-cuda-warp32", "float32"),
        ("bf16-mlx-sum", "bfloat16", "mlx-sum", "bfloat16"),
    ],
)
def test_mlx_route_validates_hr_query_chunk_and_profile(
    profile: str, dtype: str, qk: str, modulation: str
):
    mod = load_module()
    route = dict(
        mlx="0.32.4",
        device="gpu",
        attention="fast",
        query_chunk=1024,
        hr_attention_active=True,
        shape_flow_layernorm="mlx-two-pass",
        qk_norm=qk,
        torso_dtype=dtype,
        learned_modulation_dtype=modulation,
    )
    assert mod.validate_route(route, "mlx", native_profile=profile) is None
    with pytest.raises(ValueError, match="MLX route"):
        mod.validate_route(dict(route, query_chunk=2048), "mlx", native_profile=profile)
    with pytest.raises(ValueError, match="MLX route"):
        mod.validate_route(
            dict(route, hr_attention_active=False), "mlx", native_profile=profile
        )
    with pytest.raises(ValueError, match="precision"):
        mod.validate_route(
            dict(route, torso_dtype="float16" if dtype == "bfloat16" else "bfloat16"),
            "mlx",
            native_profile=profile,
        )


def test_bad_digest_fails_before_backend_with_durable_phase(tmp_path: Path, monkeypatch):
    mod = load_module()
    inp, _, _ = make_valid_inputs(tmp_path)
    monkeypatch.setattr(mod, "run_cuda", lambda *a: pytest.fail("backend must not run"))
    report = tmp_path / "report.json"
    result = mod.main(
        [
            "--backend",
            "cuda",
            "--run-id",
            "unit-negative",
            "--inputs",
            str(inp),
            "--inputs-sha256",
            "0" * 64,
            "--output-json",
            str(report),
            "--output-npz",
            str(tmp_path / "output.npz"),
        ]
    )
    r = json.loads(report.read_text())
    assert result == 1 and r["status"] == "failed"
    assert r["failure_phase"] == "input_admission"
    assert r["last_trustworthy_phase"] == "output_path_admission"
    assert "digest mismatch" in r["error"]
    assert "effective_route" not in r


def test_nonzero_neg_cond_or_bad_coords_rejected_at_input_admission(tmp_path: Path):
    mod = load_module()
    inp, coords, noise = make_valid_inputs(tmp_path)
    np.savez(
        inp,
        noise=noise,
        cond=np.ones((1, 4101, 1024), dtype=np.float32),
        neg_cond=np.ones((1, 4101, 1024), dtype=np.float32),
        coords=coords,
        t=np.array([1000.0], dtype=np.float32),
        t_prev=np.array([np.float32(33.0 / 34.0)], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="neg_cond"):
        mod.load_inputs(inp)


def test_preexisting_output_npz_is_preserved_and_rejected(tmp_path: Path, monkeypatch):
    mod = load_module()
    inp, _, _ = make_valid_inputs(tmp_path)
    monkeypatch.setattr(mod, "run_cuda", lambda *a: pytest.fail("backend must not run"))
    out = tmp_path / "output.npz"
    out.write_bytes(b"preexisting-evidence")
    report = tmp_path / "report.json"
    rc = mod.main(
        [
            "--backend",
            "cuda",
            "--run-id",
            "unit-preexisting",
            "--inputs",
            str(inp),
            "--inputs-sha256",
            mod.digest(inp),
            "--output-json",
            str(report),
            "--output-npz",
            str(out),
        ]
    )
    r = json.loads(report.read_text())
    assert rc == 1 and r["status"] == "failed"
    assert r["failure_phase"] == "output_path_admission"
    assert out.read_bytes() == b"preexisting-evidence"


def test_partial_backend_result_is_retained_but_never_completed(
    tmp_path: Path, monkeypatch
):
    mod = load_module()
    inp, coords, noise = make_valid_inputs(tmp_path, rows=2)

    def broken_backend(args, arrays, report, *unused):
        return {
            "coords": coords[:1],
            "sample_in": noise[:1],
            "pred_pos": np.ones((1, 32), dtype=np.float32),
            "pred_neg": np.zeros((1, 32), dtype=np.float32),
        }

    monkeypatch.setattr(mod, "run_cuda", broken_backend)
    report = tmp_path / "report.json"
    output = tmp_path / "output.npz"
    rc = mod.main(
        [
            "--backend",
            "cuda",
            "--run-id",
            "unit-partial",
            "--inputs",
            str(inp),
            "--inputs-sha256",
            mod.digest(inp),
            "--output-json",
            str(report),
            "--output-npz",
            str(output),
        ]
    )
    r = json.loads(report.read_text())
    assert rc == 1 and r["status"] == "failed" and r["failure_phase"] == "output_capture"
    assert output.is_file() and r["output"]["sha256"] == mod.digest(output)


def test_complete_backend_writes_verified_step1_bundle(tmp_path: Path, monkeypatch):
    mod = load_module()
    inp, coords, noise = make_valid_inputs(tmp_path, rows=2)
    pred_pos = np.full((2, 32), 0.25, dtype=np.float32)
    pred_neg = np.full((2, 32), -0.125, dtype=np.float32)

    def fake_mlx(args, arrays, report, *unused):
        report["effective_route"] = dict(
            mlx="0.32.4",
            device="gpu",
            attention="fast",
            query_chunk=1024,
            hr_attention_active=True,
            shape_flow_layernorm="mlx-two-pass",
            qk_norm="mlx-sum",
            torso_dtype="float16",
            learned_modulation_dtype="float32",
        )
        return mod.compute_step1_euler(noise, pred_pos, pred_neg, coords)

    monkeypatch.setattr(mod, "run_mlx", fake_mlx)
    report = tmp_path / "report.json"
    output = tmp_path / "output.npz"
    rc = mod.main(
        [
            "--backend",
            "mlx",
            "--native-profile",
            "fp16-control",
            "--run-id",
            "unit-complete",
            "--inputs",
            str(inp),
            "--inputs-sha256",
            mod.digest(inp),
            "--output-json",
            str(report),
            "--output-npz",
            str(output),
        ]
    )
    r = json.loads(report.read_text())
    assert rc == 0 and r["status"] == "completed"
    assert r["output"]["sha256"] == mod.digest(output)
    assert r["output"]["rows"] == 2
    with np.load(output, allow_pickle=False) as z:
        assert np.array_equal(z["pred_pos"], pred_pos)
        assert np.array_equal(z["pred_neg"], pred_neg)
        assert z["pred_final"].shape == (2, 32)
        assert z["sample_next"].shape == (2, 32)


def test_prepare_shape_first_prediction_verifies_retained_euler_step1():
    import uuid

    prep = load_module(PREPARE_SCRIPT, "prepare_shape_first_prediction")
    mod = load_module()
    assert str(uuid.UUID(prep.DEFAULT_RUN_ID)) == prep.DEFAULT_RUN_ID
    assert prep.INPUTS["noise"]["sha256"].startswith("20a23609")
    assert prep.INPUTS["conditioning"]["sha256"].startswith("8f7db969")
    assert prep.INPUTS["first_step"]["sha256"].startswith("b659b4d9")
    with np.load(prep.INPUTS["first_step"]["path"], allow_pickle=False) as e1, np.load(
        prep.INPUTS["noise"]["path"], allow_pickle=False
    ) as nz:
        coords4 = np.pad(nz["coords"].astype(np.int32), ((0, 0), (1, 0)))
        recomputed = mod.compute_step1_euler(
            e1["sample_in"], e1["pred_pos"], e1["pred_neg"], coords4
        )
        for key in (
            "pred_cfg",
            "x0_pos",
            "x0_cfg",
            "std_pos",
            "std_cfg",
            "std_ratio",
            "x0_rescaled",
            "x0_after_rescale",
            "pred_final",
            "sample_next",
        ):
            assert np.array_equal(recomputed[key], e1[key]), f"mismatch on {key}"


def test_compare_bundles_reports_exact_and_perturbed_metrics():
    mod = load_module()
    coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    sample_in = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(2, 32)
    pred_pos = np.linspace(0.1, 0.9, 64, dtype=np.float32).reshape(2, 32)
    pred_neg = np.linspace(-0.2, 0.4, 64, dtype=np.float32).reshape(2, 32)
    left = mod.compute_step1_euler(sample_in, pred_pos, pred_neg, coords)
    right = mod.compute_step1_euler(
        sample_in, pred_pos + np.float32(1e-3), pred_neg, coords
    )
    exact_cmp = mod.compare_bundles(left, left)
    assert exact_cmp["matrices"]["pred_pos"]["exact"] is True
    assert exact_cmp["matrices"]["pred_final"]["max_abs"] == 0.0
    pert_cmp = mod.compare_bundles(left, right)
    assert pert_cmp["matrices"]["pred_pos"]["exact"] is False
    assert pert_cmp["matrices"]["pred_pos"]["max_abs"] > 0.0
    assert pert_cmp["matrices"]["pred_neg"]["exact"] is True

