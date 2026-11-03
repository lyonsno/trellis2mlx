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
    tmp_path.mkdir(parents=True, exist_ok=True)
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
    [
        "cpu",
        "attention",
        "source",
        "precision",
        "runtime",
        "resolution",
        "xformers_cuda_build",
        "xformers_smoke",
    ],
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
        xformers_built_with_cuda=True,
        xformers_fp16_block_diagonal_smoke=True,
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
    elif mutation == "xformers_cuda_build":
        route["xformers_built_with_cuda"] = False
    elif mutation == "xformers_smoke":
        route["xformers_fp16_block_diagonal_smoke"] = False
    with pytest.raises(ValueError):
        mod.validate_route(route, "cuda")


def test_xformers_pip_install_argv_selects_matching_cuda_wheel_index():
    mod = load_module()
    argv_130 = mod.xformers_pip_install_argv("2.11.0+cu130", "/usr/bin/python3")
    assert argv_130 == [
        "/usr/bin/python3",
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--index-url",
        "https://download.pytorch.org/whl/cu130",
        "xformers==0.0.35",
    ]
    argv_128 = mod.xformers_pip_install_argv("2.11.0+cu128", "/usr/bin/python3")
    assert argv_128 == [
        "/usr/bin/python3",
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--index-url",
        "https://download.pytorch.org/whl/cu128",
        "xformers==0.0.35",
    ]
    with pytest.raises(ValueError, match="CUDA tag"):
        mod.xformers_pip_install_argv("2.11.0+cpu", "/usr/bin/python3")


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
    rev = "c70b27f4da56f60c7919e3ba40dbe191bd974a7a"
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
        native_revision=rev,
        native_revision_requested=rev,
        native_revision_observed=rev,
        native_tree_clean=True,
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
    with pytest.raises(ValueError, match="native revision"):
        mod.validate_route(
            dict(route, native_revision=None), "mlx", native_profile=profile
        )
    with pytest.raises(ValueError, match="native revision"):
        mod.validate_route(
            dict(route, native_revision_observed="0" * 40),
            "mlx",
            native_profile=profile,
        )
    with pytest.raises(ValueError, match="clean"):
        mod.validate_route(
            dict(route, native_tree_clean=False), "mlx", native_profile=profile
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
    rev = "c70b27f4da56f60c7919e3ba40dbe191bd974a7a"

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
            native_revision=rev,
            native_revision_requested=rev,
            native_revision_observed=rev,
            native_tree_clean=True,
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
            "--native-revision",
            rev,
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


def test_cli_output_path_collisions_preserve_protected_bytes_and_write_safe_refusal(
    tmp_path: Path,
):
    import subprocess
    import sys

    mod = load_module()
    inp, _, _ = make_valid_inputs(tmp_path / "case1")
    inp_bytes = inp.read_bytes()
    inp_sha = mod.digest(inp)

    # Case 1: --output-json equals --inputs
    proc1 = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--backend",
            "cuda",
            "--run-id",
            "collision-inputs-json",
            "--inputs",
            str(inp),
            "--inputs-sha256",
            inp_sha,
            "--output-json",
            str(inp),
            "--output-npz",
            str(tmp_path / "case1/out.npz"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc1.returncode == 1
    assert inp.read_bytes() == inp_bytes
    refusal1 = tmp_path / "case1/inputs.npz.refusal-1.json"
    assert refusal1.is_file()
    r1 = json.loads(refusal1.read_text())
    assert r1["status"] == "failed" and r1["failure_phase"] == "output_path_admission"

    # Case 2: --output-json and --output-npz both name an existing NPZ
    inp2, _, _ = make_valid_inputs(tmp_path / "case2")
    existing_npz = tmp_path / "case2/existing.npz"
    existing_npz.write_bytes(b"protected-npz-bytes")
    proc2 = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--backend",
            "cuda",
            "--run-id",
            "collision-both-existing-npz",
            "--inputs",
            str(inp2),
            "--inputs-sha256",
            mod.digest(inp2),
            "--output-json",
            str(existing_npz),
            "--output-npz",
            str(existing_npz),
        ],
        capture_output=True,
        text=True,
    )
    assert proc2.returncode == 1
    assert existing_npz.read_bytes() == b"protected-npz-bytes"
    refusal2 = tmp_path / "case2/existing.npz.refusal-1.json"
    assert refusal2.is_file()
    r2 = json.loads(refusal2.read_text())
    assert r2["status"] == "failed" and r2["failure_phase"] == "output_path_admission"

    # Case 3: Earlier report and NPZ already exist at their separate output paths
    inp3, _, _ = make_valid_inputs(tmp_path / "case3")
    existing_rep3 = tmp_path / "case3/report.json"
    existing_npz3 = tmp_path / "case3/output.npz"
    existing_rep3.write_bytes(b'{"status":"completed"}\n')
    existing_npz3.write_bytes(b"earlier-npz-bytes")
    proc3 = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--backend",
            "cuda",
            "--run-id",
            "collision-preexisting-report-and-npz",
            "--inputs",
            str(inp3),
            "--inputs-sha256",
            mod.digest(inp3),
            "--output-json",
            str(existing_rep3),
            "--output-npz",
            str(existing_npz3),
        ],
        capture_output=True,
        text=True,
    )
    assert proc3.returncode == 1
    assert existing_rep3.read_bytes() == b'{"status":"completed"}\n'
    assert existing_npz3.read_bytes() == b"earlier-npz-bytes"
    refusal3 = tmp_path / "case3/report.json.refusal-1.json"
    assert refusal3.is_file()
    r3 = json.loads(refusal3.read_text())
    assert r3["status"] == "failed" and r3["failure_phase"] == "output_path_admission"

    # Case 4: Valid input occupies the report's .tmp pathname
    inp4_src, _, _ = make_valid_inputs(tmp_path / "case4")
    rep4 = tmp_path / "case4/report.json"
    inp4_tmp = tmp_path / "case4/report.json.tmp"
    inp4_src.rename(inp4_tmp)
    inp4_bytes = inp4_tmp.read_bytes()
    proc4 = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--backend",
            "cuda",
            "--run-id",
            "collision-input-at-report-tmp",
            "--inputs",
            str(inp4_tmp),
            "--inputs-sha256",
            mod.digest(inp4_tmp),
            "--output-json",
            str(rep4),
            "--output-npz",
            str(tmp_path / "case4/output.npz"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc4.returncode == 1
    assert inp4_tmp.is_file() and inp4_tmp.read_bytes() == inp4_bytes
    assert not rep4.exists()
    refusal4 = tmp_path / "case4/report.json.refusal-1.json"
    assert refusal4.is_file()
    r4 = json.loads(refusal4.read_text())
    assert r4["status"] == "failed" and r4["failure_phase"] == "output_path_admission"


def test_preparer_preserves_existing_evidence_and_inputs_on_collision(
    tmp_path: Path, monkeypatch
):
    prep = load_module(PREPARE_SCRIPT, "prepare_shape_first_prediction")
    stage_dir = tmp_path / "staged"
    stage_dir.mkdir(parents=True)
    existing_report = stage_dir / "prepare-report.json"
    existing_capsule = stage_dir / "capsule"
    existing_capsule.mkdir()
    (existing_capsule / "inputs.npz").write_bytes(b"existing-capsule-inputs")
    existing_report.write_bytes(b'{"status":"completed","phase":"packet_prepared"}\n')
    monkeypatch.setattr(
        prep,
        "verify_retained_inputs",
        lambda *a, **kw: pytest.fail("must reject preexisting stage before reading"),
    )
    rc = prep.main(["--output-dir", str(stage_dir)])
    assert rc == 1
    assert (
        existing_report.read_bytes()
        == b'{"status":"completed","phase":"packet_prepared"}\n'
    )
    assert (existing_capsule / "inputs.npz").read_bytes() == b"existing-capsule-inputs"
    refusal = stage_dir / "prepare-report.json.refusal-1.json"
    assert refusal.is_file()
    r = json.loads(refusal.read_text())
    assert r["status"] == "failed" and r["failure_phase"] == "output_path_admission"


def test_mlx_native_revision_required_and_verified_before_imports(tmp_path: Path):
    import subprocess

    mod = load_module()
    inp, _, _ = make_valid_inputs(tmp_path / "inputs")
    inp_sha = mod.digest(inp)

    repo = tmp_path / "fake_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    tracked = repo / "tracked.py"
    tracked.write_text("x = 1\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()

    # 1. Missing --native-revision rejected before native imports
    rep_missing = tmp_path / "rep_missing.json"
    rc_missing = mod.main(
        [
            "--backend",
            "mlx",
            "--run-id",
            "missing-rev",
            "--source-root",
            str(repo),
            "--inputs",
            str(inp),
            "--inputs-sha256",
            inp_sha,
            "--output-json",
            str(rep_missing),
            "--output-npz",
            str(tmp_path / "out_missing.npz"),
        ]
    )
    r_missing = json.loads(rep_missing.read_text())
    assert rc_missing == 1 and r_missing["status"] == "failed"
    assert "native-revision" in r_missing["error"]

    # 2. Mismatched --native-revision rejected and records observed revision
    rep_mismatch = tmp_path / "rep_mismatch.json"
    rc_mismatch = mod.main(
        [
            "--backend",
            "mlx",
            "--run-id",
            "mismatch-rev",
            "--source-root",
            str(repo),
            "--native-revision",
            "0" * 40,
            "--inputs",
            str(inp),
            "--inputs-sha256",
            inp_sha,
            "--output-json",
            str(rep_mismatch),
            "--output-npz",
            str(tmp_path / "out_mismatch.npz"),
        ]
    )
    r_mismatch = json.loads(rep_mismatch.read_text())
    assert rc_mismatch == 1 and r_mismatch["status"] == "failed"
    assert r_mismatch["native_source"]["requested_revision"] == "0" * 40
    assert r_mismatch["native_source"]["observed_revision"] == head
    assert "native revision differs" in r_mismatch["error"]

    # 3. Dirty tracked source tree rejected even when revision matches HEAD
    tracked.write_text("x = 2\n")
    rep_dirty = tmp_path / "rep_dirty.json"
    rc_dirty = mod.main(
        [
            "--backend",
            "mlx",
            "--run-id",
            "dirty-rev",
            "--source-root",
            str(repo),
            "--native-revision",
            head,
            "--inputs",
            str(inp),
            "--inputs-sha256",
            inp_sha,
            "--output-json",
            str(rep_dirty),
            "--output-npz",
            str(tmp_path / "out_dirty.npz"),
        ]
    )
    r_dirty = json.loads(rep_dirty.read_text())
    assert rc_dirty == 1 and r_dirty["status"] == "failed"
    assert r_dirty["native_source"]["requested_revision"] == head
    assert r_dirty["native_source"]["observed_revision"] == head
    assert r_dirty["native_source"]["tree_clean"] is False
    assert "tracked source" in r_dirty["error"]


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


def test_compute_cuda_step1_euler_matches_standardized_step1_euler():
    mod = load_module()
    coords = np.array([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=np.int32)
    sample_in = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(2, 32)
    pred_pos = np.linspace(0.1, 0.9, 64, dtype=np.float32).reshape(2, 32)
    pred_neg = np.linspace(-0.2, 0.4, 64, dtype=np.float32).reshape(2, 32)

    class FakeTensor:
        def __init__(self, arr: np.ndarray):
            self._arr = np.asarray(arr, dtype=np.float32)

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

        def __truediv__(self, other):
            return FakeTensor(np.float32(self._arr / other._arr))

    class FakeSparse:
        ndim = 2

        def __init__(self, feats: np.ndarray):
            self._feats = np.ascontiguousarray(feats, dtype=np.float32)
            self.feats = FakeTensor(self._feats)

        def __mul__(self, other):
            val = other._arr if isinstance(other, FakeTensor) else np.float32(other)
            return FakeSparse((self._feats * val).astype(np.float32))

        def __rmul__(self, other):
            return self.__mul__(other)

        def __add__(self, other):
            return FakeSparse((self._feats + other._feats).astype(np.float32))

        def __sub__(self, other):
            return FakeSparse((self._feats - other._feats).astype(np.float32))

        def __truediv__(self, other):
            return FakeSparse((self._feats / np.float32(other)).astype(np.float32))

        def std(self, dim=None, keepdim=False):
            return FakeTensor(mod.sparse_cfg_rescale_std_fp32(self._feats))

    cuda_bundle = mod.compute_cuda_step1_euler(
        FakeSparse(sample_in),
        FakeSparse(pred_pos),
        FakeSparse(pred_neg),
        coords,
    )
    std_bundle = mod.compute_step1_euler(sample_in, pred_pos, pred_neg, coords)
    cmp = mod.compare_bundles(cuda_bundle, std_bundle)
    assert all(m["exact"] for m in cmp["matrices"].values())
    assert all(s["exact"] for s in cmp["scalars"].values())


def test_occupied_file_ancestor_rejected_with_durable_refusal_report(tmp_path: Path):
    import subprocess
    import sys

    mod = load_module()
    inp, _, _ = make_valid_inputs(tmp_path / "inputs")
    inp_sha = mod.digest(inp)

    # 1. Witness CLI: immediate parent and nested ancestor occupied by a regular file
    w_dir = tmp_path / "parent-file"
    w_dir.mkdir(parents=True)
    occupied_w = w_dir / "occupied"
    occupied_w.write_bytes(b"occupied-witness-ancestor")

    for idx, rel_report in enumerate(("occupied/report.json", "occupied/nested/report.json"), start=1):
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--backend",
                "cuda",
                "--run-id",
                f"ancestor-file-{idx}",
                "--inputs",
                str(inp),
                "--inputs-sha256",
                inp_sha,
                "--output-json",
                str(w_dir / rel_report),
                "--output-npz",
                str(w_dir / f"output-{idx}.npz"),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 1
        assert occupied_w.is_file() and occupied_w.read_bytes() == b"occupied-witness-ancestor"
        refusal = w_dir / f"report.json.refusal-{idx}.json"
        assert refusal.is_file(), f"missing refusal report {refusal}; stderr={proc.stderr}"
        rep = json.loads(refusal.read_text())
        assert rep["status"] == "failed"
        assert rep["failure_phase"] == "output_path_admission"
        assert rep["last_trustworthy_phase"] == "profile_admission"
        assert rep.get("commands", []) == []

    # 2. Preparer CLI: ancestor of --output-dir occupied by a regular file
    p_dir = tmp_path / "preparer-parent-file"
    p_dir.mkdir(parents=True)
    occupied_p = p_dir / "occupied"
    occupied_p.write_bytes(b"occupied-preparer-ancestor")
    proc_p = subprocess.run(
        [
            sys.executable,
            str(PREPARE_SCRIPT),
            "--output-dir",
            str(occupied_p / "new-stage"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc_p.returncode == 1
    assert occupied_p.is_file() and occupied_p.read_bytes() == b"occupied-preparer-ancestor"
    refusal_p = p_dir / "prepare-report.json.refusal-1.json"
    assert refusal_p.is_file(), f"missing preparer refusal {refusal_p}; stderr={proc_p.stderr}"
    rep_p = json.loads(refusal_p.read_text())
    assert rep_p["status"] == "failed"
    assert rep_p["failure_phase"] == "output_path_admission"
    assert rep_p["last_trustworthy_phase"] is None
