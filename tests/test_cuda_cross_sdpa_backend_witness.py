import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _write_witness(path: Path, *, source_tokens: int = 5) -> None:
    q = np.zeros((1, 7, 2, 4), dtype=np.float32)
    k = np.zeros((1, source_tokens, 2, 4), dtype=np.float32)
    v = np.zeros_like(k)
    reference = np.zeros_like(q)
    np.savez_compressed(
        path,
        q=q,
        k=k,
        v=v,
        reference_attention_raw=reference,
        route_identity_json=np.asarray(
            json.dumps({"effective_route": "official-source-cuda-sdpa"})
        ),
    )


def test_load_witness_accepts_cross_attention_geometry(tmp_path):
    from scripts.cuda_cross_sdpa_backend_witness import load_witness

    path = tmp_path / "witness.npz"
    _write_witness(path)

    payload = load_witness(path)

    assert payload["q"].shape == (1, 7, 2, 4)
    assert payload["k"].shape == (1, 5, 2, 4)
    assert payload["v"].shape == (1, 5, 2, 4)
    assert payload["reference_attention_raw"].shape == (1, 7, 2, 4)


def test_load_witness_rejects_reference_shape_substitution(tmp_path):
    from scripts.cuda_cross_sdpa_backend_witness import load_witness

    path = tmp_path / "witness.npz"
    _write_witness(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["reference_attention_raw"] = np.zeros((1, 5, 2, 4), dtype=np.float32)
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="reference_attention_raw"):
        load_witness(path)


def test_backend_specs_include_default_and_each_forced_cuda_family():
    from scripts.cuda_cross_sdpa_backend_witness import backend_specs

    assert backend_specs() == (
        ("default", None),
        ("math", "MATH"),
        ("efficient_attention", "EFFICIENT_ATTENTION"),
        ("flash_attention", "FLASH_ATTENTION"),
        ("cudnn_attention", "CUDNN_ATTENTION"),
    )


def test_default_backend_identification_requires_unique_matching_forced_variant():
    from scripts.cuda_cross_sdpa_backend_witness import (
        identify_default_backend,
    )

    exact = {"exact": True}
    default = {
        "name": "default",
        "status": "done",
        "profiler_events": [
            "aten::_scaled_dot_product_attention_math",
            "aten::scaled_dot_product_attention",
        ],
        "vs_source_reference": exact,
    }
    math = {
        "name": "math",
        "status": "done",
        "profiler_events": ["aten::_scaled_dot_product_attention_math"],
        "vs_source_reference": exact,
    }
    unavailable = [
        {"name": "efficient_attention", "status": "unavailable"},
        {"name": "flash_attention", "status": "unavailable"},
        {"name": "cudnn_attention", "status": "unavailable"},
    ]

    assert identify_default_backend([default, math, *unavailable]) == {
        "default_backend_effective": "math",
        "profiler_events": default["profiler_events"],
        "matching_forced_variant": "math",
        "matching_forced_variant_exact": True,
    }

    with pytest.raises(ValueError, match="no concrete CUDA SDPA backend"):
        identify_default_backend(
            [{**default, "profiler_events": ["aten::scaled_dot_product_attention"]}, math]
        )
    with pytest.raises(ValueError, match="multiple CUDA SDPA backends"):
        identify_default_backend(
            [
                {
                    **default,
                    "profiler_events": [
                        "aten::_scaled_dot_product_attention_math",
                        "aten::_scaled_dot_product_flash_attention",
                    ],
                },
                math,
                {
                    "name": "flash_attention",
                    "status": "done",
                    "vs_source_reference": exact,
                },
            ]
        )
    with pytest.raises(ValueError, match="matching forced backend"):
        identify_default_backend(
            [
                default,
                {
                    **math,
                    "vs_source_reference": {"exact": False},
                },
            ]
        )


def test_discover_split_traces_requires_one_source_and_one_kv_artifact(tmp_path):
    from scripts.cuda_cross_sdpa_backend_witness import discover_split_traces

    source_dir = tmp_path / "source"
    kv_dir = tmp_path / "kv"
    source_dir.mkdir()
    kv_dir.mkdir()
    np.savez_compressed(
        source_dir / "cuda_result.npz",
        pos_block0_cross_q_post_norm=np.zeros((1, 7, 2, 4), dtype=np.float32),
        pos_block0_cross_attention_raw=np.zeros((1, 7, 2, 4), dtype=np.float32),
    )
    np.savez_compressed(
        kv_dir / "cuda_result.npz",
        pos_block0_cross_k_pre_norm=np.zeros(
            (1, 1, 5, 2, 4), dtype=np.float32
        ),
        pos_block0_cross_v=np.zeros((1, 1, 5, 2, 4), dtype=np.float32),
        pos_block0_cross_k_post_norm=np.zeros(
            (1, 1, 5, 2, 4), dtype=np.float32
        ),
    )

    assert discover_split_traces(tmp_path) == (
        source_dir / "cuda_result.npz",
        kv_dir / "cuda_result.npz",
    )

    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    np.savez_compressed(
        duplicate / "cuda_result.npz",
        pos_block0_cross_q_post_norm=np.zeros((1, 7, 2, 4), dtype=np.float32),
        pos_block0_cross_attention_raw=np.zeros((1, 7, 2, 4), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="exactly one source trace"):
        discover_split_traces(tmp_path)


def test_load_split_witness_preserves_distinct_route_provenance(tmp_path):
    from scripts.cuda_cross_sdpa_backend_witness import load_split_witness

    source = tmp_path / "source.npz"
    kv = tmp_path / "kv.npz"
    np.savez_compressed(
        source,
        pos_block0_cross_q_post_norm=np.zeros((1, 7, 2, 4), dtype=np.float32),
        pos_block0_cross_attention_raw=np.zeros((1, 7, 2, 4), dtype=np.float32),
        route_identity_json=np.asarray(json.dumps({"route": "full-source"})),
    )
    np.savez_compressed(
        kv,
        pos_block0_cross_k_pre_norm=np.zeros(
            (1, 1, 5, 2, 4), dtype=np.float32
        ),
        pos_block0_cross_v=np.zeros((1, 1, 5, 2, 4), dtype=np.float32),
        pos_block0_cross_k_post_norm=np.zeros(
            (1, 1, 5, 2, 4), dtype=np.float32
        ),
        route_identity_json=np.asarray(json.dumps({"route": "narrow-kv"})),
    )

    payload = load_split_witness(source, kv)

    assert payload["q"].shape == (1, 7, 2, 4)
    assert payload["k"].shape == (1, 5, 2, 4)
    assert payload["v"].shape == (1, 5, 2, 4)
    assert payload["route_identity"] == {
        "source_trace": {"route": "full-source"},
        "kv_trace": {"route": "narrow-kv"},
    }
    assert set(payload["input_artifacts"]) == {"source_trace", "kv_trace"}


def test_cli_failure_writes_report_before_primary_npz(tmp_path):
    from scripts import cuda_cross_sdpa_backend_witness as witness

    report = tmp_path / "report.json"
    output = tmp_path / "output.npz"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(witness.__file__)),
            "--witness",
            str(tmp_path / "missing.npz"),
            "--output-json",
            str(report),
            "--output-npz",
            str(output),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert report.exists()
    payload = json.loads(report.read_text())
    assert payload["status"] == "failed"
    assert payload["failure_phase"] == "input_validation"
    assert payload["last_trustworthy_phase"] == "request_received"
    assert payload["primary_output_status"] == "missing"
    assert not output.exists()


def test_cli_failure_marks_preexisting_primary_as_stale(tmp_path):
    from scripts import cuda_cross_sdpa_backend_witness as witness

    report = tmp_path / "report.json"
    output = tmp_path / "output.npz"
    output.write_bytes(b"stale-output")
    stale_sha256 = witness.sha256_file(output)

    result = subprocess.run(
        [
            sys.executable,
            str(Path(witness.__file__)),
            "--witness",
            str(tmp_path / "missing.npz"),
            "--output-json",
            str(report),
            "--output-npz",
            str(output),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    payload = json.loads(report.read_text())
    assert payload["failure_phase"] == "input_validation"
    assert payload["primary_output_status"] == "stale-preexisting"
    assert payload["primary_output"] == {
        "path": str(output),
        "sha256": stale_sha256,
    }
    assert output.read_bytes() == b"stale-output"
