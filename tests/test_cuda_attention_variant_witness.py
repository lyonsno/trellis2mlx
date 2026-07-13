import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts import cuda_sparse_attention_witness as witness


def test_variant_specs_preserve_source_chunk_and_split_backend_precision():
    specs = witness.build_variant_specs(source_chunk_size=4096, manual_chunk_size=128)

    assert specs == [
        witness.AttentionVariant("source_default_chunk4096", "default", 4096, "input"),
        witness.AttentionVariant("default_full", "default", None, "input"),
        witness.AttentionVariant("default_chunk512", "default", 512, "input"),
        witness.AttentionVariant("math_chunk4096", "math", 4096, "input"),
        witness.AttentionVariant("math_chunk512", "math", 512, "input"),
        witness.AttentionVariant("manual_fp32_chunk128", "manual", 128, "float32"),
    ]


def test_load_witness_rejects_reference_with_different_element_count(tmp_path):
    path = tmp_path / "witness.npz"
    q = np.zeros((1, 2, 3, 4), dtype=np.float32)
    np.savez(
        path,
        pos_q=q,
        pos_k=q,
        pos_v=q,
        pos_reference_attention_raw=np.zeros((1, 2, 11), dtype=np.float32),
        pos_source_chunked_attention_raw=np.zeros((1, 2, 12), dtype=np.float32),
        route_identity_json=np.array(json.dumps({"branch": "pos"})),
    )

    try:
        witness.load_witness(path)
    except ValueError as exc:
        assert "reference_attention_raw" in str(exc)
        assert "element count" in str(exc)
    else:
        raise AssertionError("expected mismatched reference attention to fail")


def test_cli_failure_writes_report_before_primary_npz(tmp_path):
    script = Path(witness.__file__)
    report = tmp_path / "report.json"
    output = tmp_path / "result.npz"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
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
    assert payload["primary_output_status"] == "missing"
    assert not output.exists()
