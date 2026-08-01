import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.decoder_full_hash_ledger_contract import (
    decoder_full_hash_entry,
)
from scripts.postprocess_raw_cuda_mesh import write_binary_ply
from trellmlx.mesh_extract import decoder_output_to_mesh


RSQRT_SHA256 = "d" * 64
SILU_SHA256 = "f" * 64
MESH_OVERRIDE_SHA256 = "c" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _route() -> dict:
    return {
        "decoder_linear_backend": "turing_fda",
        "sparse_conv_matmul_backend": "turing_fda",
        "decoder_layernorm": {
            "backend": "cuda-welford-turing-t4",
            "turing_rsqrt_lut_artifact_sha256_attested": RSQRT_SHA256,
        },
        "decoder_silu": {
            "backend": "cuda-turing-t4-fp16-lut",
            "output_lut_artifact_sha256_attested": SILU_SHA256,
            "output_lut_artifact_sha256_effective": SILU_SHA256,
        },
        "decoder_output_head_backend": "mlx-native-fp32",
    }


def _mesh_fixture(tmp_path: Path) -> dict:
    coords = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.int32,
    )
    feats = np.zeros((4, 7), dtype=np.float32)
    feats[0, 3] = 1.0
    checkpoint = tmp_path / "decoder_output.npz"
    np.savez_compressed(
        checkpoint,
        feats=feats,
        coords=coords,
        decoder_route_json=np.array(json.dumps(_route(), sort_keys=True)),
    )

    vertices, faces = decoder_output_to_mesh(
        feats,
        coords,
        resolution=4,
    )
    source_ply = tmp_path / "source.raw.ply"
    write_binary_ply(source_ply, vertices, faces)
    source_report = tmp_path / "source-report.json"
    source_report.write_text(
        json.dumps(
            {
                "effective_route": {},
                "mesh_override": {
                    "sha256": MESH_OVERRIDE_SHA256,
                },
                "mesh_artifacts": [
                    {
                        "path": source_ply.name,
                        "sha256": _sha256(source_ply),
                        "size_bytes": source_ply.stat().st_size,
                        "status": "written",
                        "variant": "raw",
                        "mesh_summary": {
                            "vertices": int(vertices.shape[0]),
                            "faces": int(faces.shape[0]),
                        },
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "checkpoint": checkpoint,
        "feats": feats,
        "coords": coords,
        "source_ply": source_ply,
        "source_report": source_report,
        "output_ply": tmp_path / "local.raw.ply",
        "report_json": tmp_path / "comparison.json",
    }


def _invoke(fixture: dict, **overrides):
    from scripts.compare_decoder_mesh_to_source_ply import (
        compare_decoder_mesh_to_source_ply,
    )

    kwargs = {
        "decoder_checkpoint": fixture["checkpoint"],
        "expected_decoder_checkpoint_sha256": _sha256(
            fixture["checkpoint"]
        ),
        "expected_decoder_feats_sha256": decoder_full_hash_entry(
            "decoder_output",
            fixture["feats"],
        )["sha256"],
        "expected_decoder_coords_sha256": decoder_full_hash_entry(
            "level4_child_coords",
            fixture["coords"],
        )["sha256"],
        "expected_decoder_rsqrt_sha256": RSQRT_SHA256,
        "expected_decoder_silu_sha256": SILU_SHA256,
        "source_ply": fixture["source_ply"],
        "source_report": fixture["source_report"],
        "expected_source_ply_sha256": _sha256(fixture["source_ply"]),
        "expected_source_report_sha256": _sha256(
            fixture["source_report"]
        ),
        "expected_source_mesh_override_sha256": MESH_OVERRIDE_SHA256,
        "resolution": 4,
        "output_ply": fixture["output_ply"],
        "report_json": fixture["report_json"],
    }
    kwargs.update(overrides)
    return compare_decoder_mesh_to_source_ply(**kwargs)


def test_exact_mesh_comparison_records_route_and_reopened_output(tmp_path):
    fixture = _mesh_fixture(tmp_path)

    report = _invoke(fixture)

    assert report["status"] == "done"
    assert report["comparison"] == {
        "mesh_exact": True,
        "topology_exact": True,
        "vertices_exact": True,
    }
    assert report["effective_route"]["decoder_linear_backend"] == "turing_fda"
    assert report["effective_route"]["source_mesh_override_sha256"] == (
        MESH_OVERRIDE_SHA256
    )
    assert report["output"]["reopened_exact"] is True
    assert report["output"]["sha256"] == _sha256(fixture["output_ply"])
    assert report["last_trustworthy_phase"] == "comparison_complete"


def test_wrong_decoder_route_fails_before_extraction_and_removes_stale_output(
    tmp_path,
):
    fixture = _mesh_fixture(tmp_path)
    fixture["output_ply"].write_bytes(b"stale")
    with np.load(fixture["checkpoint"], allow_pickle=False) as arrays:
        payload = {name: arrays[name] for name in arrays.files}
    route = json.loads(payload["decoder_route_json"].item())
    route["decoder_linear_backend"] = "native"
    payload["decoder_route_json"] = np.array(json.dumps(route, sort_keys=True))
    np.savez_compressed(fixture["checkpoint"], **payload)

    with pytest.raises(ValueError, match="decoder linear backend"):
        _invoke(fixture)

    report = json.loads(fixture["report_json"].read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "decoder_validation"
    assert report["primary_output_status"] == "not_started"
    assert report["stale_output_removed"] is True
    assert not fixture["output_ply"].exists()


def test_source_report_cannot_self_authenticate_wrong_ply_digest(tmp_path):
    fixture = _mesh_fixture(tmp_path)
    source_report = json.loads(fixture["source_report"].read_text())
    source_report["mesh_artifacts"][0]["sha256"] = "0" * 64
    fixture["source_report"].write_text(
        json.dumps(source_report, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match="source report PLY SHA256"):
        _invoke(fixture)

    report = json.loads(fixture["report_json"].read_text())
    assert report["failure_phase"] == "source_validation"
    assert report["primary_output_status"] == "not_started"


def test_truncated_source_ply_fails_loud_with_durable_report(tmp_path):
    fixture = _mesh_fixture(tmp_path)
    fixture["source_ply"].write_bytes(fixture["source_ply"].read_bytes()[:-3])
    source_report = json.loads(fixture["source_report"].read_text())
    source_report["mesh_artifacts"][0]["sha256"] = _sha256(
        fixture["source_ply"]
    )
    source_report["mesh_artifacts"][0]["size_bytes"] = (
        fixture["source_ply"].stat().st_size
    )
    fixture["source_report"].write_text(
        json.dumps(source_report, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match="before all faces"):
        _invoke(fixture)

    report = json.loads(fixture["report_json"].read_text())
    assert report["failure_phase"] == "source_validation"
    assert report["primary_output_status"] == "not_started"


def test_extraction_failure_writes_phase_and_no_primary(tmp_path, monkeypatch):
    fixture = _mesh_fixture(tmp_path)

    def fail_extraction(*args, **kwargs):
        raise RuntimeError("synthetic extraction failure")

    monkeypatch.setattr(
        "scripts.compare_decoder_mesh_to_source_ply.decoder_output_to_mesh",
        fail_extraction,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="synthetic extraction failure"):
        _invoke(fixture)

    report = json.loads(fixture["report_json"].read_text())
    assert report["failure_phase"] == "mesh_extraction"
    assert report["primary_output_status"] == "not_started"
    assert not fixture["output_ply"].exists()


def test_partial_primary_output_cannot_close_comparison(tmp_path, monkeypatch):
    fixture = _mesh_fixture(tmp_path)

    def write_partial(path, vertices, faces):
        Path(path).write_bytes(b"ply\nend_header\n")

    monkeypatch.setattr(
        "scripts.compare_decoder_mesh_to_source_ply._write_binary_ply_atomic",
        write_partial,
        raising=False,
    )

    with pytest.raises(ValueError, match="binary_little_endian"):
        _invoke(fixture)

    report = json.loads(fixture["report_json"].read_text())
    assert report["failure_phase"] == "output_validation"
    assert report["primary_output_status"] == "failed_validation"
    assert not fixture["output_ply"].exists()


def test_malformed_expected_hash_still_writes_request_failure_report(tmp_path):
    fixture = _mesh_fixture(tmp_path)

    with pytest.raises(ValueError, match="canonical SHA256"):
        _invoke(
            fixture,
            expected_decoder_checkpoint_sha256="not-a-hash",
        )

    report = json.loads(fixture["report_json"].read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "request_validation"
    assert report["last_trustworthy_phase"] == "request_received"
    assert report["primary_output_status"] == "not_started"


@pytest.mark.parametrize(
    "protected_key",
    ["checkpoint", "source_ply", "source_report", "output_ply"],
)
def test_unsafe_report_alias_preserves_input_and_reroutes_failure_report(
    tmp_path,
    protected_key,
):
    fixture = _mesh_fixture(tmp_path)
    protected_path = fixture[protected_key]
    if protected_key == "output_ply":
        protected_path.write_bytes(b"protected requested output")
    protected_bytes_before = protected_path.read_bytes()
    safe_failure_report = protected_path.with_name(
        protected_path.name + ".failure.json"
    )

    with pytest.raises(ValueError, match="report JSON must have a distinct path"):
        _invoke(fixture, report_json=protected_path)

    assert protected_path.read_bytes() == protected_bytes_before
    report = json.loads(safe_failure_report.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "request_validation"
    assert report["primary_output_status"] == "not_started"
    assert report["output"]["requested_report_path"] == str(
        protected_path.resolve()
    )
    assert report["output"]["effective_report_path"] == str(
        safe_failure_report.resolve()
    )
    assert report["output"]["report_path_rerouted"] is True


def test_rerouted_failure_report_does_not_replace_foreign_sibling(tmp_path):
    fixture = _mesh_fixture(tmp_path)
    protected_path = fixture["source_report"]
    first_sibling = protected_path.with_name(
        protected_path.name + ".failure.json"
    )
    first_sibling.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.decoder_mesh_source_ply_comparison.v1",
                "output": [],
            }
        )
        + "\n"
    )
    first_sibling_before = first_sibling.read_bytes()
    effective_report = protected_path.with_name(
        protected_path.name + ".failure.1.json"
    )

    with pytest.raises(ValueError, match="report JSON must have a distinct path"):
        _invoke(fixture, report_json=protected_path)

    assert first_sibling.read_bytes() == first_sibling_before
    report = json.loads(effective_report.read_text())
    assert report["output"]["effective_report_path"] == str(
        effective_report.resolve()
    )
    assert report["output"]["report_path_rerouted"] is True


@pytest.mark.parametrize("effective_identity", ["missing", "wrong"])
def test_rerouted_failure_report_rejects_deceptive_candidate_identity(
    tmp_path,
    effective_identity,
):
    fixture = _mesh_fixture(tmp_path)
    protected_path = fixture["source_report"].resolve()
    first_sibling = protected_path.with_name(
        protected_path.name + ".failure.json"
    )
    numbered_sibling = protected_path.with_name(
        protected_path.name + ".failure.1.json"
    )
    output = {
        "report_path": str(first_sibling),
        "requested_report_path": str(protected_path),
        "report_path_rerouted": True,
    }
    if effective_identity == "wrong":
        output["effective_report_path"] = str(numbered_sibling)
    first_sibling.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.decoder_mesh_source_ply_comparison.v1",
                "status": "failed",
                "failure_phase": "request_validation",
                "primary_output_status": "not_started",
                "output": output,
            },
            sort_keys=True,
        )
        + "\n"
    )
    first_sibling_before = first_sibling.read_bytes()

    with pytest.raises(ValueError, match="report JSON must have a distinct path"):
        _invoke(fixture, report_json=protected_path)

    assert first_sibling.read_bytes() == first_sibling_before
    report = json.loads(numbered_sibling.read_text())
    assert report["output"]["effective_report_path"] == str(numbered_sibling)


def test_genuine_rerouted_failure_report_is_reused_idempotently(tmp_path):
    fixture = _mesh_fixture(tmp_path)
    protected_path = fixture["source_report"].resolve()
    first_sibling = protected_path.with_name(
        protected_path.name + ".failure.json"
    )
    numbered_sibling = protected_path.with_name(
        protected_path.name + ".failure.1.json"
    )

    for _ in range(2):
        with pytest.raises(
            ValueError,
            match="report JSON must have a distinct path",
        ):
            _invoke(fixture, report_json=protected_path)

    report = json.loads(first_sibling.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "request_validation"
    assert report["output"]["report_path"] == str(first_sibling)
    assert report["output"]["effective_report_path"] == str(first_sibling)
    assert not numbered_sibling.exists()


def test_comparator_script_help_runs_without_repo_pythonpath(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "compare_decoder_mesh_to_source_ply.py"
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--expected-source-mesh-override-sha256" in result.stdout


def test_success_report_binds_comparator_and_extractor_code(tmp_path):
    fixture = _mesh_fixture(tmp_path)

    report = _invoke(fixture)

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "compare_decoder_mesh_to_source_ply.py"
    extractor = repo_root / "trellmlx" / "mesh_extract.py"
    assert report["code_identity"]["comparator"] == {
        "path": str(script),
        "sha256": _sha256(script),
    }
    assert report["code_identity"]["extractor"] == {
        "path": str(extractor),
        "sha256": _sha256(extractor),
    }
    assert len(report["code_identity"]["repo_commit"]) == 40
    assert isinstance(report["code_identity"]["repo_dirty"], bool)


def test_producer_top_level_mesh_override_is_authenticated(tmp_path):
    fixture = _mesh_fixture(tmp_path)
    source_report = json.loads(fixture["source_report"].read_text())
    assert source_report["mesh_override"]["sha256"] == MESH_OVERRIDE_SHA256
    assert "mesh_override" not in source_report["effective_route"]

    report = _invoke(fixture)

    assert report["status"] == "done"
    assert report["effective_route"]["source_mesh_override_sha256"] == (
        MESH_OVERRIDE_SHA256
    )


def test_duplicate_mesh_override_locations_are_rejected(tmp_path):
    fixture = _mesh_fixture(tmp_path)
    source_report = json.loads(fixture["source_report"].read_text())
    source_report["effective_route"]["mesh_override"] = dict(
        source_report["mesh_override"]
    )
    fixture["source_report"].write_text(
        json.dumps(source_report, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match="ambiguous duplicate"):
        _invoke(fixture)

    report = json.loads(fixture["report_json"].read_text())
    assert report["failure_phase"] == "source_validation"
    assert report["primary_output_status"] == "not_started"


def test_nested_only_mesh_override_location_is_rejected(tmp_path):
    fixture = _mesh_fixture(tmp_path)
    source_report = json.loads(fixture["source_report"].read_text())
    source_report["effective_route"]["mesh_override"] = source_report.pop(
        "mesh_override"
    )
    fixture["source_report"].write_text(
        json.dumps(source_report, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match="must be top-level"):
        _invoke(fixture)

    report = json.loads(fixture["report_json"].read_text())
    assert report["failure_phase"] == "source_validation"
    assert report["primary_output_status"] == "not_started"
