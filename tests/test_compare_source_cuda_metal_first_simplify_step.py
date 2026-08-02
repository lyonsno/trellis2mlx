import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.compare_source_cuda_metal_first_simplify_step import run_comparison
from scripts.source_cuda_cumesh_postprocess_witness import (
    WitnessError,
    sha256_file,
    write_binary_ply,
)


COMMIT = "a" * 40


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _fixture(tmp_path: Path):
    input_ply = tmp_path / "input.ply"
    cuda_npz = tmp_path / "cuda.npz"
    cuda_report_json = tmp_path / "cuda.json"
    lut_npz = tmp_path / "lut.npz"
    output_npz = tmp_path / "metal.npz"
    report_json = tmp_path / "comparison.json"
    source_root = tmp_path / "mtlmesh"
    extension = source_root / "cumesh" / "_C.fixture.so"
    metallib = source_root / "cumesh" / "cumesh.metallib"
    extension.parent.mkdir(parents=True)
    extension.write_bytes(b"authenticated extension")
    metallib.write_bytes(b"authenticated metallib")

    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 1]], dtype=np.int32)
    write_binary_ply(input_ply, vertices, faces)
    arrays = {
        "vert2face": np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
        "post_vertices": np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        "post_faces": np.array([[0, 1, 0]], dtype=np.int32),
    }
    np.savez(cuda_npz, **arrays)
    report = {
        "schema": "trellis2mlx.source_cuda_cumesh_first_simplify_step.v1",
        "status": "done",
        "primary_output_status": "validated",
        "input_mesh": {
            "sha256": sha256_file(input_ply),
            "vertices": len(vertices),
            "faces": len(faces),
        },
        "effective_route": {
            "cuda_device_name": "Tesla T4",
            "geometry_route": "release-cumesh-first-simplify-step",
        },
        "arrays": {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": _array_sha256(array),
            }
            for name, array in arrays.items()
        },
        "output_npz": {"sha256": sha256_file(cuda_npz)},
    }
    cuda_report_json.write_text(json.dumps(report))
    lut_npz.write_bytes(b"fixture lut")
    identity = {
        "status": "available",
        "cumesh_file": str(source_root / "cumesh" / "__init__.py"),
        "cumesh_path": [str(source_root / "cumesh")],
        "metal_backend_file": str(source_root / "cumesh" / "metal_backend.py"),
        "git_root": str(source_root),
        "git_commit": COMMIT,
        "has_MtlMesh": True,
        "extension_path": str(extension),
        "extension_sha256": sha256_file(extension),
        "metallib_path": str(metallib),
        "metallib_sha256": sha256_file(metallib),
    }
    return {
        "input_ply": input_ply,
        "cuda_report_json": cuda_report_json,
        "cuda_npz": cuda_npz,
        "lut_npz": lut_npz,
        "output_npz": output_npz,
        "report_json": report_json,
        "source_root": source_root,
        "arrays": arrays,
        "identity": identity,
    }


def _kwargs(fixture):
    return {
        "input_ply": fixture["input_ply"],
        "cuda_report_json": fixture["cuda_report_json"],
        "cuda_npz": fixture["cuda_npz"],
        "turing_rsqrt_npz": fixture["lut_npz"],
        "output_npz": fixture["output_npz"],
        "report_json": fixture["report_json"],
        "expected_input_sha256": sha256_file(fixture["input_ply"]),
        "expected_cuda_report_sha256": sha256_file(
            fixture["cuda_report_json"]
        ),
        "expected_cuda_npz_sha256": sha256_file(fixture["cuda_npz"]),
        "expected_turing_rsqrt_sha256": sha256_file(fixture["lut_npz"]),
        "expected_source_root": fixture["source_root"],
        "expected_source_commit": COMMIT,
        "expected_extension_sha256": fixture["identity"]["extension_sha256"],
        "expected_metallib_sha256": fixture["identity"]["metallib_sha256"],
        "identity_probe": lambda: fixture["identity"],
        "lut_loader": lambda path, expected_sha256: (
            np.zeros(8, dtype=np.int8),
            {
                "npz_path": str(path),
                "npz_sha256": expected_sha256,
                "normalized_delta": {
                    "shape": [8],
                    "dtype": "int8",
                    "sha256": _array_sha256(np.zeros(8, dtype=np.int8)),
                },
            },
        ),
    }


def test_first_step_comparison_reports_bit_exact_metal_output(tmp_path):
    fixture = _fixture(tmp_path)
    kwargs = _kwargs(fixture)
    kwargs["runner"] = lambda vertices, faces, adjacency, delta: (
        fixture["arrays"]["post_vertices"],
        fixture["arrays"]["post_faces"],
        {
            "cuda_adjacency_segment_multisets_exact": True,
            "cuda_adjacency_readback_exact": True,
            "simplify_route": "turing-rsqrt-lut",
        },
    )

    report = run_comparison(**kwargs)

    assert report["status"] == "done"
    assert report["comparison"]["bit_exact"] is True
    assert report["effective_route"]["source_commit"] == COMMIT
    assert report["effective_route"]["simplify_route"] == "turing-rsqrt-lut"
    with np.load(fixture["output_npz"], allow_pickle=False) as archive:
        assert np.array_equal(
            archive["metal_post_faces"], fixture["arrays"]["post_faces"]
        )


def test_first_step_comparison_removes_stale_output_on_wrong_source(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["output_npz"].write_bytes(b"stale output")
    kwargs = _kwargs(fixture)
    kwargs["identity_probe"] = lambda: {
        **fixture["identity"],
        "git_root": str(tmp_path / "wrong-source"),
    }
    kwargs["runner"] = lambda *args: pytest.fail(
        "runner must not execute on a substituted source route"
    )

    with pytest.raises(WitnessError, match="does not match expected source root"):
        run_comparison(**kwargs)

    report = json.loads(fixture["report_json"].read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "runtime_validation"
    assert report["primary_output_status"] == "not_started"
    assert not fixture["output_npz"].exists()
