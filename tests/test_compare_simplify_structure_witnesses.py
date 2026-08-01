import hashlib
import json

import numpy as np

from scripts.compare_simplify_structure_witnesses import compare_witnesses
from scripts.source_cuda_cumesh_postprocess_witness import sha256_file


def _write_witness(tmp_path, role, arrays):
    npz = tmp_path / f"{role}.npz"
    report = tmp_path / f"{role}.json"
    np.savez(npz, **arrays)
    report.write_text(
        json.dumps(
            {
                "status": "done",
                "primary_output_status": "validated",
                "input_mesh": {"sha256": "a" * 64},
                "output_npz": {
                    "path": str(npz),
                    "sha256": sha256_file(npz),
                },
                "effective_route": {
                    "geometry_route": (
                        "release-cumesh-simplify-structure"
                        if role == "cuda"
                        else "metal-mtlmesh-simplify-structure"
                    ),
                    **(
                        {
                            "edge_readback": (
                                "uint64-little-endian-words-"
                                "canonicalized-to-min-max"
                            )
                        }
                        if role == "cuda"
                        else {}
                    ),
                },
                "arrays": {
                    name: {
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                    }
                    for name, array in arrays.items()
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    return report, npz


def test_comparator_isolates_adjacency_order_from_exact_structure(tmp_path):
    common = {
        "vert2face_cnt": np.array([2, 2, 2, 0], dtype=np.int32),
        "vert2face_offset": np.array([0, 2, 4, 6], dtype=np.int32),
        "edges": np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int32),
        "edge2face_cnt": np.array([2, 2, 2], dtype=np.int32),
        "boundaries": np.empty((0,), dtype=np.int32),
        "vert_is_boundary": np.zeros((3,), dtype=np.uint8),
    }
    cuda_arrays = {
        "vert2face": np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
        **common,
    }
    metal_arrays = {
        "vert2face": np.array([1, 0, 0, 1, 1, 0], dtype=np.int32),
        **common,
    }
    cuda_report, cuda_npz = _write_witness(tmp_path, "cuda", cuda_arrays)
    metal_report, metal_npz = _write_witness(tmp_path, "metal", metal_arrays)

    result = compare_witnesses(
        cuda_report_json=cuda_report,
        cuda_npz=cuda_npz,
        metal_report_json=metal_report,
        metal_npz=metal_npz,
        expected_cuda_report_sha256=sha256_file(cuda_report),
        expected_cuda_npz_sha256=sha256_file(cuda_npz),
        expected_metal_report_sha256=sha256_file(metal_report),
        expected_metal_npz_sha256=sha256_file(metal_npz),
    )

    assert result["status"] == "done"
    assert result["input_sha256_exact"] is True
    assert result["all_non_adjacency_arrays_exact"] is True
    assert result["arrays"]["vert2face"]["exact"] is False
    assert result["arrays"]["vert2face"]["differing_entries"] == 4
    assert result["vert2face_order"]["segment_multisets_exact"] is True
    assert result["vert2face_order"]["differing_vertices"] == 2
