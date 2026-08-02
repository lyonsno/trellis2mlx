import json

from scripts.compare_canonical_simplify_step_traces import compare_traces


def _report(path, *, backend, steps):
    payload = {
        "status": "done",
        "primary_output_status": "validated",
        "effective_route": {
            "device_type": backend,
            "adjacency_order": "ascending-face-id-per-vertex",
            "reuse_vertex_face_adjacency": True,
            "record_simplify_step_digests": True,
        },
        "stage_artifacts": [
            {
                "operation": "simplify_coarse",
                "details": {
                    "record_step_digests": True,
                    "simplifier_step_trace": steps,
                },
            },
            {
                "operation": "simplify_final",
                "details": {
                    "record_step_digests": True,
                    "simplifier_step_trace": [],
                },
            },
        ],
    }
    path.write_text(json.dumps(payload))
    return path


def _step(iteration, *, vertices, faces, vertex_sha, face_sha):
    return {
        "iteration": iteration,
        "input_faces": faces + 10,
        "output_vertices": vertices,
        "output_faces": faces,
        "removed_faces": 10,
        "threshold": 1e-8,
        "observation": {
            "vertices_shape": [vertices, 3],
            "vertices_dtype": "float32",
            "vertices_sha256": vertex_sha,
            "faces_shape": [faces, 3],
            "faces_dtype": "int32",
            "faces_sha256": face_sha,
        },
    }


def test_compare_traces_separates_ordered_state_and_cardinality_divergence(
    tmp_path,
):
    reference = _report(
        tmp_path / "cuda.json",
        backend="cuda",
        steps=[
            _step(
                1,
                vertices=90,
                faces=180,
                vertex_sha="a" * 64,
                face_sha="b" * 64,
            ),
            _step(
                2,
                vertices=80,
                faces=160,
                vertex_sha="c" * 64,
                face_sha="d" * 64,
            ),
            _step(
                3,
                vertices=70,
                faces=140,
                vertex_sha="e" * 64,
                face_sha="f" * 64,
            ),
        ],
    )
    candidate = _report(
        tmp_path / "metal.json",
        backend="metal",
        steps=[
            _step(
                1,
                vertices=90,
                faces=180,
                vertex_sha="a" * 64,
                face_sha="b" * 64,
            ),
            _step(
                2,
                vertices=80,
                faces=160,
                vertex_sha="1" * 64,
                face_sha="d" * 64,
            ),
            _step(
                3,
                vertices=71,
                faces=141,
                vertex_sha="2" * 64,
                face_sha="3" * 64,
            ),
        ],
    )

    result = compare_traces(reference, candidate)

    assert result["first_ordered_state_divergence"] == {
        "operation": "simplify_coarse",
        "iteration": 2,
        "vertices_exact": False,
        "faces_exact": True,
    }
    assert result["first_cardinality_divergence"] == {
        "operation": "simplify_coarse",
        "iteration": 3,
    }
    assert result["stages"][0]["steps"][0]["ordered_state_exact"] is True
