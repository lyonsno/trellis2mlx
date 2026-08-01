import json
from pathlib import Path

import numpy as np
import pytest

from scripts.source_cuda_cumesh_postprocess_witness import (
    STAGE_SPECS,
    sha256_file,
    write_binary_ply,
)


def _write_report(tmp_path, name, stage_arrays, *, input_sha="a" * 64):
    stage_dir = tmp_path / name
    artifacts = []
    for index, ((operation, filename), (vertices, faces)) in enumerate(
        zip(STAGE_SPECS, stage_arrays, strict=True),
        start=1,
    ):
        path = stage_dir / filename
        write_binary_ply(path, vertices, faces)
        artifacts.append(
            {
                "index": index,
                "operation": operation,
                "output_vertices": len(vertices),
                "output_faces": len(faces),
                "path": str(path),
                "sha256": sha256_file(path),
                "status": "validated",
            }
        )
    report = {
        "status": "done",
        "primary_output_status": "validated",
        "input_mesh": {"sha256": input_sha},
        "requested_route": {"target_faces": 10},
        "stage_artifacts": artifacts,
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(report))
    return path


def _stages():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return [(vertices.copy(), faces.copy()) for _ in STAGE_SPECS]


def test_comparator_distinguishes_content_from_cardinality_divergence(tmp_path):
    from scripts.compare_cumesh_postprocess_witnesses import compare_witnesses

    reference_stages = _stages()
    candidate_stages = _stages()
    candidate_stages[1] = (
        candidate_stages[1][0][[1, 0, 2, 3]],
        candidate_stages[1][1],
    )
    candidate_stages[4] = (
        candidate_stages[4][0],
        candidate_stages[4][1][:1],
    )
    reference = _write_report(tmp_path, "reference", reference_stages)
    candidate = _write_report(tmp_path, "candidate", candidate_stages)

    report = compare_witnesses(reference, candidate)

    assert report["first_ordered_content_divergence"]["index"] == 2
    assert report["first_ordered_content_divergence"]["operation"] == "simplify_coarse"
    assert report["first_cardinality_divergence"]["index"] == 5
    assert (
        report["first_cardinality_divergence"]["operation"]
        == "remove_small_connected_components_initial"
    )
    assert report["stages"][0]["ordered_content_exact"] is True
    assert report["stages"][1]["dimensions_exact"] is True
    assert report["stages"][1]["ordered_content_exact"] is False


def test_comparator_rejects_nonvalidated_source_report(tmp_path):
    from scripts.compare_cumesh_postprocess_witnesses import (
        ComparisonError,
        compare_witnesses,
    )

    reference = _write_report(tmp_path, "reference", _stages())
    candidate = _write_report(tmp_path, "candidate", _stages())
    payload = json.loads(candidate.read_text())
    payload["primary_output_status"] = "partial"
    candidate.write_text(json.dumps(payload))

    with pytest.raises(ComparisonError, match="primary_output_status"):
        compare_witnesses(reference, candidate)


def test_comparator_rejects_changed_stage_artifact(tmp_path):
    from scripts.compare_cumesh_postprocess_witnesses import (
        ComparisonError,
        compare_witnesses,
    )

    reference = _write_report(tmp_path, "reference", _stages())
    candidate = _write_report(tmp_path, "candidate", _stages())
    payload = json.loads(candidate.read_text())
    changed = Path(payload["stage_artifacts"][0]["path"])
    changed.write_bytes(changed.read_bytes() + b"x")

    with pytest.raises(ComparisonError, match="hash mismatch"):
        compare_witnesses(reference, candidate)
