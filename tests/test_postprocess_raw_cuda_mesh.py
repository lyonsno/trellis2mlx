import json
from pathlib import Path

import numpy as np


def test_geometry_only_postprocess_labels_output_scope(monkeypatch, tmp_path):
    import scripts.postprocess_raw_cuda_mesh as postprocess_raw_cuda_mesh

    input_ply = tmp_path / "raw.ply"
    output_glb = tmp_path / "clean.glb"
    report_json = tmp_path / "report.json"

    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    postprocess_raw_cuda_mesh.write_binary_ply(input_ply, vertices, faces)

    def fake_postprocess(vertices_arg, faces_arg, target_faces, **kwargs):
        assert target_faces == 1
        assert kwargs["reference_python"] == "/ref/python"
        assert kwargs["expected_source_root"] == "/ref/mtlmesh"
        return vertices_arg[:3], faces_arg[:1], [{"operation": "fake_source_cleanup"}]

    monkeypatch.setattr(postprocess_raw_cuda_mesh, "postprocess_source_native", fake_postprocess)

    report = postprocess_raw_cuda_mesh.postprocess_raw_cuda_mesh(
        input_ply=input_ply,
        output_glb=output_glb,
        report_json=report_json,
        target_faces=1,
        reference_python="/ref/python",
        expected_source_root="/ref/mtlmesh",
    )

    assert output_glb.exists()
    assert report_json.exists()
    persisted = json.loads(report_json.read_text())
    assert persisted == report
    assert report["status"] == "done"
    assert report["artifact_scope"] == "geometry_only_raw_extraction_cleanup"
    assert report["forbidden_inferences"] == [
        "not full source o_voxel.postprocess.to_glb output",
        "not texture bake evidence",
        "not final material evidence",
        "not proof of source final winding without operator inspection",
    ]
    assert report["input_mesh"]["faces"] == 2
    assert report["output_mesh"]["faces"] == 1
    assert report["operation_trace"] == [{"operation": "fake_source_cleanup"}]
