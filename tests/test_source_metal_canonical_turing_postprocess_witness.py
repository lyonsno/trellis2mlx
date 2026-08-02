from pathlib import Path

import numpy as np

from scripts.source_metal_canonical_turing_postprocess_witness import (
    ADJACENCY_ORDER,
    GEOMETRY_ROUTE,
    MATH_PROFILE,
    _build_postprocessor,
)


def test_canonical_metal_postprocessor_routes_lut_through_both_simplifiers(
    tmp_path,
):
    lut_path = tmp_path / "rsqrt.npz"
    lut_path.write_bytes(b"fixture")
    lut_sha256 = "a" * 64
    delta = np.zeros(1 << 24, dtype=np.int8)
    calls = []

    class Mesh:
        def __init__(self):
            self.num_faces = 100

        def get_vertex_face_adjacency(self):
            calls.append("adjacency")

        def sort_vertex_face_adjacency(self):
            calls.append("sort")

        def simplify_step_turing(self, *args):
            calls.append(("step", args))
            self.num_faces = 30 if self.num_faces == 100 else 10
            return self.num_faces // 2, self.num_faces

        def read(self):
            return (
                np.zeros((self.num_faces // 2, 3), dtype=np.float32),
                np.zeros((self.num_faces, 3), dtype=np.int32),
            )

    def source_postprocessor(
        vertices,
        faces,
        target_faces,
        *,
        simplification_runner,
        **kwargs,
    ):
        mesh = Mesh()
        coarse = simplification_runner(
            mesh,
            target_faces * 3,
            lambda_edge_length=1e-2,
            lambda_skinny=1e-3,
            thresh=1e-8,
        )
        final = simplification_runner(
            mesh,
            target_faces,
            lambda_edge_length=1e-2,
            lambda_skinny=1e-3,
            thresh=1e-8,
        )
        return vertices, faces, [coarse, final]

    processor = _build_postprocessor(
        rsqrt_lut_npz=lut_path,
        expected_rsqrt_lut_sha256=lut_sha256,
        lut_loader=lambda *args, **kwargs: (
            delta,
            {"npz_path": str(lut_path), "npz_sha256": lut_sha256},
        ),
        source_postprocessor=source_postprocessor,
        tensor_factory=lambda value: value,
        record_step_digests=True,
    )

    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.zeros((1, 3), dtype=np.int32)
    _, _, trace = processor(
        vertices,
        faces,
        10,
        verbose=False,
        expected_source_root=Path("/mtlmesh"),
        stage_callback=lambda *args: None,
    )

    assert [call for call in calls if call == "adjacency"] == [
        "adjacency",
        "adjacency",
    ]
    assert [call for call in calls if call == "sort"] == ["sort", "sort"]
    step_calls = [call for call in calls if isinstance(call, tuple)]
    assert len(step_calls) == 2
    assert all(call[1][0].dtype == np.int8 for call in step_calls)
    assert all(call[1][-1] is True for call in step_calls)
    assert trace[0]["geometry_route"] == GEOMETRY_ROUTE
    assert trace[0]["adjacency_order"] == ADJACENCY_ORDER
    assert trace[0]["math_profile"] == MATH_PROFILE
    assert trace[1]["simplifier_route"] == (
        "canonical-adjacency-turing-rsqrt-step-loop"
    )
    assert trace[1]["record_step_digests"] is True
    assert trace[1]["simplifier_step_trace"][0]["observation"][
        "faces_shape"
    ] == [30, 3]
    assert trace[2]["rsqrt_lut_sha256"] == lut_sha256
