from types import SimpleNamespace

import numpy as np

from trellmlx.source_mtlmesh import (
    simplify_with_canonical_adjacency_step_loop,
)
from trellmlx.canonical_cumesh import mesh_state_digest_observer
from scripts.source_cuda_cumesh_postprocess_witness import (
    execute_geometry_sequence,
)


class _Backend:
    def __init__(self, face_counts):
        self.face_counts = iter(face_counts)
        self.num_faces_value = 1000
        self.calls = []

    def num_faces(self):
        return self.num_faces_value

    def get_vertex_face_adjacency(self):
        self.calls.append("adjacency")

    def sort_vertex_face_adjacency(self):
        self.calls.append("sort")

    def simplify_step(self, *args):
        self.calls.append(("cuda_step", args))
        self.num_faces_value = next(self.face_counts)
        return self.num_faces_value // 2, self.num_faces_value

    def simplify_step_turing(self, *args):
        self.calls.append(("metal_step", args))
        self.num_faces_value = next(self.face_counts)
        return self.num_faces_value // 2, self.num_faces_value


def test_canonical_cuda_loop_sorts_each_consumed_adjacency_and_reuses_it():
    backend = _Backend([995, 890])
    mesh = SimpleNamespace(
        cu_mesh=backend,
        num_faces=lambda: backend.num_faces_value,
    )

    trace = simplify_with_canonical_adjacency_step_loop(
        mesh,
        900,
    )

    assert backend.calls[0:3] == [
        "adjacency",
        "sort",
        ("cuda_step", (1e-2, 1e-3, 1e-8, False, True)),
    ]
    assert backend.calls[3:6] == [
        "adjacency",
        "sort",
        ("cuda_step", (1e-2, 1e-3, 1e-7, False, True)),
    ]
    assert [item["threshold"] for item in trace] == [1e-8, 1e-7]
    assert trace[-1]["output_faces"] == 890


def test_canonical_metal_loop_routes_turing_rsqrt_and_reuses_adjacency():
    backend = _Backend([800])
    rsqrt_lut = np.zeros(1 << 24, dtype=np.int8)

    trace = simplify_with_canonical_adjacency_step_loop(
        backend,
        900,
        rsqrt_lut=rsqrt_lut,
    )

    assert backend.calls[0:2] == ["adjacency", "sort"]
    method, args = backend.calls[2]
    assert method == "metal_step"
    assert args[0] is rsqrt_lut
    assert args[1:] == (1e-2, 1e-3, 1e-8, False, True)
    assert trace == [
        {
            "iteration": 1,
            "threshold": 1e-8,
            "input_faces": 1000,
            "output_faces": 800,
            "output_vertices": 400,
            "removed_faces": 200,
            "adjacency_order": "ascending-face-id-per-vertex",
        }
    ]


def test_canonical_loop_is_a_noop_when_target_is_already_met():
    backend = _Backend([])
    backend.num_faces_value = 20

    assert (
        simplify_with_canonical_adjacency_step_loop(backend, 20)
        == []
    )
    assert backend.calls == []


def test_canonical_loop_records_post_step_observations_without_mutating_route():
    backend = _Backend([800])
    observations = []

    def observe(mesh, step):
        observations.append((mesh, dict(step)))
        return {
            "vertices_sha256": "a" * 64,
            "faces_sha256": "b" * 64,
        }

    trace = simplify_with_canonical_adjacency_step_loop(
        backend,
        900,
        step_observer=observe,
    )

    assert observations == [
        (
            backend,
            {
                "iteration": 1,
                "threshold": 1e-8,
                "input_faces": 1000,
                "output_faces": 800,
                "output_vertices": 400,
                "removed_faces": 200,
                "adjacency_order": "ascending-face-id-per-vertex",
            },
        )
    ]
    assert trace[0]["observation"] == {
        "vertices_sha256": "a" * 64,
        "faces_sha256": "b" * 64,
    }


def test_canonical_loop_stops_after_exact_requested_step_count():
    backend = _Backend([900, 800, 700])

    trace = simplify_with_canonical_adjacency_step_loop(
        backend,
        1,
        max_steps=2,
    )

    assert [item["iteration"] for item in trace] == [1, 2]
    assert trace[-1]["output_faces"] == 800
    assert backend.calls == [
        "adjacency",
        "sort",
        ("cuda_step", (1e-2, 1e-3, 1e-8, False, True)),
        "adjacency",
        "sort",
        ("cuda_step", (1e-2, 1e-3, 1e-8, False, True)),
    ]


def test_canonical_loop_rejects_nonpositive_step_limit():
    backend = _Backend([900])

    try:
        simplify_with_canonical_adjacency_step_loop(
            backend,
            1,
            max_steps=0,
        )
    except ValueError as error:
        assert str(error) == "max_steps must be positive when provided"
    else:
        raise AssertionError("nonpositive max_steps was accepted")


def test_mesh_state_digest_observer_binds_ordered_arrays_and_counts():
    class Tensor:
        def __init__(self, array):
            self.array = array

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    vertices = np.arange(18, dtype=np.float32).reshape(6, 3)
    faces = np.arange(12, dtype=np.int32).reshape(4, 3)
    mesh = SimpleNamespace(
        read=lambda: (Tensor(vertices), Tensor(faces)),
    )

    observation = mesh_state_digest_observer(
        mesh,
        {"output_vertices": 6, "output_faces": 4},
    )

    assert observation == {
        "vertices_shape": [6, 3],
        "vertices_dtype": "float32",
        "vertices_sha256": (
            "e2fea6553423387adcf538ee52f8365e3e2b088f73f96a62"
            "d2a0dba0bcdea1ea"
        ),
        "faces_shape": [4, 3],
        "faces_dtype": "int32",
        "faces_sha256": (
            "a4886fc88eadb553f0300776411b64c557a02e7a09f9df7da"
            "871fb2f9f4c8278"
        ),
    }


def test_cuda_release_sequence_accepts_one_explicit_simplification_runner():
    class Mesh:
        def __init__(self):
            self.num_faces = 100

        def fill_holes(self, **kwargs):
            pass

        def remove_duplicate_faces(self):
            pass

        def repair_non_manifold_edges(self):
            pass

        def remove_small_connected_components(self, threshold):
            pass

        def unify_face_orientations(self):
            pass

    mesh = Mesh()
    targets = []
    stages = []

    def runner(actual_mesh, target_faces):
        targets.append(target_faces)
        actual_mesh.num_faces = target_faces
        return {
            "simplifier_route": "canonical_adjacency_step_loop",
            "simplifier_step_trace": [{"target": target_faces}],
        }

    execute_geometry_sequence(
        mesh,
        10,
        lambda operation, input_faces, output_faces, details: stages.append(
            (operation, input_faces, output_faces, details)
        ),
        simplification_runner=runner,
    )

    assert targets == [30, 10]
    assert stages[1][0] == "simplify_coarse"
    assert stages[1][3]["simplifier_route"] == (
        "canonical_adjacency_step_loop"
    )
    assert stages[6][0] == "simplify_final"
