"""Generate.py mesh postprocess sequencing contracts."""

import pytest


class FaceBag:
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count


def test_postprocess_skips_final_simplify_if_cleanup_drops_below_target():
    from generate import _cleanup_and_simplify_mesh

    vertices = FaceBag(10)
    cleanup_outputs = [FaceBag(1_000_000), FaceBag(150_000), FaceBag(150_000)]
    simplify_calls = []
    cleanup_calls = []

    def cleanup_mesh(v, faces, keep_largest=False, do_fix_normals=True, verbose=True):
        cleanup_calls.append((len(faces), do_fix_normals, verbose))
        return v, cleanup_outputs.pop(0)

    def simplify(v, faces, target_reduction):
        simplify_calls.append(target_reduction)
        if target_reduction < 0:
            raise AssertionError(f"negative reduction {target_reduction}")
        return v, FaceBag(600_000)

    out_vertices, out_faces = _cleanup_and_simplify_mesh(
        vertices,
        FaceBag(1_000_000),
        target_faces=200_000,
        no_cleanup=False,
        cleanup_mesh=cleanup_mesh,
        simplify=simplify,
        log=lambda *args, **kwargs: None,
    )

    assert out_vertices is vertices
    assert len(out_faces) == 150_000
    assert simplify_calls == [pytest.approx(0.4)]
    assert cleanup_calls == [
        (1_000_000, False, True),
        (600_000, False, False),
        (150_000, True, False),
    ]


def test_postprocess_target_faces_zero_still_runs_final_normals_cleanup():
    from generate import _cleanup_and_simplify_mesh

    vertices = FaceBag(10)
    cleanup_outputs = [FaceBag(500_000), FaceBag(500_000)]
    cleanup_calls = []

    def cleanup_mesh(v, faces, keep_largest=False, do_fix_normals=True, verbose=True):
        cleanup_calls.append((len(faces), do_fix_normals, verbose))
        return v, cleanup_outputs.pop(0)

    out_vertices, out_faces = _cleanup_and_simplify_mesh(
        vertices,
        FaceBag(500_000),
        target_faces=0,
        no_cleanup=False,
        cleanup_mesh=cleanup_mesh,
        simplify=None,
        log=lambda *args, **kwargs: None,
    )

    assert out_vertices is vertices
    assert len(out_faces) == 500_000
    assert cleanup_calls == [
        (500_000, False, True),
        (500_000, True, False),
    ]


def test_postprocess_no_cleanup_still_simplifies_without_cleanup_import():
    from generate import _cleanup_and_simplify_mesh

    vertices = FaceBag(10)
    simplify_calls = []

    def cleanup_mesh(*args, **kwargs):
        raise AssertionError("cleanup should not run with no_cleanup=True")

    def simplify(v, faces, target_reduction):
        simplify_calls.append(target_reduction)
        if len(simplify_calls) == 1:
            return v, FaceBag(600_000)
        return v, FaceBag(200_000)

    out_vertices, out_faces = _cleanup_and_simplify_mesh(
        vertices,
        FaceBag(1_000_000),
        target_faces=200_000,
        no_cleanup=True,
        cleanup_mesh=cleanup_mesh,
        simplify=simplify,
        log=lambda *args, **kwargs: None,
    )

    assert out_vertices is vertices
    assert len(out_faces) == 200_000
    assert simplify_calls == [pytest.approx(0.4), pytest.approx(2 / 3)]


def test_postprocess_reference_cleanup_matches_gpu_source_order_and_orientation():
    from generate import _cleanup_and_simplify_mesh

    vertices = FaceBag(10)
    cleanup_outputs = [FaceBag(500_000), FaceBag(190_000)]
    cleanup_calls = []
    simplify_calls = []
    orient_calls = []

    def fill_holes(v, faces, max_hole_perimeter=3e-2, verbose=True):
        raise AssertionError("reference GPU source does not standalone-fill holes before coarse simplify")

    def cleanup_mesh(v, faces, keep_largest=False, do_fix_normals=True, verbose=True):
        cleanup_calls.append((len(faces), do_fix_normals, verbose))
        return v, cleanup_outputs.pop(0)

    def simplify(v, faces, target_reduction=None, target_count=None):
        simplify_calls.append((target_reduction, target_count))
        if target_count == 600_000:
            return v, FaceBag(600_000)
        if target_count == 200_000:
            return v, FaceBag(200_000)
        raise AssertionError(f"unexpected simplify target {target_count}")

    def orient_faces_by_adjacency(v, faces, verbose=True):
        orient_calls.append((len(faces), verbose))
        return v, faces

    out_vertices, out_faces = _cleanup_and_simplify_mesh(
        vertices,
        FaceBag(1_000_000),
        target_faces=200_000,
        no_cleanup=False,
        reference_cleanup=True,
        cleanup_mesh=cleanup_mesh,
        fill_holes=fill_holes,
        simplify=simplify,
        orient_faces_by_adjacency=orient_faces_by_adjacency,
        log=lambda *args, **kwargs: None,
    )

    assert out_vertices is vertices
    assert len(out_faces) == 190_000
    assert simplify_calls == [(None, 600_000), (None, 200_000)]
    assert cleanup_calls == [
        (600_000, False, True),
        (200_000, False, False),
    ]
    assert orient_calls == [(190_000, False)]


def test_postprocess_reference_cleanup_records_operation_trace():
    from generate import _cleanup_and_simplify_mesh

    vertices = FaceBag(10)
    cleanup_outputs = [FaceBag(500_000), FaceBag(190_000)]
    operation_trace = []

    def cleanup_mesh(v, faces, keep_largest=False, do_fix_normals=True, verbose=True):
        return v, cleanup_outputs.pop(0)

    def simplify(v, faces, target_reduction=None, target_count=None):
        if target_count == 600_000:
            return v, FaceBag(590_000)
        if target_count == 200_000:
            return v, FaceBag(180_000)
        raise AssertionError(f"unexpected simplify target {target_count}")

    def orient_faces_by_adjacency(v, faces, verbose=True):
        return v, faces

    _cleanup_and_simplify_mesh(
        vertices,
        FaceBag(1_000_000),
        target_faces=200_000,
        no_cleanup=False,
        reference_cleanup=True,
        cleanup_mesh=cleanup_mesh,
        fill_holes=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no prefill")),
        simplify=simplify,
        orient_faces_by_adjacency=orient_faces_by_adjacency,
        operation_trace=operation_trace,
        log=lambda *args, **kwargs: None,
    )

    assert operation_trace == [
        {
            "operation": "simplify_coarse",
            "input_faces": 1_000_000,
            "requested_target_faces": 600_000,
            "output_faces": 590_000,
        },
        {
            "operation": "cleanup_initial",
            "input_faces": 590_000,
            "output_faces": 500_000,
            "do_fix_normals": False,
        },
        {
            "operation": "simplify_final",
            "input_faces": 500_000,
            "requested_target_faces": 200_000,
            "output_faces": 180_000,
        },
        {
            "operation": "cleanup_final",
            "input_faces": 180_000,
            "output_faces": 190_000,
            "do_fix_normals": False,
        },
        {
            "operation": "orient_faces_by_adjacency",
            "input_faces": 190_000,
            "output_faces": 190_000,
        },
    ]


def test_postprocess_reference_cleanup_rejects_qem_simplify_until_parity_gate():
    from generate import _cleanup_and_simplify_mesh

    with pytest.raises(ValueError, match="reference_cleanup.*qem_simplify"):
        _cleanup_and_simplify_mesh(
            FaceBag(10),
            FaceBag(1_000_000),
            target_faces=200_000,
            no_cleanup=False,
            reference_cleanup=True,
            qem_simplify=True,
            cleanup_mesh=lambda v, faces, **kwargs: (v, faces),
            simplify=lambda v, faces, **kwargs: (v, faces),
            log=lambda *args, **kwargs: None,
        )


def test_postprocess_reference_cleanup_accepts_source_native_qem_backend():
    from generate import _cleanup_and_simplify_mesh

    calls = []
    cleanup_outputs = [250_000, 190_000]

    def cleanup_mesh(v, faces, **kwargs):
        return v, FaceBag(cleanup_outputs.pop(0))

    def source_simplify(v, faces, target_faces, **kwargs):
        calls.append((len(faces), target_faces, kwargs))
        return v, FaceBag(target_faces)

    vertices, faces = _cleanup_and_simplify_mesh(
        FaceBag(10),
        FaceBag(1_000_000),
        target_faces=200_000,
        no_cleanup=False,
        reference_cleanup=True,
        qem_simplify=True,
        qem_backend="source-native",
        cleanup_mesh=cleanup_mesh,
        source_native_simplify=source_simplify,
        orient_faces_by_adjacency=lambda v, faces, **kwargs: (v, faces),
        simplify=lambda v, faces, **kwargs: (v, FaceBag(kwargs["target_count"])),
        log=lambda *args, **kwargs: None,
    )

    assert len(faces) == 190_000
    assert calls == [
        (1_000_000, 600_000, {"verbose": True}),
        (250_000, 200_000, {"verbose": True}),
    ]


def test_postprocess_source_native_qem_backend_receives_expected_source_root():
    from generate import _cleanup_and_simplify_mesh

    calls = []
    cleanup_outputs = [250_000, 190_000]

    def cleanup_mesh(v, faces, **kwargs):
        return v, FaceBag(cleanup_outputs.pop(0))

    def source_simplify(v, faces, target_faces, **kwargs):
        calls.append((len(faces), target_faces, kwargs))
        return v, FaceBag(target_faces)

    _cleanup_and_simplify_mesh(
        FaceBag(10),
        FaceBag(1_000_000),
        target_faces=200_000,
        no_cleanup=False,
        reference_cleanup=True,
        qem_simplify=True,
        qem_backend="source-native",
        source_native_source_root="/Users/noahlyons/dev/trellis-mac/deps/mtlmesh",
        cleanup_mesh=cleanup_mesh,
        source_native_simplify=source_simplify,
        orient_faces_by_adjacency=lambda v, faces, **kwargs: (v, faces),
        simplify=lambda v, faces, **kwargs: (v, FaceBag(kwargs["target_count"])),
        log=lambda *args, **kwargs: None,
    )

    assert calls == [
        (
            1_000_000,
            600_000,
            {
                "verbose": True,
                "expected_source_root": "/Users/noahlyons/dev/trellis-mac/deps/mtlmesh",
            },
        ),
        (
            250_000,
            200_000,
            {
                "verbose": True,
                "expected_source_root": "/Users/noahlyons/dev/trellis-mac/deps/mtlmesh",
            },
        )
    ]


def test_postprocess_source_native_qem_backend_uses_source_native_orientation():
    from generate import _cleanup_and_simplify_mesh

    operation_trace = []
    calls = []

    def cleanup_mesh(v, faces, **kwargs):
        return v, FaceBag(len(faces))

    def source_simplify(v, faces, target_faces, **kwargs):
        return v, FaceBag(target_faces)

    def source_orient(v, faces, **kwargs):
        calls.append((len(faces), kwargs))
        return v, FaceBag(len(faces))

    def local_orient(*args, **kwargs):
        raise AssertionError("source-native reference cleanup must not use local orientation")

    _cleanup_and_simplify_mesh(
        FaceBag(10),
        FaceBag(1_000_000),
        target_faces=200_000,
        no_cleanup=False,
        reference_cleanup=True,
        qem_simplify=True,
        qem_backend="source-native",
        source_native_source_root="/Users/noahlyons/dev/trellis-mac/deps/mtlmesh",
        cleanup_mesh=cleanup_mesh,
        source_native_simplify=source_simplify,
        source_native_orient=source_orient,
        orient_faces_by_adjacency=local_orient,
        operation_trace=operation_trace,
        simplify=lambda v, faces, **kwargs: (v, FaceBag(kwargs["target_count"])),
        log=lambda *args, **kwargs: None,
    )

    assert calls == [
        (
            200_000,
            {
                "verbose": False,
                "expected_source_root": "/Users/noahlyons/dev/trellis-mac/deps/mtlmesh",
            },
        )
    ]
    assert operation_trace[-1] == {
        "operation": "orient_faces_source_native",
        "input_faces": 200_000,
        "output_faces": 200_000,
    }


def test_postprocess_source_native_qem_backend_uses_source_native_cleanup():
    from generate import _cleanup_and_simplify_mesh

    operation_trace = []
    cleanup_calls = []

    def local_cleanup(*args, **kwargs):
        raise AssertionError("source-native reference cleanup must not use local cleanup")

    def source_simplify(v, faces, target_faces, **kwargs):
        return v, FaceBag(target_faces)

    def source_cleanup(v, faces, **kwargs):
        cleanup_calls.append((len(faces), kwargs))
        return v, FaceBag(len(faces))

    _cleanup_and_simplify_mesh(
        FaceBag(10),
        FaceBag(1_000_000),
        target_faces=200_000,
        no_cleanup=False,
        reference_cleanup=True,
        qem_simplify=True,
        qem_backend="source-native",
        source_native_source_root="/Users/noahlyons/dev/trellis-mac/deps/mtlmesh",
        cleanup_mesh=local_cleanup,
        source_native_simplify=source_simplify,
        source_native_cleanup=source_cleanup,
        source_native_orient=lambda v, faces, **kwargs: (v, faces),
        operation_trace=operation_trace,
        simplify=lambda v, faces, **kwargs: (v, FaceBag(kwargs["target_count"])),
        log=lambda *args, **kwargs: None,
    )

    assert cleanup_calls == [
        (
            600_000,
            {
                "verbose": True,
                "expected_source_root": "/Users/noahlyons/dev/trellis-mac/deps/mtlmesh",
            },
        ),
        (
            200_000,
            {
                "verbose": False,
                "expected_source_root": "/Users/noahlyons/dev/trellis-mac/deps/mtlmesh",
            },
        ),
    ]
    assert [entry["operation"] for entry in operation_trace] == [
        "simplify_coarse_source_native_qem",
        "cleanup_initial_source_native",
        "simplify_final_source_native_qem",
        "cleanup_final_source_native",
        "orient_faces_source_native",
    ]


def test_postprocess_source_native_qem_backend_uses_combined_source_route_by_default():
    from generate import _cleanup_and_simplify_mesh

    operation_trace = []
    calls = []

    def source_postprocess(v, faces, target_faces, **kwargs):
        calls.append((len(faces), target_faces, kwargs))
        return (
            v,
            FaceBag(target_faces + 7),
            [
                {
                    "operation": "source_full_combined",
                    "input_faces": len(faces),
                    "requested_target_faces": target_faces,
                    "output_faces": target_faces + 7,
                }
            ],
        )

    def local_cleanup(*args, **kwargs):
        raise AssertionError("combined source-native route must not call local cleanup")

    out_vertices, out_faces = _cleanup_and_simplify_mesh(
        FaceBag(10),
        FaceBag(1_000_000),
        target_faces=200_000,
        no_cleanup=False,
        reference_cleanup=True,
        qem_simplify=True,
        qem_backend="source-native",
        source_native_source_root="/Users/noahlyons/dev/trellis-mac/deps/mtlmesh",
        source_native_postprocess=source_postprocess,
        cleanup_mesh=local_cleanup,
        simplify=lambda v, faces, **kwargs: (_ for _ in ()).throw(AssertionError("no local simplify")),
        operation_trace=operation_trace,
        log=lambda *args, **kwargs: None,
    )

    assert len(out_vertices) == 10
    assert len(out_faces) == 200_007
    assert calls == [
        (
            1_000_000,
            200_000,
            {
                "verbose": True,
                "expected_source_root": "/Users/noahlyons/dev/trellis-mac/deps/mtlmesh",
            },
        )
    ]
    assert operation_trace == [
        {
            "operation": "source_full_combined",
            "input_faces": 1_000_000,
            "requested_target_faces": 200_000,
            "output_faces": 200_007,
        }
    ]


def test_postprocess_combined_source_route_honors_keep_largest(monkeypatch):
    from generate import _cleanup_and_simplify_mesh
    import trellmlx.mesh_cleanup as mesh_cleanup

    operation_trace = []
    keep_largest_calls = []

    def source_postprocess(v, faces, target_faces, **kwargs):
        return (
            v,
            FaceBag(target_faces + 7),
            [
                {
                    "operation": "source_full_combined",
                    "input_faces": len(faces),
                    "requested_target_faces": target_faces,
                    "output_faces": target_faces + 7,
                }
            ],
        )

    def keep_largest_component(v, faces, verbose=True):
        keep_largest_calls.append((len(faces), verbose))
        return v, FaceBag(150_000)

    monkeypatch.setattr(mesh_cleanup, "keep_largest_component", keep_largest_component)

    out_vertices, out_faces = _cleanup_and_simplify_mesh(
        FaceBag(10),
        FaceBag(1_000_000),
        target_faces=200_000,
        no_cleanup=False,
        keep_largest=True,
        reference_cleanup=True,
        qem_simplify=True,
        qem_backend="source-native",
        source_native_postprocess=source_postprocess,
        cleanup_mesh=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no local cleanup")),
        simplify=lambda v, faces, **kwargs: (_ for _ in ()).throw(AssertionError("no local simplify")),
        operation_trace=operation_trace,
        log=lambda *args, **kwargs: None,
    )

    assert len(out_vertices) == 10
    assert len(out_faces) == 150_000
    assert keep_largest_calls == [(200_007, False)]
    assert operation_trace == [
        {
            "operation": "source_full_combined",
            "input_faces": 1_000_000,
            "requested_target_faces": 200_000,
            "output_faces": 200_007,
        },
        {
            "operation": "keep_largest_component",
            "input_faces": 200_007,
            "output_faces": 150_000,
        },
    ]


def test_postprocess_rejects_unknown_qem_backend():
    from generate import _cleanup_and_simplify_mesh

    with pytest.raises(ValueError, match="unknown qem_backend"):
        _cleanup_and_simplify_mesh(
            FaceBag(10),
            FaceBag(1_000_000),
            target_faces=200_000,
            no_cleanup=False,
            qem_simplify=True,
            qem_backend="definitely-not-real",
            cleanup_mesh=lambda v, faces, **kwargs: (v, faces),
            log=lambda *args, **kwargs: None,
        )
