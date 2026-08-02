import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.source_cuda_cumesh_qem_cost_witness as witness_module
from scripts.source_cuda_cumesh_postprocess_witness import (
    WitnessError,
    sha256_file,
    write_binary_ply,
)
from scripts.source_cuda_cumesh_qem_cost_witness import run_witness


def _write_input(path: Path) -> None:
    write_binary_ply(
        path,
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        np.array([[0, 1, 2], [0, 2, 1]], dtype=np.int32),
    )


def _runtime(
    patch_sha256: str,
    schema: str = "trellis2mlx.cumesh_qem_cost_instrumentation.v1",
    *,
    patches: list[dict[str, str]] | None = None,
    changed_files: list[str] | None = None,
):
    instrumentation = {
        "schema": schema,
        "patch_sha256": patch_sha256,
        "changed_files": changed_files or [
            "cumesh/cumesh.py",
            "src/cumesh.h",
            "src/ext.cpp",
            "src/simplify.cu",
        ],
    }
    if patches is not None:
        instrumentation["patches"] = patches
    return SimpleNamespace(
        effective_route={
            "trellis_commit": "5565d240c4a494caaf9ece7a554542b76ffa36d3",
            "trellis_postprocess_sha256": (
                "ef51a1ba0f2748ffb4c265b47d382cee956f23c6a52d0f3587e6d8beccb7e54a"
            ),
            "trellis_source_clean": True,
            "cumesh_commit": "c4ad6125924fcedfd13f0bd61520ca2d24eb7a87",
            "cumesh_source_clean_before_build": True,
            "cuda_available": True,
            "cuda_device_name": "Tesla T4",
            "cuda_capability": [7, 5],
            "device_type": "cuda",
            "cumesh_instrumentation": instrumentation,
        }
    )


def _arrays() -> dict[str, np.ndarray]:
    return {
        "vert2face": np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
        "qems": np.arange(30, dtype=np.float32).reshape(3, 10),
        "edge_collapse_costs": np.array([0.25, np.inf, 0.5], dtype=np.float32),
    }


def _component_arrays() -> dict[str, np.ndarray]:
    total = np.array([0.25, np.inf, 0.5], dtype=np.float32)
    return {
        "vert2face": np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
        "qems": np.arange(30, dtype=np.float32).reshape(3, 10),
        "edge_collapse_costs": total,
        "component_edge_collapse_costs": total.copy(),
        "qem_costs": np.array([0.2, 0.3, 0.4], dtype=np.float32),
        "edge_length2": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        "skinny_avgs": np.array([10.0, np.inf, 20.0], dtype=np.float32),
        "skinny_terms": np.array([0.01, np.inf, 0.04], dtype=np.float32),
    }


def test_qem_cost_witness_records_instrumented_t4_route_and_reopens_npz(tmp_path):
    input_ply = tmp_path / "input.ply"
    patch = tmp_path / "trace.patch"
    output_npz = tmp_path / "trace.npz"
    output_json = tmp_path / "trace.json"
    _write_input(input_ply)
    patch.write_text("instrumentation patch\n")
    patch_sha256 = sha256_file(patch)

    report = run_witness(
        input_ply=input_ply,
        instrumentation_patch=patch,
        output_npz=output_npz,
        output_json=output_json,
        expected_input_sha256=sha256_file(input_ply),
        expected_patch_sha256=patch_sha256,
        work_dir=tmp_path / "build",
        runtime_factory=lambda **kwargs: _runtime(patch_sha256),
        collector=lambda runtime, vertices, faces: _arrays(),
    )

    assert report["status"] == "done"
    assert report["effective_route"]["cuda_device_name"] == "Tesla T4"
    assert (
        report["effective_route"]["cumesh_instrumentation"]["patch_sha256"]
        == patch_sha256
    )
    assert report["arrays"]["qems"]["shape"] == [3, 10]
    assert report["arrays"]["edge_collapse_costs"]["nonfinite"] == 1
    with np.load(output_npz, allow_pickle=False) as archive:
        for name, expected in _arrays().items():
            assert np.array_equal(archive[name], expected, equal_nan=True)
    assert json.loads(output_json.read_text()) == report


def test_qem_cost_witness_rejects_substituted_patch_before_runtime(tmp_path):
    input_ply = tmp_path / "input.ply"
    patch = tmp_path / "trace.patch"
    output_npz = tmp_path / "trace.npz"
    output_json = tmp_path / "trace.json"
    _write_input(input_ply)
    patch.write_text("substituted\n")

    with pytest.raises(WitnessError, match="instrumentation patch SHA256"):
        run_witness(
            input_ply=input_ply,
            instrumentation_patch=patch,
            output_npz=output_npz,
            output_json=output_json,
            expected_input_sha256=sha256_file(input_ply),
            expected_patch_sha256="a" * 64,
            work_dir=tmp_path / "build",
            runtime_factory=lambda **kwargs: pytest.fail(
                "runtime should not be created"
            ),
            collector=lambda *args: pytest.fail("collector should not run"),
        )

    report = json.loads(output_json.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "patch_validation"
    assert report["primary_output_status"] == "not_started"
    assert not output_npz.exists()


def test_component_witness_rejects_backend_self_inconsistency_before_output(
    tmp_path,
):
    input_ply = tmp_path / "input.ply"
    patch = tmp_path / "trace.patch"
    output_npz = tmp_path / "trace.npz"
    output_json = tmp_path / "trace.json"
    _write_input(input_ply)
    patch.write_text("component instrumentation patch\n")
    patch_sha256 = sha256_file(patch)
    arrays = _component_arrays()
    arrays["component_edge_collapse_costs"][0] = np.nextafter(
        arrays["component_edge_collapse_costs"][0],
        np.float32(np.inf),
    )

    with pytest.raises(WitnessError, match="component total differs"):
        run_witness(
            input_ply=input_ply,
            instrumentation_patch=patch,
            output_npz=output_npz,
            output_json=output_json,
            expected_input_sha256=sha256_file(input_ply),
            expected_patch_sha256=patch_sha256,
            work_dir=tmp_path / "build",
            component_trace=True,
            runtime_factory=lambda **kwargs: _runtime(
                patch_sha256,
                "trellis2mlx.cumesh_qem_cost_component_instrumentation.v2",
            ),
            collector=lambda runtime, vertices, faces: arrays,
        )

    report = json.loads(output_json.read_text())
    assert report["failure_phase"] == "backend_self_consistency"
    assert report["primary_output_status"] == "not_started"
    assert not output_npz.exists()


def test_component_witness_records_self_consistent_arrays(tmp_path):
    input_ply = tmp_path / "input.ply"
    patch = tmp_path / "trace.patch"
    output_npz = tmp_path / "trace.npz"
    output_json = tmp_path / "trace.json"
    _write_input(input_ply)
    patch.write_text("component instrumentation patch\n")
    patch_sha256 = sha256_file(patch)

    report = run_witness(
        input_ply=input_ply,
        instrumentation_patch=patch,
        output_npz=output_npz,
        output_json=output_json,
        expected_input_sha256=sha256_file(input_ply),
        expected_patch_sha256=patch_sha256,
        work_dir=tmp_path / "build",
        component_trace=True,
        runtime_factory=lambda **kwargs: _runtime(
            patch_sha256,
            "trellis2mlx.cumesh_qem_cost_component_instrumentation.v2",
        ),
        collector=lambda runtime, vertices, faces: _component_arrays(),
    )

    assert report["schema"].endswith(".v2")
    assert report["backend_self_consistency"]["bit_exact"] is True
    assert report["effective_route"]["geometry_route"].endswith(
        "qem-cost-component-trace-instrumented"
    )
    with np.load(output_npz, allow_pickle=False) as archive:
        assert set(archive.files) == set(_component_arrays())


def test_component_witness_records_canonical_three_patch_route(tmp_path):
    input_ply = tmp_path / "input.ply"
    qem_patch = tmp_path / "qem.patch"
    canonical_patch = tmp_path / "canonical.patch"
    trace_sort_patch = tmp_path / "trace-sort.patch"
    output_npz = tmp_path / "trace.npz"
    output_json = tmp_path / "trace.json"
    _write_input(input_ply)
    for path, text in (
        (qem_patch, "qem\n"),
        (canonical_patch, "canonical\n"),
        (trace_sort_patch, "trace sort\n"),
    ):
        path.write_text(text)
    patches = [
        {
            "role": "canonical_adjacency",
            "path": str(canonical_patch),
            "sha256": sha256_file(canonical_patch),
        },
        {
            "role": "qem_component_trace",
            "path": str(qem_patch),
            "sha256": sha256_file(qem_patch),
        },
        {
            "role": "trace_local_adjacency_sort",
            "path": str(trace_sort_patch),
            "sha256": sha256_file(trace_sort_patch),
        },
    ]
    changed_files = [
        "cumesh/cumesh.py",
        "src/connectivity.cu",
        "src/cumesh.h",
        "src/ext.cpp",
        "src/simplify.cu",
    ]

    report = run_witness(
        input_ply=input_ply,
        instrumentation_patch=qem_patch,
        output_npz=output_npz,
        output_json=output_json,
        expected_input_sha256=sha256_file(input_ply),
        expected_patch_sha256=sha256_file(qem_patch),
        work_dir=tmp_path / "build",
        component_trace=True,
        canonical_adjacency_patch=canonical_patch,
        expected_canonical_adjacency_patch_sha256=sha256_file(
            canonical_patch
        ),
        trace_adjacency_patch=trace_sort_patch,
        expected_trace_adjacency_patch_sha256=sha256_file(
            trace_sort_patch
        ),
        runtime_factory=lambda **kwargs: _runtime(
            sha256_file(qem_patch),
            (
                "trellis2mlx.cumesh_canonical_qem_cost_component_"
                "instrumentation.v1"
            ),
            patches=patches,
            changed_files=changed_files,
        ),
        collector=lambda runtime, vertices, faces: _component_arrays(),
    )

    instrumentation = report["effective_route"]["cumesh_instrumentation"]
    assert instrumentation["patches"] == patches
    assert instrumentation["changed_files"] == changed_files
    assert report["effective_route"]["geometry_route"] == (
        "release-cumesh-canonical-adjacency-qem-cost-component-"
        "trace-instrumented"
    )
    assert report["effective_route"]["adjacency_order"] == (
        "ascending-face-id-per-vertex"
    )


def test_canonical_component_witness_rejects_partial_patch_identity(tmp_path):
    input_ply = tmp_path / "input.ply"
    qem_patch = tmp_path / "qem.patch"
    canonical_patch = tmp_path / "canonical.patch"
    _write_input(input_ply)
    qem_patch.write_text("qem\n")
    canonical_patch.write_text("canonical\n")

    with pytest.raises(WitnessError, match="canonical QEM route requires"):
        run_witness(
            input_ply=input_ply,
            instrumentation_patch=qem_patch,
            output_npz=tmp_path / "trace.npz",
            output_json=tmp_path / "trace.json",
            expected_input_sha256=sha256_file(input_ply),
            expected_patch_sha256=sha256_file(qem_patch),
            work_dir=tmp_path / "build",
            component_trace=True,
            canonical_adjacency_patch=canonical_patch,
            expected_canonical_adjacency_patch_sha256=sha256_file(
                canonical_patch
            ),
            runtime_factory=lambda **kwargs: pytest.fail(
                "runtime should not be created"
            ),
        )


def test_component_witness_can_preserve_explicitly_masked_non_global_trace(
    tmp_path,
):
    input_ply = tmp_path / "input.ply"
    patch = tmp_path / "trace.patch"
    output_npz = tmp_path / "trace.npz"
    output_json = tmp_path / "trace.json"
    _write_input(input_ply)
    patch.write_text("component instrumentation patch\n")
    patch_sha256 = sha256_file(patch)
    arrays = _component_arrays()
    arrays["component_edge_collapse_costs"][0] = np.nextafter(
        arrays["component_edge_collapse_costs"][0],
        np.float32(np.inf),
    )

    report = run_witness(
        input_ply=input_ply,
        instrumentation_patch=patch,
        output_npz=output_npz,
        output_json=output_json,
        expected_input_sha256=sha256_file(input_ply),
        expected_patch_sha256=patch_sha256,
        work_dir=tmp_path / "build",
        component_trace=True,
        allow_masked_attribution=True,
        runtime_factory=lambda **kwargs: _runtime(
            patch_sha256,
            "trellis2mlx.cumesh_qem_cost_component_instrumentation.v2",
        ),
        collector=lambda runtime, vertices, faces: arrays,
    )

    assert report["status"] == "done"
    assert report["backend_self_consistency"]["bit_exact"] is False
    assert report["component_attribution"] == {
        "global_admitted": False,
        "masked_admitted": True,
        "rejected_edge_count": 1,
        "mask_predicate": (
            "component_edge_collapse_costs bits equal "
            "edge_collapse_costs bits"
        ),
    }
    assert report["primary_output_status"] == "validated"
    assert output_npz.is_file()


def test_instrumentation_callback_resolves_patch_before_git_changes_directory(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    commands = []

    def fake_run(command, report):
        commands.append(command)
        stdout = ""
        if "status" in command:
            stdout = "".join(
                f" M {name}\n"
                for name in witness_module.INSTRUMENTED_FILES
            )
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(witness_module, "_run", fake_run)
    callback = witness_module._instrumentation_callback(
        Path("cumesh-qem-cost-trace.patch"),
        "a" * 64,
    )

    callback(tmp_path / "CuMesh", {"setup_commands": []})

    assert commands[0][-1] == str(
        (tmp_path / "cumesh-qem-cost-trace.patch").resolve()
    )
