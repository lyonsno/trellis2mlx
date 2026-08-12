"""Contracts for honest local-QEM termination reporting."""

from __future__ import annotations

import numpy as np
import pytest

import trellmlx.simplify_qem_metal as qem


def _mesh(face_count: int = 6) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
    )
    faces = np.tile(np.asarray([[0, 1, 2]], dtype=np.int32), (face_count, 1))
    return vertices, faces


def test_default_api_remains_two_arrays(monkeypatch) -> None:
    vertices, faces = _mesh()
    monkeypatch.setattr(qem, "_simplify_step", lambda v, f, **kwargs: (v, f[:2]))

    result = qem.simplify_qem(vertices, faces, 2, max_iterations=3, verbose=False)

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_receipt_reports_input_already_at_target() -> None:
    vertices, faces = _mesh(2)

    out_vertices, out_faces, receipt = qem.simplify_qem(
        vertices,
        faces,
        2,
        return_receipt=True,
        verbose=False,
    )

    np.testing.assert_array_equal(out_vertices, vertices)
    np.testing.assert_array_equal(out_faces, faces)
    assert receipt == {
        "route": "trellis2mlx-sequential-qem-v1",
        "scheduler": "sequential-mutating-conflict-check",
        "source_faces": 2,
        "requested_target_faces": 2,
        "achieved_faces": 2,
        "target_satisfied": True,
        "termination_reason": "input_at_or_below_target",
        "iterations": 0,
        "max_iterations": 500,
        "zero_removal_iterations": 0,
        "final_iteration_removed_faces": 0,
        "initial_threshold": 1e-8,
        "final_executed_threshold": None,
        "next_scheduled_threshold": 1e-8,
    }


def test_receipt_exposes_hard_cap_without_calling_it_fixed_point(monkeypatch) -> None:
    vertices, faces = _mesh(6)
    monkeypatch.setattr(qem, "_simplify_step", lambda v, f, **kwargs: (v, f))

    _, out_faces, receipt = qem.simplify_qem(
        vertices,
        faces,
        1,
        max_iterations=3,
        return_receipt=True,
        verbose=False,
    )

    assert len(out_faces) == 6
    assert receipt["target_satisfied"] is False
    assert receipt["termination_reason"] == "max_iterations_reached"
    assert receipt["iterations"] == 3
    assert receipt["max_iterations"] == 3
    assert receipt["zero_removal_iterations"] == 3
    assert receipt["final_iteration_removed_faces"] == 0
    assert "final_threshold" not in receipt
    assert receipt["final_executed_threshold"] == pytest.approx(1e-6)
    assert receipt["next_scheduled_threshold"] == pytest.approx(1e-5)
    assert "fixed" not in receipt["termination_reason"]


def test_receipt_reports_target_reached_and_iteration_evidence(monkeypatch) -> None:
    vertices, faces = _mesh(6)

    def remove_two(v, f, **kwargs):
        return v, f[:-2]

    monkeypatch.setattr(qem, "_simplify_step", remove_two)

    _, out_faces, receipt = qem.simplify_qem(
        vertices,
        faces,
        2,
        max_iterations=10,
        return_receipt=True,
        verbose=False,
    )

    assert len(out_faces) == 2
    assert receipt["target_satisfied"] is True
    assert receipt["termination_reason"] == "target_reached"
    assert receipt["iterations"] == 2
    assert receipt["zero_removal_iterations"] == 0
    assert receipt["final_iteration_removed_faces"] == 2


@pytest.mark.parametrize("max_iterations", [0, -1, 1.5, True])
def test_invalid_iteration_cap_fails_before_work(max_iterations) -> None:
    vertices, faces = _mesh()

    with pytest.raises(ValueError, match="max_iterations"):
        qem.simplify_qem(
            vertices,
            faces,
            2,
            max_iterations=max_iterations,
            return_receipt=True,
            verbose=False,
        )
