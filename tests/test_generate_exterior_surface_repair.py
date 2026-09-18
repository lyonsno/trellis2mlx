import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_generate_exposes_single_command_exterior_repair_flag():
    result = subprocess.run(
        [sys.executable, str(ROOT / "generate.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--repair-exterior-surface" in result.stdout


def test_generate_applies_requested_exterior_repair(monkeypatch):
    import generate
    import trellmlx.exterior_surface_repair as repair_module

    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    repaired = faces[:, [0, 2, 1]]
    receipt = {
        "component_orientation": {
            "flipped_components": 2,
            "flipped_faces": 7,
        },
        "orientation": {"selection_status": "selected", "reversed_faces": 1},
    }
    calls = []
    logs = []

    def fake_repair(actual_vertices, actual_faces, *, component_orientation_confidence):
        calls.append(
            (
                actual_vertices.copy(),
                actual_faces.copy(),
                component_orientation_confidence,
            )
        )
        return repaired, receipt

    monkeypatch.setattr(repair_module, "repair_exterior_surface", fake_repair)

    actual_faces, actual_receipt = generate._apply_exterior_surface_repair(
        vertices,
        faces,
        enabled=True,
        component_orientation_confidence=0.75,
        log=lambda message, **__: logs.append(message),
    )

    np.testing.assert_array_equal(actual_faces, repaired)
    assert actual_receipt == receipt
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], vertices)
    np.testing.assert_array_equal(calls[0][1], faces)
    assert calls[0][2] == 0.75
    assert logs == [
        "  Exterior surface repair: oriented 2 components (7 faces); "
        "dominant sheet selected (1 faces reversed)"
    ]


def test_generate_leaves_faces_unchanged_when_repair_is_not_requested():
    import generate

    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)

    actual_faces, receipt = generate._apply_exterior_surface_repair(
        vertices,
        faces,
        enabled=False,
        component_orientation_confidence=0.5,
        log=lambda *_, **__: None,
    )

    assert actual_faces is faces
    assert receipt is None
