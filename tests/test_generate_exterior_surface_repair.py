import subprocess
import sys
import hashlib
import json
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


def test_final_glb_checkpoint_replaces_stale_binding_for_resumed_output(tmp_path):
    import generate

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    output = tmp_path / "output.glb"
    output.write_bytes(b"current resumed output")
    np.savez(
        checkpoint_dir / "final_glb.npz",
        output_path=np.asarray("stale.glb"),
        output_sha256=np.asarray("stale"),
    )
    postprocess_route = {
        "repair_exterior_surface": True,
        "exterior_orientation_confidence": 0.5,
        "exterior_surface_repair": {"schema": "test-receipt"},
    }
    resume_identity = {"mesh_raw.npz": "a" * 64, "texture.npz": "b" * 64}

    generate._save_final_glb_checkpoint(
        checkpoint_dir,
        output,
        producer_mode="resume",
        postprocess_route=postprocess_route,
        route_arrays={
            "resume_source_identity_json": np.asarray(
                json.dumps(resume_identity, sort_keys=True)
            )
        },
    )

    with np.load(checkpoint_dir / "final_glb.npz", allow_pickle=False) as data:
        assert data["output_path"].item() == str(output)
        assert data["output_sha256"].item() == hashlib.sha256(
            output.read_bytes()
        ).hexdigest()
        assert data["producer_mode"].item() == "resume"
        assert json.loads(data["postprocess_route_json"].item()) == postprocess_route
        assert json.loads(data["resume_source_identity_json"].item()) == resume_identity
