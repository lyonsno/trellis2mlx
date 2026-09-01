import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "rebake_frozen_texture_on_mesh.py"
SPEC = importlib.util.spec_from_file_location("rebake_frozen_texture_on_mesh", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_require_sha256_accepts_exact_digest(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"fixed product")

    digest = MODULE.sha256_file(artifact)

    assert MODULE.require_sha256(artifact, digest) == digest


def test_require_sha256_rejects_wrong_digest(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"fixed product")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        MODULE.require_sha256(artifact, "0" * 64)


def test_texture_backend_is_bound_to_production_vocabulary(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--mesh",
            "mesh.ply",
            "--texture-checkpoint-dir",
            "checkpoints",
            "--output",
            "output.glb",
            "--report",
            "report.json",
            "--mesh-grid-size",
            "512",
            "--texture-backend",
            "gpu",
        ],
    )

    assert MODULE.parse_args().texture_backend == "gpu"

    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--mesh",
            "mesh.ply",
            "--texture-checkpoint-dir",
            "checkpoints",
            "--output",
            "output.glb",
            "--report",
            "report.json",
            "--mesh-grid-size",
            "512",
            "--texture-backend",
            "metal",
        ],
    )
    with pytest.raises(SystemExit):
        MODULE.parse_args()


def test_xatlas_winding_repair_is_explicit_and_rejects_other_uv_routes(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--mesh",
            "mesh.ply",
            "--texture-checkpoint-dir",
            "checkpoints",
            "--output",
            "output.glb",
            "--report",
            "report.json",
            "--mesh-grid-size",
            "512",
            "--uv-method",
            "xatlas",
            "--xatlas-fix-winding",
        ],
    )

    args = MODULE.parse_args()
    assert args.xatlas_fix_winding is True
    unwrap = MODULE.select_unwrap(
        args.uv_method,
        xatlas_fix_winding=args.xatlas_fix_winding,
    )
    assert unwrap.keywords == {"fix_winding": True}

    with pytest.raises(ValueError, match="only applies to xatlas"):
        MODULE.select_unwrap("cube", xatlas_fix_winding=True)


def test_connected_component_orientation_flips_only_confidently_inward_sheet():
    octahedron = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    outward = np.array(
        [
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 2],
            [1, 3, 2],
            [1, 4, 3],
            [1, 5, 4],
            [1, 2, 5],
        ],
        dtype=np.int32,
    )
    vertices = np.concatenate([octahedron, octahedron])
    inward = outward[:, [0, 2, 1]] + len(octahedron)
    faces = np.concatenate([outward, inward])

    oriented, receipt = MODULE.orient_connected_components_outward(
        vertices,
        faces,
        min_confidence=0.5,
    )

    assert np.array_equal(oriented[:8], outward)
    assert np.array_equal(oriented[8:], outward + len(octahedron))
    assert receipt == {
        "components": 2,
        "flipped_components": 1,
        "flipped_faces": 8,
        "min_confidence": 0.5,
    }


def test_connected_component_orientation_threshold_rejects_ambiguous_inward_sheet():
    vertices = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    outward = np.array(
        [
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 2],
            [1, 3, 2],
            [1, 4, 3],
            [1, 5, 4],
            [1, 2, 5],
        ],
        dtype=np.int32,
    )
    mostly_inward = outward[:, [0, 2, 1]].copy()
    mostly_inward[0] = outward[0]

    low_threshold, low_receipt = MODULE.orient_connected_components_outward(
        vertices,
        mostly_inward,
        min_confidence=0.5,
    )
    high_threshold, high_receipt = MODULE.orient_connected_components_outward(
        vertices,
        mostly_inward,
        min_confidence=0.9,
    )

    assert np.array_equal(low_threshold, mostly_inward[:, [0, 2, 1]])
    assert low_receipt["flipped_components"] == 1
    assert low_receipt["flipped_faces"] == 8
    assert np.array_equal(high_threshold, mostly_inward)
    assert high_receipt["flipped_components"] == 0
    assert high_receipt["flipped_faces"] == 0


def test_connected_component_orientation_preserves_empty_face_dtype():
    vertices = np.zeros((0, 3), dtype=np.float32)
    faces = np.zeros((0, 3), dtype=np.uint32)

    oriented, receipt = MODULE.orient_connected_components_outward(vertices, faces)

    assert oriented.shape == (0, 3)
    assert oriented.dtype == np.uint32
    assert receipt["components"] == 0


@pytest.mark.parametrize("confidence", [-0.01, 1.01, np.nan, np.inf, -np.inf])
def test_connected_component_orientation_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError, match="between 0 and 1"):
        MODULE.orient_connected_components_outward(
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            min_confidence=confidence,
        )


def test_prepare_faces_for_bake_default_off_does_not_invoke_orientation(monkeypatch):
    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("disabled route invoked orientation")

    monkeypatch.setattr(
        MODULE,
        "orient_connected_components_outward",
        fail_if_called,
    )

    prepared, receipt = MODULE.prepare_faces_for_bake(
        vertices,
        faces,
        orient_connected_components=False,
        orientation_confidence=0.5,
    )

    assert np.array_equal(prepared, faces)
    assert prepared is not faces
    assert prepared.dtype == faces.dtype
    assert receipt == {
        "requested": False,
        "applied": False,
        "components": None,
        "flipped_components": 0,
        "flipped_faces": 0,
        "min_confidence": 0.5,
    }


def test_prepare_faces_for_bake_enabled_records_effective_counts(monkeypatch):
    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    flipped = faces[:, [0, 2, 1]]

    def fake_orientation(actual_vertices, actual_faces, *, min_confidence):
        assert actual_vertices is vertices
        assert actual_faces is faces
        assert min_confidence == 0.75
        return flipped, {
            "components": 3,
            "flipped_components": 2,
            "flipped_faces": 17,
            "min_confidence": 0.75,
        }

    monkeypatch.setattr(
        MODULE,
        "orient_connected_components_outward",
        fake_orientation,
    )

    prepared, receipt = MODULE.prepare_faces_for_bake(
        vertices,
        faces,
        orient_connected_components=True,
        orientation_confidence=0.75,
    )

    assert np.array_equal(prepared, flipped)
    assert receipt == {
        "requested": True,
        "applied": True,
        "components": 3,
        "flipped_components": 2,
        "flipped_faces": 17,
        "min_confidence": 0.75,
    }


def test_implementation_identity_is_snapshotted_before_later_file_mutation(tmp_path):
    repo_root = tmp_path / "repo"
    script = repo_root / "scripts" / "rebake.py"
    dependency = repo_root / "trellmlx" / "texture_bake.py"
    script.parent.mkdir(parents=True)
    dependency.parent.mkdir(parents=True)
    script.write_text("original script\n")
    dependency.write_text("original dependency\n")

    identity = MODULE.snapshot_implementation_identity(
        repo_root=repo_root,
        files=(script, dependency),
        git_head="test-base",
    )
    original_script_sha = MODULE.sha256_file(script)
    original_manifest_sha = identity["manifest_sha256"]

    script.write_text("mutated after startup\n")
    later_identity = MODULE.snapshot_implementation_identity(
        repo_root=repo_root,
        files=(script, dependency),
        git_head="test-base",
    )

    assert identity["git_head"] == "test-base"
    assert identity["files"][0] == {
        "path": "scripts/rebake.py",
        "sha256": original_script_sha,
    }
    assert identity["manifest_sha256"] == original_manifest_sha
    assert later_identity["files"][0]["sha256"] == MODULE.sha256_file(script)
    assert later_identity["manifest_sha256"] != original_manifest_sha


def test_prebake_orientation_is_explicit_and_defaults_off(monkeypatch):
    base_args = [
        str(SCRIPT),
        "--mesh",
        "mesh.ply",
        "--texture-checkpoint-dir",
        "checkpoints",
        "--output",
        "output.glb",
        "--report",
        "report.json",
        "--mesh-grid-size",
        "512",
    ]
    monkeypatch.setattr("sys.argv", base_args)
    args = MODULE.parse_args()
    assert args.orient_connected_components_outward is False
    assert args.orientation_confidence == 0.5

    monkeypatch.setattr(
        "sys.argv",
        base_args
        + [
            "--orient-connected-components-outward",
            "--orientation-confidence",
            "0.75",
        ],
    )
    args = MODULE.parse_args()
    assert args.orient_connected_components_outward is True
    assert args.orientation_confidence == 0.75


def test_validate_output_paths_rejects_aliases_and_existing_outputs(tmp_path):
    mesh = (tmp_path / "mesh.ply").resolve()
    texture_npz = (tmp_path / "texture.npz").resolve()
    texture_json = (tmp_path / "texture.json").resolve()
    output = (tmp_path / "output.glb").resolve()
    report = (tmp_path / "report.json").resolve()

    with pytest.raises(ValueError, match="protected input"):
        MODULE.validate_output_paths(
            mesh_path=mesh,
            texture_npz=texture_npz,
            texture_json=texture_json,
            output_path=mesh,
            report_path=report,
            overwrite=False,
        )

    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="overwrite"):
        MODULE.validate_output_paths(
            mesh_path=mesh,
            texture_npz=texture_npz,
            texture_json=texture_json,
            output_path=output,
            report_path=report,
            overwrite=False,
        )

    MODULE.validate_output_paths(
        mesh_path=mesh,
        texture_npz=texture_npz,
        texture_json=texture_json,
        output_path=output,
        report_path=report,
        overwrite=True,
    )
