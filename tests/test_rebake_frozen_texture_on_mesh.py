import importlib.util
import hashlib
import json
import os
import stat
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


def test_face_reversal_manifest_flips_only_named_source_faces(tmp_path):
    faces = np.array(
        [
            [0, 1, 2],
            [2, 3, 0],
            [4, 5, 6],
        ],
        dtype=np.uint32,
    )
    source_sha256 = "6" * 64
    manifest = tmp_path / "crown-subsheet.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.face-reversal-manifest.v1",
                "semantic_name": "feature-animation-81414-crown-root-subsheet",
                "source_mesh": {
                    "sha256": source_sha256,
                    "faces": 3,
                },
                "face_indices": [0, 2],
            }
        )
    )

    prepared, receipt = MODULE.apply_face_reversal_manifest(
        faces,
        manifest_path=manifest,
        source_mesh_sha256=source_sha256,
    )

    assert np.array_equal(prepared[0], faces[0, [0, 2, 1]])
    assert np.array_equal(prepared[1], faces[1])
    assert np.array_equal(prepared[2], faces[2, [0, 2, 1]])
    assert prepared.dtype == faces.dtype
    assert receipt == {
        "requested": True,
        "applied": True,
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": MODULE.sha256_file(manifest),
        "manifest_schema": "trellis2mlx.face-reversal-manifest.v1",
        "semantic_name": "feature-animation-81414-crown-root-subsheet",
        "source_mesh_sha256": source_sha256,
        "source_mesh_faces": 3,
        "reversed_faces": 2,
    }


def test_face_reversal_manifest_default_off_preserves_faces():
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    prepared, receipt = MODULE.apply_face_reversal_manifest(
        faces,
        manifest_path=None,
        source_mesh_sha256="7" * 64,
    )

    assert np.array_equal(prepared, faces)
    assert prepared is not faces
    assert receipt == {
        "requested": False,
        "applied": False,
        "manifest_path": None,
        "manifest_sha256": None,
        "manifest_schema": None,
        "semantic_name": None,
        "source_mesh_sha256": "7" * 64,
        "source_mesh_faces": 1,
        "reversed_faces": 0,
    }


def test_face_reversal_manifest_hashes_and_parses_one_captured_byte_sequence(
    tmp_path,
    monkeypatch,
):
    manifest = (tmp_path / "changing-manifest.json").resolve()
    first_payload = json.dumps(
        {
            "schema": "trellis2mlx.face-reversal-manifest.v1",
            "semantic_name": "first captured selection",
            "source_mesh": {"sha256": "6" * 64, "faces": 2},
            "face_indices": [0],
        }
    ).encode()
    later_payload = json.dumps(
        {
            "schema": "trellis2mlx.face-reversal-manifest.v1",
            "semantic_name": "later substituted selection",
            "source_mesh": {"sha256": "6" * 64, "faces": 2},
            "face_indices": [1],
        }
    ).encode()
    read_count = 0

    def changing_read_bytes(path):
        nonlocal read_count
        assert path == manifest
        read_count += 1
        return first_payload if read_count == 1 else later_payload

    def reject_split_read(*args, **kwargs):
        raise AssertionError("manifest must be hashed and parsed from one byte capture")

    monkeypatch.setattr(Path, "read_bytes", changing_read_bytes)
    monkeypatch.setattr(Path, "read_text", reject_split_read)
    monkeypatch.setattr(MODULE, "sha256_file", reject_split_read)

    faces = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int32)
    prepared, receipt = MODULE.apply_face_reversal_manifest(
        faces,
        manifest_path=manifest,
        source_mesh_sha256="6" * 64,
    )

    assert read_count == 1
    assert np.array_equal(prepared[0], faces[0, [0, 2, 1]])
    assert np.array_equal(prepared[1], faces[1])
    assert receipt["semantic_name"] == "first captured selection"
    assert receipt["manifest_sha256"] == hashlib.sha256(first_payload).hexdigest()


@pytest.mark.parametrize(
    ("manifest_update", "message"),
    [
        ({"schema": "wrong.schema"}, "unsupported face reversal manifest schema"),
        ({"semantic_name": ""}, "semantic_name must be a non-empty string"),
        ({"source_mesh": {"sha256": "8" * 64, "faces": 3}}, "source mesh SHA256 mismatch"),
        ({"source_mesh": {"sha256": "6" * 64, "faces": 4}}, "source mesh face count mismatch"),
        ({"face_indices": []}, "face_indices must be a non-empty list"),
        ({"face_indices": [0, 0]}, "face_indices must be unique"),
        ({"face_indices": [0, 3]}, "face index out of range"),
        ({"face_indices": [True]}, "face_indices must contain integers"),
    ],
)
def test_face_reversal_manifest_rejects_untrusted_selection(
    tmp_path,
    manifest_update,
    message,
):
    payload = {
        "schema": "trellis2mlx.face-reversal-manifest.v1",
        "semantic_name": "feature-animation-81414-crown-root-subsheet",
        "source_mesh": {
            "sha256": "6" * 64,
            "faces": 3,
        },
        "face_indices": [0, 2],
    }
    payload.update(manifest_update)
    manifest = tmp_path / "crown-subsheet.json"
    manifest.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        MODULE.apply_face_reversal_manifest(
            np.array([[0, 1, 2], [2, 3, 0], [4, 5, 6]], dtype=np.int32),
            manifest_path=manifest,
            source_mesh_sha256="6" * 64,
        )


def test_feature_animation_crown_manifest_preserves_proven_selection():
    manifest_path = (
        Path(__file__).parents[1]
        / "assets"
        / "repairs"
        / "feature-animation-81414-crown-root-subsheet.json"
    )
    payload = json.loads(manifest_path.read_text())

    assert payload["schema"] == "trellis2mlx.face-reversal-manifest.v1"
    assert payload["source_mesh"] == {
        "sha256": "6477dfef060bea007efac964b048cb086a98d84054d78b8a2fed528d06441029",
        "faces": 99837,
    }
    assert len(payload["face_indices"]) == 666
    assert len(set(payload["face_indices"])) == 666


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
    assert args.face_reversal_manifest is None

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

    monkeypatch.setattr(
        "sys.argv",
        base_args + ["--face-reversal-manifest", "crown-subsheet.json"],
    )
    args = MODULE.parse_args()
    assert args.face_reversal_manifest == Path("crown-subsheet.json")


def test_validate_output_paths_rejects_aliases_and_existing_outputs(tmp_path):
    mesh = (tmp_path / "mesh.ply").resolve()
    texture_npz = (tmp_path / "texture.npz").resolve()
    texture_json = (tmp_path / "texture.json").resolve()
    output = (tmp_path / "output.glb").resolve()
    report = (tmp_path / "report.json").resolve()
    manifest = (tmp_path / "face-reversal.json").resolve()

    with pytest.raises(ValueError, match="protected input"):
        MODULE.validate_output_paths(
            mesh_path=mesh,
            texture_npz=texture_npz,
            texture_json=texture_json,
            output_path=mesh,
            report_path=report,
            overwrite=False,
        )

    with pytest.raises(ValueError, match="protected input"):
        MODULE.validate_output_paths(
            mesh_path=mesh,
            texture_npz=texture_npz,
            texture_json=texture_json,
            face_reversal_manifest=manifest,
            output_path=manifest,
            report_path=report,
            overwrite=True,
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


@pytest.mark.parametrize("staging_owner", ["output", "report"])
def test_validate_output_paths_rejects_manifest_alias_with_internal_staging_name(
    tmp_path,
    staging_owner,
):
    mesh = (tmp_path / "mesh.ply").resolve()
    texture_npz = (tmp_path / "texture.npz").resolve()
    texture_json = (tmp_path / "texture.json").resolve()
    output = (tmp_path / "output.glb").resolve()
    report = (tmp_path / "report.json").resolve()
    owner = output if staging_owner == "output" else report
    manifest = owner.with_name(owner.name + ".tmp")

    with pytest.raises(ValueError, match="protected input"):
        MODULE.validate_output_paths(
            mesh_path=mesh,
            texture_npz=texture_npz,
            texture_json=texture_json,
            face_reversal_manifest=manifest,
            output_path=output,
            report_path=report,
            overwrite=True,
        )


def test_manifest_rejection_writes_durable_cli_failure_report(tmp_path, monkeypatch):
    mesh = tmp_path / "mesh.ply"
    mesh.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "element face 1",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "3 0 1 2",
            ]
        )
        + "\n"
    )
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    (checkpoint / "texture.npz").write_bytes(b"not reached")
    (checkpoint / "texture.json").write_text("{}\n")
    manifest = tmp_path / "wrong-source.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.face-reversal-manifest.v1",
                "semantic_name": "wrong source selection",
                "source_mesh": {"sha256": "0" * 64, "faces": 1},
                "face_indices": [0],
            }
        )
    )
    output = tmp_path / "output.glb"
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--mesh",
            str(mesh),
            "--texture-checkpoint-dir",
            str(checkpoint),
            "--output",
            str(output),
            "--report",
            str(report),
            "--mesh-grid-size",
            "512",
            "--face-reversal-manifest",
            str(manifest),
            "--texture-backend",
            "cpu",
        ],
    )

    with pytest.raises(ValueError, match="source mesh SHA256 mismatch"):
        MODULE.main()

    failure = json.loads(report.read_text())
    mesh_sha256 = MODULE.sha256_file(mesh)
    assert failure["schema"] == "trellis2mlx.frozen-texture-rebake.failure.v1"
    assert failure["status"] == "failed"
    assert failure["failure"]["phase"] == "face_reversal_manifest"
    assert failure["failure"]["type"] == "ValueError"
    assert "source mesh SHA256 mismatch" in failure["failure"]["message"]
    assert failure["effective"]["mesh"] == {
        "path": str(mesh.resolve()),
        "sha256": mesh_sha256,
        "vertices": 3,
        "faces": 1,
    }
    assert failure["effective"]["face_reversal"] == {
        "requested": True,
        "applied": False,
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": MODULE.sha256_file(manifest),
        "source_mesh_sha256": mesh_sha256,
        "source_mesh_faces": 1,
        "reversed_faces": 0,
    }
    assert failure["primary_output"] == {
        "path": str(output.resolve()),
        "exists_after": False,
        "existed_before": False,
        "produced_by_attempt": False,
    }
    assert failure["requested"]["route"]["face_reversal_manifest"] == str(
        manifest.resolve()
    )
    assert not output.exists()


def test_invalid_mesh_carrier_failure_report_retains_known_input_identity(
    tmp_path,
    monkeypatch,
):
    import trimesh

    mesh = tmp_path / "scene.glb"
    mesh.write_bytes(b"carrier identity established before load")
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    (checkpoint / "texture.npz").write_bytes(b"not reached")
    (checkpoint / "texture.json").write_text("{}\n")
    output = tmp_path / "output.glb"
    report = tmp_path / "report.json"
    monkeypatch.setattr(trimesh, "load", lambda *args, **kwargs: trimesh.Scene())
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--mesh",
            str(mesh),
            "--texture-checkpoint-dir",
            str(checkpoint),
            "--output",
            str(output),
            "--report",
            str(report),
            "--mesh-grid-size",
            "512",
            "--texture-backend",
            "cpu",
        ],
    )

    with pytest.raises(TypeError, match="expected one Trimesh"):
        MODULE.main()

    failure = json.loads(report.read_text())
    assert failure["failure"]["phase"] == "mesh_load"
    assert failure["effective"]["mesh"] == {
        "path": str(mesh.resolve()),
        "sha256": MODULE.sha256_file(mesh),
        "carrier_type": "Scene",
        "vertices": None,
        "faces": None,
    }
    assert failure["primary_output"]["produced_by_attempt"] is False
    assert failure["primary_output"]["exists_after"] is False


def test_no_overwrite_refusal_preserves_existing_report_and_writes_sibling_failure(
    tmp_path,
    monkeypatch,
):
    mesh = tmp_path / "mesh.ply"
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    output = tmp_path / "output.glb"
    report = tmp_path / "report.json"
    sentinel = b'{"admitted": "prior durable evidence"}\n'
    report.write_bytes(sentinel)
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--mesh",
            str(mesh),
            "--texture-checkpoint-dir",
            str(checkpoint),
            "--output",
            str(output),
            "--report",
            str(report),
            "--mesh-grid-size",
            "512",
        ],
    )

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        MODULE.main()

    assert report.read_bytes() == sentinel
    assert not output.exists()
    failure_paths = list(tmp_path.glob("report.failure-*.json"))
    assert len(failure_paths) == 1
    failure = json.loads(failure_paths[0].read_text())
    assert failure["failure"]["phase"] == "output_path_validation"
    assert failure["failure"]["type"] == "FileExistsError"
    assert failure["publication"] == {
        "overwrite_authorized": False,
        "requested_report_path": str(report.resolve()),
        "requested_report_preexisting": True,
        "effective_failure_report_path": str(failure_paths[0].resolve()),
        "mode": "exclusive_sibling_no_clobber",
    }
    assert failure["primary_output"] == {
        "path": str(output.resolve()),
        "exists_after": False,
        "existed_before": False,
        "produced_by_attempt": False,
    }


def test_overwrite_authorization_allows_requested_failure_report_replacement(
    tmp_path,
    monkeypatch,
):
    mesh = tmp_path / "mesh.ply"
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    output = tmp_path / "output.glb"
    report = tmp_path / "report.json"
    report.write_text('{"stale": true}\n')
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--mesh",
            str(mesh),
            "--texture-checkpoint-dir",
            str(checkpoint),
            "--output",
            str(output),
            "--report",
            str(report),
            "--mesh-grid-size",
            "512",
            "--overwrite",
        ],
    )

    def fail_after_authorized_validation(args, state):
        state["phase"] = "authorized_failure_probe"
        raise RuntimeError("authorized replacement probe")

    monkeypatch.setattr(MODULE, "run", fail_after_authorized_validation)

    with pytest.raises(RuntimeError, match="authorized replacement probe"):
        MODULE.main()

    failure = json.loads(report.read_text())
    assert failure["publication"] == {
        "overwrite_authorized": True,
        "requested_report_path": str(report.resolve()),
        "requested_report_preexisting": True,
        "effective_failure_report_path": str(report.resolve()),
        "mode": "requested_report_atomic_replace",
    }
    assert list(tmp_path.glob("report.failure-*.json")) == []


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink", "unrelated"])
def test_exclusive_output_staging_ignores_preplanted_deterministic_name(
    tmp_path,
    alias_kind,
):
    manifest = tmp_path / "manifest.json"
    manifest_bytes = b'{"face_indices": [1, 2, 3]}\n'
    manifest.write_bytes(manifest_bytes)
    output = tmp_path / "output.glb"
    planted = tmp_path / "output.glb.tmp"
    if alias_kind == "symlink":
        planted.symlink_to(manifest)
    elif alias_kind == "hardlink":
        os.link(manifest, planted)
    else:
        planted.write_bytes(b"unrelated caller-owned staging file")
    planted_lstat = planted.lstat()

    MODULE.write_bytes_atomically(output, b"exclusive glb payload")

    assert output.read_bytes() == b"exclusive glb payload"
    assert manifest.read_bytes() == manifest_bytes
    assert planted.lstat().st_ino == planted_lstat.st_ino
    if alias_kind == "unrelated":
        assert planted.read_bytes() == b"unrelated caller-owned staging file"


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink", "unrelated"])
def test_exclusive_success_report_staging_ignores_preplanted_deterministic_name(
    tmp_path,
    alias_kind,
):
    manifest = tmp_path / "manifest.json"
    manifest_bytes = b'{"face_indices": [1, 2, 3]}\n'
    manifest.write_bytes(manifest_bytes)
    report = tmp_path / "report.json"
    planted = tmp_path / "report.json.tmp"
    if alias_kind == "symlink":
        planted.symlink_to(manifest)
    elif alias_kind == "hardlink":
        os.link(manifest, planted)
    else:
        planted.write_bytes(b"unrelated caller-owned staging file")
    planted_lstat = planted.lstat()

    MODULE.publish_success_report(report, {"status": "succeeded"})

    assert json.loads(report.read_text()) == {"status": "succeeded"}
    assert manifest.read_bytes() == manifest_bytes
    assert planted.lstat().st_ino == planted_lstat.st_ino
    if alias_kind == "unrelated":
        assert planted.read_bytes() == b"unrelated caller-owned staging file"


def test_interrupted_failure_serialization_never_exposes_final_pattern(
    tmp_path,
    monkeypatch,
):
    requested_report = tmp_path / "report.json"
    payload = {
        "schema": "trellis2mlx.frozen-texture-rebake.failure.v1",
        "publication": {"effective_failure_report_path": None},
    }

    def interrupt_after_prefix(value, handle, **kwargs):
        handle.write('{"partial":')
        handle.flush()
        raise KeyboardInterrupt("serialization interrupted")

    monkeypatch.setattr(MODULE.json, "dump", interrupt_after_prefix)

    with pytest.raises(KeyboardInterrupt, match="serialization interrupted"):
        MODULE.write_json_exclusively_beside(requested_report, payload)

    assert list(tmp_path.glob("report.failure-*.json")) == []
    assert list(tmp_path.glob(".report.failure-*.tmp")) == []


def test_atomic_publishers_follow_controlled_umask_for_fresh_and_replaced_files(
    tmp_path,
):
    output = tmp_path / "output.glb"
    success_report = tmp_path / "success.json"
    requested_report = tmp_path / "requested.json"
    requested_report.write_text('{"stale": true}\n')
    expected_mode = 0o640
    original_umask = os.umask(0o027)
    try:
        MODULE.write_bytes_atomically(output, b"glb bytes")
        MODULE.publish_success_report(success_report, {"status": "succeeded"})
        failure_payload = {
            "schema": "trellis2mlx.frozen-texture-rebake.failure.v1",
            "publication": {"effective_failure_report_path": None},
        }
        failure_report = MODULE.write_json_exclusively_beside(
            requested_report,
            failure_payload,
        )
        MODULE.write_json_atomically(
            requested_report,
            {"status": "authorized replacement"},
        )
    finally:
        os.umask(original_umask)

    assert stat.S_IMODE(output.stat().st_mode) == expected_mode
    assert stat.S_IMODE(success_report.stat().st_mode) == expected_mode
    assert stat.S_IMODE(failure_report.stat().st_mode) == expected_mode
    assert stat.S_IMODE(requested_report.stat().st_mode) == expected_mode
