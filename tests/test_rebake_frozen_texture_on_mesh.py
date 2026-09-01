import importlib.util
from pathlib import Path

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
