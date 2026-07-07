from pathlib import Path
import sys
import types

import pytest


def test_source_route_identity_rejects_hunyuan_backend_path():
    from trellmlx.source_route_identity import SourceRouteIdentityError, validate_source_route_identity

    identity = {
        "status": "available",
        "python": "/Users/noahlyons/dev/trellis-mac/.venv/bin/python",
        "cumesh_path": [
            "/Users/noahlyons/dev/trellis-mac/.venv/lib/python3.11/site-packages/cumesh"
        ],
        "metal_backend_file": (
            "/private/tmp/Hunyuan3D-MLX-trellis2mlx-qem-cost-readback-0706/"
            "libraries/mtlmesh/cumesh/metal_backend.py"
        ),
        "git_root": "/private/tmp/Hunyuan3D-MLX-trellis2mlx-qem-cost-readback-0706",
        "git_remote": "https://github.com/ZimengXiong/Hunyuan3D-MLX.git",
        "git_commit": "481acc3c73cd6645ea0b6537d274f8bc0306765b",
    }

    with pytest.raises(SourceRouteIdentityError, match="forbidden source route"):
        validate_source_route_identity(identity)


def test_source_route_identity_accepts_expected_trellis_mac_mtlmesh_root():
    from trellmlx.source_route_identity import validate_source_route_identity

    root = Path("/Users/noahlyons/dev/trellis-mac/deps/mtlmesh")
    identity = {
        "status": "available",
        "python": "/Users/noahlyons/dev/trellis-mac/.venv/bin/python",
        "cumesh_path": [
            "/Users/noahlyons/dev/trellis-mac/.venv/lib/python3.11/site-packages/cumesh"
        ],
        "metal_backend_file": str(root / "cumesh/metal_backend.py"),
        "git_root": str(root),
        "git_remote": "https://github.com/pedronaugusto/mtlmesh.git",
        "git_commit": "212079e55772cff3d648a21372392c37e0643f3b",
    }

    validated = validate_source_route_identity(identity, expected_root=root)

    assert validated["status"] == "available"
    assert validated["source_root_match"] is True
    assert validated["forbidden_source_route"] is False


def test_source_route_identity_rejects_wrong_root_even_without_hunyuan_marker():
    from trellmlx.source_route_identity import SourceRouteIdentityError, validate_source_route_identity

    identity = {
        "status": "available",
        "metal_backend_file": "/tmp/other-mtlmesh/cumesh/metal_backend.py",
        "git_root": "/tmp/other-mtlmesh",
    }

    with pytest.raises(SourceRouteIdentityError, match="expected source root"):
        validate_source_route_identity(
            identity,
            expected_root="/Users/noahlyons/dev/trellis-mac/deps/mtlmesh",
        )


def test_source_native_loader_rejects_hunyuan_import_route(monkeypatch):
    import trellmlx.source_mtlmesh as source_mtlmesh

    cumesh = types.ModuleType("cumesh")
    cumesh.__path__ = ["/tmp/site-packages/cumesh"]
    metal_backend = types.ModuleType("cumesh.metal_backend")
    metal_backend.__file__ = (
        "/private/tmp/Hunyuan3D-MLX-trellis2mlx-qem-cost-readback-0706/"
        "libraries/mtlmesh/cumesh/metal_backend.py"
    )
    metal_backend.MtlMesh = type("MtlMesh", (), {})

    monkeypatch.setitem(sys.modules, "cumesh", cumesh)
    monkeypatch.setitem(sys.modules, "cumesh.metal_backend", metal_backend)

    with pytest.raises(RuntimeError, match="source-native.*rejected route"):
        source_mtlmesh._load_source_mesh_class()
