import pytest


def test_required_model_names_for_512_postcond_decode():
    from scripts.source_cuda_postcond_full_decode_timing import required_model_names

    assert required_model_names("512") == (
        "sparse_structure_decoder",
        "sparse_structure_flow_model",
        "shape_slat_decoder",
        "shape_slat_flow_model_512",
        "tex_slat_decoder",
        "tex_slat_flow_model_512",
    )


def test_required_model_names_for_1024_cascade_postcond_decode():
    from scripts.source_cuda_postcond_full_decode_timing import required_model_names

    assert required_model_names("1024_cascade") == (
        "sparse_structure_decoder",
        "sparse_structure_flow_model",
        "shape_slat_decoder",
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
        "tex_slat_decoder",
        "tex_slat_flow_model_1024",
    )


def test_required_model_names_rejects_unknown_pipeline_type():
    from scripts.source_cuda_postcond_full_decode_timing import required_model_names

    with pytest.raises(ValueError, match="unsupported pipeline_type"):
        required_model_names("1536_cascade")


def test_resolve_model_ref_uses_pipeline_repo_for_relative_models():
    from scripts.source_cuda_postcond_full_decode_timing import resolve_model_ref

    assert (
        resolve_model_ref("microsoft/TRELLIS.2-4B", "ckpts/example")
        == "microsoft/TRELLIS.2-4B/ckpts/example"
    )
    assert (
        resolve_model_ref("microsoft/TRELLIS.2-4B", "microsoft/TRELLIS-image-large/ckpts/ss_dec")
        == "microsoft/TRELLIS-image-large/ckpts/ss_dec"
    )


def test_postcond_decode_runner_defaults_to_dependency_free_sparse_conv_backend():
    from scripts.source_cuda_postcond_full_decode_timing import build_parser

    args = build_parser().parse_args(["--output-json", "out.json", "--output-npz", "out.npz"])

    assert args.sparse_conv_backend == "none"
    assert args.sparse_attn_backend == "sdpa"


def test_apply_sparse_backend_env_sets_dense_attention_alias(monkeypatch):
    from scripts.source_cuda_postcond_full_decode_timing import apply_sparse_backend_env

    applied = apply_sparse_backend_env("none", "sdpa")

    assert applied == {
        "SPARSE_CONV_BACKEND": "none",
        "SPARSE_ATTN_BACKEND": "sdpa",
        "ATTN_BACKEND": "sdpa",
    }
    assert applied["ATTN_BACKEND"] == "sdpa"


def test_postcond_decode_runner_defaults_mesh_override_input():
    from pathlib import Path

    from scripts.source_cuda_postcond_full_decode_timing import build_parser

    args = build_parser().parse_args(["--output-json", "out.json", "--output-npz", "out.npz"])

    assert args.mesh_override == Path("o_voxel_override_convert.py")
    assert args.output_mesh_state is None


def test_install_mesh_override_copies_into_source_stubs(tmp_path):
    from scripts.source_cuda_postcond_full_decode_timing import install_mesh_override

    source_root = tmp_path / "source"
    override = tmp_path / "o_voxel_override_convert.py"
    override.write_text("SENTINEL = 1\n")

    result = install_mesh_override(source_root, override)

    installed = source_root / "stubs" / "o_voxel_override_convert.py"
    assert installed.read_text() == "SENTINEL = 1\n"
    assert result["status"] == "installed"
    assert result["path"] == str(installed)
    assert result["source"] == str(override)


def test_write_binary_mesh_ply_preserves_vertices_and_faces(tmp_path):
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import write_binary_mesh_ply

    class Mesh:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)

    output = tmp_path / "mesh.ply"

    write_binary_mesh_ply(output, Mesh())

    payload = output.read_bytes()
    header, body = payload.split(b"end_header\n", 1)
    assert b"format binary_little_endian 1.0" in header
    assert b"element vertex 3" in header
    assert b"element face 1" in header

    vertices = np.frombuffer(body[: 3 * 3 * 4], dtype="<f4").reshape(3, 3)
    faces = np.frombuffer(
        body[3 * 3 * 4 :],
        dtype=np.dtype([("count", "u1"), ("indices", "<i4", (3,))]),
    )
    np.testing.assert_allclose(vertices, Mesh.vertices)
    assert faces["count"].tolist() == [3]
    assert faces["indices"].tolist() == [[0, 1, 2]]


def test_write_mesh_state_npz_preserves_voxel_payload(tmp_path):
    import json
    import numpy as np

    from scripts.source_cuda_postcond_full_decode_timing import write_mesh_state_npz

    class Mesh:
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        faces = np.array([[0, 1, 1]], dtype=np.int64)
        attrs = np.array([[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]], dtype=np.float32)
        coords = np.array([[3, 4, 5]], dtype=np.int64)
        origin = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        voxel_size = 1 / 512
        voxel_shape = (1, 64, 64, 64)
        layout = {
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        }

    output = tmp_path / "mesh_state.npz"

    write_mesh_state_npz(output, Mesh())

    with np.load(output) as data:
        np.testing.assert_allclose(data["vertices"], Mesh.vertices)
        np.testing.assert_array_equal(data["faces"], Mesh.faces.astype(np.int32))
        np.testing.assert_allclose(data["attrs"], Mesh.attrs)
        np.testing.assert_array_equal(data["coords"], Mesh.coords.astype(np.int32))
        np.testing.assert_allclose(data["origin"], Mesh.origin)
        assert float(data["voxel_size"]) == Mesh.voxel_size
        np.testing.assert_array_equal(data["voxel_shape"], np.array(Mesh.voxel_shape, dtype=np.int64))
        layout = json.loads(str(data["layout_json"]))

    assert layout == {
        "base_color": [0, 3, None],
        "metallic": [3, 4, None],
        "roughness": [4, 5, None],
        "alpha": [5, 6, None],
    }
