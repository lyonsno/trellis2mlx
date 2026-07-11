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
