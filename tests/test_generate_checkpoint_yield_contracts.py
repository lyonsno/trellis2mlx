from pathlib import Path


GENERATE_SOURCE = Path(__file__).resolve().parents[1] / "generate.py"


def test_generate_exposes_checkpoint_stop_file_cli():
    source = GENERATE_SOURCE.read_text()

    assert "--checkpoint-stop-file" in source
    assert "--save-checkpoints is required when --checkpoint-stop-file is set" in source


def test_generate_checks_checkpoint_yield_after_durable_checkpoint_saves():
    source = GENERATE_SOURCE.read_text()

    mesh_save = source.index('save_checkpoint(args.save_checkpoints, "mesh_raw"')
    mesh_yield = source.index('completed_stage="mesh_raw"')
    clean_save = source.index('save_checkpoint(args.save_checkpoints, "mesh_clean"', mesh_yield)
    clean_yield = source.index('completed_stage="mesh_clean"', clean_save)
    texture_save = source.index('save_checkpoint(args.save_checkpoints, "texture"', clean_yield)
    texture_yield = source.index('completed_stage="texture"', texture_save)
    uv_save = source.index('save_checkpoint(args.save_checkpoints, "mesh_uv"', texture_yield)
    uv_yield = source.index('completed_stage="mesh_uv"', uv_save)

    assert source.count("maybe_checkpoint_yield(") >= 4
    assert mesh_save < mesh_yield
    assert mesh_yield < clean_save < clean_yield
    assert clean_yield < texture_save < texture_yield
    assert texture_save < texture_yield
    assert texture_yield < uv_save < uv_yield


def test_generate_mesh_stage_checkpoints_have_attribution_arrays():
    source = GENERATE_SOURCE.read_text()

    mesh_yield = source.index('completed_stage="mesh_raw"')
    clean_save = source.index('save_checkpoint(args.save_checkpoints, "mesh_clean"', mesh_yield)
    clean_block = source[clean_save:source.index('completed_stage="mesh_clean"', clean_save)]
    assert "vertices=vertices" in clean_block
    assert "faces=faces" in clean_block
    assert "mesh_grid_size=mesh_grid_size" in clean_block

    texture_yield = source.index('completed_stage="texture"', clean_save)
    uv_save = source.index('save_checkpoint(args.save_checkpoints, "mesh_uv"', texture_yield)
    uv_block = source[uv_save:source.index('completed_stage="mesh_uv"', uv_save)]
    assert "vertices=uv_verts" in uv_block
    assert "faces=uv_faces" in uv_block
    assert "uvs=uvs" in uv_block
    assert "vmapping=vmapping" in uv_block


def test_generate_marks_mesh_raw_yield_as_not_resume_supported_yet():
    source = GENERATE_SOURCE.read_text()

    mesh_yield = source.index('completed_stage="mesh_raw"')
    texture_yield = source.index('completed_stage="texture"')
    mesh_block = source[mesh_yield:texture_yield]

    assert "resume_supported=False" in mesh_block
    assert "mesh-only resume is not implemented" in mesh_block
