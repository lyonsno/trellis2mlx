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
    texture_save = source.index('save_checkpoint(args.save_checkpoints, "texture"')
    texture_yield = source.index('completed_stage="texture"')

    assert source.count("maybe_checkpoint_yield(") >= 2
    assert mesh_save < mesh_yield
    assert texture_save < texture_yield


def test_generate_marks_mesh_raw_yield_as_not_resume_supported_yet():
    source = GENERATE_SOURCE.read_text()

    mesh_yield = source.index('completed_stage="mesh_raw"')
    texture_yield = source.index('completed_stage="texture"')
    mesh_block = source[mesh_yield:texture_yield]

    assert "resume_supported=False" in mesh_block
    assert "mesh-only resume is not implemented" in mesh_block
