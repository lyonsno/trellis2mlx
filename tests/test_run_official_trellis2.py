from types import SimpleNamespace


def test_build_run_kwargs_records_seed_steps_pipeline_and_preprocess_switch():
    from scripts.run_official_trellis2 import _build_run_kwargs

    args = SimpleNamespace(
        seed=101,
        steps=8,
        pipeline_type="512",
        no_preprocess=True,
    )

    assert _build_run_kwargs(args) == {
        "seed": 101,
        "pipeline_type": "512",
        "preprocess_image": False,
        "sparse_structure_sampler_params": {"steps": 8},
        "shape_slat_sampler_params": {"steps": 8},
        "tex_slat_sampler_params": {"steps": 8},
    }
