import importlib


def test_official_runner_records_raw_512_contract(tmp_path):
    runner = importlib.import_module("scripts.run_official_trellis2")

    image = tmp_path / "source.png"
    image.write_bytes(b"not a real png; parser/identity test only")
    output_dir = tmp_path / "out"

    args = runner.build_parser().parse_args(
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--save-raw-mesh",
            "--seed",
            "42",
            "--steps",
            "8",
            "--pipeline-type",
            "512",
            "--target-faces",
            "350000",
            "--texture-size",
            "4096",
            "--shared-noise",
            "",
        ]
    )

    identity = runner.build_route_identity(args, command=["run_official_trellis2.py"])

    assert identity["schema"] == "trellis2mlx.official_trellis2_route.v1"
    assert identity["route"]["family"] == "local-reference/trellis-mac"
    assert identity["route"]["pipeline_type"] == "512"
    assert identity["route"]["seed"] == 42
    assert identity["route"]["steps"] == 8
    assert identity["route"]["target_faces"] == 350000
    assert identity["route"]["texture_size"] == 4096
    assert identity["source"]["image_path"] == str(image)
    assert identity["source"]["image_sha256"]
    assert identity["requested_outputs"]["raw_mesh"] is True
    assert identity["requested_outputs"]["decoder_output"] is True
    assert identity["requested_outputs"]["final_glb"] is False
    assert identity["forbidden_inferences"] == [
        "not Microsoft CUDA TRELLIS.2 evidence",
        "not final-GLB parity evidence",
        "not texture/bake parity evidence",
    ]


def test_stop_after_raw_mesh_uses_shape_only_cut(monkeypatch, tmp_path):
    runner = importlib.import_module("scripts.run_official_trellis2")
    image = tmp_path / "source.png"
    image.write_bytes(b"parser only")
    args = runner.build_parser().parse_args(
        [
            "--image",
            str(image),
            "--output-dir",
            str(tmp_path / "out"),
            "--save-raw-mesh",
            "--stop-after-raw-mesh",
            "--pipeline-type",
            "512",
        ]
    )
    calls = []

    class Pipeline:
        def run(self, *_args, **_kwargs):  # pragma: no cover - should not be reached
            raise AssertionError("stop-after-raw-mesh must not enter full pipeline.run")

    def fake_raw_only(pipeline, loaded_image, parsed_args):
        calls.append((pipeline, loaded_image, parsed_args))
        return ["raw-mesh"]

    monkeypatch.setattr(runner, "_run_raw_mesh_only", fake_raw_only)

    assert runner._run_mesh_pipeline(Pipeline(), object(), args) == ["raw-mesh"]
    assert calls and calls[0][2] is args
