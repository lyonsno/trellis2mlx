import hashlib
import json
from pathlib import Path
import types

import pytest


EXPECTED_STAGES = (
    "preprocessed_image",
    "conditioning_512",
    "sparse_flow",
    "sparse_support",
    "shape_flow",
    "shape_slat",
    "texture_flow",
    "decoder_raw_mesh",
    "texture_voxels",
    "pipeline_filled_mesh",
    "postprocess_stage11_pre_orientation",
    "postprocess_stage12_post_orientation",
    "consumer_glb",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_args(tmp_path: Path, image: Path) -> list[str]:
    return [
        "--image",
        str(image),
        "--expected-image-sha256",
        _sha256(image),
        "--output-dir",
        str(tmp_path / "outputs"),
        "--work-dir",
        str(tmp_path / "runtime"),
    ]


def test_parser_defaults_bind_the_authorized_native_route():
    from scripts.source_cuda_native_image_to_glb_witness import (
        CUMESH_COMMIT,
        CUMESH_REPOSITORY,
        DINOV3_REVISION,
        EXPECTED_TORCH_VERSION,
        MODEL_REVISION,
        NVDIFFRAST_COMMIT,
        REMBG_REVISION,
        SPARSE_DECODER_REVISION,
        TRELLIS_COMMIT,
        TRELLIS_REPOSITORY,
        build_parser,
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "--image",
            "image.png",
            "--expected-image-sha256",
            "a" * 64,
            "--output-dir",
            "outputs",
            "--work-dir",
            "runtime",
        ]
    )

    assert TRELLIS_REPOSITORY == "https://github.com/microsoft/TRELLIS.2.git"
    assert TRELLIS_COMMIT == "5565d240c4a494caaf9ece7a554542b76ffa36d3"
    assert CUMESH_REPOSITORY == "https://github.com/JeffreyXiang/CuMesh.git"
    assert CUMESH_COMMIT == "c4ad6125924fcedfd13f0bd61520ca2d24eb7a87"
    assert EXPECTED_TORCH_VERSION == "2.10.0+cu128"
    assert MODEL_REVISION == "af44b45f2e35a493886929c6d786e563ec68364d"
    assert SPARSE_DECODER_REVISION == "25e0d31ffbebe4b5a97464dd851910efc3002d96"
    assert DINOV3_REVISION == "ea8dc2863c51be0a264bab82070e3e8836b02d51"
    assert REMBG_REVISION == "5df4c9c76d8170882c34f6986e848ee07fd0ba43"
    assert NVDIFFRAST_COMMIT == "253ac4fcea7de5f396371124af597e6cc957bfae"
    assert args.pipeline_type == "512"
    assert args.seed == 42
    assert args.steps == 8
    assert args.target_faces == 350000
    assert args.texture_size == 1024
    assert args.attention_backend == "xformers"
    assert args.sparse_conv_backend == "flex_gemm"


def test_wrong_image_digest_fails_before_touching_existing_output(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import main

    image = tmp_path / "image.png"
    image.write_bytes(b"not-the-authorized-image")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    stale = output_dir / "consumer.glb"
    stale.write_bytes(b"preserve-until-request-validates")

    args = _base_args(tmp_path, image)
    args[args.index("--expected-image-sha256") + 1] = "0" * 64
    rc = main(args + ["--no-download"])

    report = json.loads((output_dir / "report.json").read_text())
    assert rc == 1
    assert report["status"] == "failed"
    assert report["failure_phase"] == "request_validation"
    assert report["last_trustworthy_phase"] == "arguments_parsed"
    assert report["primary_output_status"] == "not_attempted"
    assert "image SHA256" in report["error"]
    assert stale.read_bytes() == b"preserve-until-request-validates"


def test_no_download_preflight_records_exact_route_without_primary_artifacts(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import main

    image = tmp_path / "image.png"
    image.write_bytes(b"authorized-image-fixture")
    args = _base_args(tmp_path, image)

    rc = main(args + ["--no-download"])

    output_dir = tmp_path / "outputs"
    report = json.loads((output_dir / "report.json").read_text())
    assert rc == 0
    assert report["status"] == "preflight_stopped"
    assert report["failure_phase"] is None
    assert report["last_trustworthy_phase"] == "request_validated"
    assert report["primary_output_status"] == "not_written_no_download"
    assert report["effective_route"]["device_type"] == "not_loaded_no_download"
    assert report["effective_route"]["trellis_source_clean"] == "not_checked_no_download"
    assert report["requested_route"]["native_conditioning"] is True
    assert report["requested_route"]["native_rng"] is True
    assert report["requested_route"]["pipeline_run_called_once"] is True
    assert report["requested_route"]["sampler_steps"] == {
        "shape": 8,
        "sparse": 8,
        "texture": 8,
    }
    assert report["requested_route"]["postprocess"] == {
        "decimation_target": 350000,
        "remesh": False,
        "texture_size": 1024,
    }
    assert report["expected_capture_order"] == list(EXPECTED_STAGES)
    assert list(output_dir.iterdir()) == [output_dir / "report.json"]


def test_recorder_publishes_each_boundary_to_progress_report(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import ArtifactRecorder

    observations = []

    def publish(recorder):
        observations.append(
            {
                "capture_order": list(recorder.capture_order),
                "artifacts": dict(recorder.artifacts),
            }
        )

    recorder = ArtifactRecorder(tmp_path / "outputs", on_capture=publish)
    recorder.save_image(
        "preprocessed_image",
        type(
            "FakeImage",
            (),
            {
                "mode": "RGBA",
                "size": (2, 2),
                "save": lambda self, path, format: Path(path).write_bytes(b"png"),
            },
        )(),
    )

    assert observations == [
        {
            "capture_order": ["preprocessed_image"],
            "artifacts": {
                "preprocessed_image": recorder.artifacts["preprocessed_image"]
            },
        }
    ]


def test_orientation_observer_does_not_mutate_native_extension_class():
    from scripts.source_cuda_native_image_to_glb_witness import (
        _restore_orientation_observer,
        install_orientation_observer,
    )

    class ImmutableNativeType(type):
        def __setattr__(cls, name, value):
            raise TypeError("immutable extension type")

    class NativeCuMesh(metaclass=ImmutableNativeType):
        def read(self):
            return [1, 2, 3], [[0, 1, 2]]

        def unify_face_orientations(self):
            return "native-return"

    class Recorder:
        def __init__(self):
            self.stages = []

        def save_npz(self, stage, arrays):
            self.stages.append((stage, arrays))

    module = types.SimpleNamespace(CuMesh=NativeCuMesh)
    recorder = Recorder()

    observer = install_orientation_observer(module, recorder)
    observed = module.CuMesh()
    result = observed.unify_face_orientations()
    _restore_orientation_observer(observer)

    assert result == "native-return"
    assert module.CuMesh is NativeCuMesh
    assert [stage for stage, _ in recorder.stages] == [
        "postprocess_stage11_pre_orientation",
        "postprocess_stage12_post_orientation",
    ]
    assert observer["state"] == {
        "call_count": 1,
        "native_method_return_preserved": True,
        "pre_readback_written": True,
        "post_readback_written": True,
    }


def _write_completed_fixture(tmp_path: Path) -> Path:
    from scripts.source_cuda_native_image_to_glb_witness import (
        CUMESH_COMMIT,
        TRELLIS_COMMIT,
    )

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    artifacts = {}
    for index, stage in enumerate(EXPECTED_STAGES):
        suffix = ".glb" if stage == "consumer_glb" else ".npz"
        path = output_dir / f"{index:02d}-{stage}{suffix}"
        path.write_bytes(f"artifact:{stage}".encode())
        artifacts[stage] = {
            "path": path.name,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }

    report = {
        "schema": "trellis2mlx.source_cuda_native_image_to_glb.v1",
        "status": "completed",
        "failure_phase": None,
        "last_trustworthy_phase": "consumer_glb_validated",
        "primary_output_status": "validated",
        "effective_route": {
            "device_type": "cuda",
            "cuda_device_name": "Tesla T4",
            "torch_version": "2.10.0+cu128",
            "attention_backend": "xformers",
            "sparse_attention_backend": "xformers",
            "sparse_conv_backend": "flex_gemm",
            "trellis_commit": TRELLIS_COMMIT,
            "trellis_source_clean": True,
            "cumesh_commit": CUMESH_COMMIT,
            "cumesh_source_clean_before_build": True,
            "nvdiffrast_commit": "253ac4fcea7de5f396371124af597e6cc957bfae",
            "model_revision": "af44b45f2e35a493886929c6d786e563ec68364d",
            "sparse_decoder_revision": "25e0d31ffbebe4b5a97464dd851910efc3002d96",
            "dinov3_revision": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
            "rembg_revision": "5df4c9c76d8170882c34f6986e848ee07fd0ba43",
            "pipeline_type": "512",
            "seed": 42,
            "sampler_steps": {"sparse": 8, "shape": 8, "texture": 8},
            "pipeline_run_call_count": 1,
            "native_conditioning": True,
            "native_rng": True,
            "observation_only_instrumentation": True,
        },
        "capture_order": list(EXPECTED_STAGES),
        "orientation_observer": {
            "call_count": 1,
            "native_method_return_preserved": True,
            "pre_readback_written": True,
            "post_readback_written": True,
        },
        "artifacts": artifacts,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")
    return report_path


def test_completed_report_admission_reopens_every_boundary(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import (
        validate_completed_report,
    )

    report_path = _write_completed_fixture(tmp_path)
    admitted = validate_completed_report(report_path)

    assert admitted["status"] == "completed"
    assert admitted["primary_output_status"] == "validated"
    assert admitted["capture_order"] == list(EXPECTED_STAGES)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_stage", "capture order"),
        ("wrong_digest", "digest mismatch"),
        ("wrong_device", "device_type"),
        ("dirty_source", "trellis_source_clean"),
        ("orientation_not_observed", "orientation observer"),
    ],
)
def test_completed_report_admission_rejects_false_closure(tmp_path, mutation, match):
    from scripts.source_cuda_native_image_to_glb_witness import (
        validate_completed_report,
    )

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    if mutation == "missing_stage":
        report["capture_order"].remove("shape_flow")
    elif mutation == "wrong_digest":
        report["artifacts"]["shape_flow"]["sha256"] = "0" * 64
    elif mutation == "wrong_device":
        report["effective_route"]["device_type"] = "cpu"
    elif mutation == "dirty_source":
        report["effective_route"]["trellis_source_clean"] = False
    elif mutation == "orientation_not_observed":
        report["orientation_observer"]["call_count"] = 0
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=match):
        validate_completed_report(report_path)
