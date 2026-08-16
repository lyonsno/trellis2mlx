import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
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


def test_actual_kaggle_runner_reaches_witness_request_validation(tmp_path, monkeypatch):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, prepare_packet
    from scripts import source_cuda_native_image_to_glb_witness as witness

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    entrypoint = capsule / "source_cuda_native_image_to_glb_witness.py"
    entrypoint.write_text(Path(witness.__file__).read_text())
    image = capsule / "9_img.png"
    image.write_bytes(b"synthetic-transport-contract-image")
    expected_outputs = tuple(
        f"{index:02d}-{stage}{'.png' if stage == 'preprocessed_image' else '.glb' if stage == 'consumer_glb' else '.npz'}"
        for index, stage in enumerate(EXPECTED_STAGES)
    )
    packet = prepare_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/native-image-anchor-inputs",
            kernel_id="operator/native-image-anchor-cuda",
            title="Native Image Anchor CUDA",
            entrypoint=entrypoint.name,
            inputs=(entrypoint.name, image.name),
            output_json="report.json",
            output_npz=None,
            expected_outputs=expected_outputs,
            entrypoint_args=(
                "--image",
                image.name,
                "--expected-image-sha256",
                _sha256(image),
                "--output-dir",
                ".",
                "--work-dir",
                str(tmp_path / "runtime"),
                "--no-download",
            ),
        )
    )
    fake_torch = types.SimpleNamespace(
        __version__="2.10.0+cu128",
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: "Tesla T4",
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text().replace(
        'Path("/kaggle/input")',
        f"Path({str(packet.dataset_dir)!r})",
    )
    namespace = {"__name__": "runner_test"}
    exec(runner, namespace)

    rc = namespace["main"]()

    report = json.loads((work / "report.json").read_text())
    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    assert rc != 0
    assert report["status"] == "preflight_stopped"
    assert report["last_trustworthy_phase"] == "request_validated"
    assert receipt["failure_phase"] == "output"
    assert receipt["effective_command"][:3] == [
        sys.executable,
        entrypoint.name,
        "--output-json",
    ]
    assert set(packet.expected_outputs).issuperset(expected_outputs)


@pytest.mark.parametrize(
    "mutated_role",
    ("image", "model.safetensors", "config.json", "preprocessor_config.json"),
)
def test_admitted_inputs_reject_requested_path_substitution_before_use(
    tmp_path,
    mutated_role,
):
    from scripts.source_cuda_native_image_to_glb_witness import (
        admit_run_inputs,
        verify_admitted_inputs_before_use,
    )

    image = tmp_path / "image.png"
    image.write_bytes(b"image-authority")
    dino = tmp_path / "dino"
    dino.mkdir()
    dino_files = {
        "model.safetensors": b"model-authority",
        "config.json": b"config-authority",
        "preprocessor_config.json": b"preprocessor-authority",
    }
    for name, payload in dino_files.items():
        (dino / name).write_bytes(payload)
    args = types.SimpleNamespace(
        image=image,
        expected_image_sha256=_sha256(image),
        dinov3_model_path=dino,
        work_dir=tmp_path / "runtime",
        run_id="11111111-1111-4111-8111-111111111111",
    )
    report = {}
    admitted = admit_run_inputs(
        args,
        report,
        expected_dinov3_files={name: _sha256(dino / name) for name in dino_files},
    )
    requested = image if mutated_role == "image" else dino / mutated_role
    requested.write_bytes(b"substituted-after-admission")

    with pytest.raises(RuntimeError, match=mutated_role):
        verify_admitted_inputs_before_use(
            admitted,
            report,
            expected_dinov3_files={name: _sha256(admitted.dinov3_model_path / name) for name in dino_files},
        )

    assert Path(admitted.image).read_bytes() == b"image-authority"
    for name, payload in dino_files.items():
        assert (Path(admitted.dinov3_model_path) / name).read_bytes() == payload


def _write_completed_fixture(tmp_path: Path) -> Path:
    from scripts.source_cuda_native_image_to_glb_witness import (
        CUMESH_COMMIT,
        CUMESH_REPOSITORY,
        DINOV3_FILES,
        DINOV3_REPOSITORY,
        DINOV3_REVISION,
        FLEX_GEMM_COMMIT,
        FLEX_GEMM_REPOSITORY,
        MODEL_PIPELINE_SHA256,
        MODEL_REPOSITORY,
        MODEL_REVISION,
        NVDIFFRAST_COMMIT,
        NVDIFFRAST_REPOSITORY,
        REMBG_REPOSITORY,
        REMBG_REVISION,
        SPARSE_DECODER_REPOSITORY,
        SPARSE_DECODER_REVISION,
        TRELLIS_COMMIT,
        TRELLIS_REPOSITORY,
    )
    from PIL import Image
    import trimesh

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    run_id = "11111111-1111-4111-8111-111111111111"
    image_sha256 = "a" * 64
    coords = np.asarray(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
        dtype=np.int32,
    )
    voxel_coords = coords[:, 1:].copy()
    features = np.arange(12, dtype=np.float32).reshape(4, 3)
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        dtype=np.int32,
    )
    arrays_by_stage = {
        "conditioning_512": {
            "cond": np.ones((1, 2, 3), dtype=np.float32),
            "neg_cond": np.zeros((1, 2, 3), dtype=np.float32),
        },
        "sparse_flow": {
            "noise": np.ones((1, 2, 2, 2, 2), dtype=np.float32),
            "sample_next": np.ones((8, 1, 2, 2, 2, 2), dtype=np.float32),
            "pred_x0": np.zeros((8, 1, 2, 2, 2, 2), dtype=np.float32),
        },
        "sparse_support": {"coords": coords},
        "shape_flow": {
            "noise_feats": features,
            "noise_coords": coords,
            "coords": coords,
            "sample_next": np.repeat(features[None], 8, axis=0),
            "pred_x0": np.repeat((features + 1)[None], 8, axis=0),
        },
        "shape_slat": {"shape_slat_feats": features, "shape_slat_coords": coords},
        "texture_flow": {
            "noise_feats": features,
            "noise_coords": coords,
            "coords": coords,
            "sample_next": np.repeat(features[None], 8, axis=0),
            "pred_x0": np.repeat((features + 1)[None], 8, axis=0),
        },
        "decoder_raw_mesh": {"vertices": vertices, "faces": faces},
        "texture_voxels": {
            "texture_voxels_feats": features,
            "texture_voxels_coords": coords,
        },
        "pipeline_filled_mesh": {
            "vertices": vertices,
            "faces": faces,
            "texture_coords": voxel_coords,
            "texture_attrs": np.ones((4, 6), dtype=np.float32),
        },
        "postprocess_stage11_pre_orientation": {"vertices": vertices, "faces": faces},
        "postprocess_stage12_post_orientation": {"vertices": vertices, "faces": faces},
    }
    artifacts = {}
    for index, stage in enumerate(EXPECTED_STAGES):
        suffix = ".png" if stage == "preprocessed_image" else ".glb" if stage == "consumer_glb" else ".npz"
        path = output_dir / f"{index:02d}-{stage}{suffix}"
        metadata = {}
        if stage == "preprocessed_image":
            image = Image.new("RGB", (2, 2), (80, 120, 160))
            image.save(path, format="PNG")
            metadata = {"mode": "RGB", "size": [2, 2]}
        elif stage == "consumer_glb":
            uv = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.new("RGB", (2, 2), (200, 100, 50))
            )
            visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                visual=visual,
                process=False,
            )
            path.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
        else:
            arrays = arrays_by_stage[stage]
            np.savez(path, **arrays)
            metadata = {
                "arrays": {
                    name: {
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                        "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                    }
                    for name, value in arrays.items()
                }
            }
        artifacts[stage] = {
            "path": path.name,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "run_id": run_id,
            **metadata,
        }

    report = {
        "schema": "trellis2mlx.source_cuda_native_image_to_glb.v1",
        "run_id": run_id,
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
            "run_id": run_id,
        },
        "requested_route": {"run_id": run_id, "image_sha256": image_sha256},
        "effective_inputs": {
            "run_id": run_id,
            "image": {"path": "/run/image.png", "sha256": image_sha256, "size_bytes": 100},
            "dinov3": {
                "path": "/run/dinov3",
                "files": {
                    name: {"path": f"/run/dinov3/{name}", "sha256": digest, "size_bytes": 100}
                    for name, digest in DINOV3_FILES.items()
                },
            },
        },
        "model_assets": {
            "trellis": {
                "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "pipeline_json_sha256": MODEL_PIPELINE_SHA256,
            },
            "sparse_decoder": {
                "repository": SPARSE_DECODER_REPOSITORY,
                "revision": SPARSE_DECODER_REVISION,
            },
            "dinov3": {
                "repository": DINOV3_REPOSITORY,
                "revision": DINOV3_REVISION,
                "files": dict(DINOV3_FILES),
            },
            "rembg": {"repository": REMBG_REPOSITORY, "revision": REMBG_REVISION},
            "path_rewrite_only": True,
        },
        "source_identities_after_build": {
            "trellis": {"repository": TRELLIS_REPOSITORY, "commit": TRELLIS_COMMIT, "clean": True},
            "cumesh": {"repository": CUMESH_REPOSITORY, "commit": CUMESH_COMMIT, "clean": True},
            "flex_gemm": {"repository": FLEX_GEMM_REPOSITORY, "commit": FLEX_GEMM_COMMIT, "clean": True},
            "nvdiffrast": {"repository": NVDIFFRAST_REPOSITORY, "commit": NVDIFFRAST_COMMIT, "clean": True},
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


def test_completed_report_rejects_plaintext_npz_and_glb_bytes(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import validate_completed_report

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    record = report["artifacts"]["shape_flow"]
    path = report_path.parent / record["path"]
    path.write_bytes(b"plain text pretending to be npz")
    record["sha256"] = _sha256(path)
    record["size_bytes"] = path.stat().st_size
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="NPZ|GLB|PNG|artifact structure"):
        validate_completed_report(report_path)


def test_completed_report_rejects_malformed_glb_even_with_matching_receipt(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import validate_completed_report

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    record = report["artifacts"]["consumer_glb"]
    path = report_path.parent / record["path"]
    path.write_bytes(b"plain text pretending to be a glb")
    record["sha256"] = _sha256(path)
    record["size_bytes"] = path.stat().st_size
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="GLB"):
        validate_completed_report(report_path)


def test_completed_report_rejects_duplicate_stage_path_even_with_matching_receipt(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import validate_completed_report

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    duplicate = report["artifacts"]["shape_flow"]
    report["artifacts"]["texture_flow"] = dict(duplicate)
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="distinct|duplicate"):
        validate_completed_report(report_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_array", "NPZ keys"),
        ("retyped_array", "NPZ dtype"),
        ("empty_array", "NPZ array is empty"),
    ],
)
def test_completed_report_rejects_invalid_npz_array_contract(tmp_path, mutation, match):
    from scripts.source_cuda_native_image_to_glb_witness import validate_completed_report

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    record = report["artifacts"]["conditioning_512"]
    path = report_path.parent / record["path"]
    with np.load(path, allow_pickle=False) as reopened:
        arrays = {name: np.ascontiguousarray(reopened[name]) for name in reopened.files}
    if mutation == "missing_array":
        arrays.pop("neg_cond")
    elif mutation == "retyped_array":
        arrays["cond"] = arrays["cond"].astype(np.float64)
    elif mutation == "empty_array":
        arrays["cond"] = np.empty((0, 2, 3), dtype=np.float32)
    np.savez(path, **arrays)
    record["sha256"] = _sha256(path)
    record["size_bytes"] = path.stat().st_size
    record["arrays"] = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
        for name, value in arrays.items()
    }
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=match):
        validate_completed_report(report_path)


def test_completed_report_rejects_stale_bundle_from_another_run(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import validate_completed_report

    report_path = _write_completed_fixture(tmp_path)

    with pytest.raises(ValueError, match="run identity mismatch"):
        validate_completed_report(
            report_path,
            expected_run_id="22222222-2222-4222-8222-222222222222",
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_model_assets", "model assets"),
        ("wrong_effective_image", "effective image"),
        ("missing_run_id", "run identity"),
    ],
)
def test_completed_report_rejects_missing_or_wrong_authority(tmp_path, mutation, match):
    from scripts.source_cuda_native_image_to_glb_witness import validate_completed_report

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    if mutation == "missing_model_assets":
        report.pop("model_assets", None)
    elif mutation == "wrong_effective_image":
        report.setdefault("effective_inputs", {})["image"] = {
            "sha256": "0" * 64,
            "size_bytes": 1,
        }
    elif mutation == "missing_run_id":
        report.pop("run_id", None)
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=match):
        validate_completed_report(report_path)


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
