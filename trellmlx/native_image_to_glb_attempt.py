"""Structured native image-to-GLB CUDA attempt construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import uuid

from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket


class AttemptSpecError(ValueError):
    """Raised when structured attempt data cannot be admitted."""


@dataclass(frozen=True)
class AttemptAsset:
    source: Path
    coordinate: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class NativeImageToGLBAttemptSpec:
    run_id: str
    dataset_id: str
    kernel_id: str
    title: str
    capsule_dir: Path
    output_dir: Path
    entrypoint: AttemptAsset
    authority_helper: AttemptAsset
    image: AttemptAsset
    dinov3_files: Mapping[str, AttemptAsset]
    rembg_files: Mapping[str, AttemptAsset]
    expected_outputs: tuple[str, ...]
    capture_profile: str = "full"
    model_kernel_source: str | None = None
    pipeline_type: str = "512"
    seed: int = 42
    steps: int = 8
    target_faces: int = 350000
    texture_size: int = 1024
    request_settings_bound: bool = False
    output_coordinate: str = "outputs"
    work_coordinate: str = "runtime"
    accelerator: str = "NvidiaTeslaT4"
    enable_internet: bool = True


REMBG_ARGUMENTS = {
    "model.safetensors": "--rembg-model-file",
    "config.json": "--rembg-config-file",
    "birefnet.py": "--rembg-birefnet-file",
    "BiRefNet_config.py": "--rembg-birefnet-config-file",
}
DINOV3_FILENAMES = (
    "model.safetensors",
    "config.json",
    "preprocessor_config.json",
)
DINOV3_MODEL_COORDINATE = "."
ATTEMPT_MANIFEST = "native-image-to-glb-attempt.json"
ATTEMPT_SPEC_SCHEMA_V2 = "trellis2mlx.native_image_to_glb_attempt_spec.v2"
ATTEMPT_SPEC_SCHEMA = "trellis2mlx.native_image_to_glb_attempt_spec.v3"
ATTEMPT_SPEC_SCHEMA_V4 = "trellis2mlx.native_image_to_glb_attempt_spec.v4"
ATTEMPT_SPEC_SCHEMA_V5 = "trellis2mlx.native_image_to_glb_attempt_spec.v5"
ATTEMPT_MANIFEST_SCHEMA_V2 = "trellis2mlx.native_image_to_glb_attempt.v2"
ATTEMPT_MANIFEST_SCHEMA = "trellis2mlx.native_image_to_glb_attempt.v3"
ATTEMPT_MANIFEST_SCHEMA_V4 = "trellis2mlx.native_image_to_glb_attempt.v4"
ATTEMPT_MANIFEST_SCHEMA_V5 = "trellis2mlx.native_image_to_glb_attempt.v5"
CAPTURE_PROFILE_OUTPUTS = {
    "full": (
        "00-preprocessed_image.png",
        "01-conditioning_512.npz",
        "02-sparse_flow.npz",
        "03-sparse_support.npz",
        "04-shape_flow.npz",
        "05-shape_slat.npz",
        "06-texture_flow.npz",
        "07-decoder_raw_mesh.npz",
        "08-texture_voxels.npz",
        "09-pipeline_filled_mesh.npz",
        "10-postprocess_stage11_pre_orientation.npz",
        "11-postprocess_stage12_post_orientation.npz",
        "12-consumer_glb.glb",
    ),
    "final-consumer": (
        "07-decoder_raw_mesh.npz",
        "10-postprocess_stage11_pre_orientation.npz",
        "11-postprocess_stage12_post_orientation.npz",
        "12-consumer_glb.glb",
    ),
}


@dataclass(frozen=True)
class CaptureContract:
    capture_profile: str
    expected_outputs: tuple[str, ...]
    manifest_schema: str
    profile_is_explicit: bool
    profile_binds_outputs: bool


def resolve_capture_contract(
    *,
    capture_profile: object,
    expected_outputs: object,
    profile_is_explicit: bool,
    allow_explicit_full: bool = False,
    context: str,
) -> CaptureContract:
    if not isinstance(expected_outputs, (list, tuple)) or not all(
        isinstance(value, str) and value for value in expected_outputs
    ):
        raise AttemptSpecError(f"{context} expected output list is invalid")
    outputs = tuple(expected_outputs)
    if not outputs or len(set(outputs)) != len(outputs):
        raise AttemptSpecError(f"{context} expected outputs are missing or duplicated")

    if not profile_is_explicit:
        if capture_profile not in {None, "full"}:
            raise AttemptSpecError(f"{context} implicit capture profile is invalid")
        return CaptureContract(
            capture_profile="full",
            expected_outputs=outputs,
            manifest_schema=ATTEMPT_MANIFEST_SCHEMA_V2,
            profile_is_explicit=False,
            profile_binds_outputs=False,
        )

    if capture_profile == "full" and not allow_explicit_full:
        raise AttemptSpecError(
            f"{context} cannot declare an explicit full capture profile"
        )
    if capture_profile not in CAPTURE_PROFILE_OUTPUTS:
        raise AttemptSpecError(f"{context} capture profile is invalid")
    if outputs != CAPTURE_PROFILE_OUTPUTS[capture_profile]:
        raise AttemptSpecError(
            f"{context} expected outputs do not match capture profile"
        )
    return CaptureContract(
        capture_profile=capture_profile,
        expected_outputs=outputs,
        manifest_schema=(
            ATTEMPT_MANIFEST_SCHEMA_V5
            if capture_profile == "full"
            else ATTEMPT_MANIFEST_SCHEMA
        ),
        profile_is_explicit=True,
        profile_binds_outputs=True,
    )


def capture_contract_from_spec_payload(payload: Mapping[str, Any]) -> CaptureContract:
    schema = payload.get("schema")
    return resolve_capture_contract(
        capture_profile=payload.get("capture_profile"),
        expected_outputs=payload.get("expected_outputs"),
        profile_is_explicit=schema in {
            ATTEMPT_SPEC_SCHEMA,
            ATTEMPT_SPEC_SCHEMA_V4,
            ATTEMPT_SPEC_SCHEMA_V5,
        },
        allow_explicit_full=schema == ATTEMPT_SPEC_SCHEMA_V5,
        context="attempt spec",
    )


def capture_contract_from_manifest(payload: Mapping[str, Any]) -> CaptureContract:
    schema = payload.get("schema")
    return resolve_capture_contract(
        capture_profile=payload.get("capture_profile"),
        expected_outputs=payload.get("expected_outputs"),
        profile_is_explicit=schema in {
            ATTEMPT_MANIFEST_SCHEMA,
            ATTEMPT_MANIFEST_SCHEMA_V4,
            ATTEMPT_MANIFEST_SCHEMA_V5,
        },
        allow_explicit_full=schema == ATTEMPT_MANIFEST_SCHEMA_V5,
        context="attempt manifest",
    )


def capture_contract_from_profile(
    capture_profile: str,
    expected_outputs: tuple[str, ...],
    *,
    bind_full: bool = False,
    context: str,
) -> CaptureContract:
    return resolve_capture_contract(
        capture_profile=capture_profile,
        expected_outputs=expected_outputs,
        profile_is_explicit=capture_profile != "full" or bind_full,
        allow_explicit_full=bind_full,
        context=context,
    )


def capture_contract_from_entrypoint_args(
    entrypoint_args: Sequence[str],
    expected_outputs: Sequence[str],
    *,
    context: str,
) -> CaptureContract:
    declarations: list[str] = []
    malformed_declaration = False
    for index, argument in enumerate(entrypoint_args):
        if argument == "--capture-profile":
            if (
                index + 1 >= len(entrypoint_args)
                or entrypoint_args[index + 1].startswith("--")
            ):
                malformed_declaration = True
            else:
                declarations.append(entrypoint_args[index + 1])
        elif argument.startswith("--capture-profile="):
            capture_profile = argument.removeprefix("--capture-profile=")
            if capture_profile:
                declarations.append(capture_profile)
            else:
                malformed_declaration = True

    if malformed_declaration or len(declarations) > 1:
        raise AttemptSpecError(f"{context} capture profile is ambiguous")
    if not declarations:
        capture_profile = "full"
    else:
        capture_profile = declarations[0]
    return resolve_capture_contract(
        capture_profile=capture_profile,
        expected_outputs=expected_outputs,
        profile_is_explicit=bool(declarations),
        context=context,
    )


def load_attempt_spec_bytes(
    data: bytes,
    *,
    source_path: Path,
) -> NativeImageToGLBAttemptSpec:
    path = Path(source_path).resolve()
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttemptSpecError(f"attempt spec is missing or invalid: {path}") from exc
    expected_fields = {
        "schema",
        "run_id",
        "dataset_id",
        "kernel_id",
        "title",
        "capsule_dir",
        "output_dir",
        "entrypoint",
        "authority_helper",
        "image",
        "dinov3_files",
        "rembg_files",
        "expected_outputs",
        "output_coordinate",
        "work_coordinate",
        "accelerator",
        "enable_internet",
    }
    schema = payload.get("schema") if isinstance(payload, dict) else None
    if schema in {ATTEMPT_SPEC_SCHEMA, ATTEMPT_SPEC_SCHEMA_V4}:
        expected_fields.add("capture_profile")
    if schema == ATTEMPT_SPEC_SCHEMA_V4:
        expected_fields.add("model_kernel_source")
    if schema == ATTEMPT_SPEC_SCHEMA_V5:
        expected_fields.update(
            {
                "capture_profile",
                "model_kernel_source",
                "pipeline_type",
                "seed",
                "steps",
                "target_faces",
                "texture_size",
            }
        )
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise AttemptSpecError("attempt spec field set is incomplete or contains unknown fields")
    if schema not in {
        ATTEMPT_SPEC_SCHEMA_V2,
        ATTEMPT_SPEC_SCHEMA,
        ATTEMPT_SPEC_SCHEMA_V4,
        ATTEMPT_SPEC_SCHEMA_V5,
    }:
        raise AttemptSpecError(f"unexpected attempt spec schema: {payload.get('schema')!r}")

    def resolved_path(value: object, label: str) -> Path:
        if not isinstance(value, str) or not value:
            raise AttemptSpecError(f"{label} path is missing")
        candidate = Path(value)
        return (path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

    def asset(value: object, label: str) -> AttemptAsset:
        if not isinstance(value, dict) or set(value) != {
            "source",
            "coordinate",
            "sha256",
            "size_bytes",
        }:
            raise AttemptSpecError(f"{label} asset field set is invalid")
        return AttemptAsset(
            source=resolved_path(value["source"], f"{label} source"),
            coordinate=value["coordinate"],
            sha256=value["sha256"],
            size_bytes=value["size_bytes"],
        )

    dinov3_payload = payload["dinov3_files"]
    if not isinstance(dinov3_payload, dict):
        raise AttemptSpecError("attempt DINOv3 file map is invalid")
    rembg_payload = payload["rembg_files"]
    if not isinstance(rembg_payload, dict):
        raise AttemptSpecError("attempt RMBG file map is invalid")
    capture_contract = capture_contract_from_spec_payload(payload)
    if schema in {ATTEMPT_SPEC_SCHEMA_V4, ATTEMPT_SPEC_SCHEMA_V5} and (
        not isinstance(payload["model_kernel_source"], str)
        or not payload["model_kernel_source"]
    ):
        raise AttemptSpecError("attempt model kernel source is missing")
    if type(payload["enable_internet"]) is not bool:
        raise AttemptSpecError("attempt enable_internet must be boolean")
    scalar_fields = (
        "run_id",
        "dataset_id",
        "kernel_id",
        "title",
        "output_coordinate",
        "work_coordinate",
        "accelerator",
    )
    if any(not isinstance(payload[field], str) or not payload[field] for field in scalar_fields):
        raise AttemptSpecError("attempt scalar identity field is missing")
    return NativeImageToGLBAttemptSpec(
        run_id=payload["run_id"],
        dataset_id=payload["dataset_id"],
        kernel_id=payload["kernel_id"],
        title=payload["title"],
        capsule_dir=resolved_path(payload["capsule_dir"], "capsule"),
        output_dir=resolved_path(payload["output_dir"], "output"),
        entrypoint=asset(payload["entrypoint"], "entrypoint"),
        authority_helper=asset(payload["authority_helper"], "authority helper"),
        image=asset(payload["image"], "image"),
        dinov3_files={
            role: asset(value, f"DINOv3 {role}")
            for role, value in dinov3_payload.items()
        },
        rembg_files={
            role: asset(value, f"RMBG {role}")
            for role, value in rembg_payload.items()
        },
        expected_outputs=capture_contract.expected_outputs,
        capture_profile=capture_contract.capture_profile,
        model_kernel_source=payload.get("model_kernel_source"),
        pipeline_type=payload.get("pipeline_type", "512"),
        seed=payload.get("seed", 42),
        steps=payload.get("steps", 8),
        target_faces=payload.get("target_faces", 350000),
        texture_size=payload.get("texture_size", 1024),
        request_settings_bound=schema == ATTEMPT_SPEC_SCHEMA_V5,
        output_coordinate=payload["output_coordinate"],
        work_coordinate=payload["work_coordinate"],
        accelerator=payload["accelerator"],
        enable_internet=payload["enable_internet"],
    )


def load_attempt_spec(path: Path) -> NativeImageToGLBAttemptSpec:
    path = Path(path).resolve()
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise AttemptSpecError(f"attempt spec is missing or invalid: {path}") from exc
    return load_attempt_spec_bytes(data, source_path=path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_asset(asset: AttemptAsset, *, role: str) -> None:
    source = Path(asset.source)
    coordinate = PurePosixPath(asset.coordinate)
    if (
        coordinate.is_absolute()
        or len(coordinate.parts) != 1
        or coordinate.name in {"", ".", ".."}
        or coordinate.as_posix() != asset.coordinate
    ):
        raise AttemptSpecError(f"{role} coordinate must be canonical, flat, and relative")
    if not source.is_file() or source.stat().st_size <= 0:
        raise AttemptSpecError(f"{role} source is missing or blank: {source}")
    if type(asset.size_bytes) is not int or asset.size_bytes <= 0:
        raise AttemptSpecError(f"{role} declared size is invalid")
    actual_size = source.stat().st_size
    actual_sha256 = _sha256_file(source)
    if actual_size != asset.size_bytes or actual_sha256 != asset.sha256:
        raise AttemptSpecError(
            f"{role} source digest or size drift: "
            f"sha256={actual_sha256}, size={actual_size}"
        )


def _validate_flat_relative_coordinate(value: str, *, role: str) -> None:
    coordinate = PurePosixPath(value)
    if (
        coordinate.is_absolute()
        or len(coordinate.parts) != 1
        or coordinate.name in {"", ".", ".."}
        or coordinate.as_posix() != value
    ):
        raise AttemptSpecError(f"attempt {role} must be canonical, flat, and relative")


def _validate_execution_coordinates(spec: NativeImageToGLBAttemptSpec) -> None:
    _validate_flat_relative_coordinate(
        spec.output_coordinate,
        role="output coordinate",
    )
    _validate_flat_relative_coordinate(
        spec.work_coordinate,
        role="work coordinate",
    )
    if spec.output_coordinate == spec.work_coordinate:
        raise AttemptSpecError("attempt output and work coordinates must be distinct")

    published_outputs = ("report.json", *spec.expected_outputs)
    if len(set(published_outputs)) != len(published_outputs):
        raise AttemptSpecError("attempt expected outputs collide with the report output")
    for output in published_outputs:
        _validate_flat_relative_coordinate(output, role="expected output")
    reserved_wrapper_outputs = {
        "kaggle_cuda_witness_receipt.json",
        "kaggle_cuda_witness_child_report.json",
    }
    if reserved_wrapper_outputs & set(published_outputs):
        raise AttemptSpecError(
            "attempt expected outputs collide with a witness wrapper output"
        )

    asset_coordinates = {
        spec.entrypoint.coordinate,
        spec.authority_helper.coordinate,
        spec.image.coordinate,
        *(asset.coordinate for asset in spec.dinov3_files.values()),
        *(asset.coordinate for asset in spec.rembg_files.values()),
        ATTEMPT_MANIFEST,
    }
    reserved = {
        spec.output_coordinate,
        spec.work_coordinate,
    }
    if reserved & set(published_outputs):
        raise AttemptSpecError(
            "attempt output/work coordinates collide with a published output"
        )
    if reserved & asset_coordinates:
        raise AttemptSpecError(
            "attempt output/work coordinates collide with a staged input"
        )


def _validate_request_settings(spec: NativeImageToGLBAttemptSpec) -> None:
    if spec.pipeline_type != "512":
        raise AttemptSpecError("attempt pipeline_type must be '512'")
    integer_settings = {
        "seed": spec.seed,
        "steps": spec.steps,
        "target_faces": spec.target_faces,
        "texture_size": spec.texture_size,
    }
    if any(type(value) is not int for value in integer_settings.values()):
        raise AttemptSpecError("attempt integer request settings are invalid")
    if spec.seed < 0:
        raise AttemptSpecError("attempt seed must be nonnegative")
    if spec.steps != 8:
        raise AttemptSpecError("attempt steps must equal 8")
    if spec.target_faces <= 0:
        raise AttemptSpecError("attempt target_faces must be positive")
    if spec.texture_size <= 0 or spec.texture_size & (spec.texture_size - 1):
        raise AttemptSpecError("attempt texture_size must be a positive power of two")


def _manifest(spec: NativeImageToGLBAttemptSpec, assets: Mapping[str, AttemptAsset]) -> dict:
    capture_contract = capture_contract_from_profile(
        spec.capture_profile,
        spec.expected_outputs,
        bind_full=spec.request_settings_bound,
        context="attempt",
    )
    payload = {
        "schema": capture_contract.manifest_schema,
        "run_id": spec.run_id,
        "dataset_id": spec.dataset_id,
        "kernel_id": spec.kernel_id,
        "title": spec.title,
        "accelerator": spec.accelerator,
        "enable_internet": spec.enable_internet,
        "output_coordinate": spec.output_coordinate,
        "work_coordinate": spec.work_coordinate,
        "expected_outputs": list(capture_contract.expected_outputs),
        "assets": {
            role: {
                "coordinate": asset.coordinate,
                "sha256": asset.sha256,
                "size_bytes": asset.size_bytes,
            }
            for role, asset in assets.items()
        },
    }
    if capture_contract.profile_is_explicit:
        payload["capture_profile"] = capture_contract.capture_profile
    if spec.model_kernel_source is not None:
        payload["schema"] = ATTEMPT_MANIFEST_SCHEMA_V4
        payload["model_kernel_source"] = spec.model_kernel_source
    if spec.request_settings_bound:
        payload.update(
            {
                "schema": ATTEMPT_MANIFEST_SCHEMA_V5,
                "capture_profile": capture_contract.capture_profile,
                "pipeline_type": spec.pipeline_type,
                "seed": spec.seed,
                "steps": spec.steps,
                "target_faces": spec.target_faces,
                "texture_size": spec.texture_size,
            }
        )
    return payload


def _paths_overlap(left: Path, right: Path) -> bool:
    left = Path(left).resolve()
    right = Path(right).resolve()
    return left == right or left in right.parents or right in left.parents


def validate_attempt_topology(spec: NativeImageToGLBAttemptSpec) -> None:
    capsule = Path(spec.capsule_dir).resolve()
    output = Path(spec.output_dir).resolve()
    if _paths_overlap(capsule, output):
        raise AttemptSpecError(
            f"attempt managed path topology overlaps: capsule={capsule}, output={output}"
        )
    assets = (
        spec.entrypoint,
        spec.authority_helper,
        spec.image,
        *spec.dinov3_files.values(),
        *spec.rembg_files.values(),
    )
    for asset in assets:
        source = Path(asset.source).resolve()
        if _paths_overlap(source, capsule) or _paths_overlap(source, output):
            raise AttemptSpecError(
                f"attempt source is inside or overlaps a managed path: {source}"
            )


def load_attempt_manifest_bytes(data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttemptSpecError("attempt manifest is missing or invalid") from exc
    expected_fields = {
        "schema",
        "run_id",
        "dataset_id",
        "kernel_id",
        "title",
        "accelerator",
        "enable_internet",
        "output_coordinate",
        "work_coordinate",
        "expected_outputs",
        "assets",
    }
    schema = payload.get("schema") if isinstance(payload, dict) else None
    if schema in {ATTEMPT_MANIFEST_SCHEMA, ATTEMPT_MANIFEST_SCHEMA_V4}:
        expected_fields.add("capture_profile")
    if schema == ATTEMPT_MANIFEST_SCHEMA_V4:
        expected_fields.add("model_kernel_source")
    if schema == ATTEMPT_MANIFEST_SCHEMA_V5:
        expected_fields.update(
            {
                "capture_profile",
                "model_kernel_source",
                "pipeline_type",
                "seed",
                "steps",
                "target_faces",
                "texture_size",
            }
        )
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise AttemptSpecError("attempt manifest field set is invalid")
    if schema not in {
        ATTEMPT_MANIFEST_SCHEMA_V2,
        ATTEMPT_MANIFEST_SCHEMA,
        ATTEMPT_MANIFEST_SCHEMA_V4,
        ATTEMPT_MANIFEST_SCHEMA_V5,
    }:
        raise AttemptSpecError("attempt manifest schema is invalid")
    capture_contract_from_manifest(payload)
    return payload


def load_attempt_manifest(path: Path) -> dict[str, Any]:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise AttemptSpecError("attempt manifest is missing or invalid") from exc
    return load_attempt_manifest_bytes(data)


def validate_attempt_manifest(
    packet: KaggleCudaWitnessPacket,
    payload: Mapping[str, Any],
    *,
    file_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    expected_identity = {
        "run_id": packet.run_id,
        "dataset_id": packet.dataset_id,
        "kernel_id": packet.kernel_id,
        "title": packet.title,
        "accelerator": packet.accelerator,
        "enable_internet": packet.enable_internet,
        "expected_outputs": list(packet.expected_outputs),
    }
    for field, expected in expected_identity.items():
        if payload.get(field) != expected:
            raise AttemptSpecError(
                f"attempt manifest {field} or run identity mismatch: "
                f"expected {expected!r}, got {payload.get(field)!r}"
            )
    capture_contract = capture_contract_from_manifest(payload)
    model_kernel_source = payload.get("model_kernel_source")
    if model_kernel_source is None:
        if packet.kernel_sources:
            raise AttemptSpecError(
                "attempt manifest is missing its packet kernel source identity"
            )
    elif packet.kernel_sources != (model_kernel_source,):
        raise AttemptSpecError(
            "attempt manifest model kernel source does not match packet sources"
        )

    def argument(flag: str) -> str:
        positions = [
            index for index, value in enumerate(packet.entrypoint_args) if value == flag
        ]
        if len(positions) != 1 or positions[0] + 1 >= len(packet.entrypoint_args):
            raise AttemptSpecError(f"attempt packet is missing exactly one {flag}")
        return packet.entrypoint_args[positions[0] + 1]

    packet_capture_contract = capture_contract_from_entrypoint_args(
        packet.entrypoint_args,
        packet.expected_outputs,
        context="attempt packet",
    )
    if (
        packet_capture_contract.capture_profile != capture_contract.capture_profile
        or (
            packet_capture_contract.profile_is_explicit
            != capture_contract.profile_is_explicit
            and payload.get("schema") != ATTEMPT_MANIFEST_SCHEMA_V5
        )
    ):
        raise AttemptSpecError("attempt manifest capture profile argument mismatch")

    if payload.get("schema") == ATTEMPT_MANIFEST_SCHEMA_V5:
        setting_flags = {
            "pipeline_type": "--pipeline-type",
            "seed": "--seed",
            "steps": "--steps",
            "target_faces": "--target-faces",
            "texture_size": "--texture-size",
        }
        for field, flag in setting_flags.items():
            packet_value = argument(flag)
            expected_value = str(payload.get(field))
            if packet_value != expected_value:
                raise AttemptSpecError(
                    f"attempt manifest {field} request setting mismatch"
                )

    if payload.get("output_coordinate") != argument("--output-dir"):
        raise AttemptSpecError("attempt manifest output coordinate mismatch")
    if payload.get("work_coordinate") != argument("--work-dir"):
        raise AttemptSpecError("attempt manifest work coordinate mismatch")
    if argument("--dinov3-model-path") != DINOV3_MODEL_COORDINATE:
        raise AttemptSpecError("attempt manifest DINOv3 model coordinate mismatch")
    assets = payload.get("assets")
    if not isinstance(assets, dict):
        raise AttemptSpecError("attempt manifest assets are missing")
    expected_roles = {
        "entrypoint": packet.entrypoint,
        "authority_helper": "witness_authority.py",
        "image": argument("--image"),
        **{f"dinov3:{role}": role for role in DINOV3_FILENAMES},
        **{
            f"rembg:{role}": argument(flag)
            for role, flag in REMBG_ARGUMENTS.items()
        },
    }
    if set(assets) != set(expected_roles):
        raise AttemptSpecError("attempt manifest asset role set mismatch")
    for role, coordinate in expected_roles.items():
        record = assets[role]
        if not isinstance(record, dict) or set(record) != {
            "coordinate",
            "sha256",
            "size_bytes",
        }:
            raise AttemptSpecError(f"attempt manifest asset record is invalid for {role}")
        if record.get("coordinate") != coordinate or coordinate not in packet.inputs:
            raise AttemptSpecError(f"attempt manifest asset coordinate mismatch for {role}")
        if file_records is not None:
            outer = file_records.get(coordinate)
            if not isinstance(outer, Mapping) or any(
                outer.get(field) != record.get(field)
                for field in ("sha256", "size_bytes")
            ):
                raise AttemptSpecError(
                    f"attempt manifest asset digest or size mismatch for {role}"
                )


def build_attempt_packet(spec: NativeImageToGLBAttemptSpec):
    try:
        parsed_run_id = uuid.UUID(spec.run_id)
    except (ValueError, TypeError, AttributeError) as exc:
        raise AttemptSpecError("attempt run identity is missing or invalid") from exc
    if str(parsed_run_id) != spec.run_id:
        raise AttemptSpecError("attempt run identity is not canonical")
    if set(spec.dinov3_files) != set(DINOV3_FILENAMES):
        raise AttemptSpecError("attempt DINOv3 role set is incomplete")
    if set(spec.rembg_files) != set(REMBG_ARGUMENTS):
        raise AttemptSpecError("attempt RMBG role set is incomplete")
    capture_contract = capture_contract_from_profile(
        spec.capture_profile,
        spec.expected_outputs,
        bind_full=spec.request_settings_bound,
        context="attempt",
    )
    default_settings = ("512", 42, 8, 350000, 1024)
    request_settings = (
        spec.pipeline_type,
        spec.seed,
        spec.steps,
        spec.target_faces,
        spec.texture_size,
    )
    if not spec.request_settings_bound and request_settings != default_settings:
        raise AttemptSpecError(
            "non-default request settings require a v5-bound attempt"
        )
    _validate_request_settings(spec)
    if type(spec.request_settings_bound) is not bool:
        raise AttemptSpecError("attempt request_settings_bound must be boolean")
    if spec.request_settings_bound and spec.model_kernel_source is None:
        raise AttemptSpecError("v5-bound attempt requires a model kernel source")
    if spec.model_kernel_source is not None:
        if not isinstance(spec.model_kernel_source, str):
            raise AttemptSpecError("attempt model kernel source is invalid")
        parts = spec.model_kernel_source.split("/")
        if len(parts) != 2 or not all(parts):
            raise AttemptSpecError(
                "attempt model kernel source must be a Kaggle ref like owner/slug"
            )
    _validate_execution_coordinates(spec)
    validate_attempt_topology(spec)

    ordered_assets: dict[str, AttemptAsset] = {
        "entrypoint": spec.entrypoint,
        "authority_helper": spec.authority_helper,
        "image": spec.image,
        **{f"dinov3:{role}": spec.dinov3_files[role] for role in DINOV3_FILENAMES},
        **{f"rembg:{role}": spec.rembg_files[role] for role in REMBG_ARGUMENTS},
    }
    for role in DINOV3_FILENAMES:
        if spec.dinov3_files[role].coordinate != role:
            raise AttemptSpecError(
                f"DINOv3 {role} coordinate must equal its canonical filename"
            )
    dinov3_sources = [
        Path(spec.dinov3_files[role].source).resolve()
        for role in DINOV3_FILENAMES
    ]
    if len(set(dinov3_sources)) != len(DINOV3_FILENAMES):
        raise AttemptSpecError("attempt DINOv3 source identities must be distinct")
    coordinates = [asset.coordinate for asset in ordered_assets.values()]
    if ATTEMPT_MANIFEST in coordinates or len(set(coordinates)) != len(coordinates):
        raise AttemptSpecError("attempt asset coordinates must be distinct")
    for role, asset in ordered_assets.items():
        _validate_asset(asset, role=role)

    capsule = Path(spec.capsule_dir).resolve()
    capsule.parent.mkdir(parents=True, exist_ok=True)
    candidate = Path(
        tempfile.mkdtemp(
            prefix=f".{capsule.name}.candidate-",
            dir=capsule.parent,
        )
    )
    backup: Path | None = None
    try:
        for role, asset in ordered_assets.items():
            destination = candidate / asset.coordinate
            shutil.copy2(asset.source, destination)
            _validate_asset(
                AttemptAsset(
                    source=destination,
                    coordinate=asset.coordinate,
                    sha256=asset.sha256,
                    size_bytes=asset.size_bytes,
                ),
                role=f"staged {role}",
            )
        (candidate / ATTEMPT_MANIFEST).write_text(
            json.dumps(_manifest(spec, ordered_assets), indent=2, sort_keys=True)
            + "\n"
        )
        if capsule.exists():
            backup = capsule.with_name(f".{capsule.name}.backup-{uuid.uuid4().hex}")
            os.replace(capsule, backup)
        try:
            os.replace(candidate, capsule)
        except BaseException:
            if backup is not None:
                os.replace(backup, capsule)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    finally:
        shutil.rmtree(candidate, ignore_errors=True)
        if backup is not None and not capsule.exists():
            os.replace(backup, capsule)

    rembg_arguments = tuple(
        item
        for role, flag in REMBG_ARGUMENTS.items()
        for item in (flag, spec.rembg_files[role].coordinate)
    )
    return KaggleCudaWitnessPacket(
        capsule_dir=capsule,
        output_dir=spec.output_dir,
        dataset_id=spec.dataset_id,
        kernel_id=spec.kernel_id,
        title=spec.title,
        entrypoint=spec.entrypoint.coordinate,
        inputs=(
            spec.entrypoint.coordinate,
            spec.authority_helper.coordinate,
            spec.image.coordinate,
            *(spec.dinov3_files[role].coordinate for role in DINOV3_FILENAMES),
            *(spec.rembg_files[role].coordinate for role in REMBG_ARGUMENTS),
            ATTEMPT_MANIFEST,
        ),
        entrypoint_args=(
            "--image",
            spec.image.coordinate,
            "--expected-image-sha256",
            spec.image.sha256,
            "--run-id",
            spec.run_id,
            "--output-dir",
            spec.output_coordinate,
            "--work-dir",
            spec.work_coordinate,
            *(
                ("--capture-profile", capture_contract.capture_profile)
                if capture_contract.profile_is_explicit
                and capture_contract.capture_profile != "full"
                else ()
            ),
            *(
                (
                    "--pipeline-type",
                    spec.pipeline_type,
                    "--seed",
                    str(spec.seed),
                    "--steps",
                    str(spec.steps),
                    "--target-faces",
                    str(spec.target_faces),
                    "--texture-size",
                    str(spec.texture_size),
                )
                if spec.request_settings_bound
                else ()
            ),
            "--dinov3-model-path",
            DINOV3_MODEL_COORDINATE,
            *rembg_arguments,
        ),
        run_id=spec.run_id,
        expected_image_sha256=spec.image.sha256,
        accelerator=spec.accelerator,
        enable_internet=spec.enable_internet,
        output_json="report.json",
        output_npz=None,
        expected_outputs=capture_contract.expected_outputs,
        attempt_manifest=ATTEMPT_MANIFEST,
        kernel_sources=(
            (spec.model_kernel_source,)
            if spec.model_kernel_source is not None
            else ()
        ),
    )
