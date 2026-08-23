#!/usr/bin/env python3
"""Prepare the canonical private CPU Kaggle model-source publisher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

try:
    from scripts.source_cuda_native_image_to_glb_witness import (
        MODEL_BLOB_MANIFEST,
        MODEL_PIPELINE_SHA256,
        MODEL_REPOSITORY,
        MODEL_REQUIRED_BYTES,
        MODEL_REVISION,
        MODEL_SOURCE_MARKER,
        SPARSE_DECODER_REPOSITORY,
        SPARSE_DECODER_REVISION,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from source_cuda_native_image_to_glb_witness import (
        MODEL_BLOB_MANIFEST,
        MODEL_PIPELINE_SHA256,
        MODEL_REPOSITORY,
        MODEL_REQUIRED_BYTES,
        MODEL_REVISION,
        MODEL_SOURCE_MARKER,
        SPARSE_DECODER_REPOSITORY,
        SPARSE_DECODER_REVISION,
    )
from trellmlx.kaggle_model_source_publisher import (
    ModelSourcePublisherError,
    ModelSourcePublisherFile,
    ModelSourcePublisherPacket,
    ModelSourcePublisherRepository,
    prepare_publisher_packet,
    sha256_file,
)


PUBLISHER_KERNEL_PREFIX = "noahboo/t2mlx-r11-source"
_REPOSITORY_AUTHORITY = {
    "trellis": (MODEL_REPOSITORY, MODEL_REVISION),
    "sparse_decoder": (SPARSE_DECODER_REPOSITORY, SPARSE_DECODER_REVISION),
}


def canonical_repositories() -> tuple[ModelSourcePublisherRepository, ...]:
    repositories = []
    for family, manifest in MODEL_BLOB_MANIFEST.items():
        repo_id, revision = _REPOSITORY_AUTHORITY[family]
        repositories.append(
            ModelSourcePublisherRepository(
                name=family,
                repo_id=repo_id,
                revision=revision,
                cache_dir=manifest["cache_dir"],
                files=tuple(
                    ModelSourcePublisherFile(
                        coordinate=coordinate,
                        blob=identity["blob"],
                        sha256=identity["sha256"],
                        size_bytes=identity["size_bytes"],
                    )
                    for coordinate, identity in manifest["files"].items()
                ),
            )
        )
    return tuple(sorted(repositories))


def build_packet(
    *,
    output_dir: Path,
    attempt_id: str | None = None,
) -> ModelSourcePublisherPacket:
    effective_attempt_id = attempt_id or uuid.uuid4().hex
    kernel_id = f"{PUBLISHER_KERNEL_PREFIX}-{effective_attempt_id}"
    return ModelSourcePublisherPacket(
        output_dir=output_dir,
        kernel_id=kernel_id,
        title=kernel_id.split("/", 1)[1],
        attempt_id=effective_attempt_id,
        marker=MODEL_SOURCE_MARKER,
        marker_sha256=MODEL_PIPELINE_SHA256,
        repositories=canonical_repositories(),
    )


def validate_r11_publisher_authority(packet: ModelSourcePublisherPacket) -> None:
    expected_kernel_id = f"{PUBLISHER_KERNEL_PREFIX}-{packet.attempt_id}"
    expected = {
        "kernel_id": expected_kernel_id,
        "title": expected_kernel_id.split("/", 1)[1],
        "marker": MODEL_SOURCE_MARKER,
        "marker_sha256": MODEL_PIPELINE_SHA256,
        "repositories": canonical_repositories(),
        "blob_count": 17,
        "published_blob_bytes": MODEL_REQUIRED_BYTES,
    }
    actual = {
        "kernel_id": packet.kernel_id,
        "title": packet.title,
        "marker": packet.marker,
        "marker_sha256": packet.marker_sha256,
        "repositories": packet.repositories,
        "blob_count": packet.blob_count,
        "published_blob_bytes": packet.published_blob_bytes,
    }
    for field, value in expected.items():
        if actual[field] != value:
            raise ModelSourcePublisherError(
                f"R11 publisher authority mismatch: {field}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packet = build_packet(output_dir=args.output_dir.resolve())
    validate_r11_publisher_authority(packet)
    prepare_publisher_packet(packet)
    manifest_path = packet.output_dir / "packet.json"
    print(
        json.dumps(
            {
                "status": "prepared",
                "packet_dir": str(packet.output_dir),
                "packet_manifest": str(manifest_path),
                "packet_manifest_sha256": sha256_file(manifest_path),
                "kernel_id": packet.kernel_id,
                "requested_accelerator": None,
                "attempt_id": packet.attempt_id,
                "packet_contract_sha256": packet.packet_contract_sha256,
                "repository_count": len(packet.repositories),
                "blob_count": packet.blob_count,
                "published_blob_bytes": packet.published_blob_bytes,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
