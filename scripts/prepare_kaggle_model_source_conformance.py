#!/usr/bin/env python3
"""Prepare the private CPU Kaggle model-source conformance packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

try:
    from scripts.source_cuda_native_image_to_glb_witness import (
        MODEL_BLOB_MANIFEST,
        MODEL_PIPELINE_SHA256,
        MODEL_REQUIRED_BYTES,
        MODEL_SOURCE_MARKER,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from source_cuda_native_image_to_glb_witness import (
        MODEL_BLOB_MANIFEST,
        MODEL_PIPELINE_SHA256,
        MODEL_REQUIRED_BYTES,
        MODEL_SOURCE_MARKER,
    )
from trellmlx.kaggle_model_source_conformance import (
    ModelSourceBlob,
    ModelSourceConformanceError,
    ModelSourceConformancePacket,
    prepare_packet,
    sha256_file,
)


R11_SOURCE_KERNEL = "noahboo/t2mlx-native-pixal9-t4-f6446f9-r10"


def canonical_blobs() -> tuple[ModelSourceBlob, ...]:
    blobs = []
    for family_manifest in MODEL_BLOB_MANIFEST.values():
        cache_dir = family_manifest["cache_dir"]
        for expected in family_manifest["files"].values():
            blobs.append(
                ModelSourceBlob(
                    relative_path=f"{cache_dir}/blobs/{expected['blob']}",
                    sha256=expected["sha256"],
                    size_bytes=expected["size_bytes"],
                )
            )
    return tuple(sorted(blobs))


def validate_r11_authority(packet: ModelSourceConformancePacket) -> None:
    expected = {
        "source_kernel": R11_SOURCE_KERNEL,
        "marker": MODEL_SOURCE_MARKER,
        "marker_sha256": MODEL_PIPELINE_SHA256,
        "blobs": canonical_blobs(),
        "mounted_blob_bytes": MODEL_REQUIRED_BYTES,
    }
    actual = {
        "source_kernel": packet.source_kernel,
        "marker": packet.marker,
        "marker_sha256": packet.marker_sha256,
        "blobs": packet.blobs,
        "mounted_blob_bytes": packet.mounted_blob_bytes,
    }
    for field, value in expected.items():
        if actual[field] != value:
            raise ModelSourceConformanceError(
                f"R11 model-source authority mismatch: {field}"
            )


def build_packet(
    *,
    output_dir: Path,
    kernel_id: str,
    source_kernel: str,
    title: str,
    attempt_id: str | None = None,
) -> ModelSourceConformancePacket:
    packet = ModelSourceConformancePacket(
        output_dir=output_dir,
        kernel_id=kernel_id,
        title=title,
        source_kernel=source_kernel,
        marker=MODEL_SOURCE_MARKER,
        marker_sha256=MODEL_PIPELINE_SHA256,
        blobs=canonical_blobs(),
        attempt_id=attempt_id or uuid.uuid4().hex,
    )
    if packet.mounted_blob_bytes != MODEL_REQUIRED_BYTES:
        raise RuntimeError(
            "model-source packet byte total diverges from the native witness: "
            f"packet={packet.mounted_blob_bytes}, native={MODEL_REQUIRED_BYTES}"
        )
    return packet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--source-kernel", default=R11_SOURCE_KERNEL)
    parser.add_argument("--title", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packet = build_packet(
        output_dir=args.output_dir.resolve(),
        kernel_id=args.kernel_id,
        source_kernel=args.source_kernel,
        title=args.title,
    )
    validate_r11_authority(packet)
    prepare_packet(packet)
    manifest_path = packet.output_dir / "packet.json"
    summary = {
        "status": "prepared",
        "packet_dir": str(packet.output_dir),
        "packet_manifest": str(manifest_path),
        "packet_manifest_sha256": sha256_file(manifest_path),
        "kernel_id": packet.kernel_id,
        "source_kernel": packet.source_kernel,
        "requested_accelerator": None,
        "attempt_id": packet.attempt_id,
        "packet_contract_sha256": packet.packet_contract_sha256,
        "blob_count": len(packet.blobs),
        "mounted_blob_bytes": packet.mounted_blob_bytes,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
