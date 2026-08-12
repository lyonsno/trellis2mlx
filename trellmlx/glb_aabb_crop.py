"""Memory-bounded, provenance-bearing AABB crops for simple triangle GLBs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import mmap
from pathlib import Path
import shutil
import struct
from typing import Any

import numpy as np


ROUTE = "glb-mmap-triangle-bounds-aabb-v1"
GLB_MAGIC = b"glTF"
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


class CropError(RuntimeError):
    def __init__(self, phase: str, message: str, evidence: dict[str, Any] | None = None):
        super().__init__(message)
        self.phase = phase
        self.evidence = evidence or {}


@dataclass
class GlbMeshView:
    path: Path
    document: dict[str, Any]
    file_handle: Any
    mapping: mmap.mmap
    vertices: np.ndarray
    faces: np.ndarray
    position_accessor: int
    index_accessor: int
    binary_offset: int

    def close(self) -> None:
        del self.vertices
        del self.faces
        self.mapping.close()
        self.file_handle.close()

    def __enter__(self) -> "GlbMeshView":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


@dataclass(frozen=True)
class CropSelection:
    source_face_indices: np.ndarray
    source_vertex_indices: np.ndarray
    core_face_mask: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    core_min: np.ndarray
    core_max: np.ndarray
    outer_min: np.ndarray
    outer_max: np.ndarray
    source_face_count: int
    source_vertex_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_accessor_array(
    *,
    mapping: mmap.mmap,
    document: dict[str, Any],
    accessor_index: int,
    binary_offset: int,
    binary_length: int,
    declared_buffer_length: int,
    expected_type: str,
    component_types: dict[int, np.dtype],
) -> np.ndarray:
    accessors = document.get("accessors", [])
    views = document.get("bufferViews", [])
    if not isinstance(accessor_index, int) or not 0 <= accessor_index < len(accessors):
        raise CropError("parse_glb", f"invalid accessor index {accessor_index}")
    accessor = accessors[accessor_index]

    try:
        view_index = int(accessor["bufferView"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CropError(
            "parse_glb", f"accessor {accessor_index} has an invalid bufferView index"
        ) from exc
    if view_index < 0 or view_index >= len(views):
        raise CropError(
            "parse_glb", f"accessor {accessor_index} has invalid bufferView index {view_index}"
        )
    view = views[view_index]

    if accessor.get("sparse") is not None:
        raise CropError("parse_glb", f"sparse accessor {accessor_index} is unsupported")
    if accessor.get("type") != expected_type:
        raise CropError(
            "parse_glb",
            f"accessor {accessor_index} must be {expected_type}, found {accessor.get('type')}",
        )
    if view.get("buffer", 0) != 0:
        raise CropError("parse_glb", "only the embedded GLB buffer is supported")

    component_type = accessor.get("componentType")
    if component_type not in component_types:
        raise CropError(
            "parse_glb",
            f"unsupported component type {component_type} for accessor {accessor_index}",
        )
    dtype = np.dtype(component_types[component_type])
    width = 3 if expected_type == "VEC3" else 1
    packed_stride = dtype.itemsize * width
    try:
        stride = int(view.get("byteStride", packed_stride))
        count = int(accessor["count"])
        view_offset = int(view.get("byteOffset", 0))
        view_length = int(view["byteLength"])
        accessor_offset = int(accessor.get("byteOffset", 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise CropError(
            "parse_glb", f"accessor {accessor_index} has invalid integer metadata"
        ) from exc
    if stride < packed_stride:
        raise CropError("parse_glb", f"accessor {accessor_index} has invalid byte stride {stride}")
    if min(count, view_offset, view_length, accessor_offset) < 0:
        raise CropError("parse_glb", f"accessor {accessor_index} has negative bounds metadata")
    view_end = view_offset + view_length
    if view_end > declared_buffer_length or view_end > binary_length:
        raise CropError(
            "parse_glb", f"bufferView {view_index} exceeds the declared GLB BIN buffer"
        )
    accessor_end = accessor_offset + max(count - 1, 0) * stride + packed_stride
    if accessor_offset > view_length or accessor_end > view_length:
        raise CropError(
            "parse_glb", f"accessor {accessor_index} exceeds bufferView {view_index}"
        )

    offset = binary_offset + view_offset + accessor_offset
    final_byte = binary_offset + view_offset + accessor_end
    if final_byte > binary_offset + binary_length or final_byte > len(mapping):
        raise CropError("parse_glb", f"accessor {accessor_index} exceeds the GLB BIN chunk")

    shape = (count, width) if width > 1 else (count,)
    strides = (stride, dtype.itemsize) if width > 1 else (stride,)
    return np.ndarray(shape=shape, dtype=dtype, buffer=mapping, offset=offset, strides=strides)


def open_triangle_glb(path: Path) -> GlbMeshView:
    if not path.exists():
        raise CropError("load_glb", f"input GLB does not exist: {path}")

    file_handle = path.open("rb")
    mapping: mmap.mmap | None = None
    try:
        mapping = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)
        if len(mapping) < 20:
            raise CropError("parse_glb", "GLB is shorter than its container header")
        magic, version, declared_length = struct.unpack_from("<4sII", mapping, 0)
        if magic != GLB_MAGIC or version != 2:
            raise CropError("parse_glb", f"expected GLB v2, found magic={magic!r} version={version}")
        if declared_length != len(mapping):
            raise CropError(
                "parse_glb",
                f"declared GLB length {declared_length} does not match file size {len(mapping)}",
            )

        json_length, json_type = struct.unpack_from("<II", mapping, 12)
        if json_type != JSON_CHUNK:
            raise CropError("parse_glb", "first GLB chunk is not JSON")
        json_start = 20
        json_end = json_start + json_length
        if json_end + 8 > len(mapping):
            raise CropError("parse_glb", "GLB JSON chunk is truncated")
        document = json.loads(bytes(mapping[json_start:json_end]).rstrip(b" \t\r\n\x00"))

        binary_length, binary_type = struct.unpack_from("<II", mapping, json_end)
        if binary_type != BIN_CHUNK:
            raise CropError("parse_glb", "second GLB chunk is not BIN")
        binary_offset = json_end + 8
        if binary_offset + binary_length > len(mapping):
            raise CropError("parse_glb", "GLB binary chunk is truncated")
        buffers = document.get("buffers", [])
        if len(buffers) != 1 or buffers[0].get("uri") is not None:
            raise CropError("parse_glb", "only one embedded GLB buffer is supported")
        try:
            declared_buffer_length = int(buffers[0]["byteLength"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CropError("parse_glb", "embedded GLB buffer has invalid byteLength") from exc
        if declared_buffer_length < 0 or declared_buffer_length > binary_length:
            raise CropError("parse_glb", "declared GLB buffer exceeds the BIN chunk")

        meshes = document.get("meshes", [])
        if len(meshes) != 1 or len(meshes[0].get("primitives", [])) != 1:
            raise CropError("parse_glb", "expected exactly one mesh with one primitive")
        primitive = meshes[0]["primitives"][0]
        if primitive.get("mode", 4) != 4:
            raise CropError("parse_glb", "mesh primitive must use TRIANGLES mode")
        if "indices" not in primitive:
            raise CropError("parse_glb", "mesh primitive must be indexed")
        try:
            position_accessor = int(primitive["attributes"]["POSITION"])
            index_accessor = int(primitive["indices"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CropError("parse_glb", "mesh primitive lacks valid POSITION/indices accessors") from exc

        vertices = _read_accessor_array(
            mapping=mapping,
            document=document,
            accessor_index=position_accessor,
            binary_offset=binary_offset,
            binary_length=binary_length,
            declared_buffer_length=declared_buffer_length,
            expected_type="VEC3",
            component_types={5126: np.dtype("<f4")},
        )
        indices = _read_accessor_array(
            mapping=mapping,
            document=document,
            accessor_index=index_accessor,
            binary_offset=binary_offset,
            binary_length=binary_length,
            declared_buffer_length=declared_buffer_length,
            expected_type="SCALAR",
            component_types={
                5121: np.dtype("u1"),
                5123: np.dtype("<u2"),
                5125: np.dtype("<u4"),
            },
        )
        if indices.size == 0 or indices.size % 3:
            raise CropError("parse_glb", f"index count {indices.size} is not a nonzero triangle list")
        faces = indices.reshape(-1, 3)
        if int(faces.max(initial=0)) >= len(vertices):
            raise CropError("parse_glb", "mesh indices exceed the POSITION accessor")
        if not np.isfinite(vertices).all():
            raise CropError("parse_glb", "POSITION accessor contains non-finite coordinates")

        return GlbMeshView(
            path=path,
            document=document,
            file_handle=file_handle,
            mapping=mapping,
            vertices=vertices,
            faces=faces,
            position_accessor=position_accessor,
            index_accessor=index_accessor,
            binary_offset=binary_offset,
        )
    except Exception:
        if mapping is not None:
            mapping.close()
        file_handle.close()
        raise


def validate_request(
    core_min: np.ndarray,
    core_max: np.ndarray,
    halo_fraction: float,
    chunk_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
    core_min = np.asarray(core_min, dtype=np.float64)
    core_max = np.asarray(core_max, dtype=np.float64)
    if core_min.shape != (3,) or core_max.shape != (3,):
        raise CropError("validate_request", "core bounds must each contain exactly three values")
    if not np.isfinite(core_min).all() or not np.isfinite(core_max).all():
        raise CropError("validate_request", "core bounds must be finite")
    if np.any(core_min >= core_max):
        raise CropError("validate_request", "every core minimum must be less than its maximum")
    if not np.isfinite(halo_fraction) or halo_fraction < 0:
        raise CropError("validate_request", "halo fraction must be finite and nonnegative")
    if chunk_faces <= 0:
        raise CropError("validate_request", "chunk face count must be positive")
    span = core_max - core_min
    return core_min - halo_fraction * span, core_max + halo_fraction * span


def select_aabb_crop(
    view: GlbMeshView,
    *,
    core_min: np.ndarray,
    core_max: np.ndarray,
    halo_fraction: float,
    chunk_faces: int,
) -> CropSelection:
    outer_min, outer_max = validate_request(core_min, core_max, halo_fraction, chunk_faces)
    core_min = np.asarray(core_min, dtype=np.float64)
    core_max = np.asarray(core_max, dtype=np.float64)
    selected_ids: list[np.ndarray] = []
    selected_core: list[np.ndarray] = []

    for start in range(0, len(view.faces), chunk_faces):
        end = min(start + chunk_faces, len(view.faces))
        face_chunk = np.asarray(view.faces[start:end], dtype=np.int64)
        triangles = np.asarray(view.vertices[face_chunk], dtype=np.float32)
        triangle_min = triangles.min(axis=1)
        triangle_max = triangles.max(axis=1)
        in_outer = np.all(triangle_max >= outer_min, axis=1) & np.all(
            triangle_min <= outer_max, axis=1
        )
        if not in_outer.any():
            continue
        in_core = np.all(triangle_max >= core_min, axis=1) & np.all(
            triangle_min <= core_max, axis=1
        )
        local_ids = np.nonzero(in_outer)[0]
        selected_ids.append(local_ids.astype(np.int64, copy=False) + start)
        selected_core.append(in_core[local_ids].astype(np.bool_, copy=False))

    if not selected_ids:
        raise CropError(
            "select_faces",
            "outer AABB selected no triangles",
            {
                "source_faces": int(len(view.faces)),
                "source_vertices": int(len(view.vertices)),
                "outer_min": outer_min.tolist(),
                "outer_max": outer_max.tolist(),
            },
        )

    source_face_indices = np.concatenate(selected_ids)
    core_face_mask = np.concatenate(selected_core)
    source_faces = np.asarray(view.faces[source_face_indices], dtype=np.uint32)
    source_vertex_indices = np.unique(source_faces.reshape(-1)).astype(np.int64, copy=False)
    vertices = np.asarray(view.vertices[source_vertex_indices], dtype=np.float32).copy()
    faces = np.searchsorted(source_vertex_indices, source_faces).astype(np.uint32, copy=False)

    return CropSelection(
        source_face_indices=source_face_indices,
        source_vertex_indices=source_vertex_indices,
        core_face_mask=core_face_mask,
        vertices=vertices,
        faces=faces,
        core_min=core_min,
        core_max=core_max,
        outer_min=outer_min,
        outer_max=outer_max,
        source_face_count=int(len(view.faces)),
        source_vertex_count=int(len(view.vertices)),
    )


def _pad4(data: bytes, padding: bytes) -> bytes:
    return data + padding * ((-len(data)) % 4)


def write_geometry_glb(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    vertices = np.ascontiguousarray(vertices, dtype="<f4")
    faces = np.ascontiguousarray(faces, dtype="<u4")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise CropError("write_output", "output vertices must be nonempty VEC3 float32")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise CropError("write_output", "output faces must be nonempty triangles")

    index_bytes = faces.tobytes(order="C")
    position_bytes = vertices.tobytes(order="C")
    binary = index_bytes + position_bytes
    document = {
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5125,
                "count": int(faces.size),
                "max": [int(faces.max())],
                "min": [int(faces.min())],
                "type": "SCALAR",
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": int(len(vertices)),
                "max": vertices.max(axis=0).astype(float).tolist(),
                "min": vertices.min(axis=0).astype(float).tolist(),
                "type": "VEC3",
            },
        ],
        "asset": {"generator": "trellis2mlx glb_aabb_crop", "version": "2.0"},
        "bufferViews": [
            {"buffer": 0, "byteLength": len(index_bytes), "byteOffset": 0, "target": 34963},
            {
                "buffer": 0,
                "byteLength": len(position_bytes),
                "byteOffset": len(index_bytes),
                "target": 34962,
            },
        ],
        "buffers": [{"byteLength": len(binary)}],
        "meshes": [
            {
                "name": "aabb_crop",
                "primitives": [{"attributes": {"POSITION": 1}, "indices": 0, "mode": 4}],
            }
        ],
        "nodes": [{"mesh": 0, "name": "aabb_crop"}],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
    }
    json_bytes = _pad4(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8"), b" "
    )
    binary_bytes = _pad4(binary, b"\x00")
    total_length = 12 + 8 + len(json_bytes) + 8 + len(binary_bytes)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_temporary_path(path)
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as stream:
        stream.write(struct.pack("<4sII", GLB_MAGIC, 2, total_length))
        stream.write(struct.pack("<II", len(json_bytes), JSON_CHUNK))
        stream.write(json_bytes)
        stream.write(struct.pack("<II", len(binary_bytes), BIN_CHUNK))
        stream.write(binary_bytes)
        stream.flush()
    temporary.replace(path)


def write_provenance(directory: Path, selection: CropSelection) -> None:
    temporary = provenance_temporary_path(directory)
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    np.save(temporary / "source_face_indices.npy", selection.source_face_indices)
    np.save(temporary / "source_vertex_indices.npy", selection.source_vertex_indices)
    np.save(temporary / "core_face_mask.npy", selection.core_face_mask)
    manifest = {
        "route": ROUTE,
        "source_faces": selection.source_face_count,
        "source_vertices": selection.source_vertex_count,
        "selected_faces": int(len(selection.faces)),
        "selected_vertices": int(len(selection.vertices)),
        "core_faces": int(selection.core_face_mask.sum()),
        "halo_only_faces": int((~selection.core_face_mask).sum()),
        "core_min": selection.core_min.tolist(),
        "core_max": selection.core_max.tolist(),
        "outer_min": selection.outer_min.tolist(),
        "outer_max": selection.outer_max.tolist(),
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if directory.exists():
        shutil.rmtree(directory)
    temporary.replace(directory)


def remove_output_surface(output_path: Path, provenance_dir: Path) -> dict[str, bool]:
    output_existed = output_path.exists()
    provenance_existed = provenance_dir.exists()
    if output_existed and (output_path.is_dir() or not output_path.is_file()):
        raise CropError(
            "cleanup_output_surface",
            f"existing output GLB is not a removable file: {output_path}",
            {
                "preexisting_output_removed": False,
                "preexisting_provenance_removed": False,
            },
        )
    if provenance_existed and (not provenance_dir.is_dir() or provenance_dir.is_symlink()):
        raise CropError(
            "cleanup_output_surface",
            f"existing provenance surface is not a removable directory: {provenance_dir}",
            {
                "preexisting_output_removed": False,
                "preexisting_provenance_removed": False,
            },
        )
    if output_existed:
        output_path.unlink()
    if provenance_existed:
        shutil.rmtree(provenance_dir)
    return {
        "preexisting_output_removed": output_existed,
        "preexisting_provenance_removed": provenance_existed,
    }


def remove_partial_output_surface(output_path: Path, provenance_dir: Path) -> dict[str, bool]:
    """Remove only file/dir shapes owned by this tool during failure recovery."""
    output_removed = False
    provenance_removed = False
    if output_path.is_file() or output_path.is_symlink():
        output_path.unlink()
        output_removed = True
    if provenance_dir.is_dir() and not provenance_dir.is_symlink():
        shutil.rmtree(provenance_dir)
        provenance_removed = True
    return {
        "partial_output_removed": output_removed,
        "partial_provenance_removed": provenance_removed,
        "output_cleanup_blocked_by_type": output_path.exists(),
        "provenance_cleanup_blocked_by_type": provenance_dir.exists(),
    }


def output_temporary_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".tmp")


def provenance_temporary_path(provenance_dir: Path) -> Path:
    return provenance_dir.with_name(provenance_dir.name + ".tmp")


def validate_output_paths(
    *, input_path: Path, output_path: Path, report_path: Path, provenance_dir: Path
) -> None:
    source = input_path.resolve()
    output = output_path.resolve()
    report = report_path.resolve()
    provenance = provenance_dir.resolve()
    output_temporary = output_temporary_path(output).resolve()
    provenance_temporary = provenance_temporary_path(provenance).resolve()
    named_paths = {
        "input GLB": source,
        "output GLB": output,
        "output temporary": output_temporary,
        "report JSON": report,
        "provenance directory": provenance,
        "provenance temporary": provenance_temporary,
    }
    values = list(named_paths.values())
    if len(set(values)) != len(values):
        collisions = [
            name
            for name, value in named_paths.items()
            if values.count(value) > 1
        ]
        raise CropError(
            "validate_output_paths",
            "crop paths must be distinct, including derived temporary paths: "
            + ", ".join(collisions),
        )

    for directory_name, directory in (
        ("provenance directory", provenance),
        ("provenance temporary", provenance_temporary),
    ):
        for file_name, file_path in (
            ("input GLB", source),
            ("output GLB", output),
            ("output temporary", output_temporary),
            ("report JSON", report),
        ):
            if file_path.is_relative_to(directory):
                raise CropError(
                    "validate_output_paths",
                    f"{directory_name} must not contain {file_name}",
                )
