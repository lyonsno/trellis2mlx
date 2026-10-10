"""Run pinned official to_glb on saved raw mesh/appearance.

No inference or local cleanup. Stage observers forward native calls unchanged.
An optional, explicitly reported reconstruction override changes only the
remesher's resolution and padded scale, preserving the appearance grid.
Each finishing arm has an independent CUDA process; failures preserve reports
and intermediate arrays and do not suppress the other arm.
"""

import argparse
import copy
import hashlib
import importlib.util
import io
import inspect
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np

SOURCES = {
    "TRELLIS.2": ("https://github.com/microsoft/TRELLIS.2.git",
                 "5565d240c4a494caaf9ece7a554542b76ffa36d3"),
    "CuMesh": ("https://github.com/JeffreyXiang/CuMesh.git",
               "c4ad6125924fcedfd13f0bd61520ca2d24eb7a87"),
    "FlexGEMM": ("https://github.com/JeffreyXiang/FlexGEMM.git",
                 "6dd94a859c26ee8246888502eada3dd8ad85532e"),
    "nvdiffrast": ("https://github.com/NVlabs/nvdiffrast.git",
                  "253ac4fcea7de5f396371124af597e6cc957bfae"),
}
POSTPROCESS_SHA = "ef51a1ba0f2748ffb4c265b47d382cee956f23c6a52d0f3587e6d8beccb7e54a"
MUTATIONS = {"init", "fill_holes", "simplify", "remove_duplicate_faces",
             "repair_non_manifold_edges", "remove_small_connected_components",
             "unify_face_orientations"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_inputs(mesh_path, texture_path, mesh_sha, texture_sha, grid_size):
    for path, expected in ((mesh_path, mesh_sha), (texture_path, texture_sha)):
        if digest(path) != expected:
            raise ValueError(f"input digest mismatch: {path}")
    with np.load(mesh_path, allow_pickle=False) as data:
        vertices, faces = data["vertices"], data["faces"]
    with np.load(texture_path, allow_pickle=False) as data:
        attrs, coords = data["tex_np"], data["tex_coords_spatial"]
    for name, array, width in (("vertices", vertices, 3), ("faces", faces, 3),
                               ("attrs", attrs, 6), ("coords", coords, 3)):
        if array.ndim != 2 or array.shape[1] != width or len(array) == 0:
            raise ValueError(f"invalid {name} shape: {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"nonfinite {name}")
    if faces.dtype.kind not in "iu" or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("invalid triangle indices")
    if coords.dtype.kind not in "iu" or len(coords) != len(attrs):
        raise ValueError("invalid sparse appearance coordinates")
    if grid_size <= 0 or coords.min() < 0 or coords.max() >= grid_size:
        raise ValueError("appearance coordinates outside grid")
    return dict(vertices=vertices, faces=faces, attrs=attrs, coords=coords)


def verify_source(path, expected=POSTPROCESS_SHA):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"official source digest mismatch: {actual}")
    return actual


def validate_glb(path, texture_size):
    from PIL import Image
    blob = Path(path).read_bytes()
    if len(blob) < 20 or struct.unpack_from("<4sII", blob) != (b"glTF", 2, len(blob)):
        raise ValueError("invalid or incomplete GLB")
    size, kind = struct.unpack_from("<I4s", blob, 12)
    if kind != b"JSON":
        raise ValueError("GLB missing JSON chunk")
    doc = json.loads(blob[20:20 + size])
    primitives = [p for mesh in doc.get("meshes", []) for p in mesh.get("primitives", [])]
    if not primitives or not doc.get("accessors"):
        raise ValueError("GLB missing geometry")
    for p in primitives:
        if not {"POSITION", "NORMAL", "TEXCOORD_0"} <= p.get("attributes", {}).keys():
            raise ValueError("GLB missing textured geometry attributes")
        if doc["accessors"][p["indices"]]["count"] <= 0 or "material" not in p:
            raise ValueError("GLB missing geometry/material")
    binary_offset = 20 + size
    if len(blob) < binary_offset + 8:
        raise ValueError("GLB missing binary data")
    binary_size, binary_kind = struct.unpack_from("<I4s", blob, binary_offset)
    if binary_kind != b"BIN\x00" or binary_offset + 8 + binary_size != len(blob):
        raise ValueError("GLB invalid binary chunk")
    binary = memoryview(blob)[binary_offset + 8:]
    image_sizes = []
    for image in doc.get("images", []):
        view = doc["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        with Image.open(io.BytesIO(binary[start:start + view["byteLength"]])) as decoded:
            decoded.load()
            image_sizes.append(list(decoded.size))
    if len(image_sizes) < 2 or any(s != [texture_size, texture_size] for s in image_sizes):
        raise ValueError(f"GLB texture resolution mismatch: {image_sizes}")
    return {"sha256": digest(path), "size_bytes": len(blob), "image_sizes": image_sizes,
            "faces": sum(doc["accessors"][p["indices"]]["count"] // 3 for p in primitives),
            "vertices": sum(doc["accessors"][p["attributes"]["POSITION"]]["count"] for p in primitives)}


def observe_mesh_class(native_class, capture, boundary=None):
    class Observed:
        def __init__(self, *args, **kwargs):
            self._native = native_class(*args, **kwargs)

        def __getattr__(self, name):
            original = getattr(self._native, name)
            if name not in MUTATIONS | {"uv_unwrap", "compute_vertex_normals"}:
                return original

            def call(*args, **kwargs):
                if boundary:
                    boundary(name, "before")
                try:
                    result = original(*args, **kwargs)
                except Exception:
                    if boundary:
                        boundary(name, "failed")
                    raise
                if boundary:
                    boundary(name, "after")
                if name in MUTATIONS:
                    vertices, faces = self._native.read()
                    capture(name, {"vertices": vertices, "faces": faces})
                elif name == "uv_unwrap":
                    capture(name, dict(zip(("vertices", "faces", "uvs", "vmaps"), result)))
                else:
                    capture(name, {"normals": self._native.read_vertex_normals()})
                return result
            return call
    return Observed


def observe_remesher(native, resolution_override, capture):
    if resolution_override is not None and resolution_override <= 0:
        raise ValueError("reconstruction resolution must be positive")

    def call(*args, **kwargs):
        source_resolution = kwargs["resolution"]
        source_scale = kwargs["scale"]
        band = kwargs.get("band", 1)
        effective = dict(kwargs)
        if resolution_override is not None:
            domain_scale = source_scale / ((source_resolution + 3 * band) / source_resolution)
            effective["resolution"] = resolution_override
            effective["scale"] = domain_scale * ((resolution_override + 3 * band) / resolution_override)
        capture({"source_resolution": source_resolution,
                 "effective_resolution": effective["resolution"],
                 "source_scale": source_scale, "effective_scale": effective["scale"],
                 "band": band, "project_back": kwargs.get("project_back", 0),
                 "resolution_override": resolution_override is not None})
        return native(*args, **effective)
    return call


def run_reported(path, report, operation):
    started = time.perf_counter()
    write_json(path, report)
    code = 0
    try:
        operation(report)
        report.update(status="completed", phase="completed", failure_phase=None)
    except Exception as exc:
        code = 1
        report.update(status="failed", failure_phase=report.get("phase"),
                      error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(path, report)
    return code


def command(argv, report, cwd=None):
    print("+", " ".join(map(str, argv)), flush=True)
    start = time.perf_counter()
    result = subprocess.run(list(map(str, argv)), cwd=cwd, capture_output=True, text=True)
    record = dict(argv=list(map(str, argv)), cwd=str(cwd) if cwd else None,
                  exit_code=result.returncode, elapsed_seconds=time.perf_counter() - start,
                  stdout=result.stdout, stderr=result.stderr)
    report.setdefault("commands", []).append(record)
    print(result.stdout, flush=True)
    print(result.stderr, file=sys.stderr, flush=True)
    result.check_returncode()
    return result.stdout.strip()


def bootstrap(args, report):
    report["phase"] = "bootstrap"
    root = args.runtime.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report["source_identities"] = {}
    for name, (url, sha) in SOURCES.items():
        target = root / name
        command(["git", "clone", "--filter=blob:none", "--no-checkout", url, target], report)
        command(["git", "checkout", "--detach", sha], report, target)
        if name in {"CuMesh", "FlexGEMM"}:
            command(["git", "submodule", "update", "--init", "--recursive"], report, target)
        actual = command(["git", "rev-parse", "HEAD"], report, target)
        dirty = command(["git", "status", "--porcelain", "--untracked-files=no"], report, target)
        if actual != sha or dirty:
            raise ValueError(f"source identity mismatch: {name}")
        report["source_identities"][name] = {"repository": url, "commit": actual, "tracked_clean": True}
    command([sys.executable, "-m", "pip", "install", "ninja", "numpy", "pillow",
             "opencv-python-headless", "trimesh", "tqdm", "easydict"], report)
    for name in ("CuMesh", "FlexGEMM", "nvdiffrast"):
        command([sys.executable, "-m", "pip", "install", "--no-build-isolation",
                 "--no-deps", str(root / name)], report)
    report["postprocess_sha256"] = verify_source(root / "TRELLIS.2/o-voxel/o_voxel/postprocess.py")


def finishing_settings(args):
    return dict(grid_size=args.grid_size, aabb=[[-.5, -.5, -.5], [.5, .5, .5]],
                decimation_target=args.target_faces, texture_size=args.texture_size,
                remesh=args.arm == "remesh", remesh_band=1,
                remesh_project=args.remesh_project, verbose=True, use_tqdm=False)


def load_reconstruction_checkpoint(args):
    """Admit a saved post-projection mesh, not an arbitrary replacement mesh."""
    if args.arm != "remesh" or getattr(args, "projection_batch_vertices", None) is not None:
        raise ValueError("checkpoint continuation requires remesh without fresh projection")
    if digest(args.resume_report) != args.resume_report_sha:
        raise ValueError("checkpoint origin report digest mismatch")
    origin = json.loads(args.resume_report.read_text())
    if origin.get("source_pins") != {k: list(v) for k, v in SOURCES.items()}:
        raise ValueError("checkpoint source pins mismatch")
    requested, route = origin["requested"], origin["effective_route"]
    if (route.get("postprocess_sha256") != POSTPROCESS_SHA or
            route.get("model_inference") is not False or route.get("local_cleanup") is not False or
            route.get("device") != "Tesla T4" or route.get("capability") != [7, 5]):
        raise ValueError("checkpoint effective source route mismatch")
    for field in ("mesh_sha", "appearance_sha", "grid_size", "target_faces", "texture_size", "remesh_project"):
        if requested.get(field) != getattr(args, field):
            raise ValueError(f"checkpoint requested {field} mismatch")
    if route.get("settings") != finishing_settings(args):
        raise ValueError("checkpoint effective finishing settings mismatch")
    reconstruction = route.get("reconstruction")
    call_evidence = "producer-observed-remesher-call"
    if reconstruction is None:
        if (route.get("finishing") != "unmodified-official-to_glb" or
                requested.get("remesh_resolution") is not None or
                requested.get("projection_batch_vertices") is not None):
            raise ValueError("missing observed reconstruction identity")
        reconstruction = {"appearance_grid_size": args.grid_size, "calls": [{
            "source_resolution": args.grid_size, "effective_resolution": args.grid_size,
            "source_scale": (args.grid_size + 3) / args.grid_size,
            "effective_scale": (args.grid_size + 3) / args.grid_size,
            "band": 1, "project_back": args.remesh_project, "resolution_override": False}]}
        call_evidence = "pinned-unmodified-to_glb-derived-settings"
    calls = reconstruction["calls"]
    resolution = args.remesh_resolution or args.grid_size
    if (reconstruction["appearance_grid_size"] != args.grid_size or len(calls) != 1 or
            calls[0]["source_resolution"] != args.grid_size or calls[0]["effective_resolution"] != resolution or
            calls[0]["band"] != 1 or calls[0]["project_back"] != args.remesh_project):
        raise ValueError("checkpoint reconstruction settings mismatch")
    stages = origin["stages"]
    if [s["label"] for s in stages[:3]] != ["init", "fill_holes", "init"]:
        raise ValueError("checkpoint stage boundary mismatch")
    arrays = []
    for path, stage in ((args.resume_rebuilt, stages[2]), (args.resume_filled, stages[1])):
        if digest(path) != stage["sha256"]:
            raise ValueError("checkpoint geometry digest mismatch")
        with np.load(path, allow_pickle=False) as data:
            values = {key: data[key] for key in ("vertices", "faces")}
        vertices, faces = values["vertices"], values["faces"]
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0 or
                vertices.dtype != np.float32 or not np.isfinite(vertices).all() or
                faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or
                faces.dtype != np.int32 or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("invalid complete checkpoint geometry")
        arrays.append(values)
    for key, value in arrays[0].items():
        if stages[2]["arrays"][key] != {"shape": list(value.shape), "dtype": str(value.dtype)}:
            raise ValueError("checkpoint shape differs from producer receipt")
    memory = route.get("projection_memory")
    if route.get("finishing") == "official-to_glb-with-deferred-batched-projection" and memory is None:
        raise ValueError("missing checkpoint projection account")
    if memory is not None:
        projected = memory.get("calls", [])
        count = len(arrays[0]["vertices"])
        if (memory.get("conformance", {}).get("exact_equal") is not True or len(projected) != 1 or
                projected[0].get("processed_vertices") != count or projected[0].get("vertices") != count or
                projected[0].get("project_back") != args.remesh_project):
            raise ValueError("checkpoint projection incomplete or unverified")
    receipt = {"origin_report_sha256": args.resume_report_sha,
               "rebuilt_sha256": stages[2]["sha256"], "filled_sha256": stages[1]["sha256"],
               "reconstruction_executed_this_run": False, "origin_reconstruction": calls[0],
               "origin_call_evidence": call_evidence,
               "vertices": len(arrays[0]["vertices"]), "faces": len(arrays[0]["faces"])}
    return arrays[0], arrays[1], receipt


def resume_remesher(rebuilt, filled, receipt, to_cuda, equal, capture):
    def call(vertices, faces, **kwargs):
        source = receipt["origin_reconstruction"]
        for key, expected in (("resolution", source["effective_resolution"]),
                              ("scale", source["effective_scale"]), ("band", source["band"]),
                              ("project_back", source["project_back"])):
            if kwargs.get(key) != expected:
                raise ValueError(f"checkpoint effective remesher {key} mismatch")
        if kwargs.get("bvh") is None:
            raise ValueError("checkpoint continuation missing original BVH")
        for key, value in (("vertices", vertices), ("faces", faces)):
            current = value.detach().cpu().numpy() if hasattr(value, "detach") else value
            if not equal(current, filled[key]):
                raise ValueError("checkpoint original surface differs before BVH reuse")
        capture(dict(receipt, original_surface_exact=True))
        return to_cuda(rebuilt["vertices"]), to_cuda(rebuilt["faces"])
    return call


def memory_boundary(torch, report, path, release_unused):
    def snapshot(boundary):
        try:
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
            row = {"boundary": boundary, "global_free_bytes": free, "global_total_bytes": total,
                   "torch_allocated_bytes": allocated, "torch_reserved_bytes": reserved,
                   "torch_reserved_unused_bytes": reserved - allocated,
                   "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        except Exception as exc:
            row = {"boundary": boundary, "measurement_error": f"{type(exc).__name__}: {exc}"}
        report.setdefault("cuda_memory", []).append(row)
        write_json(path, report)

    def boundary(name, state):
        report["phase"] = f"native_{name}"
        snapshot(f"{state}-{name}")
        if name == "simplify" and state == "before" and release_unused:
            torch.cuda.empty_cache()
            snapshot("after-empty-cache-before-simplify")
    return boundary


def project_vertices_batched(mesh_vertices, vertices, faces, bvh, project_back, batch_vertices):
    """Apply the pinned native interpolation, retaining every query and output row."""
    if batch_vertices <= 0:
        raise ValueError("projection batch size must be positive")
    processed = batches = 0
    for start in range(0, len(mesh_vertices), batch_vertices):
        batch = mesh_vertices[start:start + batch_vertices]
        distance, face_id, uvw = bvh.unsigned_distance(batch, return_uvw=True)
        orig_tri_verts = vertices[faces[face_id.long()]]
        projected_verts = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
        batch -= project_back * (batch - projected_verts)
        processed += len(batch)
        batches += 1
        # Release the previous batch before the next query allocates its outputs.
        del distance, face_id, uvw, orig_tri_verts, projected_verts, batch
    return dict(vertices=len(mesh_vertices), processed_vertices=processed,
                batch_vertices=batch_vertices, batches=batches, project_back=project_back)


def projection_conformance(queries, vertices, faces, bvh, project_back, batch_vertices,
                           equal, capture=None):
    """Compare the actual BVH's batched query against the pinned full-query formula."""
    expected = queries.clone()
    distance, face_id, uvw = bvh.unsigned_distance(expected, return_uvw=True)
    orig_tri_verts = vertices[faces[face_id.long()]]
    projected_verts = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
    expected -= project_back * (expected - projected_verts)
    del distance, face_id, uvw, orig_tri_verts, projected_verts
    actual = queries.clone()
    receipt = project_vertices_batched(actual, vertices, faces, bvh, project_back, batch_vertices)
    receipt["exact_equal"] = bool(equal(expected, actual))
    if capture is not None:
        capture(queries, expected, actual, receipt)
    if not receipt["exact_equal"]:
        raise ValueError("CUDA projection batch equivalence failed; reconstruction not launched")
    return expected, actual, receipt


def defer_remesh_projection(native, batch_vertices, capture, verify=None):
    """Return from unchanged native reconstruction before its optional projection."""
    if batch_vertices <= 0:
        raise ValueError("projection batch size must be positive")
    signature = inspect.signature(native)

    def call(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        project_back = bound.arguments.get("project_back", 0)
        if project_back <= 0:
            return native(*args, **kwargs)
        bvh = bound.arguments.get("bvh")
        if bvh is None:
            raise ValueError("deferred projection requires the original reusable BVH")
        vertices, faces = bound.arguments["vertices"], bound.arguments["faces"]
        if verify is not None:
            verify(vertices, faces, bvh, project_back, batch_vertices)
        bound.arguments["project_back"] = 0
        mesh_vertices, mesh_triangles = native(*bound.args, **bound.kwargs)
        # The native function frame and its reconstruction temporaries are now gone.
        receipt = project_vertices_batched(mesh_vertices, vertices, faces, bvh,
                                           project_back, batch_vertices)
        receipt["native_project_back"] = 0
        capture(receipt)
        return mesh_vertices, mesh_triangles
    return call


def run_arm(args, report):
    import torch
    import cumesh
    import flex_gemm
    import nvdiffrast.torch
    from importlib.metadata import version
    report["phase"] = "input_admission"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no fallback admitted")
    arrays = load_inputs(args.mesh, args.appearance, args.mesh_sha, args.appearance_sha, args.grid_size)
    source_path = args.runtime.resolve() / "TRELLIS.2/o-voxel/o_voxel/postprocess.py"
    source_sha = verify_source(source_path)
    spec = importlib.util.spec_from_file_location("original_trellis_postprocess", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = finishing_settings(args)
    resolution_override = getattr(args, "remesh_resolution", None)
    batch_vertices = getattr(args, "projection_batch_vertices", None)
    if args.arm != "remesh" and resolution_override is not None:
        raise ValueError("reconstruction override requires the remesh arm")
    if batch_vertices is not None and (args.arm != "remesh" or batch_vertices <= 0):
        raise ValueError("positive projection batch size requires the remesh arm")
    resume_fields = (args.resume_rebuilt, args.resume_filled, args.resume_report, args.resume_report_sha)
    if any(v is not None for v in resume_fields) and not all(v is not None for v in resume_fields):
        raise ValueError("checkpoint continuation requires rebuilt, filled and hash-bound origin report")
    checkpoint = load_reconstruction_checkpoint(args) if all(v is not None for v in resume_fields) else None
    report["effective_route"] = {
        "entry_boundary": "saved-mlx-raw-mesh-and-decoded-appearance-before-local-cleanup",
        "finishing": ("official-to_glb-with-reconstruction-resolution-override"
                      if resolution_override is not None else "unmodified-official-to_glb"),
        "model_inference": False, "local_cleanup": False,
        "postprocess_path": str(source_path), "postprocess_sha256": source_sha,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
        "settings": settings, "trimesh_version": version("trimesh"),
        "cumesh_module": str(Path(cumesh.__file__).resolve()),
        "flex_gemm_module": str(Path(flex_gemm.__file__).resolve()),
        "nvdiffrast_module": str(Path(nvdiffrast.torch.__file__).resolve()),
    }
    report["effective_route"]["reconstruction"] = {
        "requested_resolution_override": resolution_override,
        "appearance_grid_size": args.grid_size,
        "calls": [],
    }
    if batch_vertices is not None:
        report["effective_route"]["finishing"] = "official-to_glb-with-deferred-batched-projection"
        report["effective_route"]["projection_memory"] = {
            "batch_vertices": batch_vertices, "all_vertices_required": True,
            "native_reconstruction_project_back": 0,
            "final_project_back": args.remesh_project, "calls": [],
            "execution": "unchanged-native-reconstruction-return-then-original-BVH-interpolation",
        }
    directory = args.output_dir / args.arm
    directory.mkdir(parents=True, exist_ok=False)
    report["stages"] = []
    report_path = args.output_dir / f"{args.arm}-report.json"

    def capture(label, values):
        torch.cuda.synchronize()
        converted = {key: value.detach().cpu().numpy() if hasattr(value, "detach") else value
                     for key, value in values.items()}
        path = directory / f"{len(report['stages']):02d}-{label}.npz"
        np.savez_compressed(path, **converted)
        report["stages"].append({"label": label, "path": str(path), "sha256": digest(path),
                                 "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                                            for k, v in converted.items()}})
        report["last_trustworthy_phase"] = f"captured-{label}"
        write_json(report_path, report)

    report["phase"] = "official_to_glb"
    native_class = cumesh.CuMesh
    native_remesher = cumesh.remeshing.remesh_narrow_band_dc

    def capture_remesher(record):
        if record["source_resolution"] != args.grid_size:
            raise ValueError("source remesher resolution differs from appearance grid")
        report["effective_route"]["reconstruction"]["calls"].append(record)
        report["last_trustworthy_phase"] = "native-remesher-call-settings-captured"
        write_json(report_path, report)

    remesher = native_remesher
    if checkpoint is not None:
        rebuilt, filled, receipt = checkpoint
        report["effective_route"]["finishing"] = "official-to_glb-with-saved-reconstruction-continuation"
        report["effective_route"]["checkpoint_continuation"] = receipt

        def capture_resume(record):
            report["effective_route"]["checkpoint_continuation"] = record
            report["last_trustworthy_phase"] = "checkpoint-and-original-BVH-surface-admitted"
            write_json(report_path, report)

        remesher = resume_remesher(rebuilt, filled, receipt,
            lambda value: torch.as_tensor(value, device="cuda"), np.array_equal, capture_resume)
    if batch_vertices is not None:
        memory = report["effective_route"]["projection_memory"]

        def verify_projection(vertices, faces, bvh, project_back, batch_size):
            report["phase"] = "cuda_projection_conformance"
            write_json(report_path, report)
            # Explicit diagnostic fixture spanning two full batches plus a tail.
            # This selects calibration queries, never limits reconstructed output.
            query_count = min(len(vertices), 2 * batch_size + 17)
            queries = vertices[:query_count].clone()
            queries += vertices.new_tensor([-.00071, .00043, .00029])

            def save_conformance(queries, expected, actual, receipt):
                torch.cuda.synchronize()
                path = directory / "projection-conformance.npz"
                np.savez_compressed(path, queries=queries.cpu().numpy(),
                                    original_formula=expected.cpu().numpy(),
                                    batched_formula=actual.cpu().numpy())
                memory["conformance"] = dict(receipt, path=str(path), sha256=digest(path),
                    fixture="retained-source-vertices-prefix-with-fixed-offset",
                    criterion="bitwise-equal-full-query-vs-batched-on-actual-CUDA-BVH",
                    claim_ceiling="calibration-query equivalence, not independent reconstruction parity")
                write_json(report_path, report)

            expected, actual, receipt = projection_conformance(
                queries, vertices, faces, bvh, project_back, batch_size,
                torch.equal, save_conformance)
            del expected, actual, queries
            report["last_trustworthy_phase"] = "actual-CUDA-projection-calibration-exact"
            report["phase"] = "official_to_glb"
            write_json(report_path, report)

        def capture_projection(receipt):
            memory["calls"].append(receipt)
            report["last_trustworthy_phase"] = "all-reconstructed-vertices-projected"
            write_json(report_path, report)

        remesher = defer_remesh_projection(native_remesher, batch_vertices,
                                            capture_projection, verify_projection)
    observed_remesher = observe_remesher(remesher, resolution_override, capture_remesher)
    report["effective_route"]["release_unused_torch_cache_before_simplify"] = args.release_torch_cache_before_simplify
    cumesh.CuMesh = observe_mesh_class(native_class, capture,
        memory_boundary(torch, report, report_path, args.release_torch_cache_before_simplify))
    cumesh.remeshing.remesh_narrow_band_dc = observed_remesher
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        mesh = module.to_glb(
            vertices=torch.as_tensor(arrays["vertices"], dtype=torch.float32, device="cuda"),
            faces=torch.as_tensor(arrays["faces"], dtype=torch.int32, device="cuda"),
            attr_volume=torch.as_tensor(arrays["attrs"], dtype=torch.float32, device="cuda"),
            coords=torch.as_tensor(arrays["coords"], dtype=torch.int32, device="cuda"),
            attr_layout={"base_color": slice(0, 3), "metallic": slice(3, 4),
                         "roughness": slice(4, 5), "alpha": slice(5, 6)}, **settings)
    finally:
        cumesh.CuMesh = native_class
        cumesh.remeshing.remesh_narrow_band_dc = native_remesher
        report["finishing_seconds_including_stage_capture"] = time.perf_counter() - started
        report["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        write_json(report_path, report)
    torch.cuda.synchronize()
    report["finishing_seconds_including_stage_capture"] = time.perf_counter() - started
    report["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["phase"] = "glb_export"
    path = directory / "warrior.glb"
    mesh.export(path, extension_webp=True)
    report["glb"] = validate_glb(path, args.texture_size)
    report["glb"]["path"] = str(path)
    labels = [s["label"] for s in report["stages"]]
    expected = (["init", "fill_holes", "init", "simplify", "uv_unwrap", "compute_vertex_normals"]
                if args.arm == "remesh" else
                ["init", "fill_holes", "simplify", "remove_duplicate_faces",
                 "repair_non_manifold_edges", "remove_small_connected_components", "fill_holes",
                 "simplify", "remove_duplicate_faces", "repair_non_manifold_edges",
                 "remove_small_connected_components", "fill_holes", "unify_face_orientations",
                 "uv_unwrap", "compute_vertex_normals"])
    if labels != expected:
        raise ValueError(f"effective stage order mismatch: {labels}")
    remesh_calls = report["effective_route"]["reconstruction"]["calls"]
    if args.arm == "remesh":
        expected_resolution = resolution_override if resolution_override is not None else args.grid_size
        if len(remesh_calls) != 1 or remesh_calls[0]["effective_resolution"] != expected_resolution:
            raise ValueError("effective reconstruction resolution mismatch or missing native call")
        if checkpoint is not None:
            if report["effective_route"]["checkpoint_continuation"].get("original_surface_exact") is not True:
                raise ValueError("missing original surface checkpoint conformance")
        if batch_vertices is not None and args.remesh_project > 0:
            memory = report["effective_route"]["projection_memory"]
            calls = memory["calls"]
            reconstructed = report["stages"][2]["arrays"]["vertices"]["shape"][0]
            if (memory.get("conformance", {}).get("exact_equal") is not True or
                    len(calls) != 1 or calls[0]["processed_vertices"] != reconstructed or
                    calls[0]["vertices"] != reconstructed):
                raise ValueError("missing projection equivalence or incomplete reconstructed output")
    elif remesh_calls:
        raise ValueError("unexpected reconstruction call in non-remesh arm")
    report["last_trustworthy_phase"] = "glb-and-stage-order-validated"


def child_command(args, arm):
    options = []
    for flag, field in (("--remesh-resolution", "remesh_resolution"),
                        ("--projection-batch-vertices", "projection_batch_vertices"),
                        ("--resume-rebuilt", "resume_rebuilt"), ("--resume-filled", "resume_filled"),
                        ("--resume-report", "resume_report"), ("--resume-report-sha", "resume_report_sha")):
        value = getattr(args, field, None)
        if arm == "remesh" and value is not None:
            options += [flag, str(value)]
    if getattr(args, "release_torch_cache_before_simplify", False):
        options += ["--release-torch-cache-before-simplify"]
    return [sys.executable, "-u", str(Path(__file__).resolve()),
            "--mesh", str(args.mesh.resolve()), "--appearance", str(args.appearance.resolve()),
            "--mesh-sha", args.mesh_sha, "--appearance-sha", args.appearance_sha,
            "--grid-size", str(args.grid_size), "--target-faces", str(args.target_faces),
            "--texture-size", str(args.texture_size), "--runtime", str(args.runtime.resolve()),
            "--output-dir", str(args.output_dir.resolve()),
            "--remesh-project", str(args.remesh_project), *options, "--arm", arm]


def run_both(args, report):
    report["phase"] = "input_admission"
    arrays = load_inputs(args.mesh, args.appearance, args.mesh_sha, args.appearance_sha, args.grid_size)
    report["inputs"] = {key: {"shape": list(array.shape), "dtype": str(array.dtype)}
                        for key, array in arrays.items()}
    del arrays
    args.output_dir.mkdir(parents=True, exist_ok=False)
    bootstrap(args, report)
    report["phase"] = "finishing_arms"
    report["arms"] = {}
    for arm in args.arms:
        argv = child_command(args, arm)
        log_path = args.output_dir / f"{arm}.log"
        with log_path.open("w") as stream:
            result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT)
        path = args.output_dir / f"{arm}-report.json"
        child = json.loads(path.read_text()) if path.is_file() else {"status": "missing-report"}
        report["arms"][arm] = {"exit_code": result.returncode, "report": child}
        write_json(args.output_json, report)
    report["phase"] = "artifact_bundle"
    bundle = args.output_dir.parent / "finishing-bundle.tar"
    with tarfile.open(bundle, "x") as archive:
        archive.add(args.output_dir, arcname="finishing")
    report["bundle"] = {"path": str(bundle), "sha256": digest(bundle), "size_bytes": bundle.stat().st_size}
    if any(a["exit_code"] != 0 or a["report"].get("status") != "completed" for a in report["arms"].values()):
        raise RuntimeError("one or more finishing arms failed; partial evidence retained")
    report["last_trustworthy_phase"] = "selected-official-finishing-arms-exported"


def check_replay_gate(child, reference, reference_sha):
    route = child.get("effective_route", {})
    if (child.get("status") != "completed" or route.get("device") != "Tesla T4" or
            route.get("capability") != [7, 5] or
            route.get("finishing") != "official-to_glb-with-saved-reconstruction-continuation" or
            route.get("checkpoint_continuation", {}).get("original_surface_exact") is not True):
        raise ValueError("768 continuation replay route failed or unverified")
    if [s["label"] for s in child["stages"]] != ["init", "fill_holes", "init", "simplify", "uv_unwrap", "compute_vertex_normals"]:
        raise ValueError("768 continuation replay stage order mismatch")
    if child.get("glb", {}).get("image_sizes") != [[4096, 4096], [4096, 4096]]:
        raise ValueError("768 continuation replay missing full 4K output")
    stage = child["stages"][3]
    if digest(reference) != reference_sha or digest(stage["path"]) != stage["sha256"]:
        raise ValueError("768 continuation replay reduced geometry digest mismatch")
    with np.load(reference, allow_pickle=False) as expected, np.load(stage["path"], allow_pickle=False) as actual:
        if not all(np.array_equal(expected[key], actual[key]) for key in ("vertices", "faces")):
            raise ValueError("768 continuation reduced arrays differ from saved native control; target deferred")
    return {"reduced_arrays_exact": True, "reference_sha256": reference_sha,
            "replayed_reduction_sha256": stage["sha256"],
            "claim_ceiling": "same reduced geometry and original finishing route; not bitwise UV/bake parity or visual closure"}


def run_checkpoint_suite(args, report):
    """Reuse one native build; gate the large continuation on a known replay."""
    report["phase"] = "checkpoint_suite_input_admission"
    raw = load_inputs(args.mesh, args.appearance, args.mesh_sha, args.appearance_sha, args.grid_size)
    del raw
    control = copy.copy(args)
    control.arm, control.remesh_resolution = "remesh", None
    control.resume_rebuilt, control.resume_filled = args.control_rebuilt, args.control_filled
    control.resume_report, control.resume_report_sha = args.control_report, args.control_report_sha
    target = copy.copy(args)
    target.arm = "remesh"
    for selected in (control, target):
        admitted = load_reconstruction_checkpoint(selected)
        del admitted
    origin = json.loads(control.resume_report.read_text())
    if origin.get("status") != "completed" or origin["stages"][3]["label"] != "simplify":
        raise ValueError("768 replay source lacks a completed native reduction")
    reference_sha = origin["stages"][3]["sha256"]
    if digest(args.control_reduced) != reference_sha:
        raise ValueError("768 replay reference digest mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    bootstrap(args, report)
    report["continuations"] = {}
    try:
        for name, selected in (("control-768", control), ("target-1024", target)):
            report["phase"] = name
            selected.output_dir = args.output_dir / name
            selected.output_dir.mkdir(exist_ok=False)
            argv = child_command(selected, "remesh")
            report["continuations"][name] = {"command": argv, "status": "running"}
            write_json(args.output_json, report)
            with (selected.output_dir / "remesh.log").open("w") as stream:
                result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT)
            path = selected.output_dir / "remesh-report.json"
            child = json.loads(path.read_text()) if path.is_file() else {"status": "missing-report"}
            report["continuations"][name].update(exit_code=result.returncode, report=child, status=child["status"])
            write_json(args.output_json, report)
            if result.returncode or child["status"] != "completed":
                raise RuntimeError(f"{name} failed; stage and memory evidence retained")
            if (child["script_sha256"] != digest(__file__) or
                    child["effective_route"]["postprocess_sha256"] != POSTPROCESS_SHA or
                    child["requested"]["resume_report_sha"] != selected.resume_report_sha):
                raise ValueError("continuation producer identity mismatch")
            if name == "control-768":
                report["phase"] = "control-768-replay-conformance"
                report["replay_gate"] = check_replay_gate(child, args.control_reduced, reference_sha)
                write_json(args.output_json, report)
    finally:
        bundle = args.output_dir.parent / "finishing-bundle.tar"
        with tarfile.open(bundle, "x") as archive:
            archive.add(args.output_dir, arcname="finishing")
        report["bundle"] = {"path": str(bundle), "sha256": digest(bundle), "size_bytes": bundle.stat().st_size}
        write_json(args.output_json, report)
    report["last_trustworthy_phase"] = "768-replay-exact-and-1024-checkpoint-finished"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--appearance", type=Path, required=True)
    p.add_argument("--mesh-sha", required=True)
    p.add_argument("--appearance-sha", required=True)
    p.add_argument("--grid-size", type=int, required=True)
    p.add_argument("--target-faces", type=int, default=1000000)
    p.add_argument("--texture-size", type=int, default=4096)
    p.add_argument("--remesh-project", type=float, default=0)
    p.add_argument("--remesh-resolution", type=int,
                   help="Override reconstruction resolution only; appearance keeps --grid-size")
    p.add_argument("--projection-batch-vertices", type=int,
                   help="Memory-only deferred projection, all vertices retained; remesh arm only")
    p.add_argument("--resume-rebuilt", type=Path)
    p.add_argument("--resume-filled", type=Path)
    p.add_argument("--resume-report", type=Path)
    p.add_argument("--resume-report-sha")
    p.add_argument("--release-torch-cache-before-simplify", action="store_true")
    p.add_argument("--checkpoint-suite", action="store_true")
    p.add_argument("--control-rebuilt", type=Path)
    p.add_argument("--control-filled", type=Path)
    p.add_argument("--control-report", type=Path)
    p.add_argument("--control-report-sha")
    p.add_argument("--control-reduced", type=Path)
    p.add_argument("--arms", nargs="+", choices=("non-remesh", "remesh"),
                   default=["non-remesh", "remesh"])
    p.add_argument("--runtime", type=Path, default=Path("/kaggle/working/cuda-finishing-runtime"))
    p.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/finishing"))
    p.add_argument("--output-json", type=Path, default=Path("/kaggle/working/finishing-report.json"))
    p.add_argument("--arm", choices=("non-remesh", "remesh"))
    return p


def main():
    args = parser().parse_args()
    # Match the current CUDA device; do not change a caller's explicit setting.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
    report = {"schema": "trellis2mlx.original-cuda-finishing.v1", "status": "running",
              "phase": "start", "last_trustworthy_phase": None,
              "requested": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "script_sha256": digest(__file__), "source_pins": SOURCES}
    path = args.output_dir / f"{args.arm}-report.json" if args.arm else args.output_json
    operation = run_arm if args.arm else run_checkpoint_suite if args.checkpoint_suite else run_both
    return run_reported(path, report, lambda r: operation(args, r))


if __name__ == "__main__":
    raise SystemExit(main())
