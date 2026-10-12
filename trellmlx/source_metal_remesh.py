"""Opt-in original remeshing branch, using Torch-facing Metal primitives.

Ported from Microsoft TRELLIS.2 5565d240, o-voxel/o_voxel/postprocess.py
(MIT). The ordinary MLX finalizer is unchanged. This is not CUDA arithmetic
parity or a pure-MLX implementation: host tensors are CPU tensors and the
explicit mesh/BVH/raster/sampling primitives dispatch Metal work.
"""
from __future__ import annotations

import importlib
import inspect
import math
import platform
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_backend():
    """Require the Metal route; never substitute CPU/CUDA fallback exporters."""
    if platform.system() != "Darwin":
        raise RuntimeError("original Metal finishing requires macOS")
    import torch
    import cumesh
    import cumesh.metal_backend as mesh_module
    import cumesh.metal_remeshing as remesh_module
    import mtlbvh
    import mtldiffrast.torch as dr
    from flex_gemm.ops.grid_sample import grid_sample_3d
    from flex_gemm.ops.grid_sample import grid_sample as sample_module

    if cumesh.CuMesh is not mesh_module.MtlMesh or cumesh.cuBVH is not mtlbvh.MtlBVH:
        raise RuntimeError("cumesh did not select the Metal mesh/BVH classes")
    if cumesh.remeshing.remesh_narrow_band_dc is not remesh_module.remesh_narrow_band_dc:
        raise RuntimeError("cumesh did not select Metal remeshing")
    if not sample_module.kernels.cuda.__name__.endswith(".metal") or not sample_module.kernels.triton.__name__.endswith(".metal"):
        raise RuntimeError("FlexGEMM did not select Metal sparse sampling")
    # The caller records these effective files and their digests before compute.
    names = {
        "cumesh", "cumesh._C", "cumesh.metal_backend", "cumesh.metal_remeshing",
        "mtlbvh", "mtldiffrast.torch", "flex_gemm.ops.grid_sample",
        "flex_gemm.kernels.metal", "cumesh._xatlas", "mtlbvh._C",
        "mtldiffrast._C", "flex_gemm.kernels.metal._C"}
    # Follow defining modules rather than hashing package wrappers alone.
    names.update(obj.__module__ for obj in (
        cumesh.CuMesh, cumesh.cuBVH, remesh_module.remesh_narrow_band_dc,
        dr.rasterize, dr.interpolate, grid_sample_3d, sample_module.GridSample3dFunction))
    modules = {name: importlib.import_module(name) for name in sorted(names)}
    identity = {name: inspect.getfile(module) for name, module in modules.items()}
    native_libraries = sorted({str(Path(path).resolve()) for path in identity.values()
                               if Path(path).suffix in {".so", ".dylib"}})
    # Source-side adjacent shader files are useful provenance, but no runtime
    # loader API authenticates the library object. Do not promote that claim.
    declared_metallibs = sorted(str(Path(identity[name]).resolve().parent / filename)
                               for name, filename in (
                                   ("cumesh._C", "cumesh.metallib"),
                                   ("mtlbvh._C", "mtlbvh.metallib"),
                                   ("mtldiffrast._C", "mtldiffrast.metallib"),
                                   ("flex_gemm.kernels.metal._C", "flex_gemm.metallib")))
    return SimpleNamespace(torch=torch, mesh=cumesh.CuMesh, bvh=cumesh.cuBVH,
                           remesh=remesh_module.remesh_narrow_band_dc,
                           dr=dr, sample=grid_sample_3d, identity=identity,
                           native_libraries=native_libraries,
                           declared_metallibs=declared_metallibs)


def verify_bvh_source_surface(vertices, faces, bvh):
    """Reject a BVH that misses vertices actually referenced by its own mesh."""
    import torch

    referenced = torch.zeros(len(vertices), dtype=torch.bool, device=vertices.device)
    referenced[faces.reshape(-1).long()] = True
    distances, _, _ = bvh.unsigned_distance(vertices[referenced])
    if len(distances) != int(referenced.sum()) or not torch.isfinite(distances).all():
        raise RuntimeError("source-surface BVH self-distance returned incomplete/nonfinite results")
    bad = distances > 1e-6
    if bad.any():
        raise RuntimeError(
            "source-surface BVH self-distance failed: "
            f"{int(bad.sum())}/{len(distances)} referenced vertices exceed 1e-6 "
            f"(maximum {float(distances.max()):.9g})")


def rebuild_surface(vertices, faces, *, backend, grid_size, target_faces,
                    band=1., project_back=.9, capture=None, phase=None, verbose=False):
    """Prefill→original BVH→DC/project→direct reduction; no cleanup loop."""
    if grid_size <= 0 or target_faces <= 0 or band <= 0:
        raise ValueError("grid, target and band must be positive")
    if not math.isfinite(band) or not math.isfinite(project_back) or not 0 <= project_back <= 1:
        raise ValueError("invalid band or project-back factor")
    mesh = backend.mesh()
    notify = phase or (lambda value: None)
    notify("mesh_init")
    mesh.init(vertices, faces)
    if capture:
        capture("raw", dict(zip(("vertices", "faces"), mesh.read())))
    notify("prefill")
    mesh.fill_holes(max_hole_perimeter=.03)
    original_v, original_f = mesh.read()
    if capture:
        capture("filled_original", {"vertices": original_v, "faces": original_f})
    notify("bvh")
    bvh = backend.bvh(original_v, original_f)
    notify("bvh_self_distance")
    verify_bvh_source_surface(original_v, original_f, bvh)
    # AABB is the model's fixed [-.5,.5]^3 domain, not the mesh's bounding box.
    center = vertices.new_zeros(3)
    notify("remesh")
    rebuilt = backend.remesh(
        original_v, original_f, center=center,
        scale=(grid_size + 3 * band) / grid_size,
        resolution=grid_size, band=band, project_back=project_back,
        verbose=verbose, bvh=bvh)
    mesh.init(*rebuilt)
    del rebuilt
    if capture:
        capture("rebuilt", dict(zip(("vertices", "faces"), mesh.read())))
    notify("simplify")
    mesh.simplify(target_faces, verbose=verbose)
    if capture:
        capture("reduced", dict(zip(("vertices", "faces"), mesh.read())))
    return mesh, original_v, original_f, bvh


def original_surface_positions(positions, vertices, faces, bvh):
    _, face_ids, barycentric = bvh.unsigned_distance(positions, return_uvw=True)
    return (vertices[faces[face_ids.long()]] * barycentric.unsqueeze(-1)).sum(dim=1)


def to_glb(vertices, faces, attrs, coords, *, backend, grid_size,
           target_faces=1000000, texture_size=4096, band=1., project_back=.9,
           capture=None, phase=None, verbose=False):
    """Complete original remesh treatment from saved mesh and appearance."""
    import cv2
    from PIL import Image
    import trimesh

    torch, dr = backend.torch, backend.dr
    if texture_size <= 0:
        raise ValueError("texture size must be positive")
    # Custom Metal modules use CPU host tensors backed by shared Metal buffers.
    vertices = vertices.to(device="cpu", dtype=torch.float32).contiguous()
    faces = faces.to(device="cpu", dtype=torch.int32).contiguous()
    coords = coords.to(device="cpu", dtype=torch.int32).contiguous()
    attrs = attrs.to(device="cpu", dtype=torch.float32).contiguous()
    notify = phase or (lambda value: None)
    notify("geometry")
    mesh, original_v, original_f, bvh = rebuild_surface(
        vertices, faces, backend=backend, grid_size=grid_size,
        target_faces=target_faces, band=band, project_back=project_back,
        capture=capture, phase=notify, verbose=verbose)
    notify("uv")
    out_v, out_f, uv, vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={"threshold_cone_half_angle_rad": np.radians(90.),
                               "refine_iterations": 0, "global_iterations": 1,
                               "smooth_strength": 1},
        return_vmaps=True, verbose=verbose)
    out_v, out_f, uv, vmaps = [x.cpu().contiguous() for x in (out_v, out_f, uv, vmaps)]
    if capture:
        capture("uv", {"vertices": out_v, "faces": out_f, "uvs": uv, "vmaps": vmaps})
    # Normals belong to the unexpanded mesh, then transfer through UV seams.
    mesh.compute_vertex_normals()
    source_normals = mesh.read_vertex_normals()
    normals = source_normals[vmaps.long()]
    if capture:
        capture("normals", {"normals": source_normals})
    notify("raster")
    ctx = dr.MtlRasterizeContext()
    uv_clip = torch.cat([uv * 2 - 1, torch.zeros_like(uv[:, :1]),
                         torch.ones_like(uv[:, :1])], dim=-1).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), dtype=torch.float32)
    # Original upstream chunking: every face participates, no face-count cap.
    for offset in range(0, len(out_f), 100000):
        chunk, _ = dr.rasterize(ctx, uv_clip, out_f[offset:offset + 100000],
                               resolution=[texture_size, texture_size])
        covered = chunk[..., 3:4] > 0
        chunk[..., 3:4] += offset
        rast = torch.where(covered, chunk, rast)
    mask = rast[0, ..., 3] > 0
    if not mask.any():
        raise RuntimeError("Metal UV rasterizer produced no covered texels")
    notify("original_surface_transfer")
    positions = dr.interpolate(out_v.unsqueeze(0), rast, out_f)[0][0][mask]
    positions = original_surface_positions(positions, original_v, original_f, bvh)
    notify("attribute_sample")
    texels = torch.zeros((texture_size, texture_size, attrs.shape[1]), dtype=torch.float32)
    texels[mask] = backend.sample(
        attrs, torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attrs.shape[1], grid_size, grid_size, grid_size]),
        grid=((positions + .5) * grid_size).reshape(1, -1, 3), mode="trilinear")
    if not torch.isfinite(texels).all():
        raise RuntimeError("nonfinite baked texture")
    notify("material")
    baked = np.clip(texels.numpy() * 255, 0, 255).astype(np.uint8)
    missing = (~mask.numpy()).astype(np.uint8)
    rgb = cv2.inpaint(baked[..., :3], missing, 3, cv2.INPAINT_TELEA)
    metallic, roughness, alpha = [
        cv2.inpaint(baked[..., channel], missing, 1, cv2.INPAINT_TELEA)[..., None]
        for channel in (3, 4, 5)]
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([rgb, alpha], axis=-1)),
        baseColorFactor=np.array([255] * 4, dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate(
            [np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1., roughnessFactor=1., alphaMode="OPAQUE", doubleSided=False)
    v_np, f_np, uv_np, n_np = [x.numpy().copy() for x in (out_v, out_f, uv, normals)]
    v_np[:, 1], v_np[:, 2] = v_np[:, 2].copy(), -v_np[:, 1].copy()
    n_np[:, 1], n_np[:, 2] = n_np[:, 2].copy(), -n_np[:, 1].copy()
    uv_np[:, 1] = 1 - uv_np[:, 1]
    return trimesh.Trimesh(vertices=v_np, faces=f_np, vertex_normals=n_np,
                          process=False,
                          visual=trimesh.visual.TextureVisuals(uv=uv_np, material=material))
