# Remesh Parity Blocker: SDF Winding Number Fails on Non-Watertight Meshes

**Date:** 2026-06-29
**Branch:** `cc/molten-remesh-parity-0629`
**Diaulos:** trellis-remesh-parity

## Blocker Statement

CPU SDF-based remesh (igl signed_distance + marching cubes or dual contouring) cannot produce correct results on TRELLIS.2 decoder output because the winding number sign computation fails catastrophically on meshes with thousands of holes and non-manifold edges. The result is large false-interior blobs that engulf the actual object geometry.

This is a fundamental limitation of the SDF approach on non-watertight input, not a parameter tuning or implementation bug. The official TRELLIS.2 `cumesh.remeshing.remesh_narrow_band_dc` avoids this by using a GPU BVH with local normal-based sign computation that is robust to non-watertight input.

**Required fix:** Port `cumesh.remeshing.remesh_narrow_band_dc` (or equivalent local-sign dual contouring) to CPU/Metal/MLX.

## Evidence Chain

### 1. Topology metrics confirm the problem is structural

| Metric | Raw (3.25M faces) | Cleanup-first (220K) |
|--------|----:|----:|
| Boundary loops (holes) | 5,652 | 1,602 |
| Non-manifold edges | 300,026 | 347 |
| Boundary edges | 236,877 | 100,983 |

The raw mesh has 5,652 holes and 300K non-manifold edges. Cleanup fills 3,499 small holes but 1,565 are too large.

### 2. SDF remesh fixes topology but creates blobs

| Variant | Boundary loops | Non-manifold | Visual |
|---------|----:|----:|------|
| MC-128, project=0.9, pre-simplified SDF | 4 | 0 | Asymmetric blob on ~1/3 of object |
| MC-256, project=0.0, pre-simplified SDF | 349 | 12 | Same blob region |
| DC-256, project=0.0, pre-simplified SDF | 2,702 | 62 | Same blob region |
| DC-256, project=0.0, full-cleaned SDF | 2,080 | 34 | **Worse** — two large spherical blobs |

Same asymmetric blob appeared across MC and DC, confirming the extraction algorithm is not the cause. Full-cleaned SDF produced *larger* blobs than pre-simplified SDF.

### 3. SDF divergence diagnostic pinpointed the mechanism

Pre-simplified vs full-cleaned mesh SDF comparison at 128^3:
- **16.33% sign disagreement** near the surface (1 in 6 voxels)
- 72,515 false-exterior points (simplified says outside, full says inside)
- 7,810 false-interior points (the blob seed)
- Full-cleaned mesh has 590K "inside" points vs 525K pre-simplified

Both meshes produce wrong signs, just in different locations. The winding number is unreliable on *any* version of this mesh because the surface has thousands of holes that break the global inside/outside assumption.

### 4. Visual confirmation

Screenshot shows two large spherical blobs (lower-left and upper-right) engulfing the T-shape object. These are false-interior regions from winding number sign errors extracted as solid geometry by both MC and DC.

## What the Official Path Does Differently

`cumesh.remeshing.remesh_narrow_band_dc` (CUDA):
1. Builds a GPU BVH on the full mesh (after initial hole-fill)
2. Computes a narrow-band distance field using the BVH
3. Determines sign using **local face normals and BVH ray queries**, not global winding number
4. Runs dual contouring on the narrow-band SDF
5. Projects vertices back to original surface via BVH (with `project_back=0` in all official callers)

The critical difference is step 3: local normal-based sign computation is robust to non-watertight input because it only needs consistent normals in the local neighborhood, not a globally consistent inside/outside classification.

## What We Built (Useful Going Forward)

- **`trellmlx/topology_metrics.py`** — Topology metric harness (boundary edges/loops, non-manifold edges, connected components, degenerate faces). Sound, tested, reusable.
- **`trellmlx/dual_contouring.py`** — CPU dual contouring on regular SDF grid. Works correctly when given a correct SDF. Reusable if we get a correct sign computation.
- **`trellmlx/remesh_reconstruct.py`** — SDF computation pipeline (igl signed_distance, pre-cleaning). The SDF distance magnitudes are correct; only the signs are wrong.
- **`scripts/mesh_parity_harness.py`** — Comparative metrics harness with GLB export and Kaminos integration.
- **`scripts/sdf_divergence_diagnostic.py`** — SDF sign divergence analysis tool.
- **17 passing tests** covering topology metrics, false-closure paths, DC, and remesh.

## Next Patch: cumesh Port

### Scope

Port `cumesh.remeshing.remesh_narrow_band_dc` to run on Apple Silicon without CUDA. Two possible approaches:

1. **CPU port with Metal acceleration for BVH/distance queries** — Port the algorithm to numpy + Metal kernels for the heavy parts (BVH construction, ray queries, distance field computation). The DC extraction and QEF solve can stay on CPU.

2. **Pure CPU port** — Port everything to numpy/scipy. Slower but unblocked. BVH via scipy.spatial.cKDTree or trimesh's existing BVH. Sign via ray casting against local normals.

### Pointy Blockers to Discuss

1. **cumesh is closed-source CUDA.** We can read `o_voxel/postprocess.py` which calls it, but the internals of `cumesh.remeshing.remesh_narrow_band_dc`, `cumesh.cuBVH`, `cumesh.CuMesh` are compiled CUDA. We'd be reimplementing from the algorithm (Ju et al. 2002 + local sign heuristics), not porting line-by-line.

2. **BVH construction and queries.** The official path uses `cumesh.cuBVH` for unsigned distance + closest point + ray queries. We need an equivalent. Options: trimesh's BVH (Python, slow for 3M faces), scipy.spatial.cKDTree (fast for point queries but no ray casting), or a Metal BVH kernel.

3. **Local sign computation.** This is the core missing piece. Approaches:
   - Ray casting: shoot rays from each grid point, count intersections with mesh faces, odd = inside. Robust to non-watertight if rays are short/local.
   - Normal consistency: use the closest face's normal direction relative to the query point. Simple but fails at thin features.
   - Angle-weighted pseudonormal: igl has `SIGNED_DISTANCE_TYPE_PSEUDONORMAL` which uses vertex/edge/face normals. May work better than winding number on this mesh.
   - The official cumesh likely uses a combination of BVH ray queries + normal consistency.

4. **Performance.** CPU DC at 256^3 takes ~18s (acceptable). CPU SDF at 256^3 on 3M faces takes 154s (borderline). At 512^3 it would be ~20 minutes. Metal acceleration for the SDF computation would bring this to seconds.

### Recommended First Slice

Try `igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL` first — it uses local face/edge/vertex normals instead of winding number and may be robust enough on this mesh. If that works, the existing DC pipeline is immediately usable. If not, implement local ray-casting sign computation as the next step before a full cumesh port.

## Commands

```bash
# Run topology metrics
PYTHONPATH=. .venv/bin/python scripts/mesh_parity_harness.py \
  --checkpoint /path/to/checkpoints --label my-label --skip-remesh

# Run SDF divergence diagnostic
PYTHONPATH=. .venv/bin/python scripts/sdf_divergence_diagnostic.py \
  --checkpoint /path/to/checkpoints --resolution 128

# Run remesh with current (broken) SDF path
PYTHONPATH=. .venv/bin/python scripts/mesh_parity_harness.py \
  --checkpoint /path/to/checkpoints --remesh-resolution 256 --export-glb
```
