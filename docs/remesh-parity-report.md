# Mesh Postprocess Parity Report

**Date:** 2026-06-29
**Branch:** `cc/molten-remesh-parity-0629`
**Diaulos:** trellis-remesh-parity

## Summary

The official TRELLIS.2 export path uses `cumesh.remeshing.remesh_narrow_band_dc()` to rebuild mesh topology before UV/texture export. trellis2mlx had no equivalent — it ran cleanup (dedup, non-manifold repair, component removal, hole fill) + simplification directly on the raw extracted mesh. This leaves thousands of boundary loops (holes) and hundreds of non-manifold edges.

A CPU-based narrow-band SDF remesh candidate (igl signed_distance + marching cubes + surface projection) reduces holes by **99.75%** on the official control mesh.

## Topology Metrics: Official Control (T.png, res=512)

| Metric | Raw (3.25M faces) | Cleanup-first (220K) | Remesh candidate (264K) |
|--------|----:|----:|----:|
| Boundary edges | 236,877 | 100,983 | **654** |
| Boundary loops (holes) | 5,652 | 1,602 | **4** |
| Non-manifold edges | 300,026 | 347 | **0** |
| Connected components | 496 | 7,646 | 619 |
| Degenerate faces | 0 | 0 | 45 |

## Topology Metrics: Orb/Cube (res=512)

| Metric | Raw (3.28M faces) | Cleanup-first (320K) |
|--------|----:|----:|
| Boundary edges | 182,662 | 51,122 |
| Boundary loops (holes) | 4,562 | 2,193 |
| Non-manifold edges | 119,146 | 672 |
| Connected components | 342 | 761 |

## Topology Metrics: Orb/Cube (res=768)

| Metric | Raw (7.49M faces) | Cleanup-first (468K) |
|--------|----:|----:|
| Boundary edges | 364,113 | 53,248 |
| Boundary loops (holes) | 11,728 | 3,360 |
| Non-manifold edges | 208,724 | 2,178 |
| Connected components | 843 | 461 |

## Checkpoint Paths

- Official control (T.png, res=512):
  `kaminos-trellis-official-control-20260628T2241Z/trellis2-official-T-checkpoint-res512-face350k-tex4096/checkpoints`
- Orb/cube (res=512):
  `kaminos-trellis-parity-matrix-20260628T160744Z/full-cascade-checkpoint-face350k-tex4096/checkpoints`
- Orb/cube (res=768):
  `kaminos-trellis-768-probe-20260628T1909Z/full-cascade-checkpoint-res768-face500k-tex4096/checkpoints`

All under `/Users/noahlyons/.local/state/gpu-greenroom/outputs/`.

## Commands

```bash
# Raw + cleanup metrics (fast, ~30s)
PYTHONPATH=. .venv/bin/python scripts/mesh_parity_harness.py \
  --checkpoint /path/to/checkpoints \
  --label my-label --target-faces 350000 --skip-remesh

# With remesh candidate (~100s for 128^3 grid)
PYTHONPATH=. .venv/bin/python scripts/mesh_parity_harness.py \
  --checkpoint /path/to/checkpoints \
  --label my-label --target-faces 350000 \
  --remesh-resolution 128 --export-glb

# In generate.py resume path
PYTHONPATH=. .venv/bin/python generate.py \
  --resume /path/to/checkpoints --output /tmp/remesh.glb \
  --remesh --remesh-resolution 128 --target-faces 350000
```

## Output Artifacts

- `/tmp/parity-results/official-control-512-cleanup-first.glb` — cleanup baseline
- `/tmp/parity-results/official-control-512-remesh128-cleanup-first.glb` — cleanup baseline (same)
- `/tmp/parity-results/official-control-512-remesh128-remesh-candidate.glb` — remesh candidate
- `/tmp/parity-results/*.json` — JSON reports with full metrics

## Effective Route/Config

The harness reports the exact processing pipeline in each variant's `config` field:
- `raw`: no processing
- `cleanup-first`: `simplify-first → cleanup` (fast_simplification + mesh_cleanup.cleanup_mesh)
- `remesh-candidate`: `remesh-narrow-band → simplify → cleanup` (igl SDF + marching cubes + projection + fast_simplification + cleanup)

## Visual Status

GLB exports exist for A/B comparison. **Visual A/B through Kaminos or witness images has NOT been done** — topology metrics alone do not prove visual improvement. The remesh candidate could have lost geometric detail at 128^3 resolution. Visual inspection is required before claiming the issue is fixed.

## Implementation Notes

The CPU remesh path:
1. Pre-simplifies the raw mesh to ~200K faces for SDF queries
2. Computes signed distance field using `igl.signed_distance` with fast winding number (3.3s for 2M points on the pre-simplified mesh)
3. Runs marching cubes on the SDF to extract a watertight surface
4. Projects new vertices back toward the original surface (factor=0.9)
5. Then runs the normal simplify + cleanup pipeline

The official TRELLIS.2 path uses `cumesh.remeshing.remesh_narrow_band_dc()` which is GPU-accelerated CUDA dual contouring. Our CPU version is slower (~96s vs presumably <10s on GPU) but produces comparable topology improvements.

## Next Steps

1. **Visual A/B:** Load cleanup-first and remesh-candidate GLBs in Kaminos, compare hole coverage
2. **Resolution sweep:** Test remesh at 192 and 256 resolution for detail preservation
3. **Textured export:** Run full resume with `--remesh` to get textured GLB
4. **Performance:** The CPU SDF path (96s) is the bottleneck; Metal/MLX acceleration would bring this to ~seconds
