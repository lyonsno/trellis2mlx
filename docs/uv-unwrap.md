# UV Unwrap

The pipeline uses [xatlas](https://github.com/jpcy/xatlas) for UV unwrapping
with `max_iterations=0` to skip iterative chart boundary refinement. This
prevents pathological slowdowns on complex voxel topology while producing
equivalent UV quality.

## Why `max_iterations=0`

xatlas's chart segmentation has an iterative optimization step that refines
chart boundaries for better UV packing. On smooth organic geometry (shoes,
animals) this is fast. On voxel/boxy geometry with many right-angle corners,
the optimization explores an exponentially larger search space and goes
pathological:

| Input | Faces | `iter=1` (default) | `iter=0` | Speedup |
|-------|------:|-------------------:|---------:|--------:|
| Shoe (organic) | 200K | 19-56s | ~6s | 3-9x |
| Slate ball (voxel+smooth) | 189K | 343s | 19s | 18x |
| Crevice ball (dense voxel) | 185K | 90+ min (killed) | 6.5s | >800x |
| Moss ball (cobblestone) | 462K | 93 min | ~15s (est.) | ~370x |
| MLX logo cube (pure voxel) | 1M | never finished | - | - |

## Quality tradeoff

Measured on the crevice ball checkpoint (2.88M raw vertices, simplified to
50K and 200K faces). Assay compares `max_iterations=0` vs `max_iterations=1`
on the same mesh.

### Apples-to-apples at 50K faces

| Metric | `iter=0` | `iter=1` | Delta |
|--------|----------:|----------:|------:|
| Time | **1.9s** | **105.7s** | 56x slower |
| Pixel coverage | 56.65% | 57.24% | +0.59% |
| UV utilization | 56.59% | 57.24% | +0.65% |
| Inverted triangles | 30,582 | 31,144 | +562 more |
| Area distortion (std) | 1.50 | 1.48 | -0.02 |
| Chart count | 21,459 | 20,134 | -1,325 |

The iterative optimization buys 0.6% more pixel coverage at 56x the cost.
It actually produces *more* inverted triangles. The quality difference is
negligible for voxel-sampled textures with seam inpainting.

### Production path (200K faces, `iter=0`)

| Metric | Value |
|--------|------:|
| Time | 13.4s |
| Pixel coverage | 51.5% |
| Chart count | 37,533 |
| Inverted triangles | 88,189 |
| Vertex blowup | 2.31x |

## Alternative UV methods

The pipeline includes two experimental UV unwrap methods accessible via
`--uv-method`:

- **`cube`**: Cube projection — classifies faces by dominant normal axis and
  projects onto that plane. Fast (0.1s on 200K faces) but produces shattered
  textures on smooth geometry. Only suitable for pure axis-aligned voxel meshes.

- **`lscm`**: Normal-cone chart segmentation + LSCM parameterization (libigl)
  + xatlas packing. Produces good per-chart parameterization but has visible
  seam artifacts at chart boundaries due to vertex duplication. 39.7% pixel
  coverage vs xatlas's 60.9% on smooth test geometry.

Both remain available for experimentation. The default `auto` uses xatlas with
`max_iterations=0`.

## GPU texture baking

UV rasterization uses MLX on Metal (`rasterize_uv_mlx`) with adaptive
memory-budgeted chunking. The rasterizer computes barycentric coordinates
on GPU, then scatters results to CPU output buffers. Voxel attribute sampling
uses a vectorized numpy searchsorted hash table (`sample_voxel_attrs_fast`).

Select backend with `--texture-backend gpu` (default) or `--texture-backend cpu`.

## Reference comparison

The reference TRELLIS.2 pipeline uses:
- **GPU path** (postprocess.py): cumesh chart clustering + nvdiffrast/mtldiffrast UV rasterization
- **CPU path** (postprocess_cpu.py): xatlas `parametrize()` + PyTorch MPS rasterization

Our pipeline replaces cumesh with xatlas (`max_iterations=0`) and nvdiffrast
with MLX Metal rasterization. The coordinate transforms, PBR channel layout,
and inpainting are verified equivalent to the reference
([see review artifacts](https://github.com/lyonsno/epistaxis)).
