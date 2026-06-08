# Validation Evidence

This document records local validation evidence for `trellis2mlx`: commands, hardware, timings, artifact inspection, and witness mechanics.

## M2 Pro Tahoe Run

Full native-DINO shoe run on Apple Silicon:

| Field | Value |
|---|---|
| Machine | M2 Pro, 16 GB unified memory |
| OS | macOS 26.5.1 / Tahoe |
| Input | `assets/shoe_input.png` |
| Command | `PYTHONPATH=. python generate.py --image assets/shoe_input.png --output /tmp/trellis2mlx-tahoe-shoe-full-native.glb` |
| Output GLB | `/tmp/trellis2mlx-tahoe-shoe-full-native.glb` |
| SHA256 | `608f1c3487a02b3545c8d54b4f02fedaa7deb5dd736c0020129e1a86a1033882` |
| Wall time | `1265.04s` |
| Reported total stage time | `1264.4s` |
| Peak RSS | `6.75 GB` |
| Visual inspection | Coherent red shoe form with white upper/swoosh structure and expected single-image reconstruction fragments |
| Witness PNG | [`docs/witnesses/tahoe-shoe-full-native-witness.png`](witnesses/tahoe-shoe-full-native-witness.png) |
| Witness JSON | [`docs/witnesses/tahoe-shoe-full-native-witness.json`](witnesses/tahoe-shoe-full-native-witness.json) |

Recorded stage evidence:

| Stage | Time | Observed output |
|---|---:|---|
| Native DINOv3 | load recorded `412` arrays | features `(1, 1029, 1024)` |
| Sparse structure | `116.9s` | `2,977` sparse voxels |
| LR SLat | `80.0s` | `2,977` tokens |
| Upsample to HR coords | `15.6s` | `761,916` voxels, `12,043` HR tokens |
| HR SLat | `518.0s` | `12,043` tokens |
| Shape decode | `63.2s` | `3,040,506` voxels |
| Mesh extraction and simplify | `6.0s` | `6,016,550` raw faces to `199,999` faces |
| Texture SLat | `290.6s` | `12,043` tokens |
| Texture decode | `60.1s` | 6-channel PBR attributes |
| UV unwrap and texture bake | `97.9s` | unwrap, raster, voxel sample, seam inpaint |

Recorded GLB structure:

| Field | Value |
|---|---|
| Vertices | `264,350` |
| Faces | `199,999` |
| Visual type | `TextureVisuals` |
| Material type | `PBRMaterial` |
| Base color texture | Present |

## M4 Max Reference Run

Reference full pipeline run on Apple Silicon:

| Field | Value |
|---|---|
| Machine | M4 Max, 128 GB unified memory |
| Result | Full textured shoe pipeline |
| Wall time | Approximately `8.6 min` |
| Peak memory | Approximately `3 GB` for SLat flow, approximately `5 GB` during decode |

Recorded stage evidence:

| Stage | Time | Observed output |
|---|---:|---|
| Sparse structure, 12 steps | approximately `34s` | 1.29B parameter DiT on 16^3 grid |
| LR SLat, 1.7K tokens, 12 steps | approximately `14s` | low-resolution sparse latent |
| Upsample to HR coords | approximately `6s` | `463K` voxels |
| HR SLat, 7.2K tokens, 12 steps | approximately `2 min` | 1024 cascade model |
| Shape decode, 1.9M voxels | approximately `73s` | 474M parameter sparse UNet |
| Mesh extraction and simplify | approximately `3s` | `3.7M` raw faces to `200K` faces |
| Texture SLat, 7.2K tokens, 12 steps | approximately `1.3 min` | no CFG, single pass |
| Texture decode, 1.9M voxels | approximately `29s` | 6-channel PBR attributes |
| UV unwrap and texture bake | approximately `2.2 min` | xatlas and trilinear sample |

## Witness Renderer

`scripts/render_glb_witness.py` creates a deterministic PNG witness and JSON report from a GLB without running model inference:

```bash
python scripts/render_glb_witness.py \
  --input /tmp/trellis2mlx-tahoe-shoe-full-native.glb \
  --output /tmp/trellis2mlx-tahoe-shoe-full-native-witness.png \
  --report /tmp/trellis2mlx-tahoe-shoe-full-native-witness.json
```

The renderer writes three orthographic software-projection panels:

| Panel | Projection |
|---|---|
| `front_xz` | X/Z with Y depth sorting |
| `side_yz` | Y/Z with X depth sorting |
| `top_xy` | X/Y with Z depth sorting |

The JSON report records:

| Field | Meaning |
|---|---|
| `status` | `ok` or `error` |
| `route` | Effective witness route, currently `software_projected_mesh_witness` |
| `phase` | `complete` on success, or the failing phase on error |
| `input_glb`, `output_png`, `report_json` | Exact artifact paths used for the run |
| `mesh.vertices`, `mesh.faces`, `mesh.bounds_*`, `mesh.extents` | Structural mesh evidence loaded from the GLB |
| `witness.nonblank`, `witness.pixel_std`, `witness.panels` | Render sanity checks and panel identity |
| `witness.color_route` | Effective color source: texture UV centroid, vertex colors, face colors, material color, or default material fallback |
| `last_trustworthy_evidence` | Error-report evidence available before the failure point |

Failure behavior is part of the contract. Missing inputs, empty meshes, invalid meshes, near-blank renders, and unexpected exceptions produce a JSON report with `status: error` and a phase label; failed runs do not leave a PNG witness behind.

## Test Command

Witness renderer contracts:

```bash
uv run --with pytest python -m pytest tests/test_render_glb_witness.py -v
```

Full local test suite:

```bash
uv run --with pytest python -m pytest tests/ -v
```
