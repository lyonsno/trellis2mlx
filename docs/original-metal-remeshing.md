# Original remeshing branch on Metal (experimental)

This separate saved-checkpoint command reproduces the **stage order** of
TRELLIS.2's original remeshing exporter: prefill holes, retain the original
surface/BVH, rebuild with narrow-band dual contouring, project back, reduce
directly to the face target, unwrap, transfer normals through UV vertex maps,
and bake attributes sampled from the original surface. It does not run the
additional post-reconstruction cleanup used by the historical Mac exporter.

The ordinary MLX generation and finishing defaults are unchanged. This option
uses the existing **Torch-facing Metal** CuMesh, MtlBVH, mtldiffrast and
FlexGEMM modules; they are optional and are not installed by the core package.
CPU host tensors orchestrate custom Metal kernels. This is not a pure-MLX or
Swift remesher, and stage-order agreement does not establish numerical or visual
CUDA parity. A backend failure stops the command rather than selecting a
different exporter or dense sampling fallback.

Run with an interpreter containing those modules and put this checkout and
the selected compiled CuMesh package on `PYTHONPATH`:

```bash
python -u scripts/source_metal_remesh_finishing.py \
  --checkpoints /path/to/saved/checkpoints \
  --output-dir /path/to/new/finishing-attempt \
  --mesh-sha256 RAW_NPZ_SHA --texture-sha256 TEXTURE_NPZ_SHA \
  --backend-manifest /path/to/recorded-metal-runtime.json \
  --expected-revision CLEAN_CHECKOUT_SHA --grid-size 768 \
  --target-faces 1000000 --texture-size 4096 --band 1 --project-back 0.9
```

The checkpoint must contain `mesh_raw.npz` and `texture.npz` with decoded six
channel appearance. A recorded `mesh_grid_size` must match; older checkpoints
without that field use the explicitly declared grid, labeled as such in the
report. Recover that setting from the original run, not the mesh bounds. The output directory
must be new or empty (a completion supervisor may create its parent directory).
An atomic ownership marker prevents two producers from claiming the same empty
directory; any previous state is refused without overwrite. The explicit backend manifest maps every module returned by
`load_backend().identity` to its resolved path and SHA256 and lists selected
native `.so` and `.metallib` files under `binaries`, also with path and SHA256.
Use the same manifest for the small `scripts/source_metal_remesh_smoke.py`
composition witness and the natural input runs. Record its bytes with the
execution request. The runtime validates these identities before computation.
Imported native extensions are bound through their effective module paths;
shader-library hashes are labeled adjacent-file provenance, not authentication
of the runtime's loaded library object. For automatic sequential runs, add
`--require-composition-report /path/to/smoke/finishing/report.json` to gate a
natural-input command on the same source revision, backend manifest, complete
small-witness configuration/stages and still-matching output digest. Failed or
missing composition stops before backend dispatch and leaves a failure report.

Outputs include all six mesh/UV/normal stages, an opaque one-sided PBR GLB,
and a phase-bearing `report.json`, including on failure. A nonempty existing output
is preserved and refusal is recorded in a separate sibling failure report.
Reopen/texture checks are structural admission, not visual-quality acceptance;
inspect the GLB in the same viewer and camera as its comparison.

Algorithm source: Microsoft TRELLIS.2 `5565d240c4a494caaf9ece7a554542b76ffa36d3`,
`o-voxel/o_voxel/postprocess.py` (MIT). Metal primitives: Pedro Augusto's
mtlmesh and trellis2-apple. The source's 100,000-face raster chunks consume
every face; this is memory tiling, not a face-count cap.
