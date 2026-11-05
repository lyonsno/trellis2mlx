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

The runner now also builds the upstream regression's 498,002-face nonplanar
grid with the selected Metal BVH and checks 2,000 deterministic on-surface
queries **before** dispatching either the small composition exporter or a
natural-input exporter. The known stale 24-slot traversal shader fails this
preflight; the isolated current 64-slot shader passes. The phase-bearing report
records the effective backend identity and the maximum observed distance.
This is a targeted dependency admission check, not a general BVH certificate.
Immediately after building the natural input's original-surface BVH, the runner queries every
face-referenced vertex against that same BVH and refuses reconstruction if
any unsigned distance exceeds `1e-6`, is nonfinite, or the result is incomplete.
Unreferenced vertices are excluded because they need not lie on a triangle.
That second check is a guard against positive-distance misses at source vertices, not a
general BVH correctness check: this backend can also return a zero-distance
sentinel after failing to find a triangle, and the guard does not test
off-surface queries or projected coordinates. It is not a claim that the
rebuilt mesh is watertight or CUDA-equivalent. It adds an all-referenced-vertex
distance query before remeshing; cost depends on input size. The check was
motivated by an installed mtlbvh shader whose 24-entry traversal stack missed
more than 4.6 million referenced vertices on the saved 768 warrior while a current
64-entry build missed none. Keep the effective backend manifest and this
natural-input check together; an old manifest can faithfully attest an old,
broken shader.

Outputs include all six mesh/UV/normal stages, an opaque one-sided PBR GLB,
and a phase-bearing `report.json`, including on failure. A nonempty existing output
is preserved and refusal is recorded in a separate sibling failure report.
Reopen/texture checks are structural admission, not visual-quality acceptance;
inspect the GLB in the same viewer and camera as its comparison.

Algorithm source: Microsoft TRELLIS.2 `5565d240c4a494caaf9ece7a554542b76ffa36d3`,
`o-voxel/o_voxel/postprocess.py` (MIT). Metal primitives: Pedro Augusto's
mtlmesh and trellis2-apple. The source's 100,000-face raster chunks consume
every face; this is memory tiling, not a face-count cap.

## Cooperative distance-query checkpoints

The optional `--cooperative-checkpoint` mode lets a managed Greenroom job save
its progress, exit, and resume after queued Quick work or an operator pause.
Use a command-v2 Greenroom request with `cooperative_checkpoint: true`, make
Greenroom's Python SDK available to the same interpreter, and add:

```text
--cooperative-checkpoint
--checkpoint-root /durable/run/query-state
--query-chunk-rows 32768
--continuation-registration /durable/run/register-successor.json
```

The state directory must be separate from the finishing output. The registration
file contains `{"argv": ["command", "...{job_id}..."]}`. The runner replaces
`{job_id}` literally with the future Greenroom job ID and runs the argv without a
shell. That command must register completion delivery for that exact successor;
a nonzero return prevents its publication. Owner, source root, executable,
route, arguments, backend manifest, and registration identity stay bound through
the continuation. Existing unmanaged runs keep their original behavior.

The adapter partitions the original BVH's independent distance queries and
retains every result, including triangle IDs and requested barycentrics. It
freezes the original filled surface and verifies exact query identity before
reusing a complete or partial saved prefix. No query, face, or detail is
dropped. Saved buffers are protected from the remesher's in-place arithmetic.

Handoffs currently occur in the original-surface self-distance check and narrow
band remeshing queries. Opaque native operations, including BVH construction,
simplification, and UV work, still finish their current call. The checkpoint
interval therefore depends on those operations as well as query chunk size and
state-save time; the option does not impose a wall-clock deadline. Resuming
rebuilds the original BVH and replays the outer prefix using saved query results.
That adds restart overhead but preserves the remeshing implementation and its
completed expensive queries. Different replay inputs or corrupted state stop
the command rather than silently recomputing an incompatible result.

Before a handoff, native resources unwind out of the finisher and their Python
owners must be released. Greenroom then saves the descriptor, confirms the
producer's quiescence, registers the successor, and verifies process-group exit
before another job starts. A checkpointed report explicitly says that no final
output has completed. CPU tests cover the state and caller contracts; real
Metal/query agreement and actual queue passage require the corresponding live
witness. A directory of saved meshes alone does not establish pause support.
