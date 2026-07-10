# Source-Parity Investigation Draft

Private working draft. This is not public copy and not a release claim.

## Claim Boundary

The current story is not "trellis2mlx is correct" and not "Trellis-Mac is
wrong." The actual claim under investigation is narrower:

TRELLIS.2 source parity on Apple Silicon has at least three backend routes that
must be named separately:

- Microsoft CUDA source behavior.
- Trellis-Mac / PyTorch MPS behavior.
- trellis2mlx / MLX behavior.

Trellis-Mac remains a valuable local comparator, but the port write-ups and the
local block2 witness show that PyTorch MPS can be route-specific at individual
ops. It must not silently impersonate CUDA source authority.

## External Port Clues

The relevant external write-up is the Lilting Channel TRELLIS.2 Apple Silicon
pair:

- https://lilting.ch/en/articles/trellis2-apple-silicon-mps-cuda-free
- https://lilting.ch/en/articles/trellis2-m1-max-hands-on

The articles document the CUDA dependency surface and the Mac substitutions:
FlashAttention to PyTorch SDPA, CUDA sparse convolution to Apple-compatible
paths, CUDA voxel/mesh/raster components to Python or Metal substitutes, and
hardcoded CUDA calls to routed device calls.

The current Trellis-Mac repository has moved beyond the first write-up through
Pedro Naugusto's Metal stack:

- https://github.com/shivampkumar/trellis-mac
- https://github.com/pedronaugusto/trellis2-apple

Important warnings for this investigation:

- A route can be visually useful while still numerically distinct from source.
- MPS fallback and Metal extension residency must be recorded, not assumed.
- A 512 route passing does not prove 1024 or cascade behavior.
- Tiny BF16 drift can become visible material/alpha behavior downstream.

## Current Block2 Witness

The current frozen witness isolates one SLat-flow block2 negative-branch
LayerNorm boundary:

- Input tensor: `neg_block2_after_cross`
- Modulation vectors: `neg_block2_shift_mlp`, `neg_block2_scale_mlp`
- Output tensor: `neg_block2_mlp_input`
- Shape: `[1, 7755, 1536]`
- Epsilon: `1e-6`
- Step: `0`
- Block: `2`

Reference trace:

`/Users/noahlyons/.local/state/gpu-greenroom/outputs/gribble-trellis-mac-shape-flow-block2-step0-shared-routeid-dadd1709-512-s42-steps8-20260709/shape_flow_block_trace.npz`

Candidate trace:

`/Users/noahlyons/.local/state/gpu-greenroom/outputs/gribble-mlx-shape-flow-block2-step0-refsupport-refcond-patched-source-modln-routeid-uncommitted-512-nocascade-s42-steps8-20260709/checkpoints/shape_flow_block_trace.npz`

Generated witness/census artifacts:

- MLX-side report: `/Users/noahlyons/.local/state/gpu-greenroom/outputs/gribble-block2-layernorm-census-20260709/layernorm-witness-report.json`
- Torch/MPS report: `/Users/noahlyons/.local/state/gpu-greenroom/outputs/gribble-block2-layernorm-census-20260709/layernorm-witness-report-torch-mps.json`
- Compact witness npz: `/Users/noahlyons/.local/state/gpu-greenroom/outputs/gribble-block2-layernorm-census-20260709/block2-neg-norm3-witness.npz`
- Tiny CUDA capsule: `/Users/noahlyons/.local/state/gpu-greenroom/outputs/gribble-block2-layernorm-census-20260709/run_cuda_layernorm_partial.py`

## Current Result

The input and modulation tensors match exactly between the Trellis-Mac reference
trace and trellis2mlx candidate trace:

- `after_cross_exact`: true
- `shift_mlp_exact`: true
- `scale_mlp_exact`: true

The output differs by exactly one BF16 quantum:

- Max absolute diff: `0.00390625`
- Nonzero entries: `7755`
- Differing channel count: `1`
- Differing channel: `540`
- Coverage: all `7755` tokens

The route split on the frozen witness:

- `torch_mps_layer_norm_bfloat16` exactly matches the Trellis-Mac reference.
- `torch_cpu_layer_norm_bfloat16` exactly matches the trellis2mlx/MLX candidate.
- `mlx_trellmlx_noaffine_bfloat16` exactly matches the trellis2mlx/MLX candidate.

This means the remaining block2 residual is not explained by a different input,
different modulation vector, or ordinary CPU-vs-MLX BF16 math. The live split is
PyTorch MPS BF16 LayerNorm versus PyTorch CPU BF16 / MLX BF16.

## PyTorch Source Inspection

Trellis-Mac's installed Torch build reports:

- Version: `2.12.1`
- Git commit: `7269437d655783a26cba32aa88195b741ff496aa`

The exact PyTorch source at that commit strongly supports treating the block2
LayerNorm split as an MPS route artifact unless runtime CUDA contradicts it.

CUDA forward LayerNorm:

- File: `aten/src/ATen/native/cuda/layer_norm_kernel.cu`
- BF16 dispatch uses `acc_type<scalar_t, true>`.
- The vectorized path checks `N % vec_size == 0`, with `vec_size = 4`; this
  witness has normalized width `1536`, so the width predicate is satisfied.
- The vectorized path computes row stats with `cuWelfordOnlineSum`, Welford
  partial combination, `sigma2 / float(N)`, then `rsqrt(sigma2 + eps)`.
- The output is computed in the accumulator type and stored back to BF16.
- The non-vectorized fallback also runs a rowwise moments kernel before the
  forward kernel.

CPU BF16 LayerNorm:

- Files: `aten/src/ATen/native/cpu/moments_utils.h` and
  `aten/src/ATen/native/cpu/layer_norm_kernel.cpp`
- Reduced-float CPU LayerNorm calls `RowwiseMoments`.
- `RowwiseMomentsImpl` is explicitly Welford plus cascade summation.
- BF16 vectors are converted to float for the math and converted back on store.

MPS BF16 LayerNorm:

- Files: `aten/src/ATen/native/mps/operations/Normalization.mm` and
  `aten/src/ATen/native/mps/kernels/LayerNorm.metal`
- The MPS operation dispatches custom Metal kernels, selecting the single-row
  kernel when `axis_size <= 1024 * 4` and the looped kernel otherwise.
- Both Metal kernels compute variance as `sum_sq / axis_size - mean * mean`.
- Both clamp `var < 1e-6` to zero before `metal::precise::rsqrt`.

This source split matches the local witness census: CPU and MLX agree, while
MPS is the odd route. The correct posture is therefore not to port MPS behavior
into trellis2mlx for source parity. The CUDA capsule is still valuable as a
runtime receipt, but the implementation direction is now constrained by exact
source: keep the MLX/CPU/Welford behavior unless CUDA runtime evidence
contradicts the source read.

## CUDA Budget Question

The remaining runtime-confirmation question is tiny:

Run the compact witness through PyTorch CUDA BF16 `F.layer_norm`, apply the same
shift/scale modulation, and compare against the two known outputs.

Expected interpretations:

- CUDA matches MPS: re-open the source read; this would be surprising and would
  require finding the missing route difference before changing trellis2mlx.
- CUDA matches CPU/MLX: stop chasing MPS parity at this op and record
  Trellis-Mac as route-specific here.
- CUDA is third behavior: target CUDA and record all three route behaviors.

No full CUDA TRELLIS generation is justified for this boundary yet.
