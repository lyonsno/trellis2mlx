"""SLatFlowModel — DiT over sparse tokens at occupied voxel positions.

Architecturally identical to SparseStructureFlowModel (same blocks, same
attention, same adaLN-Zero), but operates on variable-length sparse token
sequences instead of a dense 3D grid.

Input: SparseTensor with features [N, in_channels] at occupied coordinates
Output: SparseTensor with features [N, out_channels]

For the 512 and 1024 checkpoint variants, the model has:
- 30 blocks, 1536 model channels, 12 heads, 128 head dim
- 32 input/output channels (shape latent dimensionality)
- Same MLP hidden dim (8192)
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .sparse_structure_flow import (
    TimestepEmbedder,
    MultiHeadAttention,
    FeedForward,
    ModulatedBlock,
    _infer_compute_dtype,
    _cast_block_linears,
    _source_shared_modulation,
    _injections_for_block,
    _layernorm_noaffine,
)
from ..shape_flow_layernorm import (
    CUDA_WELFORD_METAL_BACKEND,
    CUDA_WELFORD_TURING_T4_BACKEND,
    get_shape_flow_layernorm_backend,
    layernorm_noaffine as _shape_flow_layernorm_noaffine,
    layernorm_noaffine_float32_output as _shape_flow_terminal_layernorm,
)
from ..modules.norm import LayerNorm32
from ..modules.attention import MultiHeadRMSNorm
from ..modules.rope import build_sparse_rope_phases


_SOURCE_CUDA_TERMINAL_LINEAR_PARTITIONS = (0, 308, 616, 924, 1232, 1536)
_SOURCE_CUDA_T4_TERMINAL_LINEAR_ROWS = (6022, 6038)
_source_cuda_terminal_linear_kernel = None


def shape_flow_terminal_linear_backend_identity(
    row_count: int,
    *,
    input_width: int,
    output_width: int,
    has_bias: bool,
    source_cuda_terminal: bool,
) -> dict:
    """Describe the effective terminal projection for the complete dispatch contract."""
    geometry = {
        "rows": row_count,
        "input_width": input_width,
        "output_width": output_width,
        "has_bias": has_bias,
    }
    if (
        source_cuda_terminal
        and row_count in _SOURCE_CUDA_T4_TERMINAL_LINEAR_ROWS
        and input_width == 1536
        and output_width == 32
        and has_bias
    ):
        return {
            "backend": "source-cuda-t4-volta-sgemm-32x128-tn-metal",
            "algorithm": "five-contiguous-fp32-fma-partitions-with-bias-epilogue",
            "experimental": True,
            "cuda_source_kernel": "volta_sgemm_32x128_tn",
            "authenticated_contract": {
                "rows": list(_SOURCE_CUDA_T4_TERMINAL_LINEAR_ROWS),
                "input_width": 1536,
                "output_width": 32,
                "input_dtype": "float32",
                "weight_dtype": "float32",
                "bias_dtype": "float32",
                "output_dtype": "float32",
                "partition_bounds": list(
                    _SOURCE_CUDA_TERMINAL_LINEAR_PARTITIONS
                ),
                "cuda_device_anchor": "Tesla T4 sm_75",
                "torch_anchor": "2.10.0+cu128",
                "source_recurrence_sha256": [
                    "5dd57e90fad742e37a345d2e19bf484298577cd5d84336371c8793f587ca947f",
                    "ebde6bc1f271813801e44a312da8077d7c46cf5092f7dfee8b0100e48e3d874c",
                ],
                "cuda_prefix_ladder_sha256": (
                    "b21bad4d52e8202efdeec5a87af4fa9b52edaa7d513bd57fddf393a6f80dd6cc"
                ),
            },
        }
    if source_cuda_terminal:
        return {
            "backend": "numpy-fp32-blas",
            "algorithm": "numpy-matmul-then-bias",
            "experimental": True,
            "effective_contract": geometry,
            "excluded_row_geometry": list(_SOURCE_CUDA_T4_TERMINAL_LINEAR_ROWS),
        }
    return {
        "backend": "mlx-native-linear",
        "algorithm": "mlx-nn-linear",
        "experimental": True,
        "effective_contract": geometry,
    }


def _shape_shared_modulation(
    t: mx.array,
    t_embedder: TimestepEmbedder,
    adaLN_modulation: nn.Sequential,
    compute_dtype: mx.Dtype,
    shape_timestep_modulation_lut=None,
) -> mx.array:
    if shape_timestep_modulation_lut is not None:
        return shape_timestep_modulation_lut.lookup_mlx(t, compute_dtype)
    return _source_shared_modulation(
        t,
        t_embedder,
        adaLN_modulation,
        compute_dtype,
    )


def _source_cuda_t4_terminal_linear(
    x: mx.array,
    linear: nn.Linear,
) -> mx.array:
    """Execute the observed T4 SGEMM reduction law for terminal projection."""
    global _source_cuda_terminal_linear_kernel

    if x.ndim != 2 or x.shape[1] != 1536:
        raise ValueError("T4 terminal linear requires rank-2 width-1536 input")
    if linear.weight.ndim != 2 or linear.weight.shape != (32, 1536):
        raise ValueError("T4 terminal linear requires 32x1536 weight")
    if linear.bias is None or linear.bias.shape != (32,):
        raise ValueError("T4 terminal linear requires width-32 bias")

    if _source_cuda_terminal_linear_kernel is None:
        source = r"""
                constexpr uint reduction = 1536;
                constexpr uint columns = 32;
                constexpr uint partition_bounds[6] = {
                    0, 308, 616, 924, 1232, 1536
                };

                uint output_index = thread_position_in_grid.x;
                uint output_count = row_count[0] * columns;
                if (output_index >= output_count) {
                    return;
                }

                uint row = output_index / columns;
                uint column = output_index - row * columns;
                uint input_offset = row * reduction;
                uint weight_offset = column * reduction;
                float accumulator = bias[column];

                for (uint partition = 0; partition < 5; ++partition) {
                    float partial = 0.0f;
                    for (uint k = partition_bounds[partition];
                         k < partition_bounds[partition + 1];
                         ++k) {
                        partial = metal::fma(
                            inp[input_offset + k],
                            weight[weight_offset + k],
                            partial);
                    }
                    accumulator = accumulator + partial;
                }
                out[output_index] = accumulator;
            """
        _source_cuda_terminal_linear_kernel = mx.fast.metal_kernel(
            name="shape_flow_source_cuda_t4_terminal_linear_fp32",
            input_names=["inp", "weight", "bias", "row_count"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )

    rows = int(x.shape[0])
    output_count = rows * 32
    grid_size = ((output_count + 255) // 256) * 256
    return _source_cuda_terminal_linear_kernel(
        inputs=[
            x.astype(mx.float32),
            linear.weight.astype(mx.float32),
            linear.bias.astype(mx.float32),
            mx.array([rows], dtype=mx.uint32),
        ],
        grid=(grid_size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, 32)],
        output_dtypes=[mx.float32],
    )[0]


def _source_cuda_terminal_linear(
    x: mx.array,
    linear: nn.Linear,
) -> mx.array:
    """Dispatch only source-authenticated terminal projection schedules."""
    if (
        x.ndim == 2
        and x.shape[0] in _SOURCE_CUDA_T4_TERMINAL_LINEAR_ROWS
        and x.shape[1] == 1536
        and linear.weight.shape == (32, 1536)
        and linear.bias is not None
        and linear.bias.shape == (32,)
    ):
        return _source_cuda_t4_terminal_linear(x, linear)

    # Preserve the pre-existing diagnostic path for non-production fixtures.
    x_np = np.array(x, dtype=np.float32)
    weight_np = np.array(linear.weight, dtype=np.float32)
    output = x_np @ weight_np.T
    if linear.bias is not None:
        output = output + np.array(linear.bias, dtype=np.float32)
    return mx.array(output.astype(np.float32, copy=False))


class SLatFlowModel(nn.Module):
    """TRELLIS.2 Structured Latent Flow Model.

    Same architecture as SparseStructureFlowModel but takes sparse tokens
    at occupied voxel coordinates instead of a dense 3D grid.

    Config (from TRELLIS.2-4B, 512 variant):
        in_channels: 32
        out_channels: 32
        model_channels: 1536
        num_heads: 12
        num_blocks: 30
        mlp_hidden: 8192
        context_channels: 1024
        resolution: 512 (coordinate space, not grid resolution)
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        model_channels: int = 1536,
        num_heads: int = 12,
        num_blocks: int = 30,
        mlp_hidden: int = 8192,
        context_channels: int = 1024,
        shape_flow_layernorm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_heads = num_heads
        self.head_dim = model_channels // num_heads
        self.shape_flow_layernorm = shape_flow_layernorm

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(model_channels)

        # Input/output projections
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.out_layer = nn.Linear(model_channels, out_channels)

        # Shared adaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 6 * model_channels),
        )

        # Transformer blocks (same as SparseStructureFlowModel)
        self.blocks = [
            ModulatedBlock(
                model_channels,
                num_heads,
                context_channels,
                mlp_hidden,
                shape_flow_layernorm=shape_flow_layernorm,
            )
            for _ in range(num_blocks)
        ]
        for block in self.blocks:
            _cast_block_linears(block, mx.bfloat16)

        # Compilation state (call .compile() to enable)
        self._compiled = False
        self._run_blocks = self._run_blocks_impl

    @classmethod
    def for_shape(cls, **kwargs) -> "SLatFlowModel":
        return cls(shape_flow_layernorm=True, **kwargs)

    @classmethod
    def for_texture(cls, **kwargs) -> "SLatFlowModel":
        return cls(
            in_channels=64,
            out_channels=32,
            shape_flow_layernorm=False,
            **kwargs,
        )

    def build_cross_kv_cache(self, cond: mx.array) -> list[tuple]:
        """Precompute cross-attention K, V for all blocks.

        Call once per conditioning input. The cache is valid for all timesteps
        since image conditioning doesn't change between ODE steps.

        Returns list of (K, V) tuples, one per block.
        """
        cond = cond.astype(_infer_compute_dtype(self))
        cache = []
        for block in self.blocks:
            k, v = block.cross_attn.project_kv(cond)
            cache.append((k, v))
        mx.eval(*[t for pair in cache for t in pair])
        return cache

    def __call__(
        self,
        x: mx.array,           # [N, in_channels] sparse token features
        t: mx.array,           # [B] timestep
        cond: mx.array,        # [B, L, context_channels]
        coords: mx.array = None,  # [N, 3] voxel coordinates for RoPE
        concat_cond: mx.array = None,  # [N, C'] features to concatenate with x
        cross_kv_cache: list = None,  # precomputed cross-attention KV per block
        shape_block_injection=None,
        shape_block_injection_branch: str | None = None,
        shape_timestep_modulation_lut=None,
    ) -> mx.array:
        input_dtype = x.dtype
        N = x.shape[0]
        B = t.shape[0] if len(t.shape) else 1
        cond_B = cond.shape[0] if cond is not None and len(cond.shape) else B
        if B != 1 or cond_B != 1:
            raise ValueError(
                f"Only B=1 supported for SLatFlowModel inference, got t B={B}, cond B={cond_B}"
            )

        if concat_cond is not None:
            x = mx.concatenate([x, concat_cond], axis=-1)

        # Project to model channels
        x = self.input_layer(x)  # [N, C]
        compute_dtype = _infer_compute_dtype(self)
        if shape_timestep_modulation_lut is not None and not self.shape_flow_layernorm:
            raise ValueError(
                "source-CUDA timestep modulation LUT is only valid for shape flow"
            )
        mod = _shape_shared_modulation(
            t,
            self.t_embedder,
            self.adaLN_modulation,
            compute_dtype,
            shape_timestep_modulation_lut,
        )
        x = x.astype(compute_dtype)
        cond = cond.astype(compute_dtype)

        # Build RoPE phases from coordinates if provided
        rope_phases = None
        if coords is not None:
            rope_phases = self._coords_to_rope_phases(coords)

        # Run through blocks (B=1 assumed)
        # KV cache is incompatible with compiled path (fixed function signature)
        if self._compiled and cross_kv_cache is None and shape_block_injection is None:
            x = self._run_blocks(x, mod[0], cond, rope_phases)
        else:
            for i, block in enumerate(self.blocks):
                block_kv = cross_kv_cache[i] if cross_kv_cache is not None else None
                block_injection = None
                if shape_block_injection is not None:
                    block_injection = _injections_for_block(shape_block_injection, i)
                if block_injection is None:
                    x = block(x, mod[0], cond, rope_phases=rope_phases,
                              cross_kv_cache=block_kv)
                else:
                    x = block.forward_with_injection(
                        x,
                        mod[0],
                        cond,
                        injection=block_injection,
                        branch=shape_block_injection_branch or "pos",
                        rope_phases=rope_phases,
                        cross_kv_cache=block_kv,
                    )
                if (i + 1) % 6 == 0:
                    mx.eval(x)

        _, x = self._final_projection(x, input_dtype)
        return x

    def _final_projection(
        self,
        x: mx.array,
        input_dtype,
    ) -> tuple[mx.array, mx.array]:
        """Return the normalized final state and projected sampler output."""

        # Source casts the terminal BF16 hidden state to FP32 before LayerNorm.
        source_cuda_terminal = (
            self.shape_flow_layernorm
            and get_shape_flow_layernorm_backend()
            in {
                CUDA_WELFORD_METAL_BACKEND,
                CUDA_WELFORD_TURING_T4_BACKEND,
            }
        )
        if source_cuda_terminal:
            x = _shape_flow_terminal_layernorm(x, eps=1e-5)
        else:
            x = x.astype(input_dtype)
            layernorm = (
                _shape_flow_layernorm_noaffine
                if self.shape_flow_layernorm
                else _layernorm_noaffine
            )
            x = layernorm(x, eps=1e-5)
        if source_cuda_terminal:
            return x, _source_cuda_terminal_linear(x, self.out_layer)
        return x, self.out_layer(x)

    def terminal_linear_backend_identity(self, row_count: int) -> dict:
        source_cuda_terminal = (
            self.shape_flow_layernorm
            and get_shape_flow_layernorm_backend()
            in {
                CUDA_WELFORD_METAL_BACKEND,
                CUDA_WELFORD_TURING_T4_BACKEND,
            }
        )
        return shape_flow_terminal_linear_backend_identity(
            row_count,
            input_width=int(self.out_layer.weight.shape[1]),
            output_width=int(self.out_layer.weight.shape[0]),
            has_bias=self.out_layer.bias is not None,
            source_cuda_terminal=source_cuda_terminal,
        )

    def _run_blocks_impl(self, x, mod, cond, rope_phases):
        """Pure forward pass through all blocks (no mx.eval)."""
        for block in self.blocks:
            x = block(x, mod, cond, rope_phases=rope_phases)
        return x

    def trace_first_block(
        self,
        x: mx.array,
        t: mx.array,
        cond: mx.array,
        coords: mx.array = None,
        cross_kv_cache: list = None,
    ) -> dict[str, mx.array]:
        return self.trace_block(
            x,
            t,
            cond,
            coords=coords,
            block_index=0,
            cross_kv_cache=cross_kv_cache,
        )

    def trace_block(
        self,
        x: mx.array,
        t: mx.array,
        cond: mx.array,
        coords: mx.array = None,
        block_index: int = 0,
        cross_kv_cache: list = None,
        shape_block_injection=None,
        shape_block_injection_branch: str | None = None,
        shape_timestep_modulation_lut=None,
    ) -> dict[str, mx.array]:
        if block_index < 0 or block_index >= len(self.blocks):
            raise ValueError(f"block_index must be in [0, {len(self.blocks) - 1}], got {block_index}")

        input_dtype = x.dtype
        B = t.shape[0] if len(t.shape) else 1
        cond_B = cond.shape[0] if cond is not None and len(cond.shape) else B
        if B != 1 or cond_B != 1:
            raise ValueError(
                f"Only B=1 supported for SLatFlowModel trace, got t B={B}, cond B={cond_B}"
            )

        x = self.input_layer(x)
        compute_dtype = _infer_compute_dtype(self)
        if shape_timestep_modulation_lut is not None and not self.shape_flow_layernorm:
            raise ValueError(
                "source-CUDA timestep modulation LUT is only valid for shape flow"
            )
        mod = _shape_shared_modulation(
            t,
            self.t_embedder,
            self.adaLN_modulation,
            compute_dtype,
            shape_timestep_modulation_lut,
        )
        x = x.astype(compute_dtype)
        cond = cond.astype(compute_dtype)

        rope_phases = None
        if coords is not None:
            rope_phases = self._coords_to_rope_phases(coords)

        trace: dict[str, mx.array] = {"input_projected": x}
        for i, block in enumerate(self.blocks):
            block_kv = cross_kv_cache[i] if cross_kv_cache is not None else None
            if i == block_index:
                trace_prefix = f"block{block_index}"
                trace[f"{trace_prefix}_input"] = x
                block_injection = None
                if shape_block_injection is not None:
                    block_injection = _injections_for_block(shape_block_injection, i)
                x_after, block_trace = block.trace(
                    x,
                    mod[0],
                    cond,
                    rope_phases=rope_phases,
                    cross_kv_cache=block_kv,
                    trace_prefix=trace_prefix,
                    injection=block_injection,
                    branch=shape_block_injection_branch or "pos",
                )
                trace.update(block_trace)
                if i == len(self.blocks) - 1:
                    trace["final_input"] = x_after
                    final_norm, final_output = self._final_projection(
                        x_after,
                        input_dtype,
                    )
                    trace["final_norm"] = final_norm
                    trace["final_out_flat"] = final_output
                    trace["final_output"] = final_output
                break
            block_injection = None
            if shape_block_injection is not None:
                block_injection = _injections_for_block(shape_block_injection, i)
            if block_injection is None:
                x = block(
                    x,
                    mod[0],
                    cond,
                    rope_phases=rope_phases,
                    cross_kv_cache=block_kv,
                )
            else:
                x = block.forward_with_injection(
                    x,
                    mod[0],
                    cond,
                    injection=block_injection,
                    branch=shape_block_injection_branch or "pos",
                    rope_phases=rope_phases,
                    cross_kv_cache=block_kv,
                )
            if (i + 1) % 6 == 0:
                mx.eval(x)
        mx.eval(*trace.values())
        return trace

    def compile(self):
        """Enable mx.compile for the transformer block loop."""
        self._compiled = True
        self._run_blocks = mx.compile(self._run_blocks_impl)

    def uncompile(self):
        """Disable mx.compile, fall back to eager with periodic eval."""
        self._compiled = False
        self._run_blocks = self._run_blocks_impl

    def _coords_to_rope_phases(self, coords: mx.array) -> mx.array:
        """Compute RoPE phases from sparse voxel coordinates.

        Args:
            coords: [N, 3] integer coordinates (z, y, x)

        Returns:
            [N, head_dim//2, 2] cos/sin pairs
        """
        return build_sparse_rope_phases(coords, head_dim=self.head_dim)
