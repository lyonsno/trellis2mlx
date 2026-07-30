import mlx.core as mx
import mlx.utils
import numpy as np


def _tiny_decoder(*, use_fp16=True):
    from trellmlx.models.shape_slat_decoder import SLatDecoder

    return SLatDecoder(
        out_channels=7,
        latent_channels=2,
        model_channels=[8, 1],
        num_blocks=[0, 0],
        pred_subdiv=True,
        use_fp16=use_fp16,
    )


def test_shape_decoder_uses_source_fp16_torso_and_fp32_boundary_layers():
    decoder = _tiny_decoder()
    parameters = dict(mlx.utils.tree_flatten(decoder.parameters()))

    assert decoder.use_fp16 is True
    assert parameters["from_latent.weight"].dtype == mx.float32
    assert parameters["output_layer.weight"].dtype == mx.float32
    assert parameters["blocks.0.0.to_subdiv.weight"].dtype == mx.float16
    assert parameters["blocks.0.0.conv1.weight"].dtype == mx.float16


def test_shape_decoder_fp16_torso_returns_source_boundary_dtypes():
    decoder = _tiny_decoder()
    decoder.blocks[0][0].to_subdiv.weight = mx.zeros(
        decoder.blocks[0][0].to_subdiv.weight.shape,
        dtype=mx.float16,
    )
    decoder.blocks[0][0].to_subdiv.bias = mx.ones(
        decoder.blocks[0][0].to_subdiv.bias.shape,
        dtype=mx.float16,
    )

    output, coords, subdivisions = decoder(
        mx.ones((1, 2), dtype=mx.float32),
        mx.array([[0, 1, 1, 1]], dtype=mx.int32),
        return_subs=True,
    )
    mx.eval(output, coords, *subdivisions)

    assert output.dtype == mx.float32
    assert coords.dtype == mx.int32
    assert len(subdivisions) == 1
    assert subdivisions[0].dtype == mx.float16


def test_sparse_conv_preserves_fp16_torso_dtype():
    from trellmlx.modules.sparse_conv import SparseConv3d

    convolution = SparseConv3d(2, 3, kernel_size=1)
    convolution.set_dtype(mx.float16)
    neighbor_map = (
        mx.array([0], dtype=mx.int32),
        mx.array([0], dtype=mx.int32),
        mx.array([0], dtype=mx.int32),
    )

    output = convolution(
        mx.ones((1, 2), dtype=mx.float16),
        neighbor_map,
    )
    mx.eval(output)

    assert output.dtype == mx.float16


def test_shape_decoder_final_layernorm_uses_source_default_epsilon():
    from trellmlx.models.shape_slat_decoder import _layernorm_noaffine

    values = mx.array(
        [[1.0, 1.0001, 0.9999, 1.0002]],
        dtype=mx.float32,
    )
    actual = _layernorm_noaffine(values)
    expected = mx.fast.layer_norm(values, None, None, 1e-5)
    wrong_epsilon = mx.fast.layer_norm(values, None, None, 1e-6)
    mx.eval(actual, expected, wrong_epsilon)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert not np.array_equal(np.asarray(actual), np.asarray(wrong_epsilon))
