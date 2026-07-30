import io
import json
import zipfile

import mlx.core as mx
import numpy as np
import pytest


TRACE_NAMES = (
    "input_feats",
    "from_latent_fp32",
    "torso_input",
    "block0_conv",
    "block0_norm",
    "block0_mlp_fc1",
    "block0_silu",
    "block0_mlp_fc2",
    "block0_output",
    "block1_output",
    "block2_output",
    "block3_output",
    "level0_subdiv_logits",
)


def _trace_arrays(
    *,
    rows=3,
    latent_channels=4,
    channels=8,
    torso_dtype=np.float16,
    offset=0.0,
):
    arrays = {
        "coords": np.arange(rows * 4, dtype=np.int32).reshape(rows, 4),
        "input_feats": np.full(
            (rows, latent_channels),
            0.25 + offset,
            dtype=np.float32,
        ),
        "from_latent_fp32": np.full(
            (rows, channels),
            0.5 + offset,
            dtype=np.float32,
        ),
    }
    for name in TRACE_NAMES[2:]:
        width = 8 if name == "level0_subdiv_logits" else channels
        if name in {"block0_mlp_fc1", "block0_silu"}:
            width = channels * 4
        arrays[name] = np.full(
            (rows, width),
            0.75 + offset,
            dtype=torso_dtype,
        )
    return arrays


def _write_report(path, *, primary, route, input_sha="a" * 64):
    import hashlib

    path.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.decoder_level0_trace_run.v1",
                "status": "passed",
                "effective_route": route,
                "input_slat_sha256": input_sha,
                "input_tensor_sha256": input_sha,
                "primary": {
                    "path": str(primary),
                    "sha256": hashlib.sha256(primary.read_bytes()).hexdigest(),
                },
            }
        )
    )


def _write_source_selective_report(
    path,
    *,
    primary,
    input_sha,
    route_overrides=None,
):
    import hashlib

    effective_route = {
        "route": "official-source-cuda-shape-decoder-level0-trace",
        "device_type": "cuda",
        "cuda_device": "Tesla T4",
        "sparse_conv_backend": "none",
        "decoder_state_only": False,
        "decoder_level0_trace": True,
        "raw_meshes": False,
        "post_fill_holes_snapshots": False,
        "mesh_conversion": False,
        "one_model_load": True,
    }
    effective_route.update(route_overrides or {})
    path.write_text(
        json.dumps(
            {
                "schema": "trellis2mlx.source_cuda_shape_slat_grid_decode.v1",
                "status": "done",
                "effective_route": effective_route,
                "decoder_trace_artifacts": [
                    {
                        "coordinate_key": "alpha-1_beta-1",
                        "path": str(primary),
                        "status": "written",
                        "sha256": hashlib.sha256(
                            primary.read_bytes()
                        ).hexdigest(),
                        "input_tensor_sha256": input_sha,
                    }
                ],
            }
        )
    )


def _write_trace_triplet(tmp_path):
    from scripts.decoder_level0_trace_contract import (
        decoder_trace_input_sha256,
        write_decoder_level0_trace_npz,
    )

    paths = {
        name: tmp_path / f"{name}.npz"
        for name in ("source", "local-fp16", "local-fp32")
    }
    reports = {
        name: tmp_path / f"{name}.json"
        for name in paths
    }
    arrays_by_name = {
        "source": _trace_arrays(),
        "local-fp16": _trace_arrays(),
        "local-fp32": _trace_arrays(torso_dtype=np.float32),
    }
    input_sha = decoder_trace_input_sha256(
        arrays_by_name["source"]["input_feats"],
        arrays_by_name["source"]["coords"],
    )
    for name, path in paths.items():
        torso_dtype = np.float32 if name == "local-fp32" else np.float16
        write_decoder_level0_trace_npz(
            path,
            arrays_by_name[name],
            latent_channels=4,
            channels=8,
            torso_dtype=torso_dtype,
        )
    for name in ("local-fp16", "local-fp32"):
        _write_report(
            reports[name],
            primary=paths[name],
            route=f"mlx-shape-decoder-level0-trace-{name.removeprefix('local-')}",
            input_sha=input_sha,
        )
    return paths, reports, input_sha


def test_mlx_level0_trace_is_exactly_the_natural_level0_forward():
    from trellmlx.decoder_level0_trace import capture_mlx_decoder_level0_trace
    from trellmlx.models.shape_slat_decoder import SLatDecoder
    from trellmlx.modules.sparse_conv import build_neighbor_map

    mx.random.seed(17)
    decoder = SLatDecoder(
        out_channels=7,
        latent_channels=4,
        model_channels=[8, 4],
        num_blocks=[4, 0],
        pred_subdiv=True,
        use_fp16=True,
    )
    feats = mx.arange(12, dtype=mx.float32).reshape(3, 4) / 11
    coords = mx.array(
        [[0, 1, 2, 3], [0, 1, 2, 4], [0, 2, 2, 3]],
        dtype=mx.int32,
    )

    arrays = capture_mlx_decoder_level0_trace(decoder, feats, coords)

    projected = decoder.from_latent(feats).astype(mx.float16)
    natural = projected
    neighbor_map = build_neighbor_map(coords)
    for block in decoder.blocks[0][:-1]:
        natural = block(natural, neighbor_map)
    logits = decoder.blocks[0][-1].to_subdiv(natural)
    mx.eval(natural, logits)

    np.testing.assert_array_equal(arrays["block3_output"], np.array(natural))
    np.testing.assert_array_equal(
        arrays["level0_subdiv_logits"],
        np.array(logits),
    )
    assert arrays["from_latent_fp32"].dtype == np.float32
    assert arrays["torso_input"].dtype == np.float16


def test_trace_contract_writes_and_reopens_every_boundary_exactly(tmp_path):
    from scripts.decoder_level0_trace_contract import (
        load_decoder_level0_trace,
        write_decoder_level0_trace_npz,
    )

    output = tmp_path / "trace.npz"
    arrays = _trace_arrays()

    summary = write_decoder_level0_trace_npz(
        output,
        arrays,
        latent_channels=4,
        channels=8,
        torso_dtype=np.float16,
    )
    reopened = load_decoder_level0_trace(
        output,
        latent_channels=4,
        channels=8,
        torso_dtype=np.float16,
    )

    assert summary["reopened_exact"] is True
    assert summary["rows"] == 3
    assert summary["trace_names"] == list(TRACE_NAMES)
    for name, expected in arrays.items():
        np.testing.assert_array_equal(reopened[name], expected)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda arrays: arrays.pop("block0_conv"),
            "missing required arrays: block0_conv",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "block0_norm",
                arrays["block0_norm"].astype(np.float32),
            ),
            "block0_norm must have dtype float16",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "block0_mlp_fc1",
                arrays["block0_mlp_fc1"][:, :-1],
            ),
            r"block0_mlp_fc1 must have shape \[3, 32\]",
        ),
        (
            lambda arrays: arrays["block2_output"].__setitem__((0, 0), np.nan),
            "block2_output contains non-finite values",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "coords",
                arrays["coords"].astype(np.int64),
            ),
            "coords must have dtype int32",
        ),
    ],
    ids=["partial", "wrong-dtype", "wrong-width", "non-finite", "coords-dtype"],
)
def test_trace_contract_rejects_false_evidence(tmp_path, mutation, message):
    from scripts.decoder_level0_trace_contract import (
        write_decoder_level0_trace_npz,
    )

    output = tmp_path / "trace.npz"
    arrays = _trace_arrays()
    mutation(arrays)

    with pytest.raises((KeyError, ValueError), match=message):
        write_decoder_level0_trace_npz(
            output,
            arrays,
            latent_channels=4,
            channels=8,
            torso_dtype=np.float16,
        )

    assert not output.exists()


def test_trace_contract_rejects_duplicate_npz_members(tmp_path):
    from scripts.decoder_level0_trace_contract import (
        load_decoder_level0_trace,
    )

    output = tmp_path / "duplicate-trace.npz"
    arrays = _trace_arrays()
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(output, "w") as archive:
            for name, values in arrays.items():
                payload = io.BytesIO()
                np.lib.format.write_array(payload, values, allow_pickle=False)
                archive.writestr(f"{name}.npy", payload.getvalue())
            duplicate = io.BytesIO()
            np.lib.format.write_array(
                duplicate,
                arrays["coords"],
                allow_pickle=False,
            )
            archive.writestr("coords.npy", duplicate.getvalue())

    with pytest.raises(ValueError, match="duplicate.*coords"):
        load_decoder_level0_trace(
            output,
            latent_channels=4,
            channels=8,
            torso_dtype=np.float16,
        )


def test_three_anchor_comparison_locates_first_fork_and_nearest_island(tmp_path):
    from scripts.compare_decoder_level0_traces import compare_level0_traces
    from scripts.decoder_level0_trace_contract import (
        write_decoder_level0_trace_npz,
    )

    source = tmp_path / "source.npz"
    local_fp16 = tmp_path / "local-fp16.npz"
    local_fp32 = tmp_path / "local-fp32.npz"
    source_report = tmp_path / "source.json"
    fp16_report = tmp_path / "local-fp16.json"
    fp32_report = tmp_path / "local-fp32.json"

    source_arrays = _trace_arrays(offset=0.0)
    fp16_arrays = _trace_arrays(offset=0.0)
    fp32_arrays = _trace_arrays(offset=0.0, torso_dtype=np.float32)
    from scripts.decoder_level0_trace_contract import (
        decoder_trace_input_sha256,
    )

    input_sha = decoder_trace_input_sha256(
        source_arrays["input_feats"],
        source_arrays["coords"],
    )
    fp16_arrays["block0_conv"] += np.float16(0.25)
    for name in TRACE_NAMES[4:]:
        fp16_arrays[name] += np.float16(0.5)
    fp32_arrays["block0_conv"] += np.float32(0.125)
    for name in TRACE_NAMES[4:]:
        fp32_arrays[name] += np.float32(0.25)

    write_decoder_level0_trace_npz(
        source,
        source_arrays,
        latent_channels=4,
        channels=8,
        torso_dtype=np.float16,
    )
    write_decoder_level0_trace_npz(
        local_fp16,
        fp16_arrays,
        latent_channels=4,
        channels=8,
        torso_dtype=np.float16,
    )
    write_decoder_level0_trace_npz(
        local_fp32,
        fp32_arrays,
        latent_channels=4,
        channels=8,
        torso_dtype=np.float32,
    )
    _write_report(
        source_report,
        primary=source,
        route="official-source-cuda-shape-decoder-level0-trace",
        input_sha=input_sha,
    )
    _write_report(
        fp16_report,
        primary=local_fp16,
        route="mlx-shape-decoder-level0-trace-fp16",
        input_sha=input_sha,
    )
    _write_report(
        fp32_report,
        primary=local_fp32,
        route="mlx-shape-decoder-level0-trace-fp32",
        input_sha=input_sha,
    )

    report = compare_level0_traces(
        source_path=source,
        source_report_path=source_report,
        local_fp16_path=local_fp16,
        local_fp16_report_path=fp16_report,
        local_fp32_path=local_fp32,
        local_fp32_report_path=fp32_report,
        latent_channels=4,
        channels=8,
    )

    assert report["first_numeric_fork"]["local_fp16"] == "block0_conv"
    assert report["first_numeric_fork"]["local_fp32"] == "block0_conv"
    assert report["stages"]["block0_conv"]["nearest_local_island"] == "local_fp32"
    assert report["stages"]["block0_conv"]["local_fp16"]["rms"] == pytest.approx(0.25)
    assert report["stages"]["block0_conv"]["local_fp32"]["rms"] == pytest.approx(0.125)


def test_three_anchor_comparison_rejects_common_forged_input_tensor_identity(
    tmp_path,
):
    from scripts.compare_decoder_level0_traces import compare_level0_traces
    from scripts.decoder_level0_trace_contract import (
        decoder_trace_input_sha256,
        write_decoder_level0_trace_npz,
    )

    paths = {
        name: tmp_path / f"{name}.npz"
        for name in ("source", "local-fp16", "local-fp32")
    }
    reports = {
        name: tmp_path / f"{name}.json"
        for name in paths
    }
    arrays_by_name = {
        "source": _trace_arrays(),
        "local-fp16": _trace_arrays(),
        "local-fp32": _trace_arrays(torso_dtype=np.float32),
    }
    actual_identity = decoder_trace_input_sha256(
        arrays_by_name["source"]["input_feats"],
        arrays_by_name["source"]["coords"],
    )
    forged_identity = "e" * 64
    assert forged_identity != actual_identity

    for name, path in paths.items():
        torso_dtype = np.float32 if name == "local-fp32" else np.float16
        write_decoder_level0_trace_npz(
            path,
            arrays_by_name[name],
            latent_channels=4,
            channels=8,
            torso_dtype=torso_dtype,
        )
        route = {
            "source": "official-source-cuda-shape-decoder-level0-trace",
            "local-fp16": "mlx-shape-decoder-level0-trace-fp16",
            "local-fp32": "mlx-shape-decoder-level0-trace-fp32",
        }[name]
        _write_report(
            reports[name],
            primary=path,
            route=route,
            input_sha=forged_identity,
        )

    with pytest.raises(ValueError, match="input tensor identity mismatch"):
        compare_level0_traces(
            source_path=paths["source"],
            source_report_path=reports["source"],
            local_fp16_path=paths["local-fp16"],
            local_fp16_report_path=reports["local-fp16"],
            local_fp32_path=paths["local-fp32"],
            local_fp32_report_path=reports["local-fp32"],
            latent_channels=4,
            channels=8,
        )


def test_three_anchor_comparison_accepts_real_source_selective_report_shape(
    tmp_path,
):
    from scripts.compare_decoder_level0_traces import compare_level0_traces

    paths, reports, input_sha = _write_trace_triplet(tmp_path)
    _write_source_selective_report(
        reports["source"],
        primary=paths["source"],
        input_sha=input_sha,
    )

    report = compare_level0_traces(
        source_path=paths["source"],
        source_report_path=reports["source"],
        local_fp16_path=paths["local-fp16"],
        local_fp16_report_path=reports["local-fp16"],
        local_fp32_path=paths["local-fp32"],
        local_fp32_report_path=reports["local-fp32"],
        latent_channels=4,
        channels=8,
    )

    source_route = report["artifacts"]["source"]["effective_route_details"]
    assert source_route["device_type"] == "cuda"
    assert source_route["sparse_conv_backend"] == "none"
    assert source_route["decoder_level0_trace"] is True
    assert source_route["raw_meshes"] is False
    assert source_route["mesh_conversion"] is False


@pytest.mark.parametrize(
    ("route_overrides", "field"),
    [
        ({"device_type": "cpu"}, "device_type"),
        ({"sparse_conv_backend": "spconv"}, "sparse_conv_backend"),
        ({"decoder_level0_trace": False}, "decoder_level0_trace"),
        ({"raw_meshes": True}, "raw_meshes"),
        ({"mesh_conversion": True}, "mesh_conversion"),
    ],
    ids=[
        "fallback-device",
        "wrong-conv-backend",
        "not-trace-mode",
        "raw-mesh-route",
        "mesh-conversion-route",
    ],
)
def test_three_anchor_comparison_rejects_source_selective_route_substitution(
    tmp_path,
    route_overrides,
    field,
):
    from scripts.compare_decoder_level0_traces import compare_level0_traces

    paths, reports, input_sha = _write_trace_triplet(tmp_path)
    _write_source_selective_report(
        reports["source"],
        primary=paths["source"],
        input_sha=input_sha,
        route_overrides=route_overrides,
    )

    with pytest.raises(
        ValueError,
        match=rf"source trace effective route field {field!r} mismatch",
    ):
        compare_level0_traces(
            source_path=paths["source"],
            source_report_path=reports["source"],
            local_fp16_path=paths["local-fp16"],
            local_fp16_report_path=reports["local-fp16"],
            local_fp32_path=paths["local-fp32"],
            local_fp32_report_path=reports["local-fp32"],
            latent_channels=4,
            channels=8,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report.__setitem__("status", "failed"),
            "source trace report status is not passed",
        ),
        (
            lambda report: report["primary"].__setitem__("sha256", "0" * 64),
            "source trace primary digest mismatch",
        ),
        (
            lambda report: report.__setitem__("input_tensor_sha256", "c" * 64),
            "trace input SLat identities do not match",
        ),
        (
            lambda report: report.__setitem__(
                "effective_route",
                "trellis-mac-mps-shape-decoder-level0-trace",
            ),
            "source trace effective route mismatch",
        ),
    ],
    ids=["failed-report", "stale-primary", "wrong-input", "fallback-route"],
)
def test_three_anchor_comparison_rejects_route_and_custody_substitution(
    tmp_path,
    mutation,
    message,
):
    from scripts.compare_decoder_level0_traces import compare_level0_traces
    from scripts.decoder_level0_trace_contract import (
        write_decoder_level0_trace_npz,
    )

    paths = {
        name: tmp_path / f"{name}.npz"
        for name in ("source", "local-fp16", "local-fp32")
    }
    reports = {
        name: tmp_path / f"{name}.json"
        for name in paths
    }
    input_sha = "d" * 64
    for name, path in paths.items():
        torso_dtype = np.float32 if name == "local-fp32" else np.float16
        write_decoder_level0_trace_npz(
            path,
            _trace_arrays(torso_dtype=torso_dtype),
            latent_channels=4,
            channels=8,
            torso_dtype=torso_dtype,
        )
        route = {
            "source": "official-source-cuda-shape-decoder-level0-trace",
            "local-fp16": "mlx-shape-decoder-level0-trace-fp16",
            "local-fp32": "mlx-shape-decoder-level0-trace-fp32",
        }[name]
        _write_report(
            reports[name],
            primary=path,
            route=route,
            input_sha=input_sha,
        )

    source_report = json.loads(reports["source"].read_text())
    mutation(source_report)
    reports["source"].write_text(json.dumps(source_report))

    with pytest.raises(ValueError, match=message):
        compare_level0_traces(
            source_path=paths["source"],
            source_report_path=reports["source"],
            local_fp16_path=paths["local-fp16"],
            local_fp16_report_path=reports["local-fp16"],
            local_fp32_path=paths["local-fp32"],
            local_fp32_report_path=reports["local-fp32"],
            latent_channels=4,
            channels=8,
        )
