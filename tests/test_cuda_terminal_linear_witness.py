import json
from argparse import Namespace
from contextlib import nullcontext
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def _arrays(*, source_sha="a" * 64, rows=2, inputs=4, outputs=3):
    route = {
        "schema": "trellis2mlx.shape_flow_terminal_linear_input.v1",
        "operation": "shape_flow.final_linear",
        "shape_flow_step_index": 6,
        "source_recurrence_sha256": source_sha,
        "mlx_exact_state_trace_sha256": "b" * 64,
        "model_checkpoint_sha256": "c" * 64,
        "logical_shapes": {
            "input": [rows, inputs],
            "weight": [outputs, inputs],
            "bias": [outputs],
            "output": [rows, outputs],
        },
    }
    return {
        "pos_final_norm": np.zeros((rows, inputs), dtype=np.float32),
        "neg_final_norm": np.zeros((rows, inputs), dtype=np.float32),
        "weight": np.zeros((outputs, inputs), dtype=np.float32),
        "bias": np.zeros(outputs, dtype=np.float32),
        "expected_pos": np.zeros((rows, outputs), dtype=np.float32),
        "expected_neg": np.zeros((rows, outputs), dtype=np.float32),
        "route_identity_json": np.asarray(json.dumps(route)),
    }


def test_terminal_linear_witness_rejects_substituted_source_identity():
    from scripts.cuda_terminal_linear_witness import validate_witness_arrays

    arrays = _arrays(source_sha="d" * 64)

    with pytest.raises(ValueError, match="source_recurrence_sha256"):
        validate_witness_arrays(
            arrays,
            expected_source_recurrence_sha256="a" * 64,
            expected_mlx_exact_state_trace_sha256="b" * 64,
            expected_model_checkpoint_sha256="c" * 64,
            expected_rows=2,
            input_channels=4,
            output_channels=3,
        )


def test_terminal_linear_witness_admits_exact_route_and_shapes():
    from scripts.cuda_terminal_linear_witness import validate_witness_arrays

    admitted = validate_witness_arrays(
        _arrays(),
        expected_source_recurrence_sha256="a" * 64,
        expected_mlx_exact_state_trace_sha256="b" * 64,
        expected_model_checkpoint_sha256="c" * 64,
        expected_rows=2,
        input_channels=4,
        output_channels=3,
    )

    assert admitted["arrays"]["weight"].shape == (3, 4)
    assert admitted["route_identity"]["shape_flow_step_index"] == 6


def test_terminal_linear_witness_rejects_nonfinite_payload():
    from scripts.cuda_terminal_linear_witness import validate_witness_arrays

    arrays = _arrays()
    arrays["pos_final_norm"][0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        validate_witness_arrays(
            arrays,
            expected_source_recurrence_sha256="a" * 64,
            expected_mlx_exact_state_trace_sha256="b" * 64,
            expected_model_checkpoint_sha256="c" * 64,
            expected_rows=2,
            input_channels=4,
            output_channels=3,
        )


def test_terminal_linear_prefix_ladder_is_uncapped():
    from scripts.cuda_terminal_linear_witness import _prefix_indices

    indices = _prefix_indices(1536)

    assert indices.start == 0
    assert indices.stop == 1537
    assert len(indices) == 1537


def test_prefix_ladder_emits_every_cumulative_linear_result():
    from scripts.cuda_terminal_linear_witness import _prefix_ladder

    class ArrayTensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        @property
        def shape(self):
            return self.value.shape

        def __getitem__(self, index):
            return ArrayTensor(self.value[index])

        def copy_(self, other):
            self.value[...] = other.value
            return self

        def index_select(self, axis, indices):
            return ArrayTensor(np.take(self.value, indices.value, axis=axis))

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    def linear(value, weight, bias):
        return ArrayTensor(value.value @ weight.value.T + bias.value)

    fake_torch = SimpleNamespace(
        zeros_like=lambda value: ArrayTensor(np.zeros_like(value.value)),
        no_grad=nullcontext,
        stack=lambda values: ArrayTensor(
            np.stack([value.value for value in values])
        ),
        nn=SimpleNamespace(functional=SimpleNamespace(linear=linear)),
    )
    pos = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    neg = -pos
    weight = np.asarray(
        [[0.5, -1.0, 2.0, 0.25], [-0.5, 0.75, 1.5, -2.0]],
        dtype=np.float32,
    )
    bias = np.asarray([0.125, -0.25], dtype=np.float32)

    prefix_pos, prefix_neg = _prefix_ladder(
        fake_torch,
        pos=ArrayTensor(pos),
        neg=ArrayTensor(neg),
        weight=ArrayTensor(weight),
        bias=ArrayTensor(bias),
        selected_rows=ArrayTensor(np.asarray([0], dtype=np.int64)),
    )

    expected_pos = []
    expected_neg = []
    for prefix in range(weight.shape[1] + 1):
        expected_pos.append(pos[:, :prefix] @ weight[:, :prefix].T + bias)
        expected_neg.append(neg[:, :prefix] @ weight[:, :prefix].T + bias)
    np.testing.assert_array_equal(prefix_pos, np.stack(expected_pos))
    np.testing.assert_array_equal(prefix_neg, np.stack(expected_neg))


def test_profiler_events_must_run_on_authenticated_cuda_device():
    from scripts.cuda_terminal_linear_witness import (
        _require_cuda_profiler_device,
    )

    with pytest.raises(ValueError, match="device mismatch"):
        _require_cuda_profiler_device(
            [
                {
                    "name": "sgemm",
                    "device_type": "DeviceType.CUDA",
                    "device_index": 1,
                }
            ],
            expected_device_index=0,
        )

    _require_cuda_profiler_device(
        [
            {
                "name": "sgemm",
                "device_type": "DeviceType.CUDA",
                "device_index": 0,
            }
        ],
        expected_device_index=0,
    )


def test_pre_cuda_failure_removes_stale_primary_before_preflight(
    tmp_path, monkeypatch
):
    import scripts.cuda_terminal_linear_witness as witness

    input_path = tmp_path / "input.npz"
    output_npz = tmp_path / "output.npz"
    output_json = tmp_path / "output.json"
    np.savez(input_path, placeholder=np.asarray(1, dtype=np.int32))
    np.savez(output_npz, stale=np.asarray(1, dtype=np.int32))
    original_sha256_file = witness.sha256_file
    primary_presence_at_preflight = []

    def observing_sha256_file(path):
        if path == input_path:
            primary_presence_at_preflight.append(output_npz.exists())
        return original_sha256_file(path)

    monkeypatch.setattr(witness, "sha256_file", observing_sha256_file)

    result = witness.run(
        Namespace(
            input=str(input_path),
            output_npz=str(output_npz),
            output_json=str(output_json),
            expected_input_sha256="0" * 64,
            expected_source_recurrence_sha256="a" * 64,
            expected_mlx_exact_state_trace_sha256="b" * 64,
            expected_model_checkpoint_sha256="c" * 64,
        )
    )

    report = json.loads(output_json.read_text())
    assert result == 1
    assert report["status"] == "failed"
    assert report["failure_phase"] == "preflight_input"
    assert report["primary_output_status"] == "not_written"
    assert report["last_trustworthy_phase"] == "invocation"
    assert primary_presence_at_preflight == [False]
    assert not output_npz.exists()


def test_run_forwards_requested_row_geometry_to_input_validation(
    tmp_path, monkeypatch
):
    import scripts.cuda_terminal_linear_witness as witness

    input_path = tmp_path / "input.npz"
    output_npz = tmp_path / "output.npz"
    output_json = tmp_path / "output.json"
    input_path.write_bytes(b"row-bound-input")
    observed = []

    monkeypatch.setattr(witness, "sha256_file", lambda path: "f" * 64)
    monkeypatch.setattr(witness, "_load_npz", lambda path: {})

    def observe_validation(*args, **kwargs):
        del args
        observed.append((kwargs["expected_rows"], kwargs["step_index"]))
        raise RuntimeError("stop after geometry validation")

    monkeypatch.setattr(witness, "validate_witness_arrays", observe_validation)

    result = witness.run(
        Namespace(
            input=str(input_path),
            output_npz=str(output_npz),
            output_json=str(output_json),
            expected_input_sha256="f" * 64,
            expected_source_recurrence_sha256="a" * 64,
            expected_mlx_exact_state_trace_sha256="b" * 64,
            expected_model_checkpoint_sha256="c" * 64,
            expected_rows=3436,
            expected_step_index=0,
        )
    )

    report = json.loads(output_json.read_text())
    assert result == 1
    assert observed == [(3436, 0)]
    assert report["requested"]["rows"] == 3436
    assert report["requested"]["shape_flow_step_index"] == 0
    assert report["failure_phase"] == "preflight_input"


def test_cuda_identity_binds_tensor_execution_to_authenticated_device_zero(
    tmp_path, monkeypatch
):
    import scripts.cuda_terminal_linear_witness as witness

    input_path = tmp_path / "input.npz"
    output_npz = tmp_path / "output.npz"
    output_json = tmp_path / "output.json"
    input_path.write_bytes(b"bound-input")
    observed_devices = []

    class FakeTensor:
        def to(self, *args, device=None, **kwargs):
            del args, kwargs
            observed_devices.append(device)
            return self

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def set_device(index):
            assert index == 0

        @staticmethod
        def get_device_name(index):
            return "Tesla T4" if index == 0 else "Unbound GPU"

        @staticmethod
        def get_device_properties(index):
            assert index == 0
            return SimpleNamespace(major=7, minor=5)

        @staticmethod
        def synchronize(device=None):
            assert device in {None, 0}
            return None

    def fail_linear(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("stop after tensor placement")

    fake_torch = SimpleNamespace(
        __version__="2.10.0+cu128",
        cuda=FakeCuda(),
        version=SimpleNamespace(cuda="12.8"),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(
                matmul=SimpleNamespace(allow_tf32=False)
            )
        ),
        from_numpy=lambda value: FakeTensor(),
        no_grad=nullcontext,
        nn=SimpleNamespace(
            functional=SimpleNamespace(linear=fail_linear)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(witness, "sha256_file", lambda path: "f" * 64)
    monkeypatch.setattr(witness, "_load_npz", lambda path: {})
    monkeypatch.setattr(
        witness,
        "validate_witness_arrays",
        lambda *args, **kwargs: {
            "arrays": {
                name: np.zeros(1, dtype=np.float32)
                for name in (
                    "pos_final_norm",
                    "neg_final_norm",
                    "weight",
                    "bias",
                    "expected_pos",
                    "expected_neg",
                )
            },
            "route_identity": {"route": "test"},
        },
    )

    result = witness.run(
        Namespace(
            input=str(input_path),
            output_npz=str(output_npz),
            output_json=str(output_json),
            expected_input_sha256="f" * 64,
            expected_source_recurrence_sha256="a" * 64,
            expected_mlx_exact_state_trace_sha256="b" * 64,
            expected_model_checkpoint_sha256="c" * 64,
        )
    )

    report = json.loads(output_json.read_text())
    assert result == 1
    assert report["failure_phase"] == "self_authenticate_source_linear"
    assert report["effective_cuda"]["device_index"] == 0
    assert observed_devices
    assert set(observed_devices) == {"cuda:0"}
