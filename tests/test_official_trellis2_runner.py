import importlib
import sys
import types

import numpy as np


def test_official_runner_records_raw_512_contract(tmp_path):
    runner = importlib.import_module("scripts.run_official_trellis2")

    image = tmp_path / "source.png"
    image.write_bytes(b"not a real png; parser/identity test only")
    output_dir = tmp_path / "out"

    args = runner.build_parser().parse_args(
        [
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
            "--save-raw-mesh",
            "--seed",
            "42",
            "--steps",
            "8",
            "--pipeline-type",
            "512",
            "--target-faces",
            "350000",
            "--texture-size",
            "4096",
            "--shared-noise",
            "",
        ]
    )

    identity = runner.build_route_identity(args, command=["run_official_trellis2.py"])

    assert identity["schema"] == "trellis2mlx.official_trellis2_route.v1"
    assert identity["route"]["family"] == "local-reference/trellis-mac"
    assert identity["route"]["pipeline_type"] == "512"
    assert identity["route"]["seed"] == 42
    assert identity["route"]["steps"] == 8
    assert identity["route"]["target_faces"] == 350000
    assert identity["route"]["texture_size"] == 4096
    assert identity["source"]["image_path"] == str(image)
    assert identity["source"]["image_sha256"]
    assert identity["requested_outputs"]["raw_mesh"] is True
    assert identity["requested_outputs"]["decoder_output"] is True
    assert identity["requested_outputs"]["final_glb"] is False
    assert identity["forbidden_inferences"] == [
        "not Microsoft CUDA TRELLIS.2 evidence",
        "not final-GLB parity evidence",
        "not texture/bake parity evidence",
    ]


def test_stop_after_raw_mesh_uses_shape_only_cut(monkeypatch, tmp_path):
    runner = importlib.import_module("scripts.run_official_trellis2")
    image = tmp_path / "source.png"
    image.write_bytes(b"parser only")
    args = runner.build_parser().parse_args(
        [
            "--image",
            str(image),
            "--output-dir",
            str(tmp_path / "out"),
            "--save-raw-mesh",
            "--stop-after-raw-mesh",
            "--pipeline-type",
            "512",
        ]
    )
    calls = []

    class Pipeline:
        def run(self, *_args, **_kwargs):  # pragma: no cover - should not be reached
            raise AssertionError("stop-after-raw-mesh must not enter full pipeline.run")

    def fake_raw_only(pipeline, loaded_image, parsed_args):
        calls.append((pipeline, loaded_image, parsed_args))
        return ["raw-mesh"]

    monkeypatch.setattr(runner, "_run_raw_mesh_only", fake_raw_only)

    assert runner._run_mesh_pipeline(Pipeline(), object(), args) == ["raw-mesh"]
    assert calls and calls[0][2] is args


def test_decoder_capture_hook_runs_base_decoder_once(monkeypatch):
    runner = importlib.import_module("scripts.run_official_trellis2")

    fake_torch = types.SimpleNamespace(no_grad=lambda: (lambda fn: fn))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    calls = {"base": 0}

    class Tensor:
        def __init__(self, array):
            self.array = np.asarray(array)
            self.shape = self.array.shape

        def __getitem__(self, key):
            return Tensor(self.array[key])

        def __mul__(self, other):
            return Tensor(self.array * other)

        __rmul__ = __mul__

        def __add__(self, other):
            return Tensor(self.array + other)

        __radd__ = __add__

        def __sub__(self, other):
            return Tensor(self.array - other)

        def __rsub__(self, other):
            return Tensor(other - self.array)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

        def __gt__(self, other):
            return Tensor(self.array > other)

    class Sparse:
        def __init__(self, feats, coords):
            self.feats = Tensor(feats)
            self.coords = Tensor(coords)

        def replace(self, feats):
            return Sparse(feats.array if isinstance(feats, Tensor) else feats, self.coords.array)

        def __iter__(self):
            yield self

    class BaseDecoder:
        def forward(self, x, **kwargs):
            calls["base"] += 1
            return Sparse(
                np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.25]], dtype=np.float32),
                np.array([[0, 0, 0, 0]], dtype=np.int32),
            )

    class Decoder(BaseDecoder):
        training = False
        voxel_margin = 0.5
        resolution = 512

        def forward(self, x, **kwargs):  # pragma: no cover - pre-patch hook calls this
            calls["original"] = calls.get("original", 0) + 1
            return super().forward(x, **kwargs)

    def fake_flexible_dual_grid_to_mesh(coords, vertices, intersected, quad_lerp, **kwargs):
        assert coords.shape == (1, 3)
        assert vertices.shape == (1, 3)
        assert intersected.shape == (1, 3)
        assert quad_lerp.shape == (1, 1)
        return "vertices", "faces"

    monkeypatch.setattr(runner, "_sigmoid", lambda value: value)
    monkeypatch.setattr(runner, "_softplus", lambda value: value)
    monkeypatch.setattr(runner, "_flexible_dual_grid_to_mesh", fake_flexible_dual_grid_to_mesh)

    captured = {}
    decoder = Decoder()
    runner._install_decoder_capture_hook(decoder, captured)
    result = decoder.forward(object())

    assert calls == {"base": 1}
    assert captured["feats"].shape == (1, 7)
    assert captured["coords"].shape == (1, 4)
    assert result == [("vertices", "faces")]


def test_records_effective_trellis_backend_identity(monkeypatch, tmp_path):
    runner = importlib.import_module("scripts.run_official_trellis2")

    attention_config = types.ModuleType("trellis2.modules.attention.config")
    attention_config.BACKEND = "sdpa"
    sparse_config = types.ModuleType("trellis2.modules.sparse.config")
    sparse_config.ATTN = "sdpa"
    monkeypatch.setitem(sys.modules, "trellis2.modules.attention.config", attention_config)
    monkeypatch.setitem(sys.modules, "trellis2.modules.sparse.config", sparse_config)

    route_identity = {"route": {}}
    runner._record_effective_backend_identity(route_identity, tmp_path)

    assert route_identity["effective_backend"] == {
        "attention_backend": "sdpa",
        "sparse_attention_backend": "sdpa",
    }
    assert (tmp_path / "route_identity.json").exists()


def test_official_runner_exposes_sparse_internals_stage():
    runner_path = importlib.import_module("scripts.run_official_trellis2").__file__
    text = open(runner_path).read()

    assert "--save-sparse-internals" in text
    assert "--stop-after-sparse-internals" in text
    assert "sparse_internals.npz" in text
    assert "z_s=" in text
    assert "logits=" in text


def test_official_runner_exposes_sparse_flow_step_stage():
    runner_path = importlib.import_module("scripts.run_official_trellis2").__file__
    text = open(runner_path).read()

    assert "--save-sparse-flow-step" in text
    assert "--stop-after-sparse-flow-step" in text
    assert "sparse_flow_step0.npz" in text
    for key in (
        "noise=",
        "pred_pos=",
        "pred_neg=",
        "pred_cfg=",
        "std_ratio=",
        "pred_final=",
        "sample_next=",
    ):
        assert key in text


def test_official_runner_exposes_sparse_flow_block_trace_stage():
    runner_path = importlib.import_module("scripts.run_official_trellis2").__file__
    text = open(runner_path).read()

    assert "--save-sparse-flow-block-trace" in text
    assert "--stop-after-sparse-flow-block-trace" in text
    assert "sparse_flow_block_trace.npz" in text
    for key in (
        "pos_input_projected=",
        "pos_block0_q_pre_norm=",
        "pos_block0_q_post_rope=",
        "pos_block0_attention_raw=",
        "pos_block0_self_attn=",
        "pos_block0_cross_attn=",
        "pos_block0_mlp=",
        "neg_input_projected=",
        "neg_block0_k_post_norm=",
        "neg_block0_after_mlp=",
    ):
        assert key in text


def test_sparse_flow_block_trace_numpy_helper_casts_before_numpy():
    runner = importlib.import_module("scripts.run_official_trellis2")

    class BFloat16Like:
        def __init__(self):
            self.cast = False

        def detach(self):
            return self

        def float(self):
            self.cast = True
            return self

        def cpu(self):
            return self

        def numpy(self):
            if not self.cast:
                raise TypeError("Got unsupported ScalarType BFloat16")
            return np.array([1.25], dtype=np.float32)

    out = runner._to_numpy_float32(BFloat16Like())

    assert out.dtype == np.float32
    assert out.tolist() == [1.25]
