"""Texture model family follows the executed shape pipeline.

Reference: microsoft/TRELLIS.2@5565d240, Trellis2ImageTo3DPipeline.run:
single-pass 512 selects tex_slat_flow_model_512; the cascades select 1024.
Model snapshot af44b45f2e35a493886929c6d786e563ec68364d/pipeline.json maps
those names to the checkpoint filenames asserted here. Native 768 is also
a cascade using the 1024 shape model, independent of export atlas size.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "no_cascade,resolution,texture_size,expected",
    [
        (False, 768, 4096, 1024),
        (False, 1024, 512, 1024),
        (False, 1536, 2048, 1024),
        (True, 512, 4096, 512),
        (True, 1024, 4096, 512),
    ],
)
def test_actual_texture_stage_loads_pipeline_model(
    no_cascade, resolution, texture_size, expected
):
    # Execute the real Stage4 prefix without allocating a model or running
    # inference. A correct unused selector cannot satisfy this witness.
    source = Path(__file__).resolve().parents[1] / "generate.py"
    tree = ast.parse(source.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    start = next(i for i, n in enumerate(main.body)
                 if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and any(isinstance(a, ast.Constant) and isinstance(a.value, str)
                         and "=== Stage 4: Texture SLat ===" in a.value
                         for a in n.value.args))
    stop = next(i for i in range(start, len(main.body))
                if isinstance(main.body[i], ast.Expr)
                and isinstance(main.body[i].value, ast.Call)
                and isinstance(main.body[i].value.func, ast.Name)
                and main.body[i].value.func.id == "load_weights")
    block = ast.fix_missing_locations(ast.Module(body=main.body[start:stop + 1], type_ignores=[]))
    model = object()
    calls = []

    def load_weights(actual_model, path, verbose):
        calls.append((actual_model, path, verbose))

    namespace = {
        "args": SimpleNamespace(no_cascade=no_cascade, resolution=resolution,
                                texture_size=texture_size,
                                shape_slat_support_mesh_grid_size=1024),
        "HF_4B": "/weights/",
        "SLatFlowModel": SimpleNamespace(for_texture=lambda: model),
        "load_weights": load_weights,
    }
    exec(compile(block, str(source), "exec"), namespace)
    assert calls == [(model, f"/weights/slat_flow_imgshape2tex_dit_1_3B_{expected}_bf16.safetensors", False)]
