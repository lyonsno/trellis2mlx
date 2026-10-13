"""The sparse-stage model references must not survive into LR shape sampling."""

import ast
from pathlib import Path


def test_sparse_models_are_unbound_before_stage_cleanup_and_lr_load():
    source = Path(__file__).resolve().parents[1] / "generate.py"
    module = ast.parse(source.read_text())
    main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    sparse_decode_line = next(
        node.lineno for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "ss_dec"
    )
    lr_load_line = next(
        node.lineno for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "for_shape"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "SLatFlowModel"
    )
    unbind_lines = {
        target.id: node.lineno
        for node in ast.walk(main) if isinstance(node, ast.Delete)
        for target in node.targets if isinstance(target, ast.Name)
        and target.id in {"ss_flow", "ss_dec"}
    }
    assert set(unbind_lines) == {"ss_flow", "ss_dec"}
    assert all(sparse_decode_line < line < lr_load_line for line in unbind_lines.values())
    assert any(
        isinstance(node.func, ast.Name)
        and node.func.id == "cleanup"
        and max(unbind_lines.values()) < node.lineno < lr_load_line
        for node in calls
    )
