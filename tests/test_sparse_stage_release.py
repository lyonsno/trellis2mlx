"""The sparse-stage model references must not survive into LR shape sampling."""

import ast
from pathlib import Path

import pytest


def _assert_sparse_stage_release(source: str):
    module = ast.parse(source)
    main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    # Walk only main's direct statement sequence. An unreachable branch or an
    # uncalled nested function must not satisfy this resource-release guard.
    body = main.body
    sparse_decode = next(
        index for index, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "ss_dec"
    )
    lr_load = next(
        index for index, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "for_shape"
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "SLatFlowModel"
    )
    release = [
        index for index, node in enumerate(body)
        if isinstance(node, ast.Delete)
        and {target.id for target in node.targets if isinstance(target, ast.Name)}
        == {"ss_flow", "ss_dec"}
    ]
    assert len(release) == 1
    release_index = release[0]
    assert sparse_decode < release_index < lr_load
    next_statement = body[release_index + 1]
    assert isinstance(next_statement, ast.Expr)
    assert isinstance(next_statement.value, ast.Call)
    assert isinstance(next_statement.value.func, ast.Name)
    assert next_statement.value.func.id == "cleanup"
    assert release_index + 1 < lr_load


def test_sparse_models_are_unbound_before_stage_cleanup_and_lr_load():
    source = Path(__file__).resolve().parents[1] / "generate.py"
    _assert_sparse_stage_release(source.read_text())


@pytest.mark.parametrize("replacement", [
    "    if False:\n        del ss_flow, ss_dec\n        cleanup()\n",
    "    def hidden_release():\n        del ss_flow, ss_dec\n        cleanup()\n",
    "    cleanup()\n    del ss_flow, ss_dec\n    if False:\n        cleanup()\n",
    "    cleanup_model(ss_flow, ss_dec)\n",
])
def test_release_guard_rejects_nonexecuting_or_misordered_mutations(replacement):
    source = (Path(__file__).resolve().parents[1] / "generate.py").read_text()
    live_release = "    del ss_flow, ss_dec\n    cleanup()\n"
    assert source.count(live_release) == 1
    with pytest.raises(AssertionError):
        _assert_sparse_stage_release(source.replace(live_release, replacement, 1))
