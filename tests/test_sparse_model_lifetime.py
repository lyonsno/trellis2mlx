"""The sparse models must be released before shape sampling begins."""

import ast
from pathlib import Path


def test_sparse_models_are_unbound_before_mlx_pool_cleanup():
    source = Path(__file__).resolve().parents[1] / "generate.py"
    module = ast.parse(source.read_text())
    main = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    # Clearing the MLX pool cannot release models that main() still owns.
    # Both bindings must be deleted before the stage-boundary cleanup call.
    for index, node in enumerate(main.body):
        if not isinstance(node, ast.Delete):
            continue
        names = {
            name.id
            for target in node.targets
            for name in ast.walk(target)
            if isinstance(name, ast.Name)
        }
        if {"ss_flow", "ss_dec"} <= names:
            following = main.body[index + 1]
            assert isinstance(following, ast.Expr)
            assert isinstance(following.value, ast.Call)
            assert isinstance(following.value.func, ast.Name)
            assert following.value.func.id == "cleanup"
            return

    raise AssertionError("main() retains the sparse models across shape sampling")
