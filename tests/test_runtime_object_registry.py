"""Runtime-only object registry for stage execution context."""

import pytest


def test_stage_execution_context_registers_and_requires_runtime_objects():
    from trellmlx.interleaved_generation import StageExecutionContext

    context = StageExecutionContext(run_id="registry-probe")
    conditioning = object()

    context.register_runtime_object("cond://seed-101", conditioning)

    assert context.runtime_object_keys == ("cond://seed-101",)
    assert context.require_runtime_object("cond://seed-101") is conditioning


def test_stage_execution_context_rejects_invalid_runtime_object_keys():
    from trellmlx.interleaved_generation import StageExecutionContext

    context = StageExecutionContext(run_id="registry-probe")

    with pytest.raises(ValueError, match="runtime object key must be a nonempty string"):
        context.register_runtime_object("", object())
    with pytest.raises(ValueError, match="runtime object key must be a nonempty string"):
        context.require_runtime_object("")


def test_stage_execution_context_rejects_missing_duplicate_and_none_runtime_objects():
    from trellmlx.interleaved_generation import StageExecutionContext

    context = StageExecutionContext(run_id="registry-probe")
    context.register_runtime_object("cond://seed-101", object())

    with pytest.raises(ValueError, match="runtime object value cannot be None"):
        context.register_runtime_object("cond://none", None)
    with pytest.raises(ValueError, match="runtime object already registered: cond://seed-101"):
        context.register_runtime_object("cond://seed-101", object())
    with pytest.raises(KeyError, match="missing runtime object: cond://missing"):
        context.require_runtime_object("cond://missing")
