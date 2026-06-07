"""Pytest configuration — Metal resource cleanup between test modules."""

import gc

import mlx.core as mx


def pytest_runtest_teardown(item, nextitem):
    """Clear MLX memory pool after each test to reduce Metal event accumulation."""
    gc.collect()
    try:
        mx.metal.clear_cache()
    except AttributeError:
        pass
