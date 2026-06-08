"""Pytest configuration — Metal resource cleanup between test modules."""

import gc

from trellmlx.cleanup import _clear_mlx_cache


def pytest_runtest_teardown(item, nextitem):
    """Clear MLX memory pool after each test to reduce Metal event accumulation."""
    gc.collect()
    _clear_mlx_cache()
