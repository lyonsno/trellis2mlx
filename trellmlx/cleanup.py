"""Metal/MLX resource cleanup utilities.

Call cleanup() at the end of any generation run to release Metal
resources and reduce event/fence exhaustion over long sessions.
"""

import gc

import mlx.core as mx


def _clear_mlx_cache():
    """Clear MLX's memory pool across old and current MLX releases."""
    clear_cache = getattr(mx, "clear_cache", None)
    if clear_cache is not None:
        clear_cache()
        return

    metal = getattr(mx, "metal", None)
    metal_clear_cache = getattr(metal, "clear_cache", None)
    if metal_clear_cache is not None:
        metal_clear_cache()


def cleanup():
    """Release Metal resources and Python garbage.

    Should be called after each generation run, especially in
    long development sessions where many processes/runs accumulate
    Metal shared events.
    """
    gc.collect()
    # Clear MLX memory pool (returns cached allocations to the system)
    _clear_mlx_cache()
    # Force synchronization to flush any pending Metal work
    try:
        mx.synchronize()
    except Exception:
        pass


def cleanup_model(*models):
    """Delete models and release their Metal resources.

    Usage:
        cleanup_model(flow_model, decoder)
    """
    for model in models:
        del model
    cleanup()
