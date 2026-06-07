"""Metal/MLX resource cleanup utilities.

Call cleanup() at the end of any generation run to release Metal
resources and reduce event/fence exhaustion over long sessions.
"""

import gc

import mlx.core as mx


def cleanup():
    """Release Metal resources and Python garbage.

    Should be called after each generation run, especially in
    long development sessions where many processes/runs accumulate
    Metal shared events.
    """
    gc.collect()
    # Clear MLX memory pool (returns cached allocations to the system)
    try:
        mx.metal.clear_cache()
    except AttributeError:
        pass  # older MLX versions may not have this
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
