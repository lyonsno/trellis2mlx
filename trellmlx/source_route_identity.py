"""Route identity checks for source-native mesh backends."""

from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


DEFAULT_FORBIDDEN_SOURCE_MARKERS = (
    "Hunyuan3D-MLX",
    "Hunyuan3D",
    "ZimengXiong/Hunyuan3D-MLX",
)
EXPECTED_ROOT_ENV = "TRELLIS2MLX_SOURCE_MTLMESH_ROOT"


class SourceRouteIdentityError(RuntimeError):
    """Raised when an imported source backend resolves to the wrong route."""

    def __init__(self, message: str, identity: dict[str, Any]):
        super().__init__(message)
        self.identity = identity


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _find_git_root(path: str | Path | None) -> str | None:
    if not path:
        return None
    current = _resolve_path(path)
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return str(candidate)
    return None


def _git_output(root: str | None, *args: str) -> str | None:
    if not root:
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", root, *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _first_remote_url(root: str | None) -> str | None:
    if not root:
        return None
    remote = _git_output(root, "remote", "get-url", "origin")
    if remote:
        return remote
    remotes = _git_output(root, "remote", "-v")
    if not remotes:
        return None
    return remotes.splitlines()[0].split()[1]


def probe_cumesh_route_identity() -> dict[str, Any]:
    """Return the effective imported cumesh route identity for this interpreter."""
    identity: dict[str, Any] = {"status": "unavailable", "python": sys.executable}
    try:
        cumesh = importlib.import_module("cumesh")
    except Exception as exc:
        identity.update({
            "reason": f"{type(exc).__name__}: {exc}",
            "error": f"{type(exc).__name__}: {exc}",
        })
        return identity

    identity.update({
        "status": "available",
        "cumesh_file": getattr(cumesh, "__file__", None),
        "cumesh_path": list(getattr(cumesh, "__path__", [])),
        "has_CuMesh": hasattr(cumesh, "CuMesh"),
    })

    try:
        metal_backend = importlib.import_module("cumesh.metal_backend")
    except Exception as exc:
        identity.update({
            "metal_backend_error": f"{type(exc).__name__}: {exc}",
            "metal_backend_file": None,
        })
    else:
        metal_backend_file = inspect.getfile(metal_backend)
        git_root = _find_git_root(metal_backend_file)
        identity.update({
            "metal_backend_file": metal_backend_file,
            "has_MtlMesh": hasattr(metal_backend, "MtlMesh"),
            "git_root": git_root,
            "git_remote": _first_remote_url(git_root),
            "git_commit": _git_output(git_root, "rev-parse", "HEAD"),
        })
    return identity


def validate_source_route_identity(
    identity: dict[str, Any],
    *,
    expected_root: str | Path | None = None,
    forbidden_markers: tuple[str, ...] = DEFAULT_FORBIDDEN_SOURCE_MARKERS,
) -> dict[str, Any]:
    """Validate that a source-native backend did not resolve to a proxy route."""
    validated = dict(identity)
    if validated.get("status") != "available":
        return validated

    route_values: list[str] = []
    for key in ("cumesh_file", "metal_backend_file", "git_root", "git_remote"):
        value = validated.get(key)
        if value:
            route_values.append(str(value))
    for value in validated.get("cumesh_path", []) or []:
        route_values.append(str(value))

    forbidden_hits = [
        value
        for value in route_values
        if any(marker in value for marker in forbidden_markers)
    ]
    validated["forbidden_source_route"] = bool(forbidden_hits)
    if forbidden_hits:
        validated["forbidden_source_hits"] = forbidden_hits
        raise SourceRouteIdentityError(
            f"forbidden source route for mtlmesh/cumesh: {forbidden_hits[0]}",
            validated,
        )

    root_value = expected_root if expected_root is not None else os.environ.get(EXPECTED_ROOT_ENV)
    if root_value:
        root = _resolve_path(root_value)
        candidate_paths = []
        for key in ("metal_backend_file", "cumesh_file", "git_root"):
            value = validated.get(key)
            if value:
                candidate_paths.append(_resolve_path(value))
        for value in validated.get("cumesh_path", []) or []:
            candidate_paths.append(_resolve_path(value))
        source_root_match = any(_is_relative_to(path, root) for path in candidate_paths)
        validated["expected_source_root"] = str(root)
        validated["source_root_match"] = source_root_match
        if not source_root_match:
            raise SourceRouteIdentityError(
                f"mtlmesh/cumesh route does not match expected source root: {root}",
                validated,
            )
    else:
        validated["source_root_match"] = None

    validated["forbidden_source_route"] = False
    return validated
