"""Shared authority-coordinate contracts for replayable witness artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class AuthorityCoordinateError(ValueError):
    """Raised when a declared coordinate cannot carry the claimed authority."""


@dataclass(frozen=True)
class AuthorityCoordinate:
    """Keep the declared coordinate separate from its resolved filesystem path."""

    raw: str
    lexical: PurePosixPath
    resolved: Path

    @classmethod
    def bind_path(
        cls,
        value: object,
        *,
        label: str,
        base: str | Path | None = None,
        expected_raw: str | None = None,
        containment_root: str | Path | None = None,
        require_absolute: bool = False,
    ) -> "AuthorityCoordinate":
        if not isinstance(value, str) or not value:
            raise AuthorityCoordinateError(f"{label} declared coordinate is missing")
        lexical = PurePosixPath(value)
        if (
            (require_absolute and not lexical.is_absolute())
            or value.startswith("//")
            or ".." in lexical.parts
            or lexical.as_posix() != value
        ):
            raise AuthorityCoordinateError(
                f"{label} declared coordinate is not canonical: {value!r}"
            )
        if expected_raw is not None and value != expected_raw:
            raise AuthorityCoordinateError(
                f"{label} declared coordinate mismatch: {value!r} != {expected_raw!r}"
            )
        path = Path(value)
        if not path.is_absolute():
            path = Path.cwd() / path if base is None else Path(base) / path
        resolved = path.resolve(strict=False)
        if containment_root is not None:
            root = Path(containment_root).resolve(strict=False)
            if resolved != root and root not in resolved.parents:
                raise AuthorityCoordinateError(
                    f"{label} resolved path escapes containment root: {resolved} not under {root}"
                )
        return cls(raw=value, lexical=lexical, resolved=resolved)

    @classmethod
    def bind_absolute(
        cls,
        value: object,
        *,
        label: str,
        expected_raw: str | None = None,
        containment_root: str | Path | None = None,
    ) -> "AuthorityCoordinate":
        return cls.bind_path(
            value,
            label=label,
            expected_raw=expected_raw,
            containment_root=containment_root,
            require_absolute=True,
        )
