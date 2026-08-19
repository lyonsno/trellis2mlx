from pathlib import Path

import pytest

from trellmlx.witness_authority import AuthorityCoordinate, AuthorityCoordinateError


@pytest.mark.parametrize(
    "alias",
    (
        "/run/admitted-inputs/run-id/./rembg",
        "/run/admitted-inputs/run-id/sibling/../rembg",
        "/run/admitted-inputs//run-id/rembg",
    ),
)
def test_authority_coordinate_rejects_lexical_aliases(alias):
    with pytest.raises(AuthorityCoordinateError, match="canonical|declared"):
        AuthorityCoordinate.bind_absolute(alias, label="effective RMBG root")


def test_authority_coordinate_preserves_raw_and_resolved_roles(tmp_path):
    root = tmp_path / "admitted-inputs" / "run-id"
    target = root / "rembg"
    target.mkdir(parents=True)

    coordinate = AuthorityCoordinate.bind_absolute(
        str(target),
        label="effective RMBG root",
        expected_raw=str(target),
        containment_root=root,
    )

    assert coordinate.raw == str(target)
    assert coordinate.lexical.as_posix() == str(target)
    assert coordinate.resolved == target.resolve()


def test_authority_coordinate_rejects_raw_mismatch_before_resolution(tmp_path):
    target = tmp_path / "run" / "rembg"
    target.mkdir(parents=True)
    alias = f"{target.parent}/./{target.name}"

    with pytest.raises(AuthorityCoordinateError, match="declared|canonical"):
        AuthorityCoordinate.bind_absolute(
            alias,
            label="effective RMBG root",
            expected_raw=str(target),
            containment_root=target.parent,
        )
