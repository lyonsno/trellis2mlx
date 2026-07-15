import subprocess
import sys
from pathlib import Path

import pytest

from generate import _shape_sampler_params


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def test_shape_sampler_defaults_preserve_existing_generation_contract():
    assert _shape_sampler_params(steps=8) == {
        "steps": 8,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.5,
        "guidance_interval": (0.6, 1.0),
        "rescale_t": 3.0,
    }


def test_shape_sampler_pressure_overrides_are_explicit():
    assert _shape_sampler_params(
        steps=6,
        guidance_strength=4.0,
        guidance_rescale=0.25,
        guidance_low=0.35,
        guidance_high=0.9,
    ) == {
        "steps": 6,
        "guidance_strength": 4.0,
        "guidance_rescale": 0.25,
        "guidance_interval": (0.35, 0.9),
        "rescale_t": 3.0,
    }


@pytest.mark.parametrize(
    ("guidance_low", "guidance_high"),
    [(-0.1, 0.8), (0.4, 1.1), (0.9, 0.3)],
)
def test_shape_sampler_rejects_invalid_guidance_interval(guidance_low, guidance_high):
    with pytest.raises(ValueError, match="shape guidance interval"):
        _shape_sampler_params(
            steps=6,
            guidance_low=guidance_low,
            guidance_high=guidance_high,
        )


def test_generate_help_exposes_shape_pressure_controls():
    result = subprocess.run(
        [str(PYTHON), "generate.py", "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--shape-guidance-strength" in result.stdout
    assert "--shape-guidance-rescale" in result.stdout
    assert "--shape-guidance-low" in result.stdout
    assert "--shape-guidance-high" in result.stdout
