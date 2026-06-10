"""Pixal3D flow-model variants with view-aligned projected conditioning."""

from __future__ import annotations

from .slat_flow import SLatFlowModel
from .sparse_structure_flow import SparseStructureFlowModel
from ..modules.proj_attention import ProjectAttention


def _install_project_attention(blocks, channels: int, proj_in_channels: int) -> None:
    for block in blocks:
        block.cross_attn = ProjectAttention(block.cross_attn, channels, proj_in_channels)


class Pixal3DSparseStructureFlowModel(SparseStructureFlowModel):
    """Sparse-structure flow that accepts Pixal3D ``{global, proj}`` context."""

    def __init__(self, *args, proj_in_channels: int = 1024, **kwargs):
        super().__init__(*args, **kwargs)
        _install_project_attention(self.blocks, self.model_channels, proj_in_channels)


class Pixal3DSLatFlowModel(SLatFlowModel):
    """Structured-latent flow that accepts Pixal3D ``{global, proj}`` context."""

    def __init__(self, *args, proj_in_channels: int = 1024, **kwargs):
        super().__init__(*args, **kwargs)
        _install_project_attention(self.blocks, self.model_channels, proj_in_channels)
