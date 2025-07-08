from quam.core import quam_dataclass
from typing import Dict, Any
from dataclasses import field

from quam_builder.architecture.superconducting.components.flux_line import FluxLine as QuamBuilderFluxLine

__all__ = ["FluxLine"]


@quam_dataclass
class FluxLine(QuamBuilderFluxLine):
    """QuAM component for a flux line with custom configuration."""

    settle_time: float = 16  # Override default settle_time to 16ns
    extras: Dict[str, Any] = field(default_factory=dict)
