from quam.core import quam_dataclass
from typing import Dict, Any
from dataclasses import field

from quam_builder.architecture.superconducting.components.flux_line import FluxLine as FluxLineBase

from iqcc_calibration_tools.quam_config.lib.qua_utils import safe_wait

__all__ = ["FluxLine"]


@quam_dataclass
class FluxLine(FluxLineBase):
    """QuAM component for a flux line with custom configuration."""

    settle_time: float = 64 # ns 
    offset_settle_time: float = 64 # ns
    extras: Dict[str, Any] = field(default_factory=dict)

    def settle(self, settle_time: float | None = None):
        """Wait for the flux bias to settle"""
        if settle_time is not None:
            safe_wait(int(settle_time) // 4)
        elif self.offset_settle_time is not None:
            safe_wait(int(self.offset_settle_time) // 4)