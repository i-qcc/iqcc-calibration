from quam.core import quam_dataclass
from quam_builder.architecture.superconducting.qubit.flux_tunable_transmon import FluxTunableTransmon
from quam_builder.architecture.superconducting.components.xy_drive import (
    XYDriveIQ,
    XYDriveMW,
)
from quam.components.channels import Pulse
from typing import Dict, Any, Union
from dataclasses import field
import numpy as np

__all__ = ["Transmon"]

@quam_dataclass
class Transmon(FluxTunableTransmon):
    """
    Optimized QuAM component for a transmon qubit, inheriting from FluxTunableTransmon.
    Only custom methods/fields not present in the base class are defined here.
    """
    # xy_SL: Union[XYDriveIQ, XYDriveMW] = None # Adding spin lockingon xy channel
    anharmonicity: float = 200e6 # default 200 MHz instead of None in base class
    extras: Dict[str, Any] = field(default_factory=dict)

    def get_output_power(self, operation, Z=50) -> float:
        power = self.xy.opx_output.full_scale_power_dbm
        amplitude = self.xy.operations[operation].amplitude
        x_mw = 10 ** (power / 10)
        x_v = amplitude * np.sqrt(2 * Z * x_mw / 1000)
        return 10 * np.log10(((x_v / np.sqrt(2)) ** 2 * 1000) / Z)

    def sigma(self, operation: Pulse):
        return operation.length / self.sigma_time_factor

    @property
    def name(self):
        return self.id if isinstance(self.id, str) else f"q{self.id}"
