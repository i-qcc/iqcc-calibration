from quam.components.ports import LFFEMAnalogOutputPort
from quam.core import quam_dataclass

from quam_builder.architecture.superconducting.components.tunable_coupler import TunableCoupler as QuamBuilderTunableCoupler

__all__ = ["TunableCoupler"]


@quam_dataclass
class TunableCoupler(QuamBuilderTunableCoupler):
    """
    Optimized QuAM component for a tunable coupler, inheriting from QuamBuilderTunableCoupler.
    Only custom methods/fields not present in the base class are defined here.
    """

    output_mode: str = "direct"  # "amplified"
    upsampling_mode: str = "pulse"

    def __post_init__(self):
        if isinstance(self.opx_output, LFFEMAnalogOutputPort):
            self.opx_output.upsampling_mode = self.upsampling_mode
            self.opx_output.output_mode = self.output_mode
