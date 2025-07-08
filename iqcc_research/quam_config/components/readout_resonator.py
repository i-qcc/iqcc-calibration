from quam.core import quam_dataclass

from quam_builder.architecture.superconducting.components.readout_resonator import (
    ReadoutResonatorBase as QuamBuilderReadoutResonatorBase,
    ReadoutResonatorIQ as QuamBuilderReadoutResonatorIQ,
    ReadoutResonatorMW as QuamBuilderReadoutResonatorMW,
)

__all__ = ["ReadoutResonator", "ReadoutResonatorIQ", "ReadoutResonatorMW"]


@quam_dataclass
class ReadoutResonatorBase(QuamBuilderReadoutResonatorBase):
    """QuAM component for a readout resonator with custom depletion time."""

    depletion_time: int = 4000  # Override default depletion time to 4000ns


@quam_dataclass
class ReadoutResonatorIQ(QuamBuilderReadoutResonatorIQ):
    @property
    def upconverter_frequency(self):
        return self.LO_frequency


@quam_dataclass
class ReadoutResonatorMW(QuamBuilderReadoutResonatorMW):
    @property
    def upconverter_frequency(self):
        return self.opx_output.upconverter_frequency


ReadoutResonator = ReadoutResonatorIQ
