from typing import Dict
from dataclasses import field

from quam.core import  quam_dataclass
from quam_builder.architecture.superconducting.qubit_pair.flux_tunable_transmon_pair import FluxTunableTransmonPair
from iqcc_calibration_tools.quam_config.components.gates.two_qubit_gates import TwoQubitGate
from qm.qua import align

__all__ = ["TransmonPair"]


# TODO : to be removed using the FluxTunableTransmonPair class directly
@quam_dataclass
class TransmonPair(FluxTunableTransmonPair):
    """
    Optimized QuAM component for a transmon pair, inheriting from FluxTunableTransmonPair.
    Only custom methods/fields not present in the base class are defined here.
    """
    gates: Dict[str, TwoQubitGate] = field(default_factory=dict)
    J2: float = 0

    @property
    def name(self):
        """The name of the transmon pair"""
        return self.id if isinstance(self.id, str) else f"q{self.id}"
    
    def align(self):
        """Custom align method with Cz gate compensations"""
        channels = [self.qubit_control.xy.name, self.qubit_control.z.name, self.qubit_control.resonator.name, self.qubit_target.xy.name, 
                  self.qubit_target.z.name, self.qubit_target.resonator.name]
        
        if self.coupler:
             channels += [self.coupler.name]
        
        if "Cz" in self.gates:
            if hasattr(self.gates['Cz'], 'compensations'):
                for compensation in self.gates['Cz'].compensations:
                    channels += [compensation["qubit"].xy.name, compensation["qubit"].z.name, compensation["qubit"].resonator.name]

        align(*channels)
            
