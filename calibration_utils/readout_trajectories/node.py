from typing import List, Union

from quam_builder.architecture.superconducting.components.readout_resonator import (
    ReadoutResonatorIQ,
    ReadoutResonatorMW,
)
from qualibration_libs.core import tracked_updates
from calibration_utils.readout_trajectories.parameters import Parameters


def patch_readout_pulse_params(
    resonators: List[Union[ReadoutResonatorIQ, ReadoutResonatorMW]],
    node_parameters: Parameters,
):
    """
    Patch readout pulse parameters for readout trajectories calibration.
    
    This function temporarily updates resonator readout pulse parameters
    (length and zero_length) based on node parameters, using tracked_updates
    to ensure they are reverted after use.
    """
    patched_resonators = []
    readout_name = node_parameters.readout_name or "readout_zero"
    
    for resonator in resonators:
        # make temporary updates before running the program and revert at the end.
        with tracked_updates(resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
            resonator.operations[readout_name].length = (
                node_parameters.square_length + node_parameters.zero_length
            )
            resonator.operations[readout_name].zero_length = node_parameters.zero_length
            patched_resonators.append(resonator)

    return patched_resonators
