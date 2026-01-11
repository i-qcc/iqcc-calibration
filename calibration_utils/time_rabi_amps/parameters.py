from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters, IdleTimeNodeParameters
from typing import List
from dataclasses import field

class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    min_wait_time_in_ns: int = 20
    max_wait_time_in_ns: int = 160
    num_time_steps: int = 100
    min_amp_factor: float = 0.1
    """Minimum amplitude factor for the operation. Default is 0.1."""
    max_amp_factor: float = 1.5
    """Maximum amplitude factor for the operation. Default is 1.5."""
    amp_factor_step: float = 0.05
    """Step size for the amplitude factor. Default is 0.05."""

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
