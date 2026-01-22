from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters, SpinLockingTimeNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1000
    min_amp_factor: float = 0.01
    """Minimum amplitude factor for the operation. Default is 0.1."""
    max_amp_factor: float = 1.0
    """Maximum amplitude factor for the operation. Default is 1.5."""
    amp_factor_step: float = 0.1
    """Step size for the amplitude factor. Default is 0.05."""

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    SpinLockingTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
