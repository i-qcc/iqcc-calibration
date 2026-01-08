from typing import Optional, Literal
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 200
    """Number of averages to perform. Default is 200."""
    operation: str = "EF_x180"
    """Type of operation to perform. Default is "EF_x180"."""
    min_amp_factor: float = 0.01
    """Minimum amplitude factor for the operation. Default is 0.0."""
    max_amp_factor: float = 1.99
    """Maximum amplitude factor for the operation. Default is 2.0."""
    amp_factor_step: float = 0.02
    """Step size for the amplitude factor. Default is 0.02."""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    """Whether to use joint or independent flux points. Default is "joint"."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass

