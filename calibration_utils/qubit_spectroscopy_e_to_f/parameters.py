from typing import Optional, Literal
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 50
    """Span of frequencies to sweep in MHz. Default is 250 MHz."""
    frequency_step_in_mhz: float = 0.25
    """Step size for frequency sweep in MHz. Default is 0.25 MHz."""
    operation: str = "saturation"
    """Type of operation to perform. Default is "saturation"."""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    """Whether to use joint or independent flux points. Default is "joint"."""
    operation_amplitude_factor: Optional[float] = 0.3


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass

