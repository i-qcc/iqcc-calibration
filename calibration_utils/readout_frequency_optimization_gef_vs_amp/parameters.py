from typing import Optional, Literal
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 7
    """Span of frequencies to sweep in MHz. Default is 7 MHz."""
    frequency_step_in_mhz: float = 0.2
    """Step size for frequency sweep in MHz. Default is 0.2 MHz."""
    amp_min: float = 0.1
    """Minimum readout amplitude prefactor. Default is 0.1."""
    amp_max: float = 1.5
    """Maximum readout amplitude prefactor. Default is 1.5."""
    amp_step: float = 0.05
    """Step size for amplitude sweep. Default is 0.05."""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    """Whether to use joint or independent flux points. Default is "joint"."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass

