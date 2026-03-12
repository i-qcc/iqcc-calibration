from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 200
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 4
    """Span of frequencies to sweep in MHz. Default is 10 MHz."""
    frequency_step_in_mhz: float = 0.025
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
