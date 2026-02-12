from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters, IdleTimeNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 500
    """Number of averages to perform. Default is 1500."""
    flux_span: float = 0.1
    """Span of flux values to sweep in volts. Default is 0.1 V."""
    flux_step: float = 0.01
    """Step size for flux sweep in volts. Default is 0.005 V."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
