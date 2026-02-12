from typing import Literal
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 2000
    """Number of runs to perform. Default is 2000."""
    operation: Literal["readout", "readout_QND", "readout_zero", "readout_zero_custom","readout_custom"] = "readout"
    """Type of operation to perform. Default is "readout"."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
