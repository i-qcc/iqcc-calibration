from typing import Literal
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 2000
    """Number of runs to perform. Default is 2000."""
    operation: Literal["readout", "readout_QND"] = "readout"
    """Type of operation to perform. Default is "readout"."""
    start_depletion_time: int = 0
    """Start depletion time in nanoseconds. Default is 1000."""
    end_depletion_time: int = 1000
    """End depletion time in nanoseconds. Default is 10000."""
    num_depletion_times: int = 100
    """Number of depletion times to sweep. Default is 10."""
    num_of_measurement: int = 5
    """Number of consecutive measurements to perform for the first measurement. Only the last measurement is saved. Default is 5."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass

