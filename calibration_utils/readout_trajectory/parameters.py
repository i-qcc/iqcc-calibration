from typing import Optional
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Readout amplitude in V. """
    readout_name: Optional[str] = "readout_zero"
    """Readout length in nanoseconds. Default is 1000 ns."""
    # leading_zero_length: Optional[int]= 0
    square_length: Optional[int] = 1000
    #duration_chunks: int = 8 # in ns
    zero_length: Optional[int]= 1000
    # readout length must equal leading_zero_length + square_length + zero_length
    readout_length_in_ns: Optional[int] = square_length + zero_length
    #Ain: Optional[complex] = -1.8110376753789537e-05+3.347973148612415e-06j

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
