from typing import Optional
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 500
    readout_name: Optional[str] = "readout_zero"
    # leading_zero_length: Optional[int]= 0
    # Default square_length: 1480 ns to match existing integration weights in qB4's readout_zero pulse
    # If you need 1500 ns, ensure integration weights are regenerated when length changes
    square_length: Optional[int] = 2000
    #duration_chunks: int = 8 # in ns
    zero_length: Optional[int] = 1000
    # readout length must equal leading_zero_length + square_length + zero_length
    # Default matches qB4's readout length: 1500 + 0 = 1500 ns
    readout_length_in_ns: Optional[int] = square_length+zero_length
    segment_length: int = 25  # Segment length for sliced measurements (in ns)
    use_custom_integration_weights: bool = False  # If True, uses optimized integration weights from previous run; If False, uses default weights

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
