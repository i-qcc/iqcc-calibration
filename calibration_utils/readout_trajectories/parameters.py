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
    
    def get_square_length(self, default: int = 1500) -> int:
        """Get square_length with a default value if None."""
        return self.square_length if self.square_length is not None else default
    
    def get_zero_length(self, default: int = 0) -> int:
        """Get zero_length with a default value if None."""
        return self.zero_length if self.zero_length is not None else default
    
    def get_full_pulse_length(self, square_default: int = 1500, zero_default: int = 0) -> int:
        """Get the full pulse length (square_length + zero_length) with defaults."""
        return self.get_square_length(square_default) + self.get_zero_length(zero_default)
    
    def get_slice_width_ns(self) -> int:
        """Get slice width in nanoseconds (segment_length * 4)."""
        return self.segment_length * 4
    
    def get_num_slices(self) -> int:
        """Calculate the number of slices from readout_length_in_ns and segment_length."""
        W = self.get_slice_width_ns()
        return int(self.readout_length_in_ns / W) if self.readout_length_in_ns else 0
    
    def get_actual_readout_name(self) -> str:
        """Get the actual readout name (with _custom suffix if use_custom_integration_weights is True)."""
        if self.use_custom_integration_weights:
            return f"{self.readout_name}_custom"
        return self.readout_name

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
