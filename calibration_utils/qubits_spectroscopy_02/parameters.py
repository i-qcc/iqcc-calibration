from typing import Optional, Literal
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 500
    """Number of averages to perform. Default is 1000."""
    frequency_span_in_mhz: float = 200
    """Span of frequencies to sweep in MHz. Default is 200 MHz."""
    frequency_step_in_mhz: float = 0.25
    """Step size for frequency sweep in MHz. Default is 0.25 MHz."""
    operation: str = "saturation"
    """Type of operation to perform. Default is "saturation"."""
    operation_amplitude_factor: Optional[float] = 6
    """Amplitude pre-factor for the operation Higher power to drive 0->2 transition. Default is 7."""
    operation_len_in_ns: Optional[int] = None
    """Length of the operation in nanoseconds. Default is the predefined pulse length."""
    initial_anharmonicity_mhz: float = 200.0
    """Default anharmonicity guess in MHz. Default is 200.0 MHz."""
    target_peak_width: Optional[float] = 2e6
    """Target peak width in Hz. Default is 2e6 Hz."""
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    """Whether to use joint or independent flux points. Default is "joint"."""
    arbitrary_flux_bias: Optional[float] = None
    """Arbitrary flux bias offset. Default is None."""
    arbitrary_qubit_frequency_in_ghz: Optional[float] = None
    """Arbitrary qubit frequency in GHz. Default is None."""
    multiplexed: bool = True
    """Whether to use multiplexed readout. Default is True."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass

