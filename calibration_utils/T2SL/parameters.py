from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters, IdleTimeNodeParameters
from typing import List
from dataclasses import field

class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 200
    # amplitude_scales: List[float] = field(default_factory=lambda: [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5])
    # amplitude_scales: List[float] = field(default_factory=lambda: [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    amplitude_scales: List[float] = field(default_factory=lambda: [0.1,0.2,0.3,0.4,0.5,0.6])
    frequency_span_in_mhz: float = 10
    """Span of frequencies to sweep in MHz. Default is 100 MHz."""
    frequency_step_in_mhz: float = 1
    """Step size for frequency sweep in MHz. Default is 0.25 MHz."""

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
