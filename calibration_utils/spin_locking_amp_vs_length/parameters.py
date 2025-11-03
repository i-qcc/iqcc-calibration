from typing import Literal
import numpy as np
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters, IdleTimeNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 50."""
    start_operation: Literal["x90", "-x90", "y90", "-y90"] = "-y90"
    end_operation: Literal["x90", "-x90", "y90", "-y90"] = "-y90"
    """Type of operation to perform. Default is "-y90"."""
    spin_locking_operation: Literal["x180_FlatTopTanhPulse"] = "x180_FlatTopTanhPulse"
    """Type of operation to perform. Default is "x180"."""
    min_amp_factor: float = 0.001
    """Minimum amplitude factor for the operation. Default is 0.001."""
    max_amp_factor: float = 1.0
    """Maximum amplitude factor for the operation. Default is 1.99."""
    amp_factor_step: float = 0.05
    """Step size for the amplitude factor. Default is 0.005."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass