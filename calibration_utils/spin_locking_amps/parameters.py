import numpy as np
from typing import Literal
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class SpinLockingTimeNodeParameters(RunnableParameters):
    """Common parameters for configuring spin locking time sweep in a quantum machine simulation or execution."""

    min_wait_time_in_ns: int = 16
    """Minimum wait time in nanoseconds. Default is 16."""
    max_wait_time_in_ns: int = 250000
    """Maximum wait time in nanoseconds. Default is 250000."""
    wait_time_num_points: int = 50
    """Number of points for the wait time scan. Default is 50."""
    log_or_linear_sweep: Literal["log", "linear"] = "log"
    """Type of sweep, either "log" (logarithmic) or "linear". Default is "log"."""


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1000
    min_amp_factor: float = 0.01
    """Minimum amplitude factor for the operation. Default is 0.1."""
    max_amp_factor: float = 1.0
    """Maximum amplitude factor for the operation. Default is 1.5."""
    amp_factor_step: float = 0.1
    """Step size for the amplitude factor. Default is 0.05."""

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    SpinLockingTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
