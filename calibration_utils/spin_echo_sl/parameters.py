import numpy as np
from typing import Literal
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters, IdleTimeNodeParameters


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


def get_sl_times_in_clock_cycles(
    node_parameters: SpinLockingTimeNodeParameters,
) -> np.ndarray:
    """
    Get the spin-locking times sweep axis according to the sweep type given by ``node.parameters.log_or_linear_sweep``.

    The spin locking time sweep is in units of clock cycles (4ns).
    The minimum is 4 clock cycles.
    """
    required_attributes = [
        "log_or_linear_sweep",
        "min_wait_time_in_ns",
        "max_wait_time_in_ns",
        "wait_time_num_points",
    ]
    if not all(hasattr(node_parameters, attr) for attr in required_attributes):
        raise ValueError(
            "The provided node parameter must contain the attributes 'log_or_linear_sweep', 'min_wait_time_in_ns', 'max_wait_time_in_ns' and 'wait_time_num_points'."
        )

    if node_parameters.log_or_linear_sweep == "linear":
        sl_times = _get_sl_times_linear_sweep_in_clock_cycles(node_parameters)
    elif node_parameters.log_or_linear_sweep == "log":
        sl_times = _get_sl_times_log_sweep_in_clock_cycles(node_parameters)
    else:
        raise ValueError(
            f"Expected sweep type to be 'log' or 'linear', got {node_parameters.log_or_linear_sweep}"
        )

    return sl_times


def _get_sl_times_linear_sweep_in_clock_cycles(
    node_parameters: SpinLockingTimeNodeParameters,
):
    return (
        np.linspace(
            node_parameters.min_wait_time_in_ns,
            node_parameters.max_wait_time_in_ns,
            node_parameters.wait_time_num_points,
        )
        // 4
    ).astype(int)


def _get_sl_times_log_sweep_in_clock_cycles(node_parameters: SpinLockingTimeNodeParameters):
    return np.unique(
        np.geomspace(
            node_parameters.min_wait_time_in_ns,
            node_parameters.max_wait_time_in_ns,
            node_parameters.wait_time_num_points,
        )
        // 4
    ).astype(int)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1500

class Parameters(
    NodeParameters,
    CommonNodeParameters,
    SpinLockingTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
