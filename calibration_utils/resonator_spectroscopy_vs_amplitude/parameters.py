from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 7
    """Span of frequencies to sweep in MHz. Default is 7 MHz."""
    frequency_step_in_mhz: float = 0.025
    """Step size for frequency sweep in MHz. Default is 0.025 MHz."""
    max_power_dbm: int = -5
    """Maximum power level in dBm. Default is -5 dBm."""
    min_power_dbm: int = -40
    """Minimum power level in dBm. Default is -40 dBm."""
    num_power_points: int = 30
    """Number of points of the readout power axis. Default is 30."""
    max_amp: float = 1.0
    """Maximum readout amplitude for the experiment. Default is 1.0."""
    derivative_crossing_threshold_in_hz_per_dbm: int = -50_000
    """Threshold for derivative crossing in Hz/dBm. Default is -50000 Hz/dBm."""
    buffer_from_crossing_threshold_in_dbm: int = 1
    """Buffer from the crossing threshold in dBm - the optimal readout power will be set to be this number in dB below
    the threshold. Default is 1 dBm."""
    outlier_clip_left_mhz: float = 1.5
    """Allowed range below the resonator frequency (negative detuning) in MHz. Points with detuning < -this value
    are excluded as outliers. Default is 1.5 MHz."""
    outlier_threshold_n_steps: int = 5
    """Outlier threshold expressed as a number of frequency steps. Points deviating more than this many steps from
    the local rolling median are excluded. Default is 5."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
