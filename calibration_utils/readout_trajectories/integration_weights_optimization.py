"""
Utility functions to optimize integration weights based on readout trajectory difference plots.

This module extracts the time-dependent distinguishability from readout trajectory measurements
and uses it to optimize integration weights for improved readout fidelity in IQ BLOBS experiments.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import xarray as xr
from quam_builder.architecture.superconducting.qubit import AnyTransmon


@dataclass
class OptimalIntegrationWindow:
    """Stores the optimal integration window parameters for a qubit."""
    
    start_time_ns: int
    """Start time of optimal integration window in nanoseconds."""
    
    end_time_ns: int
    """End time of optimal integration window in nanoseconds."""
    
    peak_time_ns: float
    """Time at which maximum distinguishability occurs in nanoseconds."""
    
    peak_diff_value: float
    """Maximum difference value at peak."""
    
    window_fraction: float
    """Fraction of total readout length covered by optimal window."""
    
    diff_threshold: float
    """Difference threshold used to determine window (as fraction of peak)."""


@dataclass
class OptimizedIntegrationWeights:
    """Stores optimized integration weights for a qubit."""
    
    cosine_weights: List[Tuple[float, int]]
    """List of (amplitude, duration_ns) tuples for cosine component."""
    
    sine_weights: List[Tuple[float, int]]
    """List of (amplitude, duration_ns) tuples for sine component."""
    
    total_length_ns: int
    """Total length of integration weights in nanoseconds."""
    
    optimal_window: OptimalIntegrationWindow
    """The optimal integration window used to generate these weights."""


def calculate_trajectory_difference(
    ds: xr.Dataset,
    qubit_name: str,
    W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the time-dependent difference between excited and ground state trajectories.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing Ie, Qe, Ig, Qg data with readout_time dimension.
    qubit_name : str
        Name of the qubit to analyze.
    W : int
        Slice width in nanoseconds (segment_length * 4).
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (time_array_ns, difference_array) where:
        - time_array_ns: Time points in nanoseconds
        - difference_array: (Ie - Ig)^2 + (Qe - Qg)^2 at each time point
    """
    if qubit_name not in ds.qubit.values:
        raise ValueError(f"Qubit {qubit_name} not found in dataset. Available qubits: {list(ds.qubit.values)}")
    
    ds_qubit = ds.sel(qubit=qubit_name)
    
    # Extract I and Q arrays
    Ie = ds_qubit["Ie"].values
    Qe = ds_qubit["Qe"].values
    Ig = ds_qubit["Ig"].values
    Qg = ds_qubit["Qg"].values
    
    # Handle different data shapes
    if Ie.ndim == 3:
        Ie = Ie[:, :, 0]
        Qe = Qe[:, :, 0]
        Ig = Ig[:, :, 0]
        Qg = Qg[:, :, 0]
    
    # Calculate averages over shots
    Ie_avg = np.mean(Ie, axis=0)
    Qe_avg = np.mean(Qe, axis=0)
    Ig_avg = np.mean(Ig, axis=0)
    Qg_avg = np.mean(Qg, axis=0)
    
    # Calculate difference: (Ie - Ig)^2 + (Qe - Qg)^2
    diff = (Ie_avg - Ig_avg) ** 2 + (Qe_avg - Qg_avg) ** 2
    
    # Create time array
    time_ns = np.arange(0, len(diff) * W, W)
    
    return time_ns, diff


def find_optimal_integration_window(
    time_ns: np.ndarray,
    diff: np.ndarray,
    threshold_fraction: float = 0.5,
    min_window_fraction: float = 0.2,
    max_window_fraction: float = 0.8,
) -> OptimalIntegrationWindow:
    """
    Find the optimal integration window based on difference plot.
    
    Parameters
    ----------
    time_ns : np.ndarray
        Time array in nanoseconds.
    diff : np.ndarray
        Difference values (Ie - Ig)^2 + (Qe - Qg)^2.
    threshold_fraction : float, optional
        Fraction of peak difference to use as threshold for window selection (default: 0.5).
        Values above this threshold will be included in the optimal window.
    min_window_fraction : float, optional
        Minimum fraction of total readout length for the window (default: 0.2).
    max_window_fraction : float, optional
        Maximum fraction of total readout length for the window (default: 0.8).
    
    Returns
    -------
    OptimalIntegrationWindow
        Object containing optimal integration window parameters.
    """
    # Find peak
    peak_idx = np.argmax(diff)
    peak_time_ns = float(time_ns[peak_idx])
    peak_diff_value = float(diff[peak_idx])
    
    # Calculate threshold
    diff_threshold = peak_diff_value * threshold_fraction
    
    # Find indices where diff exceeds threshold
    above_threshold = diff >= diff_threshold
    
    if not np.any(above_threshold):
        # If no points above threshold, use peak region
        window_size = int(len(diff) * 0.3)  # Use 30% around peak
        start_idx = max(0, peak_idx - window_size // 2)
        end_idx = min(len(diff), peak_idx + window_size // 2)
    else:
        # Find continuous region around peak
        start_idx = np.where(above_threshold)[0][0]
        end_idx = np.where(above_threshold)[0][-1] + 1
        
        # Enforce minimum window size
        min_window_size = int(len(diff) * min_window_fraction)
        current_size = end_idx - start_idx
        if current_size < min_window_size:
            # Expand symmetrically around peak
            expansion = (min_window_size - current_size) // 2
            start_idx = max(0, start_idx - expansion)
            end_idx = min(len(diff), end_idx + expansion)
        
        # Enforce maximum window size
        max_window_size = int(len(diff) * max_window_fraction)
        if end_idx - start_idx > max_window_size:
            # Center around peak
            start_idx = max(0, peak_idx - max_window_size // 2)
            end_idx = min(len(diff), peak_idx + max_window_size // 2)
    
    start_time_ns = int(time_ns[start_idx])
    end_time_ns = int(time_ns[end_idx - 1] + (time_ns[1] - time_ns[0]))  # Include last point
    
    total_length = time_ns[-1] + (time_ns[1] - time_ns[0])
    window_fraction = (end_time_ns - start_time_ns) / total_length
    
    return OptimalIntegrationWindow(
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        peak_time_ns=peak_time_ns,
        peak_diff_value=peak_diff_value,
        window_fraction=window_fraction,
        diff_threshold=diff_threshold,
    )


def create_time_weighted_integration_weights(
    time_ns: np.ndarray,
    diff: np.ndarray,
    optimal_window: OptimalIntegrationWindow,
    integration_weights_angle: float = 0.0,
    sample_rate_ns: int = 4,
) -> OptimizedIntegrationWeights:
    """
    Create time-weighted integration weights based on difference values.
    
    The weights are proportional to the difference values, acting as a matched filter
    to maximize signal-to-noise ratio for state discrimination.
    
    Parameters
    ----------
    time_ns : np.ndarray
        Time array in nanoseconds.
    diff : np.ndarray
        Difference values (Ie - Ig)^2 + (Qe - Qg)^2.
    optimal_window : OptimalIntegrationWindow
        Optimal integration window to use.
    integration_weights_angle : float, optional
        Rotation angle for integration weights in radians (default: 0.0).
        This should match the angle determined from IQ BLOBS analysis.
    sample_rate_ns : int, optional
        Sample rate in nanoseconds (default: 4, which is 1 sample per clock cycle).
    
    Returns
    -------
    OptimizedIntegrationWeights
        Object containing optimized cosine and sine integration weights.
    """
    # Find indices corresponding to optimal window
    start_idx = np.argmin(np.abs(time_ns - optimal_window.start_time_ns))
    end_idx = np.argmin(np.abs(time_ns - optimal_window.end_time_ns)) + 1
    
    # Extract difference values in optimal window
    window_diff = diff[start_idx:end_idx]
    window_time = time_ns[start_idx:end_idx]
    
    # Normalize difference values to create weights (sum to 1 for proper scaling)
    # Add small epsilon to avoid division by zero
    eps = 1e-12
    normalized_weights = (window_diff + eps) / (np.sum(window_diff) + eps * len(window_diff))
    
    # Calculate cosine and sine components with rotation
    cos_angle = np.cos(integration_weights_angle)
    sin_angle = np.sin(integration_weights_angle)
    
    # Create weight segments
    # Group consecutive samples with similar weights to reduce number of segments
    cosine_weights = []
    sine_weights = []
    
    # Convert to sample-based weights (4 ns per sample)
    current_cos_weight = normalized_weights[0] * cos_angle
    current_sin_weight = normalized_weights[0] * sin_angle
    current_duration = sample_rate_ns
    
    for i in range(1, len(normalized_weights)):
        # Calculate time step
        dt = window_time[i] - window_time[i-1]
        samples = int(round(dt / sample_rate_ns))
        
        # New weights
        new_cos_weight = normalized_weights[i] * cos_angle
        new_sin_weight = normalized_weights[i] * sin_angle
        
        # If weights are similar, accumulate duration
        weight_tolerance = 0.01  # 1% tolerance
        if (abs(new_cos_weight - current_cos_weight) < weight_tolerance and
            abs(new_sin_weight - current_sin_weight) < weight_tolerance):
            current_duration += samples * sample_rate_ns
        else:
            # Save current segment
            if current_duration > 0:
                cosine_weights.append((float(current_cos_weight), current_duration))
                sine_weights.append((float(current_sin_weight), current_duration))
            
            # Start new segment
            current_cos_weight = new_cos_weight
            current_sin_weight = new_sin_weight
            current_duration = samples * sample_rate_ns
    
    # Save last segment
    if current_duration > 0:
        cosine_weights.append((float(current_cos_weight), current_duration))
        sine_weights.append((float(current_sin_weight), current_duration))
    
    total_length_ns = int(np.sum([w[1] for w in cosine_weights]))
    
    return OptimizedIntegrationWeights(
        cosine_weights=cosine_weights,
        sine_weights=sine_weights,
        total_length_ns=total_length_ns,
        optimal_window=optimal_window,
    )


def create_windowed_integration_weights(
    optimal_window: OptimalIntegrationWindow,
    integration_weights_angle: float = 0.0,
    sample_rate_ns: int = 4,
) -> OptimizedIntegrationWeights:
    """
    Create simple windowed integration weights (uniform within optimal window).
    
    This is simpler than time-weighted weights and may be sufficient if the
    difference is relatively flat within the optimal window.
    
    Parameters
    ----------
    optimal_window : OptimalIntegrationWindow
        Optimal integration window to use.
    integration_weights_angle : float, optional
        Rotation angle for integration weights in radians (default: 0.0).
    sample_rate_ns : int, optional
        Sample rate in nanoseconds (default: 4).
    
    Returns
    -------
    OptimizedIntegrationWeights
        Object containing optimized cosine and sine integration weights.
    """
    window_length_ns = optimal_window.end_time_ns - optimal_window.start_time_ns
    
    # Round to nearest sample
    window_length_samples = int(round(window_length_ns / sample_rate_ns))
    window_length_ns = window_length_samples * sample_rate_ns
    
    # Create uniform weights within window
    cos_angle = np.cos(integration_weights_angle)
    sin_angle = np.sin(integration_weights_angle)
    
    # Normalize so total weight equals window length (for proper scaling)
    weight_amplitude = 1.0
    
    cosine_weights = [(float(weight_amplitude * cos_angle), window_length_ns)]
    sine_weights = [(float(weight_amplitude * sin_angle), window_length_ns)]
    
    return OptimizedIntegrationWeights(
        cosine_weights=cosine_weights,
        sine_weights=sine_weights,
        total_length_ns=window_length_ns,
        optimal_window=optimal_window,
    )


def optimize_integration_weights_from_trajectories(
    ds: xr.Dataset,
    qubit: AnyTransmon,
    W: int,
    threshold_fraction: float = 0.5,
    use_time_weighting: bool = True,
    integration_weights_angle: Optional[float] = None,
) -> OptimizedIntegrationWeights:
    """
    Complete workflow to optimize integration weights from trajectory data.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset from readout trajectories experiment (08e_readout_trajectories).
    qubit : AnyTransmon
        Qubit object to optimize weights for.
    W : int
        Slice width in nanoseconds (segment_length * 4).
    threshold_fraction : float, optional
        Fraction of peak difference for window threshold (default: 0.5).
    use_time_weighting : bool, optional
        If True, use time-weighted weights; if False, use uniform windowed weights (default: True).
    integration_weights_angle : float, optional
        Rotation angle in radians. If None, uses current angle from qubit (default: None).
    
    Returns
    -------
    OptimizedIntegrationWeights
        Optimized integration weights for the qubit.
    """
    qubit_name = qubit.name
    
    # Calculate difference
    time_ns, diff = calculate_trajectory_difference(ds, qubit_name, W)
    
    # Find optimal window
    optimal_window = find_optimal_integration_window(
        time_ns, diff, threshold_fraction=threshold_fraction
    )
    
    # Get integration weights angle
    if integration_weights_angle is None:
        # Try to get from qubit's readout operation
        try:
            readout_op = qubit.resonator.operations.get("readout")
            if readout_op and hasattr(readout_op, 'integration_weights_angle'):
                integration_weights_angle = float(readout_op.integration_weights_angle)
            else:
                integration_weights_angle = 0.0
        except:
            integration_weights_angle = 0.0
    
    # Create optimized weights
    if use_time_weighting:
        weights = create_time_weighted_integration_weights(
            time_ns, diff, optimal_window, integration_weights_angle
        )
    else:
        weights = create_windowed_integration_weights(
            optimal_window, integration_weights_angle
        )
    
    return weights


def get_optimal_integration_windows_for_all_qubits(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    W: int,
    threshold_fraction: float = 0.5,
) -> Dict[str, OptimalIntegrationWindow]:
    """
    Get optimal integration windows for all qubits in the dataset.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset from readout trajectories experiment.
    qubits : List[AnyTransmon]
        List of qubits to analyze.
    W : int
        Slice width in nanoseconds.
    threshold_fraction : float, optional
        Fraction of peak difference for window threshold (default: 0.5).
    
    Returns
    -------
    Dict[str, OptimalIntegrationWindow]
        Dictionary mapping qubit names to their optimal integration windows.
    """
    windows = {}
    
    for qubit in qubits:
        if qubit.name not in ds.qubit.values:
            continue
        
        time_ns, diff = calculate_trajectory_difference(ds, qubit.name, W)
        optimal_window = find_optimal_integration_window(
            time_ns, diff, threshold_fraction=threshold_fraction
        )
        windows[qubit.name] = optimal_window
    
    return windows
