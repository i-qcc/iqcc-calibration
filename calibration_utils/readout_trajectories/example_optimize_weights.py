"""
Example script showing how to use the integration weights optimization utilities.

This script demonstrates how to:
1. Load trajectory data from 08e_readout_trajectories experiment
2. Extract optimal integration windows
3. Generate optimized integration weights
4. Apply them to improve IQ BLOBS readout fidelity

Usage:
    # After running 08e_readout_trajectories experiment:
    python example_optimize_weights.py --data_id <experiment_id>
    
    Or use interactively in a notebook/IDE after loading the trajectory data.
"""

import numpy as np
import xarray as xr
from typing import List, Optional

from calibration_utils.readout_trajectories.integration_weights_optimization import (
    optimize_integration_weights_from_trajectories,
    get_optimal_integration_windows_for_all_qubits,
    OptimalIntegrationWindow,
    OptimizedIntegrationWeights,
)
from qualibration_libs.parameters import get_qubits
from qualibrate import QualibrationNode


def analyze_optimal_windows(
    ds: xr.Dataset,
    qubits: List,
    W: int,
    threshold_fraction: float = 0.5,
) -> dict:
    """
    Analyze and print optimal integration windows for all qubits.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset from readout trajectories experiment.
    qubits : List
        List of qubit objects.
    W : int
        Slice width in nanoseconds.
    threshold_fraction : float
        Threshold fraction for window selection.
    
    Returns
    -------
    dict
        Dictionary mapping qubit names to OptimalIntegrationWindow objects.
    """
    windows = get_optimal_integration_windows_for_all_qubits(
        ds, qubits, W, threshold_fraction=threshold_fraction
    )
    
    print("\n" + "="*80)
    print("OPTIMAL INTEGRATION WINDOWS")
    print("="*80)
    
    for qubit_name, window in windows.items():
        print(f"\n{qubit_name}:")
        print(f"  Peak time: {window.peak_time_ns:.1f} ns")
        print(f"  Peak diff value: {window.peak_diff_value:.2e}")
        print(f"  Optimal window: {window.start_time_ns} - {window.end_time_ns} ns")
        print(f"  Window length: {window.end_time_ns - window.start_time_ns} ns")
        print(f"  Window fraction: {window.window_fraction:.1%}")
        print(f"  Threshold used: {window.diff_threshold:.2e} ({threshold_fraction*100:.0f}% of peak)")
    
    return windows


def generate_optimized_weights(
    ds: xr.Dataset,
    qubits: List,
    W: int,
    threshold_fraction: float = 0.5,
    use_time_weighting: bool = True,
    integration_weights_angle: Optional[float] = None,
) -> dict:
    """
    Generate optimized integration weights for all qubits.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset from readout trajectories experiment.
    qubits : List
        List of qubit objects.
    W : int
        Slice width in nanoseconds.
    threshold_fraction : float
        Threshold fraction for window selection.
    use_time_weighting : bool
        Whether to use time-weighted weights (True) or uniform windowed weights (False).
    integration_weights_angle : float, optional
        Rotation angle in radians. If None, uses current angle from qubit.
    
    Returns
    -------
    dict
        Dictionary mapping qubit names to OptimizedIntegrationWeights objects.
    """
    optimized_weights = {}
    
    print("\n" + "="*80)
    print("OPTIMIZED INTEGRATION WEIGHTS")
    print("="*80)
    
    for qubit in qubits:
        if qubit.name not in ds.qubit.values:
            print(f"\nSkipping {qubit.name} (not in dataset)")
            continue
        
        try:
            weights = optimize_integration_weights_from_trajectories(
                ds,
                qubit,
                W,
                threshold_fraction=threshold_fraction,
                use_time_weighting=use_time_weighting,
                integration_weights_angle=integration_weights_angle,
            )
            
            optimized_weights[qubit.name] = weights
            
            print(f"\n{qubit.name}:")
            print(f"  Total length: {weights.total_length_ns} ns")
            print(f"  Number of segments: {len(weights.cosine_weights)}")
            print(f"  Optimal window: {weights.optimal_window.start_time_ns} - "
                  f"{weights.optimal_window.end_time_ns} ns")
            
            # Show first few weight segments
            print(f"  First 3 cosine segments:")
            for i, (amp, dur) in enumerate(weights.cosine_weights[:3]):
                print(f"    [{i}] amplitude={amp:.4f}, duration={dur} ns")
            if len(weights.cosine_weights) > 3:
                print(f"    ... ({len(weights.cosine_weights) - 3} more segments)")
        
        except Exception as e:
            print(f"\nError optimizing weights for {qubit.name}: {e}")
            continue
    
    return optimized_weights


def print_weights_for_config(
    optimized_weights: dict,
    qubit_name: str,
    weight_label: str = "optimized_cosine_weights",
):
    """
    Print integration weights in format suitable for QUAM config.
    
    Parameters
    ----------
    optimized_weights : dict
        Dictionary of OptimizedIntegrationWeights objects.
    qubit_name : str
        Name of qubit to print weights for.
    weight_label : str
        Label for the weight set in config.
    """
    if qubit_name not in optimized_weights:
        print(f"No optimized weights found for {qubit_name}")
        return
    
    weights = optimized_weights[qubit_name]
    
    print(f"\n{qubit_name} - Integration weights for config:")
    print(f'"{weight_label}": {{')
    print(f'    "cosine": {weights.cosine_weights},')
    print(f'    "sine": {weights.sine_weights},')
    print("}")


# Example usage in a QualibrationNode context:
def example_usage_in_node(node: QualibrationNode):
    """
    Example of how to use the optimization utilities within a QualibrationNode.
    
    This would typically be added as a new run_action in 08e_readout_trajectories.py
    or used in a separate analysis script.
    """
    # Get the trajectory dataset
    ds = node.results["ds_raw"]
    
    # Get qubits
    qubits = node.namespace["qubits"]
    
    # Get parameters
    W = node.parameters.segment_length * 4  # Slice width in ns
    
    # Analyze optimal windows
    windows = analyze_optimal_windows(ds, qubits, W, threshold_fraction=0.5)
    
    # Generate optimized weights
    optimized_weights = generate_optimized_weights(
        ds,
        qubits,
        W,
        threshold_fraction=0.5,
        use_time_weighting=True,  # Set to False for simpler uniform weights
    )
    
    # Store results in node
    node.results["optimal_integration_windows"] = windows
    node.results["optimized_integration_weights"] = optimized_weights
    
    return windows, optimized_weights


if __name__ == "__main__":
    # Example standalone usage
    print("This is an example script.")
    print("Import the functions and use them with your trajectory dataset.")
    print("\nExample:")
    print("  from calibration_utils.readout_trajectories import optimize_integration_weights_from_trajectories")
    print("  weights = optimize_integration_weights_from_trajectories(ds, qubit, W=40)")
