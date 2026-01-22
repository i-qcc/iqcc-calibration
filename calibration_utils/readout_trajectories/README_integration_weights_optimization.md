# Integration Weights Optimization from Readout Trajectories

This module provides utilities to optimize integration weights for IQ BLOBS experiments based on readout trajectory difference plots.

## Overview

The readout trajectory experiment (`08e_readout_trajectories.py`) measures the time-dependent separation between ground and excited states. The difference plot shows when maximum distinguishability occurs. This information can be used to:

1. **Optimize integration window**: Focus integration on the time period with maximum state separation
2. **Create time-weighted weights**: Use difference values as weights to act as a matched filter
3. **Improve readout fidelity**: Better signal-to-noise ratio for state discrimination

## Quick Start

### Step 1: Run Readout Trajectories Experiment

Run the `08e_readout_trajectories.py` experiment to collect trajectory data. The experiment automatically extracts optimal integration windows in the `extract_optimal_integration_windows` run_action.

### Step 2: Access Optimal Windows

After running the experiment, access the optimal windows:

```python
# In a new script or notebook
from qualibrate import QualibrationNode
from calibration_utils.readout_trajectories import get_optimal_integration_windows_for_all_qubits

# Load the trajectory experiment results
node = QualibrationNode.load_from_id(<experiment_id>)

# Access the optimal windows
optimal_windows = node.results["optimal_integration_windows"]

# Print information for a specific qubit
qubit_name = "Q2"
window = optimal_windows[qubit_name]
print(f"Optimal window: {window.start_time_ns} - {window.end_time_ns} ns")
print(f"Peak at: {window.peak_time_ns} ns")
```

### Step 3: Generate Optimized Integration Weights

```python
from calibration_utils.readout_trajectories import optimize_integration_weights_from_trajectories

# Get qubit and dataset
qubit = node.machine.qubits["Q2"]
ds = node.results["ds_raw"]
W = node.parameters.segment_length * 4  # Slice width in ns

# Generate optimized weights
weights = optimize_integration_weights_from_trajectories(
    ds,
    qubit,
    W,
    threshold_fraction=0.5,  # Use 50% of peak as threshold
    use_time_weighting=True,  # Use time-weighted weights (False for uniform)
)

# Access the weights
print(f"Total length: {weights.total_length_ns} ns")
print(f"Cosine weights: {weights.cosine_weights}")
print(f"Sine weights: {weights.sine_weights}")
```

## Usage Examples

### Example 1: Find Optimal Window for All Qubits

```python
from calibration_utils.readout_trajectories import get_optimal_integration_windows_for_all_qubits

windows = get_optimal_integration_windows_for_all_qubits(
    ds, qubits, W=40, threshold_fraction=0.5
)

for qubit_name, window in windows.items():
    print(f"{qubit_name}: {window.start_time_ns}-{window.end_time_ns} ns")
```

### Example 2: Create Time-Weighted Integration Weights

```python
from calibration_utils.readout_trajectories import (
    calculate_trajectory_difference,
    find_optimal_integration_window,
    create_time_weighted_integration_weights,
)

# Calculate difference
time_ns, diff = calculate_trajectory_difference(ds, "Q2", W=40)

# Find optimal window
optimal_window = find_optimal_integration_window(
    time_ns, diff, threshold_fraction=0.5
)

# Create time-weighted weights
weights = create_time_weighted_integration_weights(
    time_ns, diff, optimal_window, integration_weights_angle=0.0
)
```

### Example 3: Create Simple Windowed Weights

For simpler uniform weights within the optimal window:

```python
from calibration_utils.readout_trajectories import create_windowed_integration_weights

weights = create_windowed_integration_weights(
    optimal_window,
    integration_weights_angle=0.0,  # Rotation angle from IQ BLOBS
)
```

## Applying to IQ BLOBS Experiment

### Option 1: Modify Readout Pulse Length

Use the optimal window to set the readout pulse length:

```python
optimal_window = node.results["optimal_integration_windows"]["Q2"]
optimal_length = optimal_window.end_time_ns - optimal_window.start_time_ns

# Update readout pulse length
qubit.resonator.operations["readout"].length = optimal_length
```

### Option 2: Use Custom Integration Weights

Create custom integration weights based on the optimized weights:

```python
weights = node.results["optimized_integration_weights"]["Q2"]

# Apply to QUAM config (example structure)
integration_weights_config = {
    "optimized_cosine_weights": {
        "cosine": weights.cosine_weights,
        "sine": weights.sine_weights,
    }
}
```

## Parameters

### `threshold_fraction`
- **Default**: 0.5
- **Description**: Fraction of peak difference value to use as threshold for window selection
- **Range**: 0.0 to 1.0
- **Lower values**: Include more time points (wider window)
- **Higher values**: Include fewer time points (narrower window, more selective)

### `use_time_weighting`
- **Default**: True
- **Description**: Whether to use time-weighted weights (proportional to difference values) or uniform weights
- **True**: Matched filter approach - weights proportional to distinguishability
- **False**: Simple windowed approach - uniform weights within optimal window

### `integration_weights_angle`
- **Default**: None (uses current angle from qubit)
- **Description**: Rotation angle for integration weights in radians
- **Note**: Should match the angle determined from IQ BLOBS analysis (`iw_angle`)

## Understanding the Results

### OptimalIntegrationWindow
- `start_time_ns`: Start of optimal integration window
- `end_time_ns`: End of optimal integration window  
- `peak_time_ns`: Time at which maximum distinguishability occurs
- `peak_diff_value`: Maximum difference value
- `window_fraction`: Fraction of total readout length covered
- `diff_threshold`: Threshold used (as absolute value)

### OptimizedIntegrationWeights
- `cosine_weights`: List of (amplitude, duration_ns) tuples for cosine component
- `sine_weights`: List of (amplitude, duration_ns) tuples for sine component
- `total_length_ns`: Total length of integration weights
- `optimal_window`: The OptimalIntegrationWindow used

## Tips

1. **Start with threshold_fraction=0.5**: This typically gives a good balance
2. **Check the difference plot**: Visual inspection helps validate the optimal window
3. **Compare time-weighted vs uniform**: Try both approaches and compare readout fidelity
4. **Consider readout pulse structure**: The optimal window should align with your square/zero pulse structure
5. **Re-run IQ BLOBS**: After optimizing weights, re-run IQ BLOBS to measure improvement

## Integration with Existing Workflow

The optimization utilities are automatically called in `08e_readout_trajectories.py`:

1. Run `08e_readout_trajectories.py` experiment
2. The `extract_optimal_integration_windows` run_action automatically extracts windows
3. Access results via `node.results["optimal_integration_windows"]`
4. Use these windows to optimize your IQ BLOBS integration weights

## See Also

- `08e_readout_trajectories.py`: Readout trajectory measurement experiment
- `07_iq_blobs.py`: IQ BLOBS experiment that uses integration weights
- `example_optimize_weights.py`: Example script with more usage examples
