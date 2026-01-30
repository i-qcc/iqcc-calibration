# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode
from iqcc_calibration_tools.quam_config.components.quam_root import Quam

from calibration_utils.readout_trajectories import (
    Parameters,
    process_raw_dataset,
    plot_readout_trajectories,
    get_optimal_integration_windows_for_all_qubits,
    optimize_integration_weights_from_trajectories,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates


# %% {Node initialisation}
description = """
       readout trajectories measurement
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="08e_basic_readout",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)

# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # When this is commented out, the experiment will use machine.active_qubits if qubits is not set in the GUI
    # To specify qubits when running through the GUI, set the 'qubits' parameter in the GUI's parameter panel
    node.parameters.qubits = ["Q2","Q4","Q6"]
    pass

# Instantiate the QUAM class from the state file
node.machine = Quam.load()

# note to self: had to run the program on one qubit cause the mismatch in the readout length caused error (try to fix later)
# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    # Handle empty list case - get_qubits checks for None or "" but not []
    if isinstance(node.parameters.qubits, list) and len(node.parameters.qubits) == 0:
        node.parameters.qubits = None
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    # Extract the sweep parameters and axes from the node parameters
    # Get num_shots from NodeSpecificParameters (inherited via Parameters)
    n_runs = node.parameters.num_shots  # Number of runs
    print(f"Number of shots (n_runs): {n_runs}")
    W = node.parameters.segment_length  # Segment length for sliced measurements
    N_slices = int(node.parameters.readout_length_in_ns/(4*W))
    print(f"Number of slices: {N_slices}")
    readout_name = node.parameters.readout_name
    
    # Determine which readout operation to use
    # If use_custom_integration_weights=True, use readout_zero_custom; otherwise use readout_zero
    if node.parameters.use_custom_integration_weights:
        actual_readout_name = f"{readout_name}_custom"
    else:
        actual_readout_name = readout_name
    
    # Store actual_readout_name in namespace for use in QUA program
    node.namespace["actual_readout_name"] = actual_readout_name
    
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["tracked_resonators"] = []
   
    for q in qubits:
        resonator = q.resonator
        readout_op = resonator.operations[f"{readout_name}"]
        
        # Calculate the desired pulse parameters
        # For Square_zero_ReadoutPulse: total length = square_length + zero_length
        # Integration weights should cover square_length (the actual readout portion)
        square_len = node.parameters.square_length if node.parameters.square_length is not None else 1500
        zero_len = node.parameters.zero_length if node.parameters.zero_length is not None else 0
        desired_length = square_len + zero_len
        
        # IMPORTANT: The integration weights must match square_length
        # QUAM should regenerate them automatically when generate_config() is called,
        # but if they're cached at 1480 ns, we may need to ensure they're regenerated
        
        # Break reference if length is a reference (need to set to None first)
        # Do this BEFORE wrapping in tracked_updates so we can set to None
        try:
            # Try to set the length directly to see if it's a reference
            readout_op.length = desired_length
        except ValueError as e:
            if "is a reference" in str(e):
                # Break the reference by setting to None first
                readout_op.length = None
                # Now set the value
                readout_op.length = desired_length
            else:
                raise
        
        # Check if readout_zero_custom exists BEFORE entering tracked_updates
        # (tracked objects don't handle 'in' operator well)
        actual_readout_name = node.namespace["actual_readout_name"]
        try:
            # Try to access the operation to see if it exists
            _ = q.resonator.operations[actual_readout_name]
            # Operation exists, use it
        except KeyError:
            # Operation doesn't exist, fall back to readout_zero if it was _custom
            if actual_readout_name.endswith("_custom"):
                print(f"{q.name}: {actual_readout_name} doesn't exist yet, using {readout_name} with defaults")
                actual_readout_name = readout_name
                node.namespace["actual_readout_name"] = actual_readout_name
        
        # make temporary updates before running the program and revert at the end.
        # Now that reference is broken, we can use dont_assign_to_none=True
        with tracked_updates(resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
            # Use the appropriate readout operation (readout_zero or readout_zero_custom)
            readout_op_tracked = resonator.operations[actual_readout_name]
            # For Square_zero_ReadoutPulse, set zero_length first, then length
            # This ensures integration weights are regenerated correctly
            readout_op_tracked.zero_length = zero_len
            readout_op_tracked.length = desired_length
            
            # Always use default integration weights for readout_zero
            # readout_zero_custom will have optimized weights set in update_state
            default_weights = [(1.0, desired_length)]
            try:
                readout_op_tracked.integration_weights = default_weights
                print(f"{q.name} {node.namespace['actual_readout_name']}: Using default integration weights (uniform, {desired_length} ns)")
            except RecursionError:
                print(f"{q.name} {node.namespace['actual_readout_name']}: Recursion error setting default weights, skipping (QUAM will use its defaults)")
            except Exception as e:
                print(f"{q.name} {node.namespace['actual_readout_name']}: Error setting default weights: {type(e).__name__}: {e}")
            
            # Debug: Print the actual values to verify
            print(f"{q.name} readout_zero: length={readout_op_tracked.length}, zero_length={readout_op_tracked.zero_length}, square_length={readout_op_tracked.length - readout_op_tracked.zero_length}")
            
            node.namespace["tracked_resonators"].append(resonator)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_runs": xr.DataArray( np.arange(0, n_runs, 1), attrs={"long_name": "number of shots"}),
        "readout_time": xr.DataArray(
            np.arange(0, N_slices, 1),
            attrs={"long_name": "readout time", "units": "ns"},
        ),
    }
    
    with program() as node.namespace["qua_program"]:
        # Declare only the variables we need (n and n_st for shot counting)
        # We don't use I, I_st, Q, Q_st from declare_qua_variables since we use custom arrays for sliced measurements
        n = declare(int)
        n_st = declare_stream()
        
        # Declare arrays for sliced measurements per qubit
        IQg = [declare(fixed, size=N_slices) for _ in range(num_qubits)]
        QIg = [declare(fixed, size=N_slices) for _ in range(num_qubits)]
        IQe = [declare(fixed, size=N_slices) for _ in range(num_qubits)]
        QIe = [declare(fixed, size=N_slices) for _ in range(num_qubits)]
        Ig_slices = [declare(fixed, size=N_slices) for _ in range(num_qubits)]
        Qg_slices = [declare(fixed, size=N_slices) for _ in range(num_qubits)]
        Ie_slices = [declare(fixed, size=N_slices) for _ in range(num_qubits)]
        Qe_slices = [declare(fixed, size=N_slices) for _ in range(num_qubits)]
        
        # Declare streams per qubit
        Ig_st = [declare_stream() for _ in range(num_qubits)]
        Qg_st = [declare_stream() for _ in range(num_qubits)]
        IQg_st = [declare_stream() for _ in range(num_qubits)]
        QIg_st = [declare_stream() for _ in range(num_qubits)]
        Ie_st = [declare_stream() for _ in range(num_qubits)]
        Qe_st = [declare_stream() for _ in range(num_qubits)]
        IQe_st = [declare_stream() for _ in range(num_qubits)]
        QIe_st = [declare_stream() for _ in range(num_qubits)]
        
        k = declare(int)
        df = declare(int)  # QUA variable for the readout frequency
        amp = declare(fixed)
        assign(amp, 1.0)
        
        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()
            
            # assign(df, 0) If you need to move the readout frequency
            
            # Excited state measurements
            with for_(n, 0, n < n_runs, n + 1):
                save(n, n_st)
                
                # Qubit initialization
                for i, qubit in multiplexed_qubits.items():
                    qubit.reset("thermal", node.parameters.simulate)
                align()
                
                # Qubit excitation and readout
                for i, qubit in multiplexed_qubits.items():
                    qubit.xy.play("x180")
                    qubit.align()
                    rr = qubit.resonator
                    # rr.update_frequency(df + rr.intermediate_frequency) If you need to move the readout frequency
                    rr.measure_sliced(
                        node.namespace["actual_readout_name"],
                        stream=None,
                        qua_vars=(Ie_slices[i], IQe[i], QIe[i], Qe_slices[i]),
                        segment_length=W,
                        amplitude_scale=amp
                    )
                    # Save each slice
                    with for_(k, 0, k < N_slices, k + 1):
                        save(Ie_slices[i][k], Ie_st[i])
                        save(Qe_slices[i][k], Qe_st[i])
                        save(IQe[i][k], IQe_st[i])
                        save(QIe[i][k], QIe_st[i])
                    rr.wait(rr.depletion_time * u.ns)
                align()
            
            # Ground state measurements
            with for_(n, 0, n < n_runs, n + 1):
                save(n, n_st)
                
                # Qubit initialization
                for i, qubit in multiplexed_qubits.items():
                    qubit.reset("thermal", node.parameters.simulate)
                align()
                
                # Qubit readout
                for i, qubit in multiplexed_qubits.items():
                    rr = qubit.resonator
                    rr.update_frequency(df + rr.intermediate_frequency)
                    rr.measure_sliced(
                        node.namespace["actual_readout_name"],
                        stream=None,
                        qua_vars=(Ig_slices[i], IQg[i], QIg[i], Qg_slices[i]),
                        segment_length=W,
                        amplitude_scale=amp
                    )
                    # Save each slice
                    with for_(k, 0, k < N_slices, k + 1):
                        save(Ig_slices[i][k], Ig_st[i])
                        save(Qg_slices[i][k], Qg_st[i])
                        save(IQg[i][k], IQg_st[i])
                        save(QIg[i][k], QIg_st[i])
                    rr.wait(rr.depletion_time * u.ns)
                align()
        
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                Ie = (Ie_st[i] + IQe_st[i]).buffer(n_runs, N_slices)
                Ie.save(f"Ie{i + 1}")
                Qe = (Qe_st[i] + QIe_st[i]).buffer(n_runs, N_slices)
                Qe.save(f"Qe{i + 1}")
                Ig = (Ig_st[i] + IQg_st[i]).buffer(n_runs, N_slices)
                Ig.save(f"Ig{i + 1}")
                Qg = (Qg_st[i] + QIg_st[i]).buffer(n_runs, N_slices)
                Qg.save(f"Qg{i + 1}")
                 

# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Fetch the data
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            pass
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
  

       
# %% {Plotting}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    square = node.parameters.square_length
    zero = node.parameters.zero_length
    W = node.parameters.segment_length * 4  # Slice width in nanoseconds
    
    # Plot all readout trajectory figures
    fig_diff, fig_IQ_raw = plot_readout_trajectories(
        node.results["ds_raw"],
        node.namespace["qubits"],
        square,
        zero,
        W,
    )

    # Add node info subtitles to figures
    node.add_node_info_subtitle(fig_IQ_raw)
    node.add_node_info_subtitle(fig_diff)

    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "IQ_raw": fig_IQ_raw,
        "diff_log": fig_diff,
    }


# %% {Extract_optimal_integration_windows}

@node.run_action(skip_if=node.parameters.simulate)
def extract_optimal_integration_windows(node: QualibrationNode[Parameters, Quam]):
    """
    Extract optimal integration windows from the difference plot data.
    
    This analysis identifies the time window where state distinguishability is maximum,
    which can be used to optimize integration weights in IQ BLOBS experiments.
    """
    from dataclasses import asdict
    
    W = node.parameters.segment_length * 4  # Slice width in nanoseconds
    # Get optimal integration windows for all qubits
    optimal_windows = get_optimal_integration_windows_for_all_qubits(
        node.results["ds_raw"],
        node.namespace["qubits"],
        W,
        threshold_fraction=0.5,  # Use 50% of peak as threshold
    )
    
    # Log the results
    node.log("\n" + "="*80)
    node.log("OPTIMAL INTEGRATION WINDOWS FOR IQ BLOBS OPTIMIZATION")
    node.log("="*80)
    
    # Convert dataclasses to dicts for JSON serialization
    optimal_windows_dict = {}
    for qubit_name, window in optimal_windows.items():
        node.log(f"\n{qubit_name}:")
        node.log(f"  Peak distinguishability at: {window.peak_time_ns:.1f} ns")
        node.log(f"  Optimal integration window: {window.start_time_ns} - {window.end_time_ns} ns")
        node.log(f"  Window length: {window.end_time_ns - window.start_time_ns} ns")
        node.log(f"  Covers {window.window_fraction:.1%} of total readout length")
        node.log(f"  Peak diff value: {window.peak_diff_value:.2e}")
        
        # Convert to dict for JSON serialization
        optimal_windows_dict[qubit_name] = asdict(window)
    
    # Store in results for later use (as dicts for JSON serialization)
    node.results["optimal_integration_windows"] = optimal_windows_dict
    
    # Generate optimized integration weights
    optimized_weights_dict = {}
    for qubit in node.namespace["qubits"]:
        if qubit.name in optimal_windows:
            try:
                weights = optimize_integration_weights_from_trajectories(
                    node.results["ds_raw"],
                    qubit,
                    W,
                    threshold_fraction=0.5,
                    use_time_weighting=False,  # Use uniform windowed weights for simplicity
                )
                # Convert to dict for JSON serialization
                optimized_weights_dict[qubit.name] = {
                    "cosine_weights": weights.cosine_weights,
                    "sine_weights": weights.sine_weights,
                    "total_length_ns": weights.total_length_ns,
                    "optimal_window": asdict(weights.optimal_window),
                }
                node.log(f"\n{qubit.name}: Generated optimized integration weights")
                node.log(f"  Total length: {weights.total_length_ns} ns")
                node.log(f"  Number of segments: {len(weights.cosine_weights)}")
            except Exception as e:
                node.log(f"Warning: Could not generate optimized weights for {qubit.name}: {e}")
    
    node.results["optimized_integration_weights"] = optimized_weights_dict
    
    node.log("\n" + "="*80)
    node.log("Optimized integration weights generated and stored in node.results")
    node.log("="*80 + "\n")


# %% {Update_state}

@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """
    Create/update readout_zero_custom operation with optimized integration weights.
    
    This creates a new operation called readout_zero_custom (or updates it if it exists)
    with the optimized integration weights determined from the trajectory difference analysis.
    The original readout_zero operation remains unchanged with default integration weights.
    
    The integration weights are stored as a list of (amplitude, duration) tuples
    representing the cosine component. The rotation is handled via integration_weights_angle.
    
    Other experiments can use readout_zero_custom to benefit from the optimization.
    """
    if "optimized_integration_weights" not in node.results:
        node.log("Warning: No optimized integration weights found. Skipping state update.")
        return
    
    optimized_weights = node.results["optimized_integration_weights"]
    readout_name = node.parameters.readout_name  # This is "readout_zero"
    custom_readout_name = f"{readout_name}_custom"  # This is "readout_zero_custom"
    
    from quam.components.pulses import Square_zero_ReadoutPulse
    from copy import deepcopy
    
    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            if qubit.name not in optimized_weights:
                continue
            
            try:
                weights_data = optimized_weights[qubit.name]
                
                # Get the original readout_zero operation to copy its properties
                readout_zero_op = qubit.resonator.operations[readout_name]
                
                # IMPORTANT: Set default integration_weights for readout_zero
                # This ensures readout_zero always uses default uniform weights
                # Get pulse length for default weights
                square_len = node.parameters.square_length if node.parameters.square_length is not None else 1400
                zero_len = node.parameters.zero_length if node.parameters.zero_length is not None else 1000
                full_pulse_length = square_len + zero_len
                default_weights = [(1.0, full_pulse_length)]
                
                try:
                    current_iw = readout_zero_op.integration_weights
                    # Check if it's already default weights
                    if isinstance(current_iw, list) and len(current_iw) == 1:
                        amp, dur = current_iw[0] if isinstance(current_iw[0], (list, tuple)) else (current_iw[0], current_iw[1])
                        if amp == 1.0 and dur == full_pulse_length:
                            # Already has default weights, skip
                            node.log(f"{qubit.name}.{readout_name} already has default integration weights")
                        else:
                            # Has custom weights, replace with defaults
                            readout_zero_op.integration_weights = default_weights
                            node.log(f"Set default integration_weights for {qubit.name}.{readout_name} (replaced custom weights)")
                    elif isinstance(current_iw, str) and current_iw.startswith("#"):
                        # It's a reference, break it and set defaults
                        readout_zero_op.integration_weights = default_weights
                        node.log(f"Set default integration_weights for {qubit.name}.{readout_name} (broke reference)")
                    elif current_iw is None:
                        # Already None, set defaults
                        readout_zero_op.integration_weights = default_weights
                        node.log(f"Set default integration_weights for {qubit.name}.{readout_name}")
                    else:
                        # Has custom weights, replace with defaults
                        readout_zero_op.integration_weights = default_weights
                        node.log(f"Set default integration_weights for {qubit.name}.{readout_name} (replaced custom weights)")
                except Exception as e:
                    node.log(f"Warning: Could not set default integration_weights for {qubit.name}.{readout_name}: {e}")
                    # Try to set defaults anyway
                    try:
                        readout_zero_op.integration_weights = default_weights
                    except:
                        pass
                    # Continue anyway - we'll still create readout_zero_custom
                
                # Check if readout_zero_custom already exists
                if custom_readout_name in qubit.resonator.operations:
                    # Update existing readout_zero_custom
                    readout_custom_op = qubit.resonator.operations[custom_readout_name]
                    node.log(f"Updating existing {qubit.name}.{custom_readout_name} operation")
                else:
                    # Create new readout_zero_custom by copying readout_zero
                    node.log(f"Creating new {qubit.name}.{custom_readout_name} operation from {readout_name}")
                    
                    # Get pulse parameters from experiment (not from readout_zero to ensure consistency)
                    square_len = node.parameters.square_length if node.parameters.square_length is not None else 1400
                    zero_len = node.parameters.zero_length if node.parameters.zero_length is not None else 1000
                    full_pulse_length = square_len + zero_len
                    
                    # Copy properties from readout_zero, but use experiment parameters for length/zero_length
                    readout_custom_op = Square_zero_ReadoutPulse(
                        amplitude=readout_zero_op.amplitude,
                        length=full_pulse_length,  # Use experiment parameter
                        zero_length=zero_len,  # Use experiment parameter
                        integration_weights_angle=readout_zero_op.integration_weights_angle,
                        digital_marker=deepcopy(readout_zero_op.digital_marker) if hasattr(readout_zero_op, 'digital_marker') else None,
                    )
                    
                    # Assign to operations dict
                    qubit.resonator.operations[custom_readout_name] = readout_custom_op
                
                # Get the full pulse length (square_length + zero_length) from experiment parameters
                square_len = node.parameters.square_length if node.parameters.square_length is not None else 1400
                zero_len = node.parameters.zero_length if node.parameters.zero_length is not None else 1000
                full_pulse_length = square_len + zero_len
                
                # Ensure readout_zero_custom has the correct length and zero_length
                # Update them if they don't match the experiment parameters
                if readout_custom_op.length != full_pulse_length:
                    readout_custom_op.length = full_pulse_length
                if readout_custom_op.zero_length != zero_len:
                    readout_custom_op.zero_length = zero_len
                
                # Get the optimal window information
                optimal_window = weights_data["optimal_window"]
                window_start = optimal_window["start_time_ns"]
                window_end = optimal_window["end_time_ns"]
                window_duration = window_end - window_start
                
                # QUAM stores integration_weights as a list of (amplitude, duration) tuples
                # Format: [(amplitude1, duration1), (amplitude2, duration2), ...]
                cosine_weights = weights_data["cosine_weights"]
                
                # Convert to list of tuples as expected by QUAM
                optimized_weights_list = []
                for amp, dur in cosine_weights:
                    optimized_weights_list.append((float(amp), int(dur)))
                
                # Calculate padding needed to match full pulse length
                optimized_total_duration = sum(dur for _, dur in optimized_weights_list)
                padding_before = window_start
                padding_after = full_pulse_length - window_end
                
                # Create padded integration weights: zeros before, optimized window, zeros after
                padded_integration_weights = []
                
                # Add zeros before the optimized window
                if padding_before > 0:
                    padded_integration_weights.append((0.0, int(padding_before)))
                
                # Add the optimized weights
                padded_integration_weights.extend(optimized_weights_list)
                
                # Add zeros after the optimized window
                if padding_after > 0:
                    padded_integration_weights.append((0.0, int(padding_after)))
                
                # Verify total duration matches pulse length
                total_duration = sum(dur for _, dur in padded_integration_weights)
                if total_duration != full_pulse_length:
                    # Adjust the last segment if there's a small mismatch due to rounding
                    if abs(total_duration - full_pulse_length) <= 4:  # Allow 1 sample (4ns) tolerance
                        if padded_integration_weights:
                            last_amp, last_dur = padded_integration_weights[-1]
                            padded_integration_weights[-1] = (last_amp, last_dur + (full_pulse_length - total_duration))
                    else:
                        node.log(f"⚠ Warning: Total duration mismatch for {qubit.name}: {total_duration} ns vs {full_pulse_length} ns")
                
                # Update readout_zero_custom with optimized integration weights
                # Break reference if integration_weights is a reference (like "#./default_integration_weights")
                # Similar to how we handle length references
                try:
                    # Check if it's currently a reference
                    current_iw = readout_custom_op.integration_weights
                    if isinstance(current_iw, str) and current_iw.startswith("#"):
                        # It's a reference, break it first
                        readout_custom_op.integration_weights = None
                    # Set the actual value (padded to full pulse length)
                    readout_custom_op.integration_weights = padded_integration_weights
                except (ValueError, TypeError, AttributeError) as e:
                    # If direct assignment fails, try breaking reference first
                    try:
                        readout_custom_op.integration_weights = None
                        readout_custom_op.integration_weights = padded_integration_weights
                    except Exception as e2:
                        raise Exception(f"Failed to set integration_weights: {e2}") from e
                
                # Verify the assignment worked
                actual_iw = readout_custom_op.integration_weights
                # Compare values (handling both list and tuple formats)
                if isinstance(actual_iw, list):
                    actual_iw_tuples = [tuple(x) if isinstance(x, (list, tuple)) else x for x in actual_iw]
                    expected_tuples = padded_integration_weights
                    if actual_iw_tuples == expected_tuples:
                        success = True
                    else:
                        success = False
                else:
                    success = (actual_iw == padded_integration_weights)
                
                if success:
                    node.log(f"✓ Successfully updated integration weights for {qubit.name}.{custom_readout_name}:")
                    node.log(f"  Full pulse length: {full_pulse_length} ns")
                    node.log(f"  Optimal window: {window_start} - {window_end} ns ({window_duration} ns)")
                    node.log(f"  Padding: {padding_before} ns before, {padding_after} ns after")
                    node.log(f"  Total segments: {len(padded_integration_weights)}")
                    node.log(f"  Optimized segments: {len(optimized_weights_list)}")
                    node.log(f"  {readout_name} has been cleaned (integration_weights removed, will use defaults)")
                    node.log(f"  {custom_readout_name} has optimized weights for use in other experiments")
                else:
                    node.log(f"⚠ Warning: Integration weights assignment verification failed for {qubit.name}")
                    node.log(f"  Expected: {padded_integration_weights}")
                    node.log(f"  Got: {actual_iw} (type: {type(actual_iw)})")
                
            except Exception as e:
                node.log(f"Error updating integration weights for {qubit.name}: {e}")
                import traceback
                node.log(traceback.format_exc())
                continue


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
# %%