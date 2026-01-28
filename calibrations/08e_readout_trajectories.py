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
    node.parameters.qubits = ["qB4"]
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
        
        # make temporary updates before running the program and revert at the end.
        # Now that reference is broken, we can use dont_assign_to_none=True
        with tracked_updates(resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
            readout_op_tracked = resonator.operations[f"{readout_name}"]
            # For Square_zero_ReadoutPulse, set zero_length first, then length
            # This ensures integration weights are regenerated correctly
            # Only set zero_length if the pulse type supports it (Square_zero_ReadoutPulse)
            if hasattr(readout_op_tracked, 'zero_length'):
                readout_op_tracked.zero_length = zero_len
                readout_op_tracked.length = desired_length
                # Debug: Print the actual values to verify
                print(f"{q.name} readout: length={readout_op_tracked.length}, zero_length={readout_op_tracked.zero_length}, square_length should be {readout_op_tracked.length - readout_op_tracked.zero_length}")
            else:
                # For SquareReadoutPulse (without zero_length), just set the length
                readout_op_tracked.length = desired_length
                print(f"{q.name} readout: length={readout_op_tracked.length} (no zero_length attribute)")
            
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
                        f"{readout_name}",
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
                        f"{readout_name}",
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
    fig_diff, fig_IQ_raw, fig_IQ = plot_readout_trajectories(
        node.results["ds_raw"],
        node.namespace["qubits"],
        square,
        zero,
        W,
        apply_correction=True,
    )

    # Add node info subtitles to figures
    node.add_node_info_subtitle(fig_IQ_raw)
    node.add_node_info_subtitle(fig_IQ)
    node.add_node_info_subtitle(fig_diff)

    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "IQ_raw": fig_IQ_raw,
        "IQ": fig_IQ,
        "diff_log": fig_diff,
    }
# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
# %%