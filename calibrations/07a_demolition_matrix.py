# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam as Quam
from iqcc_calibration_tools.quam_config.macros import readout_state
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from calibration_utils.demolition_matrix import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_confusion_and_demolition_matrices,
)


# %% {Description}
description = """
        MEASUREMENT CHARACTERIZATION WITH DEPLETION TIME OPTIMIZATION
This sequence measures the confusion matrix and demolition effect of each measurement using state discrimination,
while optimizing the depletion_time parameter.
The sequence involves:
    1. Sweeping over depletion_time values
    2. For each depletion_time:
        a. Preparing the qubit in the ground state |g>, measuring, waiting depletion_time, measuring again, waiting depletion_time
        b. Preparing the qubit in the excited state |e> (via x180 pulse), measuring, waiting depletion_time, measuring again, waiting depletion_time
    
All measurements use state discrimination (unlike IQ blobs which save I/Q values).

The data is processed to determine:
    - The 1Q confusion matrix: P(measured state | prepared state)
    - The 1Q demolition error matrix: P(measured state in second measurement | measured state in first measurement)
      This quantifies how often a measurement result is repeated in consecutive measurements.
    - The optimal depletion_time that maximizes P(1|1) (minimizes demolition of the 1 state)

Prerequisites:
    - Having calibrated the readout parameters (nodes 02a, 02b and/or 02c).
    - Having calibrated the qubit x180 pulse parameters (nodes 03a_qubit_spectroscopy.py and 04b_power_rabi.py).
    - Having calibrated state discrimination thresholds (node 07_iq_blobs.py).

State update:
    - The confusion matrix: qubit.resonator.operations["readout"].confusion_matrix
    - The optimal depletion_time: qubit.resonator.depletion_time
    - The demolition error matrix: stored in node results
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="07a_demolition_matrix",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """
    Allow the user to locally set the node parameters for debugging purposes, or
    execution in the Python IDE.
    """
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["qB2", "qB4"]
    node.parameters.num_shots = 2000
    node.parameters.start_depletion_time = 1000
    node.parameters.end_depletion_time = 10000
    node.parameters.num_depletion_times = 10
    node.parameters.num_of_measurement = 5
    node.parameters.multiplexed = False
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Create the sweep axes and generate the QUA program from the pulse sequence and the
    node parameters.
    """
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_runs = node.parameters.num_shots  # Number of runs
    operation = node.parameters.operation
    # Create evenly spaced integer array for QUA from_array
    # QUA requires evenly spaced values, so we use np.arange with integer step
    if node.parameters.num_depletion_times > 1:
        # Calculate integer step size to ensure even spacing
        # Use integer division to get an integer step
        step = (node.parameters.end_depletion_time - node.parameters.start_depletion_time) // 4 //(
            node.parameters.num_depletion_times - 1
        )
        # Generate evenly spaced integers using np.arange
        depletion_times = np.arange(
            node.parameters.start_depletion_time // 4,
            node.parameters.start_depletion_time // 4 + step * node.parameters.num_depletion_times,
            step,
            dtype=np.int64
        )
    else:
        depletion_times = np.array([node.parameters.start_depletion_time // 4], dtype=np.int64)
    # Register the sweep axes to be added to the dataset when fetching data
    # Order: outer loop dimension first, then inner loop dimension (matching 08b pattern)
    
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "depletion_time": xr.DataArray(depletion_times, attrs={"long_name": "depletion time", "units": "ns"}),
        "n_runs": xr.DataArray(np.linspace(1, n_runs, n_runs), attrs={"long_name": "number of shots"}),
    }

    with program() as node.namespace["qua_program"]:
        n, n_st = declare(int), declare_stream()
        dt = declare(int)  # depletion_time variable
        num_measurements = node.parameters.num_of_measurement
        # State variables for ground state measurements
        state_g_1 = [declare(int) for _ in range(num_qubits)]
        state_g_1_st = [declare_stream() for _ in range(num_qubits)]
        state_g_2 = [declare(int) for _ in range(num_qubits)]
        state_g_2_st = [declare_stream() for _ in range(num_qubits)]
        # State variables for excited state measurements
        state_e_1 = [declare(int) for _ in range(num_qubits)]
        state_e_1_st = [declare_stream() for _ in range(num_qubits)]
        state_e_2 = [declare(int) for _ in range(num_qubits)]
        state_e_2_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(*from_array(dt, depletion_times)):
                with for_(n, 0, n < n_runs, n + 1):
                    save(n, n_st)

                    # Ground state measurements
                    # Qubit initialization
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                    align()
                    
                    # First measurement (ground state) - run one measurement and save, then extra consecutive measurements
                    for i, qubit in multiplexed_qubits.items():
                        # First measurement - save it
                        readout_state(qubit, state_g_1[i], operation, wait_depletion_time=False)
                        wait(dt, qubit.resonator.name)
                        save(state_g_1[i], state_g_1_st[i])
                        # Run extra consecutive measurements in Python loop (without saving)
                        for _ in range(num_measurements):
                            readout_state(qubit, state_g_2[i], operation, wait_depletion_time=False)
                            wait(dt, qubit.resonator.name)
                        # Save the last measurement from the extra measurements
                        save(state_g_2[i], state_g_2_st[i])
                        
                    
                    align()

                    # Excited state measurements
                    # Qubit initialization
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                    align()

                    # Apply x180 pulse to prepare excited state
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.play("x180")
                        qubit.align()
                    
                    align()
                    
                    # First measurement (excited state) - run one measurement and save, then extra consecutive measurements
                    for i, qubit in multiplexed_qubits.items():
                        # First measurement - save it
                        readout_state(qubit, state_e_1[i], operation, wait_depletion_time=False)
                        save(state_e_1[i], state_e_1_st[i])
                        # Run extra consecutive measurements in Python loop (without saving)
                        for m in range(num_measurements):
                            readout_state(qubit, state_e_2[i], operation, wait_depletion_time=False)
                            wait(dt, qubit.resonator.name)
                        # Save the last measurement from the extra measurements
                        save(state_e_2[i], state_e_2_st[i])
                        
                    
            align()

        with stream_processing():
            n_st.save("n")
            num_depletion_times = len(depletion_times)
            for i in range(num_qubits):
                # Buffer order: inner loop dimension first, then outer loop dimension (matching 08b pattern)
                # Loop order: outer is depletion_times, inner is n_runs
                state_g_1_st[i].buffer(n_runs).buffer(num_depletion_times).save(f"ag1{i + 1}")
                state_g_2_st[i].buffer(n_runs).buffer(num_depletion_times).save(f"bg2{i + 1}")
                state_e_1_st[i].buffer(n_runs).buffer(num_depletion_times).save(f"ce1{i + 1}")
                state_e_2_st[i].buffer(n_runs).buffer(num_depletion_times).save(f"de2{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw".
    """
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
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
    """
    Analyse the raw data and store the fitted data in another xarray dataset "ds_fit"
    and the fitted results in the "fit_results" dictionary.
    """
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """
    Plot the confusion matrices and demolition error matrices.
    If depletion_time optimization was performed, also plot optimization results.
    """
    plot_results = plot_confusion_and_demolition_matrices(node)
    fig_confusion, fig_demolition = plot_results[0], plot_results[1]
    fig_optimization = plot_results[2] if len(plot_results) > 2 else None
    
    node.add_node_info_subtitle(fig_confusion)
    node.add_node_info_subtitle(fig_demolition)
    
    # Store the generated figures
    node.results["figures"] = {
        "confusion_matrix": fig_confusion,
        "demolition_error_matrix": fig_demolition,
    }
    if fig_optimization is not None:
        node.add_node_info_subtitle(fig_optimization)
        node.results["figures"]["depletion_time_optimization"] = fig_optimization


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            fit_result = node.results["fit_results"][q.name]
            operation = q.resonator.operations[node.parameters.operation]
            if node.parameters.operation == "readout":
                q.resonator.confusion_matrix = fit_result["confusion_matrix"]
            # Update depletion_time if optimization was performed
            if fit_result.get("optimal_depletion_time") is not None:
                q.resonator.depletion_time = fit_result["optimal_depletion_time"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

