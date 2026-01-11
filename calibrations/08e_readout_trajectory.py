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
from iqcc_calibration_tools.quam_config.components.quam_root import Quam
from calibration_utils.readout_trajectory import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_readout_trajectory,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates

# %% {Node initialisation}
description = """
        READOUT TRAJECTORY MEASUREMENT
This sequence involves measuring the readout trajectory by sending a readout pulse with a square section
followed by a zero-amplitude ringdown section. The measurement is performed using sliced readout to
capture the time evolution of the I and Q quadratures for both ground and excited states.

The data is post-processed to visualize the trajectory in the IQ plane and analyze the separation
between ground and excited states as a function of time.

Prerequisites:
    - Having calibrated the IQ mixer/Octave connected to the readout line (node 01a_mixer_calibration.py).
    - Having calibrated the time of flight, offsets, and gains (node 01a_time_of_flight.py).
    - Having initialized the QUAM state parameters for the readout pulse amplitude and duration.
    - Having specified the readout pulse name and parameters (square_length, zero_length).

State update:
    - (Optional) The readout pulse parameters can be updated based on the analysis results.
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="08e_readout_trajectory",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots
    segment_length = 10  # Segment length in clock cycles (4ns each)
    N_slices = int(node.parameters.readout_length_in_ns / (4 * segment_length))
    
    # Update readout pulse parameters for all qubits
    node.namespace["tracked_resonators"] = []
    readout_name = node.parameters.readout_name
    
    for q in qubits:
        resonator = q.resonator
        # Make temporary updates before running the program and revert at the end.
        with tracked_updates(resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
            resonator.operations[readout_name].length = node.parameters.square_length + node.parameters.zero_length
            resonator.operations[readout_name].zero_length = node.parameters.zero_length
            node.namespace["tracked_resonators"].append(resonator)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_runs": xr.DataArray(np.arange(0, n_avg, 1), attrs={"long_name": "number of shots"}),
        "readout_time": xr.DataArray(
            np.arange(0, N_slices, 1),
            attrs={"long_name": "readout time", "units": "ns"},
        ),
    }

    # The QUA program stored in the node namespace to be transferred to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        
        # Declare variables for sliced measurements (one set per qubit, declared upfront)
        # We'll use a single set and measure qubits sequentially
        IQg = declare(fixed, size=N_slices)
        QIg = declare(fixed, size=N_slices)
        IQe = declare(fixed, size=N_slices)
        QIe = declare(fixed, size=N_slices)
        Ig_slices = declare(fixed, size=N_slices)
        Qg_slices = declare(fixed, size=N_slices)
        Ie_slices = declare(fixed, size=N_slices)
        Qe_slices = declare(fixed, size=N_slices)
        
        # Declare streams (one set per qubit) - declare individually as lists
        # QUA doesn't support dictionary indexing, so we use lists
        Ig_st = [None] * num_qubits
        Qg_st = [None] * num_qubits
        IQg_st = [None] * num_qubits
        QIg_st = [None] * num_qubits
        Ie_st = [None] * num_qubits
        Qe_st = [None] * num_qubits
        IQe_st = [None] * num_qubits
        QIe_st = [None] * num_qubits
        for i in range(num_qubits):
            Ig_st[i] = declare_stream()
            Qg_st[i] = declare_stream()
            IQg_st[i] = declare_stream()
            QIg_st[i] = declare_stream()
            Ie_st[i] = declare_stream()
            Qe_st[i] = declare_stream()
            IQe_st[i] = declare_stream()
            QIe_st[i] = declare_stream()
        
        k = declare(int)
        df = declare(int)  # QUA variable for the readout frequency
        amp = declare(fixed)
        assign(df, 0)
        assign(amp, 1.0)
        
        # Measure each qubit sequentially (one at a time, not in batches)
        # This avoids resource allocation issues with sliced measurements
        for qubit_idx, qubit in enumerate(qubits):
            # Initialize the QPU for this qubit
            node.machine.initialize_qpu(target=qubit)
            rr = qubit.resonator
            rr.update_frequency(df + rr.intermediate_frequency)
            
            # Measure excited state (|1⟩)
            with for_(n, 0, n < n_avg, n + 1):
                qubit.reset_qubit_active()
                qubit.align()
                qubit.xy.play("x180")
                qubit.align()
                rr.measure_sliced(
                    readout_name,
                    stream=None,
                    qua_vars=(Ie_slices, IQe, QIe, Qe_slices),
                    segment_length=segment_length,
                    amplitude_scale=amp,
                )
                # Save each slice
                with for_(k, 0, k < N_slices, k + 1):
                    save(Ie_slices[k], Ie_st[qubit_idx])
                    save(Qe_slices[k], Qe_st[qubit_idx])
                    save(IQe[k], IQe_st[qubit_idx])
                    save(QIe[k], QIe_st[qubit_idx])
                rr.wait(rr.depletion_time * u.ns * 10)

            # Measure ground state (|0⟩)
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                qubit.reset_qubit_active()
                qubit.align()
                rr.measure_sliced(
                    readout_name,
                    stream=None,
                    qua_vars=(Ig_slices, IQg, QIg, Qg_slices),
                    segment_length=segment_length,
                    amplitude_scale=amp,
                )
                with for_(k, 0, k < N_slices, k + 1):
                    save(Ig_slices[k], Ig_st[qubit_idx])
                    save(Qg_slices[k], Qg_st[qubit_idx])
                    save(IQg[k], IQg_st[qubit_idx])
                    save(QIg[k], QIg_st[qubit_idx])
                rr.wait(rr.depletion_time * u.ns * 10)

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                Ie = (Ie_st[i] + IQe_st[i]).buffer(n_avg, N_slices)
                Ie.save(f"Ie{i + 1}")
                Qe = (Qe_st[i] + QIe_st[i]).buffer(n_avg, N_slices)
                Qe.save(f"Qe{i + 1}")
                Ig = (Ig_st[i] + IQg_st[i]).buffer(n_avg, N_slices)
                Ig.save(f"Ig{i + 1}")
                Qg = (Qg_st[i] + QIg_st[i]).buffer(n_avg, N_slices)
                Qg.save(f"Qg{i + 1}")


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
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
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
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
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
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig = plot_readout_trajectory(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.parameters,
    )
    node.add_node_info_subtitle(fig)
    plt.show()
    # Store the generated figure
    node.results["figures"] = {"trajectory": fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes.get(q.name, "failed") == "failed":
                continue
            # Add state updates here if needed based on analysis results
            # For example, update readout pulse parameters if optimization is performed
            pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
