# %% {Imports}
import matplotlib.pyplot as plt
import xarray as xr
from dataclasses import asdict
import numpy as np

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode
from iqcc_calibration_tools.quam_config.components.quam_root import Quam
from calibration_utils.spin_echo_sl import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.parameters import get_qubits, get_sl_times_in_clock_cycles
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Description}
description = """
        T2 SL MEASUREMENT
The sequence consists in playing an SL sequence (y90 - SL(x,t) - -y90 - measurement) for 
different sl times (spin locking times).

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the qubit frequency (node 03a_qubit_spectroscopy.py).
    - Having calibrated the qubit x180 pulse parameters (nodes 03a_qubit_spectroscopy.py and 04b_power_rabi.py).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - None (this is a measurement experiment, not a calibration).
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="06c_spin_locking",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["Q6"]
    pass

# Instantiate the QUAM class from the state file
node.machine = Quam.load()

# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots  # The number of averages
    # Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
    spin_locking_times = get_sl_times_in_clock_cycles(node.parameters)
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "spin_locking_time": xr.DataArray(8*spin_locking_times, attrs={"long_name": "spin_locking_time", "units": "ns"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        shot = declare(int)
        t_sl = declare(int)

        for multiplexed_qubits in qubits.batch():
            
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()
        
            with for_(shot, 0, shot < n_avg, shot + 1):
                save(shot, n_st)
                with for_each_(t_sl, spin_locking_times):
                    # Qubit initialization
                    for i, qubit in multiplexed_qubits.items():
                        # reset_frame(qubit.xy.name)
                        qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                    align()
                    # Qubit manipulation
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy_sl.play("-y90")
                        qubit.xy_sl.update_frequency(qubit.xy_sl.intermediate_frequency +1*u.kHz)
                        qubit.xy_sl.play("x180_BlackmanIntegralPulse_Rise")
                        qubit.xy_sl.play("x180_Square",duration = 2*t_sl)
                        qubit.xy_sl.play("x180_BlackmanIntegralPulse_Fall")
                        qubit.xy_sl.update_frequency(qubit.xy_sl.intermediate_frequency -1*u.kHz)
                        qubit.xy_sl.play("-y90")
                    align()
                    # Qubit readout
                    for i, qubit in multiplexed_qubits.items():
                        # Measure the state of the resonators
                        if node.parameters.use_state_discrimination:
                            qubit.readout_state(state[i])
                            save(state[i], state_st[i])
                        else:
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            # save data
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(spin_locking_times)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(spin_locking_times)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(spin_locking_times)).average().save(f"Q{i + 1}")


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
    # Here I can change the config...
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
    # Rename for compatibility with process_raw_dataset and fit_raw_data
    ds_raw_compat = node.results["ds_raw"].rename({"spin_locking_time": "idle_time"})
    
    node.results["ds_raw"] = process_raw_dataset(ds_raw_compat, node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    
    # Revert to your custom naming for state saving and future reference
    node.results["ds_raw"] = node.results["ds_raw"].rename({"idle_time": "spin_locking_time"})
    if node.results["ds_fit"] is not None:
         node.results["ds_fit"] = node.results["ds_fit"].rename({"idle_time": "spin_locking_time"})
         
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
    # Temporarily rename 'spin_locking_time' back to 'idle_time' for the plotting utility
    # This avoids the AttributeError: 'Dataset' object has no attribute 'idle_time'
    ds_raw_compat = node.results["ds_raw"].rename({"spin_locking_time": "idle_time"})
    
    # Ensure ds_fit also has the compatible name if it exists
    ds_fit_compat = None
    if node.results.get("ds_fit") is not None:
        ds_fit_compat = node.results["ds_fit"].rename({"spin_locking_time": "idle_time"})

    # Call the plotting utility with compatible datasets
    fig_result = plot_raw_data_with_fit(
        ds_raw_compat, 
        node.namespace["qubits"], 
        ds_fit_compat
    )
    
    # Handle both single figure (state discrimination) and tuple of figures (I, Q mode)
    if isinstance(fig_result, tuple):
        fig_I, fig_Q = fig_result
        node.add_node_info_subtitle(fig_I)
        node.add_node_info_subtitle(fig_Q)
        plt.show()
        # Store the generated figures
        node.results["figures"] = {
            "T2_SL_I": fig_I,
            "T2_SL_Q": fig_Q,
        }
    else:
        node.add_node_info_subtitle(fig_result)
        plt.show()
        # Store the generated figures
        node.results["figures"] = {
            "T2_SL": fig_result,
        }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    # This is a measurement experiment, not a calibration, so no state updates are needed
    with node.record_state_updates():
        pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
