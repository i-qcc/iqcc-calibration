# %% {Imports}
import matplotlib.pyplot as plt
import xarray as xr
from dataclasses import asdict
import numpy as np

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode
from iqcc_calibration_tools.quam_config.components.quam_root import Quam
from calibration_utils.time_rabi_amps import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    extract_rabi_frequencies,
    plot_rabi_freq_vs_amplitude
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Description}
description = """
        TIME RABI vs AMPLITUDE
This sequence involves playing the qubit pulse (such as x180) while sweeping both
its amplitude and duration. For each amplitude, a time rabi experiment is performed
to determine the rabi frequency. The results show how rabi frequency changes with
amplitude, which is useful for amplitude calibration.

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the qubit frequency (node 03a_qubit_spectroscopy.py).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - None (this is a diagnostic experiment).
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="06f_time_rabi_amps",  # Name should be unique
    description=description,  # Describe what the node is doing
    parameters=Parameters(),  # Node parameters
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.min_wait_time_in_ns = 20
    node.parameters.max_wait_time_in_ns = 160
    node.parameters.num_time_steps = 100
    node.parameters.qubits = ["Q6"]
    node.parameters.min_amp_factor = 0.05
    node.parameters.max_amp_factor = 1.90
    node.parameters.amp_factor_step = 0.05
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
    
    # Duration sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
    # Note: QUA duration must be a multiple of 4ns
    dur_vec = np.unique(np.geomspace(
        node.parameters.min_wait_time_in_ns,
        node.parameters.max_wait_time_in_ns,
        node.parameters.num_time_steps)//4).astype(int)
    
    # Amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    amps = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
    )
    
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "pulse amplitude prefactor"}),
        "duration": xr.DataArray(dur_vec * 4, attrs={"long_name": "pulse duration", "units": "ns"}),
    }
    
    with program() as node.namespace["qua_program"]:
        # Use machine.declare_qua_variables() for consistency with Power Rabi
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        a = declare(fixed)  # QUA variable for the amplitude pre-factor
        t = declare(int)  # QUA variable for the duration
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()
            
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                # Outer loop: sweep amplitude
                with for_(*from_array(a, amps)):
                    # Inner loop: sweep duration
                    with for_each_(t, dur_vec): 
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()
                        # Qubit manipulation
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy_sl.play("x180_BlackmanIntegralPulse_Rise", amplitude_scale=a)
                            qubit.xy_sl.play("x180_Square", duration=t, amplitude_scale=a)
                            qubit.xy_sl.play("x180_BlackmanIntegralPulse_Fall", amplitude_scale=a)
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
                        align() # Align after all readouts

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    # Buffer order: inner loop first (duration), then outer loop (amplitude)
                    state_st[i].buffer(len(dur_vec)).buffer(len(amps)).average().save(f"state{i + 1}")
                else:
                    # Buffer order: inner loop first (duration), then outer loop (amplitude)
                    I_st[i].buffer(len(dur_vec)).buffer(len(amps)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(dur_vec)).buffer(len(amps)).average().save(f"Q{i + 1}")


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
    plt.show()


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
    
    # Extract rabi frequencies vs amplitude (before converting to dicts)
    node.results["rabi_freqs"] = extract_rabi_frequencies(fit_results)
    
    # Convert FitParameters dataclasses to dicts for storage
    # fit_results is already a dict with nested structure: {qubit: {amp: FitParameters}}
    fit_results_dict = {}
    for q_name, amp_dict in fit_results.items():
        fit_results_dict[q_name] = {}
        for amp_factor, fit_params in amp_dict.items():
            fit_results_dict[q_name][float(amp_factor)] = asdict(fit_params)
    
    node.results["fit_results"] = fit_results_dict
    
    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    
    # Use dictionary key access 'fit_result["success"]' instead of 'fit_result.success'
    # Determine outcomes based on whether we have any successful fits
    node.outcomes = {}
    for q_name, amp_dict in fit_results_dict.items():
        has_success = any(fit_result.get("success", False) for fit_result in amp_dict.values())
        node.outcomes[q_name] = "successful" if has_success else "failed"
    
# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the rabi frequency vs amplitude."""
    # Main plot: Rabi frequency vs amplitude
    fig_rabi_vs_amp = plot_rabi_freq_vs_amplitude(
        node.results["rabi_freqs"],
        node.namespace["qubits"]
    )
    node.add_node_info_subtitle(fig_rabi_vs_amp)
    plt.show()
    
    # Store the figure
    node.results["figures"] = {
        "rabi_freq_vs_amp": fig_rabi_vs_amp,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    # This is a diagnostic experiment, so we don't update the state
    # But we could potentially update amplitude based on target frequency if needed
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
