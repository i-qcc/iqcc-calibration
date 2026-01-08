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
from calibration_utils.qubits_spectroscopy_02 import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Node initialisation}
description = """
        QUBIT SPECTROSCOPY - 0->2 TRANSITION
This sequence involves sending a saturation pulse to the qubit to find the 0->2 transition frequency,
which allows estimation of the qubit anharmonicity. The sequence searches around f01 - α/2,
where f01 is the qubit frequency and α is the anharmonicity (default guess of 200 MHz).

The data is post-processed to determine the 0->2 transition frequency and calculate the actual anharmonicity.

Prerequisites:
    - Having run the qubit spectroscopy (03a) to find the 0->1 transition frequency
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the readout parameters (nodes 02a, 02b and/or 02c).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - (optional) The qubit anharmonicity: qubit.anharmonicity
    - The integration weight angle to get the state discrimination along the 'I' quadrature: qubit.resonator.operations["readout"].integration_weights_angle.
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="03e_qubits_spectroscopy_02",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["qC2"]
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

    operation = node.parameters.operation  # The qubit operation to play
    n_avg = node.parameters.num_shots  # The number of averages
    # Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
    operation_len = node.parameters.operation_len_in_ns
    # pre-factor to the value defined in the config - restricted to [-2; 2)
    operation_amp = node.parameters.operation_amplitude_factor if node.parameters.operation_amplitude_factor is not None else 1.0
    
    # Calculate expected 0->2 transition frequency (f01 - α/2)
    init_anharmonicity = node.parameters.initial_anharmonicity_mhz * u.MHz
    # Qubit detuning sweep around expected 0->2 transition
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step)

    # Calculate flux bias offsets and detunings
    qubit_freqs = {q.name: q.xy.RF_frequency for q in qubits}
    arb_flux_bias_offset = {}
    detunings = {}
    
    for q in qubits:
        if node.parameters.arbitrary_flux_bias is not None:
            arb_flux_bias_offset[q.name] = node.parameters.arbitrary_flux_bias
            detunings[q.name] = q.freq_vs_flux_01_quad_term * arb_flux_bias_offset[q.name] ** 2
        elif node.parameters.arbitrary_qubit_frequency_in_ghz is not None:
            detunings[q.name] = 1e9 * node.parameters.arbitrary_qubit_frequency_in_ghz - qubit_freqs[q.name]
            arb_flux_bias_offset[q.name] = np.sqrt(detunings[q.name] / q.freq_vs_flux_01_quad_term) if q.freq_vs_flux_01_quad_term != 0 else 0.0
        else:
            arb_flux_bias_offset[q.name] = 0.0
            detunings[q.name] = 0.0
        
        # Adjust detunings to search around 0->2 transition (f01 - α/2)
        detunings[q.name] -= init_anharmonicity / 2

    # Store in namespace for later use
    node.namespace["arb_flux_bias_offset"] = arb_flux_bias_offset
    node.namespace["detunings"] = detunings
    node.namespace["init_anharmonicity"] = init_anharmonicity

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "qubit frequency detuning", "units": "Hz"}),
    }

    flux_point = node.parameters.flux_point_joint_or_independent

    with program() as node.namespace["qua_program"]:
        # Macro to declare I, Q, n and their respective streams for a given number of qubit
        I, I_st, Q, Q_st, _, n_st = node.machine.declare_qua_variables()

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            if flux_point == "joint":
                node.machine.initialize_qpu(target=multiplexed_qubits[0])
            else:
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
            align()

            n = declare(int)  # QUA variable for the number of averages
            df = declare(int)  # QUA variable for the qubit frequency
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        # Get the duration of the operation from the node parameters or the state
                        if operation_len is not None:
                            duration = operation_len * u.ns
                        else:
                            duration = (qubit.xy.operations[operation].length + qubit.z.settle_time) * u.ns
                        
                        # Update the qubit frequency
                        qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency + detunings[qubit.name])
                        qubit.align()
                        
                        # Bring the qubit to the desired point during the saturation pulse
                        qubit.z.play(
                            "const",
                            amplitude_scale=arb_flux_bias_offset[qubit.name] / qubit.z.operations["const"].amplitude,
                            duration=duration,
                        )
                        # Play the saturation pulse
                        qubit.xy.wait(qubit.z.settle_time * u.ns)
                        qubit.xy.play(
                            operation,
                            amplitude_scale=operation_amp,
                            duration=duration,
                        )
                    align()

                    for i, qubit in multiplexed_qubits.items():
                        # readout the resonator
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        # wait for the resonator to deplete
                        qubit.resonator.wait(node.machine.depletion_time * u.ns)
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                    align()
                    
                    # Measure sequentially if not multiplexed
                    if not node.parameters.multiplexed:
                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")


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
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    node.add_node_info_subtitle(fig_raw_fit)
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "amplitude": fig_raw_fit,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            fit_result = node.results["fit_results"][q.name]
            # Update the integration weight angle
            q.resonator.operations["readout"].integration_weights_angle = fit_result["iw_angle"]
            # Optionally update the anharmonicity if the fit was successful
            # Uncomment the following line if you want to update the anharmonicity:
            q.anharmonicity = int(fit_result["anharmonicity"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
