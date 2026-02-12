# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode
from iqcc_calibration_tools.quam_config.components.quam_root import Quam
from calibration_utils.T2echo_vs_flux import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    plot_T2_vs_flux,
)
from qualibration_libs.parameters import get_qubits, get_idle_times_in_clock_cycles
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Description}
description = """
        T2 echo vs flux MEASUREMENT
The sequence consists in playing an echo sequence (x90 - idle_time - x180 - idle_time - -x90 - measurement) 
for different idle times and different flux bias points.
The qubit T2 echo is extracted by fitting the exponential decay of the measured quadratures/state for each flux point.
This allows studying how T2 echo varies with flux bias.

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the qubit frequency precisely (node 06a_ramsey.py).
    - (optional) Having optimized the readout parameters (nodes 08a, 08b and 08c).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

Next steps before going to the next node:
    - Review the T2 echo vs flux dependence.
"""

node = QualibrationNode[Parameters, Quam](name="T2_echo_vs_flux", description=description, parameters=Parameters())

# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["Q1"]
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
    idle_times = get_idle_times_in_clock_cycles(node.parameters)
    # Flux sweep parameters
    flux_span = node.parameters.flux_span
    flux_step = node.parameters.flux_step
    dcs = np.arange(0.0, flux_span / 2 + 0.001, step=flux_step)   
    # Calculate frequency shifts for each flux point (quadratic dependence)
    quads = {qubit.name: int(qubit.freq_vs_flux_01_quad_term) for qubit in qubits}
    freqs = {qubit.name: (dcs**2 * qubit.freq_vs_flux_01_quad_term).astype(int) for qubit in qubits}
    
    # Register the sweep axes to be added to the dataset when fetching data.
    #
    # IMPORTANT: XarrayDataFetcher matches the raw streamed array shape against the sweep axes
    # order (excluding the leading "qubit" axis). For this experiment the stream comes back
    # as (flux, idle_time), so we must declare sweep_axes in that same order.
    #
    # After fetching we transpose the dataset to the conventional (qubit, idle_time, flux).
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "flux": xr.DataArray(dcs, attrs={"long_name": "flux", "units": "V"}),
        "idle_time": xr.DataArray(8 * idle_times, attrs={"long_name": "idle time", "units": "ns"}),
    }
    
    # Store flux arrays in namespace for later use
    node.namespace["dcs"] = dcs
    node.namespace["freqs"] = freqs
    node.namespace["quads"] = quads

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        shot = declare(int)
        t = declare(int)
        dc = declare(fixed)  # QUA variable for the flux dc level
        freq = declare(int)
        flux_index = declare(int)
        fluxes_qua = declare(fixed, value=dcs)

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            for qubit in multiplexed_qubits.values():
                wait(1000, qubit.z.name)
            
            align()

            for i, qubit in multiplexed_qubits.items():
                # Declare frequency array for this qubit (allowed in QUA as long as it's outside for_ loops)
                freqs_qua_qubit = declare(int, value=freqs[qubit.name])
                
                with for_(shot, 0, shot < n_avg, shot + 1):
                    save(shot, n_st)

                    with for_(flux_index, 0, flux_index < len(dcs), flux_index + 1):
                        assign(dc, fluxes_qua[flux_index])
                        assign(freq, freqs_qua_qubit[flux_index])
                        
                        with for_each_(t, idle_times):
                            # Qubit initialization
                            for j, q in multiplexed_qubits.items():
                                reset_frame(q.xy.name)
                                q.reset(node.parameters.reset_type, node.parameters.simulate)
                            align()

                            # Qubit manipulation - Echo sequence
                            for j, q in multiplexed_qubits.items():
                                q.xy.play("x90")
                                q.align()
                                q.z.wait(20)
                                q.z.play("const", amplitude_scale=dc/q.z.operations["const"].amplitude, duration=t)
                                q.z.wait(20)
                                q.align()
                                q.xy.play("x180")
                                q.align()
                                q.z.wait(20)
                                q.z.play("const", amplitude_scale=dc/q.z.operations["const"].amplitude, duration=t)
                                q.z.wait(20)
                                q.align()
                                q.xy.play("-x90")
                                q.align()
                            
                            align()
                            # Qubit readout
                            for j, q in multiplexed_qubits.items():
                                # Measure the state of the resonators
                                if node.parameters.use_state_discrimination:
                                    q.readout_state(state[j])
                                    save(state[j], state_st[j])
                                else:
                                    q.resonator.measure("readout", qua_vars=(I[j], Q[j]))
                                    # save data
                                    save(I[j], I_st[j])
                                    save(Q[j], Q_st[j])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    # Buffer order in QUA: buffer dimensions match the order they appear in nested loops
                    # Loop order: outer=flux (for_), inner=idle_times (for_each_)
                    # In QUA, buffer(len(inner)).buffer(len(outer)) creates (inner, outer) = (idle_time, flux)
                    # XarrayDataFetcher expects buffer dimensions in SAME order as sweep_axes (excluding qubit)
                    # Sweep axes: (qubit, idle_time, flux) -> buffer should be (idle_time, flux) ✓
                    state_st[i].buffer(len(idle_times)).buffer(len(dcs)).average().save(f"state{i + 1}")
                else:
                    # Buffer order in QUA: buffer dimensions match the order they appear in nested loops
                    # Loop order: outer=flux (for_), inner=idle_times (for_each_)
                    # In QUA, buffer(len(inner)).buffer(len(outer)) creates (inner, outer) = (idle_time, flux)
                    # XarrayDataFetcher expects buffer dimensions in SAME order as sweep_axes (excluding qubit)
                    # Sweep axes: (qubit, idle_time, flux) -> buffer should be (idle_time, flux) ✓
                    I_st[i].buffer(len(idle_times)).buffer(len(dcs)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(idle_times)).buffer(len(dcs)).average().save(f"Q{i + 1}")


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
    # Transpose to get (qubit, idle_time, flux) order for consistency with other experiments
    if "flux" in dataset.dims and "idle_time" in dataset.dims:
        dataset = dataset.transpose("qubit", "idle_time", "flux")
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
    fig_raw_fit = plot_raw_data_with_fit(
        node.results["ds_raw"], 
        node.namespace["qubits"], 
        node.results["ds_fit"],
        sweep_type=getattr(node.parameters, 'log_or_linear_sweep', None)
    )
    
    fig_T2_vs_flux = plot_T2_vs_flux(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"]
    )
    
    # Build subtitle in the correct order: sweep, multiplexed, state discrimination, reset_type, then date/time
    subtitle_parts = []
    if hasattr(node.parameters, 'log_or_linear_sweep') and node.parameters.log_or_linear_sweep:
        subtitle_parts.append(f"sweep = {node.parameters.log_or_linear_sweep}")
    if hasattr(node.parameters, 'multiplexed'):
        subtitle_parts.append(f"multiplexed = {node.parameters.multiplexed}")
    if hasattr(node.parameters, "use_state_discrimination"):
        subtitle_parts.append(f"state discrimination = {node.parameters.use_state_discrimination}")
    if hasattr(node.parameters, 'reset_type'):
        subtitle_parts.append(f"reset type = {node.parameters.reset_type}")
    # Add date/time last
    subtitle_parts.append(f"{node.date_time} GMT+{node.time_zone} #{node.node_id}")
    
    # Get existing title and append subtitle
    for fig in [fig_raw_fit, fig_T2_vs_flux]:
        existing_title = fig._suptitle.get_text() if fig._suptitle else ""
        combined_text = f"{existing_title}\n{chr(10).join(subtitle_parts)}"
        fig.suptitle(combined_text, fontsize=10, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "raw_fit": fig_raw_fit,
        "T2_vs_flux": fig_T2_vs_flux,
    }

# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    # For T2 echo vs flux, we typically don't update the state automatically
    # as this is a characterization measurement rather than a calibration
    # But we can leave this function here for future use if needed
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
# %%
