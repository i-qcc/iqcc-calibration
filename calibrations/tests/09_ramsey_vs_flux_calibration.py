# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam as Quam
from calibration_utils.ramsey_versus_flux_calibration import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    plot_parabolas_with_fit,
    process_raw_dataset,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

# %% {Node initialisation}
description = """
        RAMSEY WITH VIRTUAL Z ROTATIONS
This program consists in playing a Ramsey sequence (x90 - idle_time - x90 - measurement)
for different idle times. Instead of detuning the qubit gates, the frame of the second
x90 pulse is rotated (de-phased) to mimic an accumulated phase acquired for a given
detuning after the idle time. This method has the advantage of playing resonant gates.

From the results, one can fit the Ramsey oscillations and precisely measure the qubit
resonance frequency and T2*.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under
      study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy,
      rabi_chevron, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude,
      duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Update the qubits frequency (f_01) in the state.
    - Save the current state by calling machine.save()
"""


node = QualibrationNode[Parameters, Quam](
    name="09_ramsey_vs_flux_calibration",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q3"]
    # node.parameters.scale_flux_span = {"qA1": 3, "qA3":3, "qA6": 3, "qD3": 3, "qB2": 3}
    node.parameters.multiplexed = True
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
    idle_times = np.arange(
        node.parameters.min_wait_time_in_ns // 4,
        node.parameters.max_wait_time_in_ns // 4,
        node.parameters.wait_time_step_in_ns // 4,
    )

    # Detuning converted into virtual Z-rotations to observe Ramsey oscillation and get the qubit frequency
    detuning = int(1e6 * node.parameters.frequency_detuning_in_mhz)
    
    # Per-qubit scale factors for flux span (default to 1 if not specified)
    scale_flux_span = node.parameters.scale_flux_span
    flux_scale_factors = {q.name: scale_flux_span.get(q.name, 1.0) for q in qubits}
    max_scale = max(flux_scale_factors.values())
    
    # Compute flux arrays: max-scaled for QUA loop, per-qubit for data coordinates
    def make_flux_array(scale):
        return np.linspace(-node.parameters.flux_span * scale / 2, node.parameters.flux_span * scale / 2, node.parameters.flux_num)
    
    fluxes = make_flux_array(max_scale)
    node.namespace["per_qubit_fluxes"] = {q.name: make_flux_array(flux_scale_factors[q.name]) for q in qubits}
    flux_scale_factors_normalized = {q: s / max_scale for q, s in flux_scale_factors.items()}
    
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "flux_bias": xr.DataArray(fluxes, attrs={"long_name": "flux bias", "units": "V"}),
        "idle_times": xr.DataArray(4 * idle_times, attrs={"long_name": "idle times", "units": "ns"}),
    }
    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        init_state = [declare(int) for _ in range(num_qubits)]
        current_state = [declare(int) for _ in range(num_qubits)]
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        t = declare(int)  # QUA variable for the idle time
        phi = declare(fixed)  # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)
        flux = declare(fixed)  # QUA variable for the flux dc level

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            for i, qubit in multiplexed_qubits.items():
                qubit.readout_state(init_state[i])

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(flux, fluxes)):
                    with for_(*from_array(t, idle_times)):
                        # Qubit manipulation
                        for i, qubit in multiplexed_qubits.items():
                            # Rotate the frame of the second x90 gate to implement a virtual Z-rotation
                            # 4*tau because tau was in clock cycles and 1e-9 because tau is ns
                            assign(phi, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))
                            # Scale flux by qubit's normalized scale factor (handles per-qubit flux spans)
                            scaled_flux = flux * flux_scale_factors_normalized[qubit.name]
                            # Ramsey sequence
                            with strict_timing_():
                                qubit.xy.play("x90")
                                qubit.xy.frame_rotation_2pi(phi)
                                qubit.xy.wait(t + 1)
                                qubit.z.wait(duration=qubit.xy.operations["x90"].length // 4)
                                qubit.z.play(
                                    "const", amplitude_scale=scaled_flux / qubit.z.operations["const"].amplitude, duration=t
                                )
                                qubit.xy.play("x90")
                        align()

                        # Qubit readout
                        for i, qubit in multiplexed_qubits.items():
                            qubit.readout_state(current_state[i])
                            assign(state[i], init_state[i] ^ current_state[i])
                            assign(init_state[i], current_state[i])
                            save(state[i], state_st[i])
                            # Reset the frame of the qubits in order not to accumulate rotations
                            reset_frame(qubit.xy.name)
                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                state_st[i].buffer(len(idle_times)).buffer(len(fluxes)).average().save(f"state{i + 1}")


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
    node.namespace["qubits"] = qubits = get_qubits(node)
    
    # Reconstruct per_qubit_fluxes from parameters (needed for analysis)
    scale = node.parameters.scale_flux_span
    half_span = node.parameters.flux_span / 2
    node.namespace["per_qubit_fluxes"] = {
        q.name: np.linspace(-half_span * scale.get(q.name, 1.0), half_span * scale.get(q.name, 1.0), node.parameters.flux_num)
        for q in qubits
    }


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    # Add per-qubit flux coordinates to the dataset
    per_qubit_fluxes = node.namespace.get("per_qubit_fluxes")
    if per_qubit_fluxes:
        qubit_names = list(per_qubit_fluxes.keys())
        flux_actual = xr.DataArray(
            [per_qubit_fluxes[q] for q in qubit_names],
            dims=["qubit", "flux_idx"],
            coords={"qubit": qubit_names},
            attrs={"long_name": "actual flux bias per qubit", "units": "V"}
        )
        node.results["ds_raw"] = node.results["ds_raw"].assign(flux_actual=flux_actual)
    
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
    fig_parabola_fit = plot_parabolas_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"], node.outcomes)
    node.add_node_info_subtitle(fig_raw_fit)
    node.add_node_info_subtitle(fig_parabola_fit)
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "raw_data": fig_raw_fit,
        "parabola_fit": fig_parabola_fit,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            flux_point = node.machine.qubits[q.name].z.flux_point
            flux_offset = node.results["fit_results"][q.name]["flux_offset"]
            target_offset = node.results["fit_results"][q.name]["target_offset"]
            freq_offset = (
                node.parameters.frequency_detuning_in_mhz * 1e6
                - node.results["ds_fit"].freq_offset.sel(qubit=q.name).values
            )
            quad_term = node.results["fit_results"][q.name]["quad_term"]
            # Use target_offset if available, otherwise use flux_offset
            offset_to_apply = target_offset if not np.isnan(target_offset) else flux_offset
            if flux_point == "independent":
                q.z.independent_offset += offset_to_apply
            elif flux_point == "joint":
                q.z.joint_offset += offset_to_apply
            else:
                raise RuntimeError("Unknown flux_point")
            # Adjust frequency correction if target_offset is used
            # The frequency at target_offset is freq_sweet_spot + target_detuning_from_sweet_spot
            if not np.isnan(target_offset):
                target_detuning = q.xy.extras["target_detuning_from_sweet_spot"]
                freq_offset -= target_detuning
            q.f_01 += freq_offset
            q.xy.RF_frequency += freq_offset
            # quad_term is stored in GHz/V^2, convert to Hz/V^2 for freq_vs_flux_01_quad_term
            q.freq_vs_flux_01_quad_term = quad_term * 1e9

# %%
