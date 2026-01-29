# %% {Imports}
import matplotlib.pyplot as plt
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam as Quam
from calibration_utils.ramsey import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.parameters import get_qubits, get_idle_times_in_clock_cycles
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.analysis import fit_oscillation_decay_exp

import numpy as np
# %% {Description}
description = """
        RAMSEY WITH VIRTUAL Z ROTATIONS
The program consists in playing a Ramsey sequence (x90 - idle_time - x90/y90 - measurement) for different idle times.
Instead of detuning the qubit gates, the frame of the second x90 pulse is rotated (de-phased) to mimic an accumulated
phase acquired for a given detuning after the idle time.
This method has the advantage of playing gates on resonance as opposed to the detuned Ramsey.

From the results, one can fit the Ramsey oscillations and precisely measure the qubit resonance frequency and T2*.

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the readout parameters (nodes 02a, 02b and/or 02c).
    - Having calibrated the qubit x180 pulse parameters (nodes 03a_qubit_spectroscopy.py and 04b_power_rabi.py).
    - (optional) Having optimized the readout parameters (nodes 08a, 08b and 08c).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - The qubit 0->1 frequency: qubit.f_01 & qubit.xy.RF_frequency
    - T2*: qubit.T2ramsey.
"""

node = QualibrationNode[Parameters, Quam](name="06a_ramsey_repeated", description=description, parameters=Parameters())


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = None
    node.parameters.max_wait_time_in_ns = 3000
    node.parameters.wait_time_num_points = 150
    node.parameters.num_shots = 50
    node.parameters.frequency_detuning_in_mhz = 2
    node.parameters.use_state_discrimination = True
    node.parameters.log_or_linear_sweep = "linear"
    pass
physical_detuning_in_Mhz = 10
num_reps = 100
delay_between_reps = 600 # sec
total_time = num_reps * delay_between_reps / 3600 # hours
print(f"Total time: {total_time} hours")

## Instantiate the QUAM class from the state file
node.machine = Quam.load()

flux_shifts = {}
for qubit in get_qubits(node):
    flux_shift = np.sqrt(-physical_detuning_in_Mhz*1e6/qubit.freq_vs_flux_01_quad_term)
    flux_shifts[qubit.name] = flux_shift

# %%

# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots

    idle_times = get_idle_times_in_clock_cycles(node.parameters)
    detuning = (node.parameters.frequency_detuning_in_mhz - physical_detuning_in_Mhz) * u.MHz

    detuning_signs = [-1, 1]
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(4 * idle_times, attrs={"long_name": "idle times", "units": "ns"}),
    }
    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        idle_time = declare(int)
        detuning_sign = declare(int)
        virtual_detuning_phases = [declare(fixed) for _ in range(num_qubits)]

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

                with for_each_(idle_time, idle_times):
                    # Qubit initialization
                    for i, qubit in multiplexed_qubits.items():
                        reset_frame(qubit.xy.name)
                        qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                    align()
                    # Qubit manipulation
                    for i, qubit in multiplexed_qubits.items():
                        assign(
                            virtual_detuning_phases[i],
                            Cast.mul_fixed_by_int(detuning * 1e-9, 4 * idle_time),
                        )


                        # with strict_timing_():
                        # qubit.xy.play("x90")
                        # qubit.xy.frame_rotation_2pi(virtual_detuning_phases[i])
                        # qubit.xy.wait(idle_time)
                        # qubit.xy.play("x90")
                        

                        qubit.xy.play("x90")
                        qubit.xy.frame_rotation_2pi(virtual_detuning_phases[i])
                        qubit.z.wait(duration=qubit.xy.operations["x180"].length)
                        
                        qubit.xy.wait(idle_time)
                        qubit.z.play("const", amplitude_scale=flux_shifts[qubit.name] / qubit.z.operations["const"].amplitude, 
                                        duration=idle_time)
                        
                        qubit.xy.play("x90")                          

                    align()
                    for i, qubit in multiplexed_qubits.items():
                        if node.parameters.use_state_discrimination:
                            qubit.readout_state(state[i])
                            save(state[i], state_st[i])
                        else:
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(idle_times)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(idle_times)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(idle_times)).average().save(f"Q{i + 1}")


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
import time

datasets = []
for i in range(num_reps):
    execute_qua_program(node)
    datasets.append(node.results["ds_raw"])
    time.sleep(delay_between_reps)
# %%
import xarray as xr

# Merge the datasets along a new axis called "repetitions" and add a coordinate for repetition index
ds_merged = xr.concat(datasets, dim="repetitions")
ds_merged = ds_merged.assign_coords(repetitions=("repetitions", list(range(len(datasets)))))
ds_merged.state.plot()
node.results["ds_merged"] = ds_merged

# %%
fit = fit_oscillation_decay_exp(ds_merged.state, "idle_time")
decay = 1/fit.sel(fit_vals = "decay")
freq = fit.sel(fit_vals = "f")

# %% {Plot_data}
import matplotlib.pyplot as plt
for q in ds_merged.qubit.values:    
    ds_merged.state.sel(qubit=q).plot()
    plt.show()

# Plot T2* as a function of repetition
plt.figure(figsize=(8, 5))
for q in ds_merged.qubit.values:
    decay = fit.sel(fit_vals="decay").sel(qubit=q)
    t2_vals = 1e-3/decay
    decay_err = np.sqrt(fit.sel(fit_vals="decay_decay").sel(qubit=q))
    t2_err = t2_vals * decay_err/decay
    plt.errorbar(ds_merged.repetitions, t2_vals, yerr=t2_err, marker='o', ls = 'none', label=f"Qubit {q}")
plt.xlabel("Repetition")
plt.ylabel("T2* (us)")
plt.title("T2* vs Repetition")
plt.legend()
plt.tight_layout()
plt.show()

# Plot frequency as a function of repetition
plt.figure(figsize=(8, 5))
for q in ds_merged.qubit.values:
    freq = fit.sel(fit_vals="f").sel(qubit=q)
    freq_err = np.sqrt(fit.sel(fit_vals="f_f").sel(qubit=q))*1e3
    freq_vals = freq*1e3
    plt.errorbar(ds_merged.repetitions, freq_vals, yerr=freq_err, marker='o', ls = 'none', label=f"Qubit {q}")
plt.xlabel("Repetition")
plt.ylabel("Frequency (MHz)")
plt.title("Frequency vs Repetition")
plt.legend()
plt.tight_layout()
plt.show()



# %% {Update_state}
# @node.run_action(skip_if=node.parameters.simulate)
# def update_state(node: QualibrationNode[Parameters, Quam]):
#     """Update the relevant parameters if the qubit data analysis was successful."""
#     with node.record_state_updates():
#         for q in node.namespace["qubits"]:
#             if node.results["fit_results"][q.name]["success"]:
#                 q.f_01 -= float(node.results["fit_results"][q.name]["freq_offset"])
#                 q.xy.RF_frequency -= float(node.results["fit_results"][q.name]["freq_offset"])
#                 q.T2ramsey = float(node.results["fit_results"][q.name]["decay"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
