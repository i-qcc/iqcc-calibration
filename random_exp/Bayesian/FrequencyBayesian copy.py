"""
Bayesian Frequency Estimation for Quantum Qubits
================================================

This experiment implements a Bayesian frequency estimation protocol to measure and track
the frequency of quantum qubits over time. The method uses sequential Bayesian updates
to estimate the qubit frequency based on measurement outcomes, providing real-time
frequency tracking and noise characterization.

Overview
--------
The experiment performs the following sequence for each qubit:
1. Applies a flux pulse to shift the qubit frequency
2. Executes a Ramsey-like sequence with variable idle time
3. Measures the qubit state
4. Updates the Bayesian probability distribution P(f) over frequency using measurement outcomes
5. Estimates the most likely frequency from the posterior distribution
6. Repeats the process to track frequency evolution over time

The results include:
- Time-resolved Bayesian probability distributions P(f|t)
- Estimated frequency trajectories
- Single-shot measurement data (optional)

Prerequisites
-------------
Before running this experiment, ensure the following calibrations are complete:
- Time of flight calibration (offsets and gains)
- IQ mixer calibration for the readout line
- Resonator spectroscopy (resonance frequency identification)
- Readout pulse amplitude and duration configuration
- Resonator depletion time specification in the state
- SPAM (State Preparation and Measurement) calibration (confusion matrix)

Parameters
----------
The experiment is configured through the Parameters class. Key parameters include:

Qubit Selection:
    qubits : Optional[List[str]]
        List of qubit names to measure. Default: None
        Example: ["qC1", "qC2"] to measure multiple qubits

Experiment Parameters:
    num_repetitions : int
        Number of measurement repetitions per time point. Default: 5000
        Higher values improve frequency resolution but increase measurement time.

    detuning : int
        Nominal detuning frequency in Hz. Default: 2 MHz
        This is the frequency offset used in the Ramsey sequence.

    physical_detuning : int
        Physical detuning applied via flux bias in Hz. Default: 5 MHz
        This creates the actual frequency shift that we want to measure.

    min_wait_time_in_ns : int
        Minimum idle time in the Ramsey sequence (nanoseconds). Default: 36 ns
        The shortest time between the two π/2 pulses.

    max_wait_time_in_ns : int
        Maximum idle time in the Ramsey sequence (nanoseconds). Default: 8000 ns
        The longest time between the two π/2 pulses.

    wait_time_step_in_ns : int
        Step size for the idle time sweep (nanoseconds). Default: 120 ns
        Controls the resolution of the time axis.

Bayesian Estimation Parameters:
    f_min : float
        Minimum frequency in the Bayesian search range (MHz). Default: 1.05 MHz
        Lower bound of the frequency prior distribution.

    f_max : float
        Maximum frequency in the Bayesian search range (MHz). Default: 1.25 MHz
        Upper bound of the frequency prior distribution.
        Note: Frequency range should be between 0 and 8 MHz due to QUA fixed variable limitations.

    df : float
        Frequency resolution step size (MHz). Default: 0.002 MHz
        Smaller values improve frequency resolution but increase computational overhead.
        The number of frequency bins is (f_max - f_min) / df.

Data Collection:
    keep_shot_data : bool
        Whether to save individual shot measurement outcomes. Default: True
        If False, only the Bayesian distributions and estimated frequencies are saved.
        Setting to False reduces memory usage for long experiments.

Execution Parameters:
    simulate : bool
        Run in simulation mode instead of hardware. Default: False
        When True, the program is compiled but not executed on hardware.

    simulation_duration_ns : int
        Duration for simulation mode (nanoseconds). Default: 2500 ns
        Only used when simulate=True.

    timeout : int
        Maximum execution time in seconds. Default: 100
        The experiment will timeout if not completed within this time.

    load_data_id : Optional[int]
        Load previously saved data instead of running new experiment. Default: None
        If provided, loads data from the specified experiment ID instead of executing.

    multiplexed : bool
        Whether to use multiplexed readout. Default: False
        Currently not fully implemented.

Output Data
----------
The experiment returns an xarray Dataset with the following variables:

    Pf : (qubit, repetition, vf)
        Bayesian probability distribution over frequency for each repetition.
        Normalized such that sum(Pf) = 1 for each repetition.

    estimated_frequency : (qubit, repetition)
        Most likely frequency estimate (MHz) for each repetition, computed as argmax(Pf).

    state : (qubit, repetition, t) [if keep_shot_data=True]
        Single-shot measurement outcomes (0 or 1) for each time point and repetition.

    time_stamp : (qubit, repetition)
        Timestamp for each measurement (seconds since experiment start).

Figures
-------
The experiment generates several visualization figures:

    PF_figure : Bayesian probability distribution P(f|t) as a function of time
    state_figure : Single-shot measurement outcomes (if keep_shot_data=True)
    estimated_frequency_figure : Estimated frequency trajectory over time

Usage Example
-------------
```python
from qualibrate import QualibrationNode

# Create node with custom parameters
node = QualibrationNode(
    name="FrequencyBayes",
    parameters=Parameters(
        qubits=["qC2"],
        num_repetitions=10000,
        f_min=1.0,
        f_max=1.5,
        df=0.001,
        keep_shot_data=True
    )
)

# Run the experiment
node.run()
```

Notes
-----
- The Bayesian update uses SPAM parameters (alpha, beta) from the qubit's confusion matrix
- Frequency estimation is limited to 0-8 MHz range due to QUA fixed variable constraints
- The flux shift is automatically calculated based on physical_detuning and qubit properties
- Frame rotation is reset after each measurement to prevent phase accumulation
- Normalization prevents numerical underflow in the Bayesian updates

References
----------
Berritta, F. et al. Real-time two-axis control of a spin qubit. Nat Commun 15, (2024).

This implementation is based on Bayesian parameter estimation techniques for quantum
metrology, adapted for real-time frequency tracking in superconducting qubits.
"""

# %% Imports
from qualibrate import QualibrationNode, NodeParameters
from iqcc_calibration_tools.quam_config.components import Quam
from iqcc_calibration_tools.analysis.plot_utils import QubitGrid, grid_iter
from iqcc_calibration_tools.quam_config.macros import qua_declaration, readout_state
from qualang_tools.units import unit
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qm.qua import *
from typing import Optional, List
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

u = unit(coerce_to_integer=True)

# %% Helper Functions

def extract_string(input_string):
    """Extract the prefix string before the first digit."""
    index = next((i for i, c in enumerate(input_string) if c.isdigit()), None)
    return input_string[:index] if index is not None else None


def fetch_results_as_xarray_arb_var(handles, qubits, measurement_axis, var_name=None):
    """
    Fetch measurement results as an xarray dataset.
    
    Parameters
    ----------
    handles : dict
        Dictionary containing stream handles from job.result_handles
    qubits : list
        List of qubit objects to fetch data for
    measurement_axis : dict
        Dictionary mapping axis names to coordinate values, e.g. {"frequency": freqs, "flux": flux_vals}
    var_name : str, optional
        Specific variable name to fetch. If None, automatically detects from handles.
    
    Returns
    -------
    xarray.Dataset
        Dataset containing the fetched measurement results with proper dimensions and coordinates
    """
    if var_name is None:
        stream_handles = handles.keys()
        meas_vars = list(set([extract_string(handle) for handle in stream_handles if extract_string(handle) is not None]))
    else:
        meas_vars = [var_name]
    values = [
        [handles.get(f"{meas_var}{i + 1}").fetch_all() for i, qubit in enumerate(qubits)] for meas_var in meas_vars
    ]
    if np.array(values).shape[-1] == 1:
        values = np.array(values).squeeze(axis=-1)
    measurement_axis["qubit"] = [qubit.name for qubit in qubits]
    measurement_axis = {key: measurement_axis[key] for key in reversed(measurement_axis.keys())}
    
    
    ds = xr.Dataset(
        {f"{meas_var}": ([key for key in measurement_axis.keys()], values[i]) for i, meas_var in enumerate(meas_vars)},
        coords=measurement_axis,
    )

    return ds


# %% Node Parameters
class Parameters(NodeParameters):
    """Configuration parameters for Bayesian frequency estimation experiment."""
    
    # Qubit selection
    qubits: Optional[List[str]] = ["qC1"]

    # Experiment parameters
    num_repetitions: int = 100
    detuning: int = 0. * u.MHz
    min_wait_time_in_ns: int = 36
    max_wait_time_in_ns: int = 6000
    wait_time_step_in_ns: int = 80
    physical_detuning: int = 20 * u.MHz

    # Bayesian estimation parameters (MHz)
    # Note: Frequency range should be between 0 and 8 MHz due to QUA fixed variable limitations
    f_min: float = 0.  # MHz
    f_max: float = 1.0 # MHz
    df: float = 0.01  # MHz
    
    # Data collection
    keep_shot_data: bool = True

    # Execution parameters
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False


# Create experiment node
node = QualibrationNode(name="FrequencyBayes", parameters=Parameters())

# %% Initialize QuAM and QOP

# Load QuAM configuration
machine = Quam.load()

# Generate hardware configurations
config = machine.generate_config()

# Connect to quantum control hardware
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# Get qubit objects
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]

num_qubits = len(qubits)


# %% QUA Program
# Set up experiment parameters
v_f = np.arange(node.parameters.f_min, node.parameters.f_max + 0.5 * node.parameters.df, node.parameters.df)

flux_shifts = {}
for qubit in qubits:
    flux_shift = np.sqrt(-node.parameters.physical_detuning/qubit.freq_vs_flux_01_quad_term)
    flux_shifts[qubit.name] = flux_shift

n_reps = node.parameters.num_repetitions
idle_times = np.arange(
    node.parameters.min_wait_time_in_ns // 4,
    node.parameters.max_wait_time_in_ns // 4,
    node.parameters.wait_time_step_in_ns // 4,
)
detuning = node.parameters.detuning - node.parameters.physical_detuning

# Define QUA program
with program() as BayesFreq:
    # Declare variables
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)

    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]

    # Bayes variables
    frequencies = declare(fixed, value=v_f.tolist())
    Pf_st = [declare_stream() for _ in range(num_qubits)]
    estimated_frequency_st = [declare_stream() for _ in range(num_qubits)]

    # Main experiment loop
    for i, qubit in enumerate(qubits):
        t = declare(int)
        phase = declare(fixed)
        estimated_frequency = declare(fixed)  # in MHz
        Pf = declare(fixed, value=(np.ones(len(v_f)) / len(v_f)).tolist())
        norm = declare(fixed)
        s = declare(int)

        t_sample = declare(fixed)  # normalization for time in us
        f = declare(fixed)
        C = declare(fixed)
        rk = declare(fixed)

        # SPAM parameters
        alpha = declare(fixed)
        beta = declare(fixed)

        # SPAM parameters from confusion matrix
        assign(alpha, qubit.resonator.confusion_matrix[0][1] - qubit.resonator.confusion_matrix[1][0])
        assign(beta, 1 - qubit.resonator.confusion_matrix[0][1] - qubit.resonator.confusion_matrix[1][0])

        # Set flux bias
        machine.set_all_fluxes(flux_point="joint", target=qubit)
        # Averaging loop
        with for_(n, 0, n < n_reps, n + 1):
            save(n, n_st)

            # Time sweep loop
            with for_(*from_array(t, idle_times)):
                assign(phase, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))

                qubit.xy.play("x90")
                qubit.xy.frame_rotation_2pi(phase)
                qubit.z.wait(duration=qubit.xy.operations["x180"].length // 4)
                qubit.xy.wait(t)
                qubit.z.play(
                    "const",
                    amplitude_scale=flux_shifts[qubit.name] / qubit.z.operations["const"].amplitude,
                    duration=t
                )
                qubit.xy.play("x90")


                # Measurement
                readout_state(qubit, state[i])
                if node.parameters.keep_shot_data:
                    save(state[i], state_st[i])
                qubit.align()
                qubit.xy.play("x180", condition=Cast.to_bool(state[i]))

                assign(rk, Cast.to_fixed(state[i]) - 0.5)
                assign(t_sample, Cast.mul_fixed_by_int(1e-3, t * 4))
                f_idx = declare(int)

                # Update P(f)
                with for_(f_idx, 0, f_idx < len(v_f), f_idx + 1):
                    assign(C, Math.cos2pi(frequencies[f_idx] * t_sample))
                    assign(
                        Pf[f_idx],
                        (0.5 + rk * (alpha + beta * C) * 0.99) * Pf[f_idx],
                    )

                # Normalize P(f)
                assign(norm, Cast.to_fixed(0.01 / Math.sum(Pf)))
                assign(norm, Math.abs(norm))
                with for_(f_idx, 0, f_idx < len(v_f), f_idx + 1):
                    assign(Pf[f_idx], Cast.mul_fixed_by_int(norm * Pf[f_idx], 100))

                qubit.align()
                reset_frame(qubit.xy.name)

            # Estimated frequency (argmax of posterior)
            # assign(f_idx, Math.argmax(Pf))
            # assign(estimated_frequency, frequencies[f_idx])
            assign(estimated_frequency, Math.dot(frequencies, Pf))

            qubit.xy.play("x90", amplitude_scale=0, duration=4, timestamp_stream=f'time_stamp{i+1}')

            # Save and reset P(f)
            with for_(f_idx, 0, f_idx < len(v_f), f_idx + 1):
                save(Pf[f_idx], Pf_st[i])
                assign(Pf[f_idx], 1 / len(v_f))

            save(estimated_frequency, estimated_frequency_st[i])

    # Stream processing
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            Pf_st[i].buffer(n_reps, len(v_f)).save(f"Pf{i + 1}")
            if node.parameters.keep_shot_data:
                state_st[i].buffer(n_reps, len(idle_times)).save(f"state{i + 1}")
            estimated_frequency_st[i].buffer(n_reps).save(f"estimated_frequency{i + 1}")
# %% Simulate or Execute
if node.parameters.simulate:
    pass
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(BayesFreq)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, node.parameters.num_repetitions, start_time=results.start_time)

# %% Data Fetching and Dataset Creation
if not node.parameters.simulate:
    if node.parameters.keep_shot_data:
        ds_single = fetch_results_as_xarray_arb_var(
            job.result_handles, qubits,
            {"t": idle_times * 4, "repetition": np.arange(1, n_reps + 1)},
            "state"
        )
    else:
        ds_single = None

    vf_array = np.arange(
        node.parameters.f_min,
        node.parameters.f_max + 0.5 * node.parameters.df,
        node.parameters.df
    )
    ds_Pf = fetch_results_as_xarray_arb_var(
        job.result_handles, qubits,
        {"vf": vf_array, "repetition": np.arange(1, n_reps + 1)},
        "Pf"
    )
    ds_estimated_frequency = fetch_results_as_xarray_arb_var(
        job.result_handles, qubits,
        {"repetition": np.arange(1, n_reps + 1)},
        "estimated_frequency"
    )
    ds_time_stamp = fetch_results_as_xarray_arb_var(
        job.result_handles, qubits,
        {"repetition": np.arange(1, n_reps + 1)},
        "time_stamp"
    )

    timestamp_values = ds_time_stamp.time_stamp.values

    ds_time_stamp = ds_time_stamp.assign(time_stamp=(ds_time_stamp.time_stamp.dims, timestamp_values))
    time_stamp = ((ds_time_stamp - ds_time_stamp.min(dim="repetition")) * 4e-9).time_stamp

    if node.parameters.keep_shot_data:
        ds = xr.merge([ds_single, ds_Pf / ds_Pf.Pf.sum(dim='vf'), ds_estimated_frequency, time_stamp])
    else:
        ds = xr.merge([ds_Pf / ds_Pf.Pf.sum(dim='vf'), ds_estimated_frequency, ds_time_stamp])

    node.results = {"ds": ds}

    # %% Plotting

    # Create qubit grid for single-shot data
    if node.parameters.keep_shot_data:
        grid_shot = QubitGrid(ds, [q.grid_location for q in qubits])
        y_data_key = "state"
        # Loop over grid axes and qubits
        for ax, qubit in grid_iter(grid_shot):
            qubit_name = qubit["qubit"]
            t_vals = ds_single.t.values
            y_vals = ds_single[y_data_key].sel(qubit=qubit_name).values

            # Plot data with pcolormesh
            X, Y = np.meshgrid(t_vals*1e-3, time_stamp.sel(qubit=qubit_name).values)
            pcm = ax.pcolormesh(X, Y, y_vals)
            ax.set_xlabel("time (µs)")
            ax.set_ylabel("time (s)")
            ax.set_title(qubit_name)
            grid_shot.fig.colorbar(pcm, ax=ax, label=f"{y_data_key}")
            ax.grid(False)

        grid_shot.fig.suptitle("Single-shot data")
        plt.tight_layout()
        node.results["state_figure"] = grid_shot.fig

    # Create qubit grid for Bayesian probability distribution
    grid_bayes = QubitGrid(ds, [q.grid_location for q in qubits])
    y_data_key = "Pf"

    for ax, qubit in grid_iter(grid_bayes):
        qubit_name = qubit["qubit"]
        da = ds_Pf[y_data_key].sel(qubit=qubit_name)
        X, Y = np.meshgrid(da.vf.values, time_stamp.sel(qubit=qubit_name).values)
        data = da.values
        vmin = np.nanpercentile(data, 1)
        vmax = np.nanpercentile(data, 99)
        pcm = ax.pcolormesh(X, Y, data, vmin=vmin, vmax=vmax)
        ax.set_xlabel("frequency (MHz)")
        ax.set_ylabel("time (s)")
        ax.set_title(qubit_name)
        grid_bayes.fig.colorbar(pcm, ax=ax, label=y_data_key)
        ax.grid(False)

    grid_bayes.fig.suptitle("Frequency Bayes distribution")
    plt.tight_layout()
    node.results["PF_figure"] = grid_bayes.fig

    # Create qubit grid for estimated frequency
    grid_freq = QubitGrid(ds, [q.grid_location for q in qubits])
    y_data_key = "estimated_frequency"
    # Loop over grid axes and qubits
    for ax, qubit in grid_iter(grid_freq):
        qubit_name = qubit["qubit"]
        y_vals = ds_estimated_frequency[y_data_key].sel(qubit=qubit_name).values
        ax.plot(time_stamp.sel(qubit=qubit_name).values, y_vals, marker='.', linestyle='-', alpha=0.5)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Estimated Frequency (MHz)")
        ax.set_title(qubit_name)
        ax.grid(False)

    grid_freq.fig.suptitle("Estimated Frequency")
    plt.tight_layout()
    node.results["estimated_frequency_figure"] = grid_freq.fig

    # %% Save Results
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%
# alpha = 0.01
# beta = 0.8
# f = 1.e6
# df = 0.0e6
# T = 5000e-9

# n_shots = 50


# mus = []
# sigmas = []
# fs = []
# mu = 0.0e6
# for n_reps in range(500):
#     mu = 0.0e6
#     sigma = 0.1e6
#     f = f + 0.03*np.random.randn()*1e6
#     for i in range(n_shots):
#         tau = (np.sqrt(16*np.pi**2*sigma**2) - 1/T)/(8 * np.pi**2 * sigma**2)
#         df = 1/(4*tau) - mu
#         ps = np.array([0.5 + m /2 *(alpha + beta * np.exp(-tau / T) * np.cos(2*np.pi*(f+df)*tau)) for m in [-1, 1]])
#         ps = ps / np.sum(ps)
#         outcome = np.random.choice([-1, 1], p=ps)
#         exponential = np.exp(-tau / T - 2*np.pi**2 * sigma**2 * tau**2)
#         mu = mu - (2 * np.pi * outcome * beta * sigma**2 * tau * exponential) / (1 + alpha * outcome)
#         sigma = np.sqrt(sigma**2 - 1.0*(4 * np.pi**2 * beta**2 * sigma**4 * tau**2 * exponential**2) / (1 + alpha * outcome)**2)
#     mus.append(mu)
#     sigmas.append(sigma)
#     fs.append(f)
#     f = f - mu
# fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# axs[0].plot(mus, label='mu', color='tab:blue')
# axs[0].plot(fs, label='f', color='tab:red')
# axs[0].set_ylabel('mu')
# axs[0].legend()

# axs[1].plot(sigmas, label='sigma', color='tab:orange')
# axs[1].set_ylabel('sigma')
# axs[1].set_xlabel('Iteration')
# axs[1].legend()

# plt.tight_layout()
# plt.show()

# # %%
