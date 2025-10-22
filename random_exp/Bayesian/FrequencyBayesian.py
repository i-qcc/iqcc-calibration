"""
        RESONATOR SPECTROSCOPY VERSUS FLUX
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures. This is done across various readout intermediate dfs and flux biases.
The resonator frequency as a function of flux bias is then extracted and fitted so that the parameters can be stored in the state.

This information can then be used to adjust the readout frequency for the maximum and minimum frequency points.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude and duration.
    - Specification of the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Update the relevant flux biases in the state.
    - Save the current state
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from iqcc_calibration_tools.quam_config.components import Quam
from iqcc_calibration_tools.analysis.plot_utils import QubitGrid, grid_iter
from iqcc_calibration_tools.quam_config.lib.qua_datasets import convert_IQ_to_V
from iqcc_calibration_tools.storage.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.units import unit
from qualang_tools.multi_user import qm_session
from qm import SimulationConfig
from qm.qua import *
from typing import Optional, List, Literal
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from qualang_tools.results import progress_counter, fetching_tool
from datetime import datetime
from iqcc_calibration_tools.quam_config.macros import qua_declaration, active_reset, readout_state
from qualang_tools.loops import from_array
u = unit(coerce_to_integer=True)
<<<<<<< HEAD
<<<<<<< HEAD
from scipy.signal import welch
=======

>>>>>>> cf92d5e (Bayesian)
=======
from scipy.signal import welch
>>>>>>> 98ee2de (Bayesian estimation scripts)
# %% {Extra functions for data fetching}

def extract_string(input_string):
    # Find the index of the first occurrence of a digit in the input string
    index = next((i for i, c in enumerate(input_string) if c.isdigit()), None)

    if index is not None:
        # Extract the substring from the start of the input string to the index
        extracted_string = input_string[:index]
        return extracted_string
    else:
        return None


def fetch_results_as_xarray_arb_var(handles, qubits, measurement_axis, var_name = None):
    """
    Fetches measurement results as an xarray dataset.
    Parameters:
    - handles : A dictionary containing stream handles, obtained through handles = job.result_handles after the execution of the program.
    - qubits (list): A list of qubits.
    - measurement_axis (dict): A dictionary containing measurement axis information, e.g. {"frequency" : freqs, "flux",}.
    Returns:
    - ds (xarray.Dataset): An xarray dataset containing the fetched measurement results.
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


# %% {Node_parameters}
class Parameters(NodeParameters):
    # Define which qubits to measure
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    qubits: Optional[List[str]] = None
=======
    qubits: Optional[List[str]] = ["Q5"]
>>>>>>> b3e97ae (noise experiemnts WIP)

    # Experiment parameters
    num_repetitions: int = 20000
    detuning: int = 7 * u.MHz
    # min_wait_time_in_ns: int = 16
    min_wait_time_in_ns: int = 36
    max_wait_time_in_ns: int = 8000
    wait_time_step_in_ns: int = 72
    
    physical_detuning: int = 5 * u.MHz

    # Bayesian parameters - frequency should be between 0 and 8 MHz due to the limitation of fixed variables. Can be modified by chagning from MHz to 10MHz units.
    
    f_min: float = 6.5 #MHz
    f_max: float = 7.5 #MHz
    df: float = 0.02 #MHz
    
    keep_shot_data: bool = True
=======
    qubits: Optional[List[str]] = ["Q5"]
=======
    qubits: Optional[List[str]] = None
>>>>>>> 98ee2de (Bayesian estimation scripts)

    # Experiment parameters
    num_repetitions: int = 2000
    detuning: int = 7 * u.MHz
    # min_wait_time_in_ns: int = 16
    min_wait_time_in_ns: int = 36
    max_wait_time_in_ns: int = 8000
    wait_time_step_in_ns: int = 72
    
    physical_detuning: int = 5 * u.MHz

    # Bayesian parameters - frequency should be between 0 and 8 MHz due to the limitation of fixed variables. Can be modified by chagning from MHz to 10MHz units.
    
    f_min: float = 6.5 #MHz
    f_max: float = 7.5 #MHz
<<<<<<< HEAD
    df: float = 0.01 #MHz

    # Control parameters
    reset_type: Literal["active", "thermal"] = "thermal"
    use_state_discrimination: bool = True
>>>>>>> cf92d5e (Bayesian)
=======
    df: float = 0.02 #MHz
    
    keep_shot_data: bool = True
>>>>>>> 98ee2de (Bayesian estimation scripts)

    # Execution parameters
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False

# Create experiment node
node = QualibrationNode(name="FrequencyBayes", parameters=Parameters())

# %% {Initialize_QuAM_and_QOP}
# Initialize unit handling
u = unit(coerce_to_integer=True)

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
    


# %% {QUA_program}
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

<<<<<<< HEAD
<<<<<<< HEAD
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]

    # Bayes variables
    frequencies = declare(fixed, value=v_f.tolist())   
    Pf_st = [declare_stream() for _ in range(num_qubits)]
=======
    if node.parameters.use_state_discrimination:
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]

    t = declare(int)
    phase = declare(fixed)
=======
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]
>>>>>>> 98ee2de (Bayesian estimation scripts)

    # Bayes variables
    frequencies = declare(fixed, value=v_f.tolist())   
    Pf_st = [declare_stream() for _ in range(num_qubits)]
<<<<<<< HEAD

    norm = declare(fixed)
    s = declare(int)  # Variable for qubit state classification

    t_sample = declare(fixed) #normalization for time in us
    f = declare(fixed)
    C = declare(fixed)
    rk = declare(fixed)

    # SPAM parameters
    alpha = declare(fixed)
    beta = declare(fixed)


    estimated_frequency = declare(fixed) #in MHz
>>>>>>> cf92d5e (Bayesian)
=======
>>>>>>> 98ee2de (Bayesian estimation scripts)
    estimated_frequency_st = [declare_stream() for _ in range(num_qubits)]

    # Main experiment loop
    for i, qubit in enumerate(qubits):
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 98ee2de (Bayesian estimation scripts)
        # align()
        t = declare(int)
        phase = declare(fixed)        
        estimated_frequency = declare(fixed) #in MHz
        Pf = declare(fixed, value=(np.ones(len(v_f)) / len(v_f)).tolist())
        norm = declare(fixed)
        s = declare(int)  # Variable for qubit state classification

        t_sample = declare(fixed) #normalization for time in us
        f = declare(fixed)
        C = declare(fixed)
        rk = declare(fixed)

        # SPAM parameters
        alpha = declare(fixed)
        beta = declare(fixed)
<<<<<<< HEAD
=======
>>>>>>> cf92d5e (Bayesian)
=======
>>>>>>> 98ee2de (Bayesian estimation scripts)

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
                
                qubit.xy.wait(t )
                qubit.z.play("const", amplitude_scale=flux_shifts[qubit.name] / qubit.z.operations["const"].amplitude, 
                                duration=t)
                
                qubit.xy.play("x90") 


                # Measurement
                readout_state(qubit, state[i])
<<<<<<< HEAD
<<<<<<< HEAD
                if node.parameters.keep_shot_data:
                    save(state[i], state_st[i])
=======
                save(state[i], state_st[i])
>>>>>>> cf92d5e (Bayesian)
=======
                if node.parameters.keep_shot_data:
                    save(state[i], state_st[i])
>>>>>>> 98ee2de (Bayesian estimation scripts)
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
                        (0.5 + rk * (alpha  + beta * C)*0.99)
                        * Pf[f_idx],
                    )
                    
                # Normalize P(f)
                assign(norm, Cast.to_fixed(0.01 / Math.sum(Pf)))
                assign(norm, Math.abs(norm))
                with for_(f_idx, 0, f_idx < len(v_f), f_idx + 1):                    
                    assign(Pf[f_idx], Cast.mul_fixed_by_int(norm *  Pf[f_idx], 100))

                qubit.align()   
                    
                reset_frame(qubit.xy.name)
            
            # Estimated frequency
            # assign(estimated_frequency, Math.dot(frequencies, Pf))
            assign(f_idx, Math.argmax(Pf))
            assign(estimated_frequency, frequencies[f_idx])
            

            qubit.xy.play("x90", amplitude_scale=0, duration = 4,  timestamp_stream=f'time_stamp{i+1}')
            # Reset P(f)
            with for_(f_idx, 0, f_idx < len(v_f), f_idx + 1):
                    save(Pf[f_idx], Pf_st[i])
                    assign(Pf[f_idx], 1 / len(v_f))
                    
            save(estimated_frequency, estimated_frequency_st[i])                    
        # Stream processing
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            Pf_st[i].buffer(n_reps,len(v_f)).save(f"Pf{i + 1}")
<<<<<<< HEAD
<<<<<<< HEAD
            if node.parameters.keep_shot_data:
                state_st[i].buffer(n_reps,len(idle_times)).save(f"state{i + 1}")
=======
            state_st[i].buffer(n_reps,len(idle_times)).save(f"state{i + 1}")
>>>>>>> cf92d5e (Bayesian)
=======
            if node.parameters.keep_shot_data:
                state_st[i].buffer(n_reps,len(idle_times)).save(f"state{i + 1}")
>>>>>>> 98ee2de (Bayesian estimation scripts)
            estimated_frequency_st[i].buffer(n_reps).save(f"estimated_frequency{i + 1}")
# %% {Simulate_or_execute}
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

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 98ee2de (Bayesian estimation scripts)
    if node.parameters.keep_shot_data:
        ds_single = fetch_results_as_xarray_arb_var(job.result_handles, qubits, {"t": idle_times*4, "repetition": np.arange(1,n_reps+1)}, "state")
    else:
        ds_single = None
<<<<<<< HEAD
    ds_Pf = fetch_results_as_xarray_arb_var(job.result_handles, qubits, { "vf" : np.arange(node.parameters.f_min, node.parameters.f_max + 0.5 * node.parameters.df, node.parameters.df),"repetition": np.arange(1,n_reps+1)}, "Pf")
    ds_estimated_frequency = fetch_results_as_xarray_arb_var(job.result_handles, qubits, {"repetition": np.arange(1,n_reps+1)}, "estimated_frequency")
    ds_time_stamp = fetch_results_as_xarray_arb_var(job.result_handles, qubits, {"repetition": np.arange(1,n_reps+1)}, "time_stamp")

    # Extract numeric values from tuples if they exist
    raw_values = ds_time_stamp.time_stamp.values
    print(f"Raw values type: {type(raw_values)}")
    print(f"Raw values dtype: {raw_values.dtype}")
    print(f"Raw values shape: {raw_values.shape}")
    print(f"First few raw values: {raw_values.flat[:3]}")
    
    if isinstance(raw_values.flat[0], tuple):
        # If values are tuples, extract the first element (assuming it's the timestamp)
        timestamp_values = np.array([val[0] if isinstance(val, tuple) else val for val in raw_values.flat]).reshape(raw_values.shape)
        print("Extracted from tuples")
    else:
        timestamp_values = np.array(raw_values)
        print("Direct conversion")
    
    print(f"Timestamp values type: {type(timestamp_values)}")
    print(f"Timestamp values dtype: {timestamp_values.dtype}")
    print(f"Timestamp values shape: {timestamp_values.shape}")
    print(f"First few timestamp values: {timestamp_values.flat[:3]}")

    ds_time_stamp = ds_time_stamp.assign(time_stamp=(ds_time_stamp.time_stamp.dims, timestamp_values))
    time_stamp = ((ds_time_stamp - ds_time_stamp.min(dim = "repetition"))*4e-9).time_stamp    
    
    if node.parameters.keep_shot_data:
        ds = xr.merge([ds_single, ds_Pf / ds_Pf.Pf.sum(dim='vf'), ds_estimated_frequency, time_stamp])
    else:
        ds = xr.merge([ds_Pf / ds_Pf.Pf.sum(dim='vf'), ds_estimated_frequency, ds_time_stamp])
=======
    ds_single = fetch_results_as_xarray_arb_var(job.result_handles, qubits, {"t": idle_times*4, "repetition": np.arange(1,n_reps+1)}, "state")
    ds_Pf = fetch_results_as_xarray_arb_var(job.result_handles, qubits, { "vf" : np.arange(node.parameters.f_min, node.parameters.f_max + 0.5 * node.parameters.df, node.parameters.df),"repetition": np.arange(1,n_reps+1)}, "Pf")
    ds_estimated_frequency = fetch_results_as_xarray_arb_var(job.result_handles, qubits, {"repetition": np.arange(1,n_reps+1)}, "estimated_frequency")
    ds_time_stamp = fetch_results_as_xarray_arb_var(job.result_handles, qubits, {"repetition": np.arange(1,n_reps+1)}, "time_stamp")
    ds = xr.merge([ds_single, ds_Pf / ds_Pf.Pf.sum(dim='vf')])
>>>>>>> cf92d5e (Bayesian)
=======
    ds_Pf = fetch_results_as_xarray_arb_var(job.result_handles, qubits, { "vf" : np.arange(node.parameters.f_min, node.parameters.f_max + 0.5 * node.parameters.df, node.parameters.df),"repetition": np.arange(1,n_reps+1)}, "Pf")
    ds_estimated_frequency = fetch_results_as_xarray_arb_var(job.result_handles, qubits, {"repetition": np.arange(1,n_reps+1)}, "estimated_frequency")
    ds_time_stamp = fetch_results_as_xarray_arb_var(job.result_handles, qubits, {"repetition": np.arange(1,n_reps+1)}, "time_stamp")

    timestamp_values = ds_time_stamp.time_stamp.values

    ds_time_stamp = ds_time_stamp.assign(time_stamp=(ds_time_stamp.time_stamp.dims, timestamp_values))
    time_stamp = ((ds_time_stamp - ds_time_stamp.min(dim = "repetition"))*4e-9).time_stamp    
    
    if node.parameters.keep_shot_data:
        ds = xr.merge([ds_single, ds_Pf / ds_Pf.Pf.sum(dim='vf'), ds_estimated_frequency, time_stamp])
    else:
        ds = xr.merge([ds_Pf / ds_Pf.Pf.sum(dim='vf'), ds_estimated_frequency, ds_time_stamp])
>>>>>>> 98ee2de (Bayesian estimation scripts)
    
    node.results = {"ds": ds}

    # %% {Data_analysis}
<<<<<<< HEAD
<<<<<<< HEAD

    # Create DataArray of estimated frequency with 'qubit' and 'repetition' dimensions
    # Use the processed time_stamp for the time axis
    estimated_frequency_xr = xr.DataArray(
        ds_estimated_frequency.estimated_frequency.values,
        dims=["qubit", "repetition"],
        coords={
            "qubit": ds_estimated_frequency.qubit.values,
            "time_stamp": (("qubit", "repetition"), time_stamp.values)
        },
        name="estimated_frequency"
    )

    # Compute FFT along the 'repetition' axis for each qubit
    estimated_frequency = ds_estimated_frequency.estimated_frequency.values  # shape: (num_qubit, num_repetitions)
    n_qubits, n_reps = estimated_frequency.shape

    # If available, use the time step between adjacent repetitions to set frequency axis
    # Assume time_stamp (ms units) is shape (num_qubit, num_reps)
    t_vals = time_stamp.values  # shape: (num_qubit, num_reps)

    # Frequency step (Hz) for each qubit (if constant sampling)
    dt = np.mean(np.diff(t_vals, axis=1), axis=1)  # convert ms to s
    # If sampling rate is irregular, ignore for now and use a representative dt
    # Take mean dt for all qubits
    mean_dt = np.mean(dt)
    freq_axis = np.fft.rfftfreq(n_reps, d=mean_dt)  # in Hz

    # Compute FFT for all qubits
    fft_data = np.fft.rfft(estimated_frequency - np.nanmean(estimated_frequency, axis=1, keepdims=True), axis=1)
    fft_data = np.abs(fft_data)   # magnitude of FFT

    ds_estimated_frequency_fft = xr.DataArray(
        fft_data,
        dims=("qubit", "frequency"),
        coords={
            "qubit": ds_estimated_frequency.qubit.values,
            "frequency": freq_axis
        },
        name="estimated_frequency_fft"
    )
    
    
    # Compute Welch's transform (Welch PSD) for each qubit and pack into an xarray


    welch_freqs_list = []
    welch_psd_list = []
    nperseg = min(1024, n_reps)
    if nperseg < 16:
        nperseg = n_reps // 2

    # Use mean_dt as dt, fs=1/mean_dt
    fs = 1.0 / mean_dt

    for qidx in range(n_qubits):
        y = estimated_frequency[qidx] - np.nanmean(estimated_frequency[qidx])
        f, Pxx = welch(
            y,
            fs=fs,
            window='hann',
            nperseg=nperseg,
            noverlap=nperseg//2,
            scaling='density',
            detrend='constant'
        )
        welch_freqs_list.append(f)
        welch_psd_list.append(Pxx)

    # Assume the Welch frequency axis is the same for all qubits, so take the first
    welch_freqs = welch_freqs_list[0]
    welch_psd_arr = np.stack(welch_psd_list, axis=0)

    ds_estimated_frequency_welch = xr.DataArray(
        welch_psd_arr,
        dims=("qubit", "frequency"),
        coords={
            "qubit": ds_estimated_frequency.qubit.values,
            "frequency": welch_freqs
        },
        name="estimated_frequency_welch_psd"
    )
    
    # Compute the integrated noise density (cumulative spectral density) for each qubit
    # Integrated noise density is the cumulative sum of PSD * df up to each frequency value
    integrated_noise_density_arr = np.cumsum(welch_psd_arr * np.diff(welch_freqs, prepend=0), axis=1)

    ds_integrated_noise_density = xr.DataArray(
        integrated_noise_density_arr,
        dims=("qubit", "frequency"),
        coords={
            "qubit": ds_estimated_frequency.qubit.values,
            "frequency": welch_freqs
        },
        name="integrated_noise_density"
    )
=======
    # Fix timestamp data - extract 'value' field from structured array
    # timestamp_values = ds_time_stamp.time_stamp.values['value']
    timestamp_values = ds_time_stamp.time_stamp.values

    ds_time_stamp = ds_time_stamp.assign(time_stamp=(ds_time_stamp.time_stamp.dims, timestamp_values))
    time_stamp = ((ds_time_stamp - ds_time_stamp.min(dim = "repetition"))*4e-6).time_stamp
>>>>>>> cf92d5e (Bayesian)
=======

    # Create DataArray of estimated frequency with 'qubit' and 'repetition' dimensions
    # Use the processed time_stamp for the time axis
    estimated_frequency_xr = xr.DataArray(
        ds_estimated_frequency.estimated_frequency.values,
        dims=["qubit", "repetition"],
        coords={
            "qubit": ds_estimated_frequency.qubit.values,
            "time_stamp": (("qubit", "repetition"), time_stamp.values)
        },
        name="estimated_frequency"
    )

    # Compute FFT along the 'repetition' axis for each qubit
    estimated_frequency = ds_estimated_frequency.estimated_frequency.values  # shape: (num_qubit, num_repetitions)
    n_qubits, n_reps = estimated_frequency.shape

    # If available, use the time step between adjacent repetitions to set frequency axis
    # Assume time_stamp (ms units) is shape (num_qubit, num_reps)
    t_vals = time_stamp.values  # shape: (num_qubit, num_reps)

    # Frequency step (Hz) for each qubit (if constant sampling)
    dt = np.mean(np.diff(t_vals, axis=1), axis=1)  # convert ms to s
    # If sampling rate is irregular, ignore for now and use a representative dt
    # Take mean dt for all qubits
    mean_dt = np.mean(dt)
    freq_axis = np.fft.rfftfreq(n_reps, d=mean_dt)  # in Hz

    # Compute FFT for all qubits
    fft_data = np.fft.rfft(estimated_frequency - np.nanmean(estimated_frequency, axis=1, keepdims=True), axis=1)
    fft_data = np.abs(fft_data)   # magnitude of FFT

    ds_estimated_frequency_fft = xr.DataArray(
        fft_data,
        dims=("qubit", "frequency"),
        coords={
            "qubit": ds_estimated_frequency.qubit.values,
            "frequency": freq_axis
        },
        name="estimated_frequency_fft"
    )
    
    
    # Compute Welch's transform (Welch PSD) for each qubit and pack into an xarray


    welch_freqs_list = []
    welch_psd_list = []
    nperseg = min(1024, n_reps)
    if nperseg < 16:
        nperseg = n_reps // 2

    # Use mean_dt as dt, fs=1/mean_dt
    fs = 1.0 / mean_dt

    for qidx in range(n_qubits):
        y = estimated_frequency[qidx] - np.nanmean(estimated_frequency[qidx])
        f, Pxx = welch(
            y,
            fs=fs,
            window='hann',
            nperseg=nperseg,
            noverlap=nperseg//2,
            scaling='density',
            detrend='constant'
        )
        welch_freqs_list.append(f)
        welch_psd_list.append(Pxx)

    # Assume the Welch frequency axis is the same for all qubits, so take the first
    welch_freqs = welch_freqs_list[0]
    welch_psd_arr = np.stack(welch_psd_list, axis=0)

    ds_estimated_frequency_welch = xr.DataArray(
        welch_psd_arr,
        dims=("qubit", "frequency"),
        coords={
            "qubit": ds_estimated_frequency.qubit.values,
            "frequency": welch_freqs
        },
        name="estimated_frequency_welch_psd"
    )
    
    # Compute the integrated noise density (cumulative spectral density) for each qubit
    # Integrated noise density is the cumulative sum of PSD * df up to each frequency value
    integrated_noise_density_arr = np.cumsum(welch_psd_arr * np.diff(welch_freqs, prepend=0), axis=1)

    ds_integrated_noise_density = xr.DataArray(
        integrated_noise_density_arr,
        dims=("qubit", "frequency"),
        coords={
            "qubit": ds_estimated_frequency.qubit.values,
            "frequency": welch_freqs
        },
        name="integrated_noise_density"
    )
>>>>>>> 98ee2de (Bayesian estimation scripts)


    # %% {Plotting}

    grid_bayes = QubitGrid(ds, [q.grid_location for q in qubits])
    y_data_key = "Pf"

    for ax, qubit in grid_iter(grid_bayes):
        qubit_name = qubit["qubit"]
        da = ds_Pf[y_data_key].sel(qubit=qubit_name)
        X, Y = np.meshgrid(da.vf.values, time_stamp.sel(qubit=qubit_name).values)
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 98ee2de (Bayesian estimation scripts)
        # Robustify the plot: set vmin/vmax to 10th and 90th percentiles of the data (ignoring NaN)
        data = da.values
        vmin = np.nanpercentile(data, 1)
        vmax = np.nanpercentile(data, 99)
        pcm = ax.pcolormesh(X, Y, data, vmin=vmin, vmax=vmax)
<<<<<<< HEAD
        ax.set_xlabel("frequency (MHz)")
        ax.set_ylabel("time (s)")
=======
        pcm = ax.pcolormesh(X, Y, da.values)
        ax.set_xlabel("frequency (MHz)")
        ax.set_ylabel("time (ms)")
>>>>>>> cf92d5e (Bayesian)
=======
        ax.set_xlabel("frequency (MHz)")
        ax.set_ylabel("time (s)")
>>>>>>> 98ee2de (Bayesian estimation scripts)
        ax.set_title(qubit_name)
        grid_bayes.fig.colorbar(pcm, ax=ax, label=y_data_key)
        ax.grid(False)

    grid_bayes.fig.suptitle("Frequency Bayes distribution")
    plt.tight_layout()
<<<<<<< HEAD
<<<<<<< HEAD
    node.results["PF_figure"] = grid_bayes.fig

    # Create qubit grid
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
=======
=======
    node.results["PF_figure"] = grid_bayes.fig
>>>>>>> 98ee2de (Bayesian estimation scripts)

    # Create qubit grid
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

<<<<<<< HEAD
    grid_shot.fig.suptitle("Single-shot data")
    plt.tight_layout()
>>>>>>> cf92d5e (Bayesian)
=======
        grid_shot.fig.suptitle("Single-shot data")
        plt.tight_layout()
        node.results["state_figure"] = grid_shot.fig
>>>>>>> 98ee2de (Bayesian estimation scripts)
    
    # Create qubit grid
    grid_freq = QubitGrid(ds, [q.grid_location for q in qubits])
    y_data_key = "estimated_frequency"
    # Loop over grid axes and qubits
    for ax, qubit in grid_iter(grid_freq):
        qubit_name = qubit["qubit"]
        y_vals = ds_estimated_frequency[y_data_key].sel(qubit=qubit_name).values
        ax.plot(time_stamp.sel(qubit=qubit_name).values, y_vals, marker='o', linestyle='-', alpha=0.5)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("Estimated Frequency (MHz)")
        ax.set_title(qubit_name)
        ax.grid(False)
        ax.set_ylim([node.parameters.f_min, node.parameters.f_max])


    grid_freq.fig.suptitle("Estimated Frequency")
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 98ee2de (Bayesian estimation scripts)
    plt.tight_layout()   
    node.results["estimated_frequency_figure"] = grid_freq.fig
    
    grid_freq = QubitGrid(ds_estimated_frequency_fft, [q.grid_location for q in qubits])
    y_data_key = "estimated_frequency_fft"
    for ax, qubit in grid_iter(grid_freq):
        qubit_name = qubit["qubit"]
        # Get the frequency and value arrays, ignoring the first value
        freqs = ds_estimated_frequency_fft.frequency.values[1:]
        fft_vals = ds_estimated_frequency_fft.sel(qubit=qubit_name).values[1:]
        ax.plot(freqs, fft_vals)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Estimated Frequency FFT")
        ax.set_title(qubit_name)
        ax.grid(False)
    
    grid_freq.fig.suptitle("Estimated Frequency FFT")
    plt.tight_layout()
    node.results["estimated_frequency_fft_figure"] = grid_freq.fig
    
    # Plot the Welch-transformed frequency (PSD) for each qubit
    grid_freq = QubitGrid(ds_estimated_frequency_fft, [q.grid_location for q in qubits])
    y_data_key = "welch_psd"
    for ax, qubit in grid_iter(grid_freq):
        qubit_name = qubit["qubit"]
        ds_estimated_frequency_welch.sel(qubit=qubit_name).plot(ax = ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Welch PSD")
        ax.set_title(f"Welch PSD - {qubit_name}")
        ax.grid(False)
    grid_freq.fig.suptitle("Welch Power Spectral Density")
    plt.tight_layout()
    node.results["welch_psd_figure"] = grid_freq.fig
    
    grid_freq = QubitGrid(ds_integrated_noise_density, [q.grid_location for q in qubits])
    y_data_key = "integrated_noise_density"
    for ax, qubit in grid_iter(grid_freq):
        qubit_name = qubit["qubit"]
        ds_integrated_noise_density.sel(qubit=qubit_name).plot(ax = ax)
        ax.set_xscale("log")
        # ax.set_yscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Integrated Noise Density")
        ax.set_title(qubit_name)
        ax.grid(False)
    node.results["integrated_noise_density_figure"] = grid_freq.fig
    grid_freq.fig.suptitle("Integrated Noise Density")
    plt.tight_layout()
         
<<<<<<< HEAD
=======
    plt.tight_layout()    
>>>>>>> cf92d5e (Bayesian)
=======
>>>>>>> 98ee2de (Bayesian estimation scripts)
    # %%

    # %% {Update_state}

    # %% {Save_results}
<<<<<<< HEAD
<<<<<<< HEAD
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
<<<<<<< HEAD
=======
    # node.results["initial_parameters"] = node.parameters.model_dump()
    # node.machine = machine
    # node.save()



# %%
estimated_frequency_values = ds_estimated_frequency.estimated_frequency.values
time_values = time_stamp.values
import matplotlib.pyplot as plt
import numpy as np

# Flatten the estimated frequency values if necessary
if estimated_frequency_values.ndim > 1:
    freq_flat = estimated_frequency_values.flatten()
else:
    freq_flat = estimated_frequency_values

# Remove NaNs for FFT
freq_flat = freq_flat[~np.isnan(freq_flat)]

# Detrend
freq_flat = freq_flat - np.mean(freq_flat)

# time_values is in ms
if time_values.ndim > 1:
    time_1d = np.sort(np.unique(time_values.flatten()))
else:
    time_1d = np.sort(np.unique(time_values))

# Sampling interval in seconds (convert ms to s)
if len(time_1d) > 1:
    dt = np.mean(np.diff(time_1d)) * 1e-3
else:
    dt = 1.0  # fallback

N = len(freq_flat)
fft_data = np.fft.fft(freq_flat)
freqs = np.fft.fftfreq(N, d=dt)

# Only positive frequencies
pos_freqs = freqs[:N//2]
pos_fft = np.abs(fft_data)[:N//2]

plt.figure(figsize=(8, 4))
plt.plot(pos_freqs[pos_freqs > 0], pos_fft[pos_freqs > 0])
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("FFT of Estimated Frequency (log-log scale)")
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()

# Calculate the integrated spectral density (cumulative sum of spectral power)
# Spectral density is |FFT|^2
power_spectrum = pos_fft ** 2
integrated_spectral_density = np.cumsum(power_spectrum) * (pos_freqs[1] - pos_freqs[0])

plt.figure(figsize=(8, 4))
plt.plot(pos_freqs[pos_freqs > 0], integrated_spectral_density[pos_freqs > 0])
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Integrated Spectral Density")
plt.title("Integrated Spectral Density (log-log scale)")
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()



# %%
# Calculate the PSD using Welch's method for better low-frequency estimation
from scipy.signal import welch

# Welch's method parameters (you can tune nperseg for your needs)
nperseg = min(256, len(freq_flat))
if nperseg < 16:
    nperseg = len(freq_flat) // 2

# freq_flat: signal, dt: sampling interval
welch_freqs, welch_psd = welch(freq_flat, fs=1/dt, window='hann', nperseg=nperseg, noverlap=nperseg//2, scaling='density', detrend='constant')

plt.figure(figsize=(8, 4))
plt.loglog(welch_freqs[welch_freqs > 0], welch_psd[welch_freqs > 0])
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (Welch, Hann window)")
plt.title("Welch PSD of Estimated Frequency (log-x)")
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()

>>>>>>> cf92d5e (Bayesian)
=======
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
>>>>>>> 98ee2de (Bayesian estimation scripts)
=======

# %% {Interactive Welch PSD Plot with Peak Detection}
# Plot ds_estimated_frequency_welch with peak detection and annotation (linear scale, no QubitGrid)

from scipy.signal import find_peaks

if not node.parameters.simulate:
    fig, axes = plt.subplots(1, num_qubits, figsize=(6*num_qubits, 5))
    if num_qubits == 1:
        axes = [axes]
    
    for i, qubit in enumerate(qubits):
        qubit_name = qubit.name
        ax = axes[i]
        
        welch_data = ds_estimated_frequency_welch.sel(qubit=qubit_name)
        freqs = welch_data.frequency.values
        psd_values = welch_data.values
        
        # Plot the Welch PSD with linear scale
        ax.plot(freqs, psd_values, 'b-', linewidth=1, alpha=0.7)
        
        # Find peaks in the PSD
        # Use lower prominence threshold to detect smaller peaks
        peaks, properties = find_peaks(psd_values, prominence=np.max(psd_values) * 0.075)
        
        # Mark peaks with their frequency values
        if len(peaks) > 0:
            peak_freqs = freqs[peaks]
            peak_psd = psd_values[peaks]
            
            # Plot peak markers
            ax.plot(peak_freqs, peak_psd, 'ro', markersize=8, alpha=0.8)
            
            # Annotate each peak with its frequency value
            for j, (freq, psd) in enumerate(zip(peak_freqs, peak_psd)):
                ax.annotate(f'{freq:.2e} Hz', 
                           xy=(freq, psd), 
                           xytext=(10, 10), 
                           textcoords='offset points',
                           fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Welch PSD")
        ax.set_title(f"Welch PSD with Peaks - {qubit_name}")
        ax.grid(True, alpha=0.3)
        
    plt.suptitle("Welch Power Spectral Density with Peak Detection (Linear Scale)")
    plt.tight_layout()
    plt.show()
    
    # Store the figure in results
    node.results["welch_psd_peaks_linear_figure"] = fig

>>>>>>> b3e97ae (noise experiemnts WIP)
# %%
