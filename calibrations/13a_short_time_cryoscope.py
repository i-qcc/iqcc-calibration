# %%
"""
        CRYOSCOPE
"""

from datetime import datetime, timezone, timedelta
from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.multi_user import qm_session
from iqcc_calibration_tools.quam_config.macros import qua_declaration, active_reset
import numpy as np
from qualang_tools.units import unit
from iqcc_calibration_tools.quam_config.components import Quam
from qualang_tools.bakery import baking
from qualang_tools.loops import from_array

from iqcc_calibration_tools.analysis.plot_utils import QubitGrid, grid_iter
from iqcc_calibration_tools.storage.save_utils import fetch_results_as_xarray
import xarray as xr
from scipy.optimize import curve_fit
from scipy.signal import lfilter, convolve
from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode, NodeParameters
from typing import Optional, Literal, List
from iqcc_calibration_tools.analysis.cryoscope_tools import (
    expdecay, 
    resample_to_target_rate, 
    conv_causal,
    analyze_and_plot_inverse_fir
)
import time
start = time.time()

# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ['Q3']    
    num_averages: int = 7500
    frequency_offset_in_mhz: float = 600
    ramsey_offset_in_mhz: float = 0
    cryoscope_len: int = 64
    num_frames: int = 17
    reset_type_active_or_thermal: Literal['active', 'thermal'] = 'active'
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    simulate: bool = False
    timeout: int = 100
    only_FIR: bool = True
    load_data_id: Optional[int] = None
    
node = QualibrationNode(
    name="13c_cryoscope_frame_2freq",
    parameters=Parameters()
)

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = Quam.load()
# machine = Quam.load()
# Get the relevant QuAM components
if node.parameters.qubits is None:
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
              
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_active_or_thermal

num_qubits = len(qubits)

# %%

####################
# Helper functions #
####################

def baked_waveform(waveform_amp, qubit):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    waveform = [waveform_amp] * 16

    for i in range(1, 17):  # from first item up to pulse_duration (16)
        with baking(config, padding_method="left") as b:
            wf = waveform[:i]
            b.add_op("flux_pulse", qubit.z.name, wf)
            b.play("flux_pulse", qubit.z.name)

        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)

    return pulse_segments

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
cryoscope_len = node.parameters.cryoscope_len  # The length of the cryoscope in nanoseconds

assert cryoscope_len % 16 == 0, 'cryoscope_len is not multiple of 16 nanoseconds'
flux_amplitudes = {qubit.name: np.sqrt(-1e6*node.parameters.frequency_offset_in_mhz / qubit.freq_vs_flux_01_quad_term) for qubit in qubits}

baked_signals = {} # Baked flux pulse segments with 1ns resolution
baked_signals = baked_waveform(float(flux_amplitudes[qubits[0].name]), qubits[0]) 

cryoscope_time = np.arange(1, cryoscope_len+1, 1)  # x-axis for plotting - must be in ns
frames = np.linspace(0, 1, node.parameters.num_frames)

# %%
with program() as cryoscope:

    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    t = declare(int)  # QUA variable for the flux pulse segment index
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]
    virtual_detuning_phases = [declare(fixed) for _ in range(num_qubits)]
    global_state = declare(int)
    idx = declare(int)
    idx2 = declare(int)
    frame = declare(fixed)
    qubit = qubits[0]
    i = 0
    
    # Bring the active qubits to the desired frequency point
    machine.set_all_fluxes(flux_point=flux_point, target=qubit)

    # Outer loop for averaging
    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        # The first 16 nanoseconds
        with for_(idx, 0, idx<16, idx+1):
            
            assign(
                virtual_detuning_phases[0],
                Cast.mul_fixed_by_int(node.parameters.frequency_offset_in_mhz * 1e-3, idx),
            )
            
            with for_(*from_array(frame, frames)):
                if reset_type == "active":
                    for qubit in qubits:
                        active_reset(qubit)
                else:
                    wait(qubit.thermalization_time * u.ns)
                align()
                # Play first X/2
                for qubit in qubits:
                    qubit.xy.play("x90")
                align()
                # Delay between x90 and the flux pulse
                wait(4)
                align()
                with switch_(idx):
                    for j in range(16):
                        with case_(j):
                            baked_signals[j].run()
                # Wait for the idle time set slightly above the maximum flux pulse duration to ensure that the 2nd x90
                # pulse arrives after the longest flux pulse
                for qubit in qubits:
                    qubit.xy.wait((cryoscope_len + 160) // 4)
                    # Play second X/2
                    qubit.xy.frame_rotation_2pi(-1*virtual_detuning_phases[0])
                    qubit.xy.frame_rotation_2pi(frame)
                    qubit.xy.play("x90")
                    
                # Measure resonator state after the sequence
                align()
                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                save(state[i], state_st[i])

        with for_(t, 4, t < cryoscope_len // 4, t + 4):

            with for_(idx, 0, idx<16, idx+1):
                assign(
                virtual_detuning_phases[i],
                Cast.mul_fixed_by_int(node.parameters.frequency_offset_in_mhz * 1e-3, idx + t * 4),
                )
                
                with for_(*from_array(frame, frames)):
                    # Initialize the qubits
                    if reset_type == "active":
                        for qubit in qubits:
                            active_reset(qubit)
                    else:
                        wait(qubit.thermalization_time * u.ns)
                    align()
                    # Play first X/2
                    for qubit in qubits:
                        qubit.xy.play("x90")
                    align()
                    # Delay between x90 and the flux pulse
                    wait(4)
                    align()
                    with switch_(idx):
                        for j in range(16):
                            with case_(j):
                                baked_signals[j].run() 
                                qubits[0].z.play('const', duration=t, amplitude_scale=flux_amplitudes[qubits[0].name] / qubits[0].z.operations["const"].amplitude)

                    # Wait for the idle time set slightly above the maximum flux pulse duration to ensure that the 2nd x90
                    # pulse arrives after the longest flux pulse
                    for qubit in qubits:
                        qubit.xy.wait((cryoscope_len + 160) // 4)
                        # Play second X/2
                        qubit.xy.frame_rotation_2pi(-1*virtual_detuning_phases[i])
                        qubit.xy.frame_rotation_2pi(frame)
                        qubit.xy.play("x90")

                    # Measure resonator state after the sequence
                    align()
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                    save(state[i], state_st[i])

    with stream_processing():
        # for the progress counter
        n_st.save("iteration")
        for i, qubit in enumerate(qubits):
            state_st[i].buffer(len(frames)).buffer(cryoscope_len).average().save(f"state{i + 1}")

# %%
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=50000)  # In clock cycles = 4ns
    job = qmm.simulate(config, cryoscope, simulation_config)
    samples = job.get_simulated_samples()
    samples.con4.plot()
    plt.show()
    samples.con5.plot()
    plt.show()

elif node.parameters.load_data_id is None:
    date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
    with qm_session(qmm, config, timeout=node.parameters.timeout ) as qm:
        job = qm.execute(cryoscope)
        data_list = ["iteration"]
        results = fetching_tool(job, data_list, mode="live")

        while results.is_processing():
            fetched_data = results.fetch_all()
            n = fetched_data[0]
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, [qubit], {"frame": frames, "time": cryoscope_time})
        plot_process = True
        node.results['ds'] = ds
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
        
end = time.time()
print(f"Script runtime: {end - start:.2f} seconds")

# %% {data analysis - extract phase, frequency, and flux}
# Find phase of sine for each time step by fitting
def extract_phase(ds):
    phases = []
    for q in range(len(qubits)):
        phase_q = []
        for i in range(len(ds.state[q, :])):
            # Get data for this time step
            y_data = ds.state[q, i, :]
            x_data = ds.frame
            
            # Fit sine wave to get phase
            def sine_fit(x, phase, A, offset):
                return A * np.sin(2*np.pi*x + phase) + offset
                
            popt, _ = curve_fit(sine_fit, x_data, y_data, p0=[0, 1, 0.5], bounds=([-np.pi, 0, -np.inf], [np.pi, np.inf, np.inf]))
            # plt.plot(x_data, y_data,'.')
            # plt.plot(x_data, sine_fit(x_data, *popt))
            # plt.show()
            phase_q.append(popt[0])
        phases.append(np.unwrap(phase_q))
    return ds.assign_coords(phase=(['qubit', 'time'], phases))

def extract_freqs(ds):
    # old way of calculating frequencies
    # freqs = ds.phase.diff('time') / ds.time.diff('time') / (2*np.pi) + node.parameters.frequency_offset_in_mhz/1e3
    
    # Use np.gradient instead of diff to avoid NaN for the first value
    # np.gradient uses forward/backward differences at boundaries and central differences in the middle
    def calc_freq(phase_values, time_values):
        # Calculate derivative using gradient, which handles boundaries properly
        dphase_dt = np.gradient(phase_values, time_values, axis=-1)
        # Convert to frequency: d(phase)/dt / (2*pi) gives frequency in cycles per unit time
        freqs = dphase_dt / (2*np.pi) + node.parameters.frequency_offset_in_mhz/1e3
        return freqs
    
    # Apply gradient along the time dimension
    freqs = xr.apply_ufunc(
        calc_freq,
        ds.phase,
        ds.time,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[['time']],
        vectorize=True
    )
    # freqs = savgol_filter(freqs, window_length=5, polyorder=2)
    return ds.assign_coords(frequencies=1e3*freqs)

def extract_flux(ds):
    fluxes = []
    for qubit in qubits:
        fluxes.append(np.sqrt(np.abs(-1e6*ds.sel(qubit = qubit.name).frequencies / qubit.freq_vs_flux_01_quad_term)))
    return ds.assign_coords(flux=(['qubit', 'time'], fluxes))

ds = extract_phase(ds)
ds = extract_freqs(ds)
ds = extract_flux(ds)

# %% {data analysis - plot phase, frequency, and flux}
print('\033[1m\033[32m PLOT STATE, PHASE, FREQUENCY, AND FLUX \033[0m')
ds.state.sel(frame = 0).plot()
node.results['figure1'] = plt.gcf()
plt.title(f'state vs time \n {date_time} GMT+3 #{node.node_id} \n reset type = {node.parameters.reset_type_active_or_thermal}')
plt.show()

ds.phase.plot()
node.results['figure2'] = plt.gcf()
plt.title(f'phase vs time \n {date_time} GMT+3 #{node.node_id} \n reset type = {node.parameters.reset_type_active_or_thermal}')
plt.show()

ds.frequencies.plot()
node.results['figure3'] = plt.gcf()
plt.title(f'frequency vs time \n {date_time} GMT+3 #{node.node_id} \n reset type = {node.parameters.reset_type_active_or_thermal}')
plt.show()

ds.flux.plot()
node.results['figure4'] = plt.gcf()
plt.title(f'flux vs time \n {date_time} GMT+3 #{node.node_id} \n reset type = {node.parameters.reset_type_active_or_thermal}')
plt.show()

# %% {data analysis - setting rise and drop indices}
print('\033[1m\033[32m SETTING RISE AND DROP INDICES \033[0m')

if not node.parameters.simulate:
    # extract the rising part of the data for analysis
    da = ds.flux.sel(qubit = qubit.name)
    first_vals = da.sel(time=slice(0, 1)).mean().values
    final_vals = da.isel(time=slice(-20, None)).mean().values  
    
# %% {data analysis - exponential fit}
print('\033[1m\033[32m EXPONENTIAL FIT \033[0m')

exponential_fit_time_interval = [10,node.parameters.cryoscope_len-1]
# Get indices corresponding to the exponential_fit_time_interval 
time_slice = da.time.sel(time=slice(*exponential_fit_time_interval))
start_index, end_index = time_slice.time.values[0], time_slice.time.values[-1]

if not node.parameters.simulate and not node.parameters.only_FIR: # exponential fit is only done when only_FIR is False
    # Filtering the data might improve the fit at the first few nS, play with range to achieve this
  
    try:
        p0 = [final_vals, -1+first_vals/final_vals, 10]
        fit, pcov, infodict, errmsg, ier = curve_fit(expdecay, da.time[start_index:end_index], da[start_index:end_index],
                p0=p0, maxfev=10000, ftol=1e-8, full_output=True)
        
        # Calculate residuals and print fit information
        y_fit = expdecay(da.time, *fit)
        residuals = da - y_fit
        chi_squared = np.sum(residuals**2)
        print("\nSingle Exponential Fit Results:")
        print(f"Number of iterations: {infodict['nfev']}")
        print(f"Final chi-squared: {chi_squared:.6f}")
        print(f"RMS of residuals: {np.sqrt(np.mean(residuals**2)):.6f}")
        print(f"Fit parameters: {fit}")
        print(f"Parameter uncertainties: {np.sqrt(np.diag(pcov))}")
    except:
        fit = p0
        print('single exp fit failed')
        
    if plot_process:
        da.plot(marker = '.')
        plt.plot(da.time, expdecay(da.time, *fit), label = 'fit single exp')
        fit_text = f's={fit[0]:.6f}\na={fit[1]:.6f}\nt={fit[2]:.6f}'
        plt.text(0.02, 0.98, fit_text, transform=plt.gca().transAxes, verticalalignment='top', fontsize=8)
        
        plt.axvline(x=exponential_fit_time_interval[0], color='red', linestyle='--', label='Exponential Fit Time Interval')
        plt.axvline(x=exponential_fit_time_interval[1], color='red', linestyle='--')
        plt.title(f'Exponential Fit - {qubit.name} \n {date_time} GMT+3 #{node.node_id} \n reset type = {node.parameters.reset_type_active_or_thermal}')
        plt.legend()
        plt.show()

    # Print fit parameters nicely (expdecay function)
    print("\nFit parameters (expdecay function):")
    print(f"s: {fit[0]:.6f}")
    print(f"a: {fit[1]:.6f}")
    print(f"t: {fit[2]:.6f}")


    print('\033[1m\033[32m CALCULATE FILTERED RESPONSE \033[0m')
    from qualang_tools.digital_filters import calc_filter_taps

    exponential_filter = list(zip([fit[1]*1.0],[fit[2]]))
    feedforward_taps_1exp, feedback_tap_1exp = calc_filter_taps(exponential=exponential_filter)

    FIR_1exp = feedforward_taps_1exp
    IIR_1exp = [1,-feedback_tap_1exp[0]]
    # flux_cryoscope_q[0] = 0
    filtered_response_long_1exp = lfilter(FIR_1exp,IIR_1exp, da)

    if plot_process:
        f,ax = plt.subplots()
        ax.plot(da.time, da, label = 'data')
        ax.plot(da.time, filtered_response_long_1exp,label = 'filtered long time 1exp')
        # ax.set_ylim([final_vals*0.9,final_vals*1.05])
        ax.legend()
        ax.set_xlabel('time (ns)')
        ax.set_ylabel('flux')
        ax.set_title(f'Filtered Response - {qubit.name} \n {date_time} GMT+3 #{node.node_id} \n reset type = {node.parameters.reset_type_active_or_thermal}')
        plt.show()
        node.results['figure6'] = plt.gcf()



# %% {data analysis - calculate filtered response combined with FIR filter}
print('\033[1m\033[32m CALCULATE FILTERED RESPONSE COMBINED WITH FIR FILTER \033[0m')


if not node.parameters.simulate:
    ####  FIR filter for the response
    if node.parameters.only_FIR:
        response_long = da.values #da.isel(time=slice(1, None)).values
    else:
        response_long = filtered_response_long_1exp.values
        long_FIR = FIR_1exp
        long_IIR = IIR_1exp
    
    # flux_q = da[1:].copy()
    # flux_q.values = response_long
    # flux_q_tp = flux_q.isel(time=slice(0, 200)) # calculate the FIR only based on the first 200 nS
    # flux_q_tp = flux_q_tp.assign_coords(
    #     time=flux_q_tp.time - 0)
    # final_vals = flux_q_tp.isel(time=slice(-20, None)).mean().values
    # step = np.ones(len(flux_q)+100)*final_vals
    # fir_est = estimate_fir_coefficients(step, flux_q_tp.values, 24)
    # if node.parameters.only_FIR:
    #     convolved_fir = fir_est
    #     long_IIR = [1]
    # else:
    #     convolved_fir = convolve(long_FIR,fir_est, mode='full')

    normalized_response_raw = response_long / response_long[-10:].mean()
    normalized_response_2gsps = resample_to_target_rate(normalized_response_raw, 1, 0.5)
    time_2gsps = np.arange(len(normalized_response_2gsps)) * 0.5 + 1
    h_fir, inv_fir = analyze_and_plot_inverse_fir(
        response=normalized_response_2gsps,
        time=time_2gsps,
        Ts=0.5,
        M=48,
        sigma_ns=0.5,
        lam_smooth=1e-2,
        method='optimization',
        verbose=plot_process
        )
    ideal_response = np.ones(len(response_long))
    predistorted_response = lfilter(inv_fir, 1, ideal_response)
    corrected_response = lfilter(h_fir, 1, predistorted_response) # TODO: Generalize to use long_IIR
    
    # filtered_response_full = lfilter(fir_est, 1, response_long)
    
    # if plot_process:
    #     da.plot(label =  'data')
    #     if not node.parameters.only_FIR:
    #         plt.plot(da.time[1:], filtered_response_long_1exp, label = 'filtered long time')
    #     plt.plot(da.time, filtered_response_full, label = 'filtered full, deconvolved')
    #     plt.axhline(final_vals*1.001, color = 'k')
    #     plt.axhline(final_vals*0.999, color = 'k')
    #     plt.ylim([final_vals*0.95,final_vals*1.05])
    #     plt.title(f'Filtered Response - {qubit.name} \n {date_time} GMT+3 #{node.node_id} \n reset type = {node.parameters.reset_type_active_or_thermal}')
    #     plt.legend()
    #     plt.show()


# %% {plot final results}
print('\033[1m\033[32m PLOT FINAL RESULTS \033[0m')
if not node.parameters.simulate:
    # plotting the results
    fig,ax = plt.subplots()
    ax.plot(da.time,da / da[-10:].mean(), label = 'data')
    if not node.parameters.only_FIR:
        ax.plot(da.time, filtered_response_long_1exp / filtered_response_long_1exp[-10:].mean(),'--', label = 'slow rise correction')
    ax.plot(da.time, corrected_response / corrected_response[-10:].mean(), '--', label = 'expected corrected response')
    ax.axhline(1.001, color = 'k')
    ax.axhline(0.999, color = 'k')
    ax.set_ylim([0.95, 1.05])
    ax.legend()
    ax.set_xlabel('time (ns)')
    ax.set_ylabel('normalized amplitude')
    ax.set_title(f'Final Results - {qubit.name} \n {date_time} GMT+3 #{node.node_id} \n reset type = {node.parameters.reset_type_active_or_thermal}')
    node.results['figure6'] = fig

# %%
if not node.parameters.simulate:
    node.results['fit_results'] = {}
    for q in qubits:
        node.results['fit_results'][q.name] = {}
        node.results['fit_results'][q.name]['fir'] = inv_fir.tolist()
        if not node.parameters.only_FIR:
            node.results['fit_results'][q.name]['iir'] = exponential_filter

# %% {Update state}
# convert_to_2GSPS = True
# if not node.parameters.simulate:
#     with node.record_state_updates():
#         for qubit in qubits:
#             #check if the filter is already set
#             if qubit.z.opx_output.feedforward_filter is None:
#                 fir_list = inv_fir.tolist()
#             else:
#                 fir_list_old = qubit.z.opx_output.feedforward_filter
#                 if convert_to_2GSPS:
#                     fir_list_old = [fir_list_old[i] for i in range(0, len(fir_list_old), 2)]
#                 fir_list = np.convolve(fir_list_old, inv_fir.tolist())[:24]

#             if convert_to_2GSPS:
#                 fir_list = [element for item in fir_list for element in (item, item)]
                
#             qubit.z.opx_output.feedforward_filter = fir_list
#             if not node.parameters.only_FIR:
#                 qubit.z.opx_output.exponential_filter = [*qubit.z.opx_output.exponential_filter, *exponential_filter]

if not node.parameters.simulate:
    with node.record_state_updates():
        for qubit in qubits:
            #check if the filter is already set
            if qubit.z.opx_output.feedforward_filter is None:
                fir_list = inv_fir.tolist()
            else:
                inv_fir_old = qubit.z.opx_output.feedforward_filter
                fir_list = conv_causal(inv_fir, inv_fir_old).tolist()
                
            qubit.z.opx_output.feedforward_filter = fir_list
            if not node.parameters.only_FIR:
                qubit.z.opx_output.exponential_filter = [*qubit.z.opx_output.exponential_filter, *exponential_filter]

# %% {save node}
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()
# %%
