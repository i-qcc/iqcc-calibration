# %%
"""
        SHORT-TIME CRYOSCOPE

This node is designed for performing a short-time cryoscope calibration experiment on a specified qubit. 
The cryoscope protocol is used to characterize and analyze the flux distortion at short time scales 
(typically less than 100 ns).
"""


from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.multi_user import qm_session
from iqcc_calibration_tools.quam_config.macros import qua_declaration, active_reset
import numpy as np
from qualang_tools.units import unit
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam as Quam
from qualang_tools.bakery import baking
from qualang_tools.loops import from_array
from iqcc_calibration_tools.analysis.plot_utils import QubitGrid, grid_iter
from iqcc_calibration_tools.storage.save_utils import fetch_results_as_xarray
import xarray as xr
from scipy.optimize import curve_fit
from scipy.signal import lfilter
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
    qubits: Optional[List[str]] = None  
    num_averages: int = 2500
    frequency_offset_in_mhz: float = 800
    cryoscope_len: int = 64
    only_baked_waveforms: bool = True # not recommended when cryoscope_len is longer than ~120ns. May cause memory issues.
    num_frames: int = 17
    reset_type_active_or_thermal: Literal['active', 'thermal'] = 'active'
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    simulate: bool = False
    timeout: int = 100
    iir_or_fir: Literal['iir', 'fir'] = 'fir'
    num_forward_firs_values: List[int] = [16, 20, 24, 28, 32, 40, 48]
    lam1_values: List[float] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    lam2_values: List[float] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    num_inverse_firs: int = 48
    method: Literal['optimization', 'analytical'] = 'optimization'
    sigma_ns: float = 0.5
    lam_smooth: float = 5e-3
    load_data_id: Optional[int] = None
    
node = QualibrationNode(
    name="13a_short_time_cryoscope",
    parameters=Parameters()
)

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = Quam.load()
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

def baked_waveform(waveform_amp, qubit, cryoscope_baking_len):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    waveform = [waveform_amp] * cryoscope_baking_len

    for i in range(0, cryoscope_baking_len):  # from first item up to pulse_duration or 16ns
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
if node.parameters.only_baked_waveforms:
    cryoscope_baking_len = node.parameters.cryoscope_len
else:
    cryoscope_baking_len = 16

assert cryoscope_len % 16 == 0, 'cryoscope_len is not multiple of 16 nanoseconds'
flux_amplitudes = {qubit.name: np.sqrt(-1e6*node.parameters.frequency_offset_in_mhz / qubit.freq_vs_flux_01_quad_term) for qubit in qubits}

baked_signals = {} # Baked flux pulse segments with 1ns resolution
baked_signals = baked_waveform(float(flux_amplitudes[qubits[0].name]), qubits[0], cryoscope_baking_len) 

cryoscope_time = np.arange(0, cryoscope_len, 1)  # x-axis for plotting - must be in ns
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
    machine.initialize_qpu(flux_point=flux_point, target=qubit)

    # Outer loop for averaging
    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        # The first 16 nanoseconds
        with for_(idx, 0, idx<cryoscope_baking_len, idx+1):
            
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
                    for j in range(cryoscope_baking_len):
                        with case_(j):
                            baked_signals[j].run()
                # Wait for the idle time set slightly above the maximum flux pulse duration to ensure that the 2nd x90
                # pulse arrives after the longest flux pulse
                for qubit in qubits:
                    qubit.xy.wait((cryoscope_baking_len + 160) // 4)
                    # Play second X/2
                    qubit.xy.frame_rotation_2pi(-1*virtual_detuning_phases[0])
                    qubit.xy.frame_rotation_2pi(frame)
                    qubit.xy.play("x90")
                    
                # Measure resonator state after the sequence
                align()
                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                save(state[i], state_st[i])
        
        if not node.parameters.only_baked_waveforms:
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
        node.results['ds'] = ds
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results['ds']
        
end = time.time()
print(f"Script runtime: {end - start:.2f} seconds")

# %% {data analysis - extract phase, frequency, and flux}
# Find phase of sine for each time step by fitting
if not node.parameters.simulate:
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
                phase_q.append(popt[0])
            phases.append(np.unwrap(phase_q))
        return ds.assign_coords(phase=(['qubit', 'time'], phases))

    def extract_freqs(ds):
        def calc_freq(phase_values, time_values):
            dphase_dt = np.gradient(phase_values, time_values, axis=-1)
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
        return ds.assign_coords(frequencies=1e3*freqs)

    def extract_flux(ds):
        fluxes = []
        for qubit in qubits:
            fluxes.append(np.sqrt(np.abs(-1e6*ds.sel(qubit = qubit.name).frequencies / qubit.freq_vs_flux_01_quad_term)))
        return ds.assign_coords(flux=(['qubit', 'time'], fluxes))

    ds = extract_phase(ds)
    ds = extract_freqs(ds)
    ds = extract_flux(ds)
    node.results['ds'] = ds
# %% {data analysis - plot phase, frequency, and flux}
if not node.parameters.simulate:
    print('\033[1m\033[32m PLOT STATE, PHASE, FREQUENCY, AND FLUX \033[0m')
    ds.state.sel(frame = 0).plot()
    plt.gca().set_xlabel('time (ns)')
    plt.gca().set_ylabel('state')
    node.results['figure1'] = plt.gcf()
    plt.title(f'state vs time \n {node.date_time} GMT+{node.time_zone} #{node.node_id} \n reset type = {node.parameters.reset_type_active_or_thermal}')
    plt.show()

    ds.phase.plot()
    plt.gca().set_xlabel('time (ns)')
    plt.gca().set_ylabel('phase (radians)')
    node.results['figure2'] = plt.gcf()
    plt.title(f'phase vs time \n {node.date_time} GMT+{node.time_zone} #{node.node_id} \n reset type = {node.parameters.reset_type_active_or_thermal}')
    plt.show()

    ds.frequencies.plot()
    plt.gca().set_xlabel('time (ns)')
    plt.gca().set_ylabel('frequency (MHz)')
    node.results['figure3'] = plt.gcf()
    plt.title(f'frequency vs time \n {node.date_time} GMT+{node.time_zone} #{node.node_id} \n reset type = {node.parameters.reset_type_active_or_thermal}')
    plt.show()

    ds.flux.plot()
    plt.gca().set_xlabel('time (ns)')
    plt.gca().set_ylabel('flux (V)')
    node.results['figure4'] = plt.gcf()
    plt.title(f'flux vs time \n {node.date_time} GMT+{node.time_zone} #{node.node_id} \n reset type = {node.parameters.reset_type_active_or_thermal}')
    plt.show()

    da = ds.flux.sel(qubit = qubit.name)
# %% {data analysis - Calculate exponential filter}

if not node.parameters.simulate and node.parameters.iir_or_fir == 'iir': 
    # extract the rising part of the data for analysis
    first_vals = da.sel(time=slice(0, 1)).mean().values
    final_vals = da.isel(time=slice(-20, None)).mean().values
    exponential_fit_time_interval = [2, node.parameters.cryoscope_len-1]
    # Get indices corresponding to the exponential_fit_time_interval 
    time_slice = da.time.sel(time=slice(*exponential_fit_time_interval))
    start_index, end_index = time_slice.time.values[0], time_slice.time.values[-1]

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
        
    da.plot(marker = '.')
    plt.plot(da.time, expdecay(da.time, *fit), label = 'fit single exp')
    fit_text = f's={fit[0]:.6f}\na={fit[1]:.6f}\nt={fit[2]:.6f}'
    plt.text(0.02, 0.98, fit_text, transform=plt.gca().transAxes, verticalalignment='top', fontsize=8)
    
    plt.axvline(x=exponential_fit_time_interval[0], color='red', linestyle='--', label='Exponential Fit Time Interval')
    plt.axvline(x=exponential_fit_time_interval[1], color='red', linestyle='--')
    plt.title(f'Exponential Fit - {qubit.name} \n {node.date_time} GMT+{node.time_zone} #{node.node_id} \n reset type = {node.parameters.reset_type_active_or_thermal}')
    plt.legend()
    plt.show()

    print("\nFit parameters (expdecay function):")
    print(f"s: {fit[0]:.6f}")
    print(f"a: {fit[1]:.6f}")
    print(f"t: {fit[2]:.6f}")

    from qualang_tools.digital_filters import calc_filter_taps

    exponential_filter = list(zip([np.round(fit[1], 6)],[np.round(fit[2], 6)]))
    feedforward_taps_1exp, feedback_tap_1exp = calc_filter_taps(exponential=exponential_filter)

    FIR_1exp = feedforward_taps_1exp
    IIR_1exp = [1,-feedback_tap_1exp[0]]
    filtered_response_long_1exp = lfilter(FIR_1exp,IIR_1exp, da)

    f,ax = plt.subplots()
    ax.plot(da.time, da, label = 'data')
    ax.plot(da.time, filtered_response_long_1exp, label = 'expected filtered response')
    ax.legend()
    ax.set_xlabel('time (ns)')
    ax.set_ylabel('flux')
    ax.set_title(f'Filtered Response - {qubit.name} \n {node.date_time} GMT+{node.time_zone} #{node.node_id} \n reset type = {node.parameters.reset_type_active_or_thermal}')
    node.results['exponential_fit_figure'] = plt.gcf()
    plt.show()

# %% {data analysis - Calculate FIR filter}
if not node.parameters.simulate and node.parameters.iir_or_fir == 'fir':
    print('\033[1m\033[32m CALCULATE FILTERED RESPONSE USING FIR FILTER \033[0m')
    
    # Check if there's an existing filter to preserve its length
    existing_filter_length = None
    
    if qubit.z.opx_output.feedforward_filter is not None:
        existing_filter_length = len(qubit.z.opx_output.feedforward_filter)
        print(f"Found existing FIR filter with length {existing_filter_length}. Will preserve this length.")
    
    normalized_response_raw = da.values / da.values[-10:].mean()
    normalized_response_2gsps = resample_to_target_rate(normalized_response_raw, 1, 0.5)
    time_2gsps = np.arange(len(normalized_response_2gsps)) * 0.5 + 0.5
    h_fir, inv_fir, best_reconstructed_response, fig_fir_fit, fig_inv_fir_fit = analyze_and_plot_inverse_fir(
        response=normalized_response_2gsps,
        time=time_2gsps,
        Ts=0.5,
        L_values=node.parameters.num_forward_firs_values,
        lam1_values=node.parameters.lam1_values,
        lam2_values=node.parameters.lam2_values,
        M=existing_filter_length,  # Preserve existing filter length if present
        sigma_ns=node.parameters.sigma_ns,
        lam_smooth=node.parameters.lam_smooth,
        method=node.parameters.method,
        verbose=True
        ) 
    ideal_response = np.ones(len(da.values))
    predistorted_response = lfilter(inv_fir, 1, ideal_response)
    corrected_response = lfilter(h_fir, 1, predistorted_response)
    node.results['figure5'] = fig_fir_fit
    node.results['figure6'] = fig_inv_fir_fit

# %% {save fit results}
node.results['fit_results'] = {}
for q in qubits:
    node.results['fit_results'][q.name] = {}
    if node.parameters.iir_or_fir == 'fir':
        node.results['fit_results'][q.name]['inverse_fir'] = inv_fir.tolist()
        node.results['fit_results'][q.name]['forward_fir'] = h_fir.tolist()
        node.results['fit_results'][q.name]['corrected_response'] = corrected_response.tolist()
        node.results['fit_results'][q.name]['best_reconstructed_response'] = best_reconstructed_response.tolist()
    elif node.parameters.iir_or_fir == 'iir':
        node.results['fit_results'][q.name]['filtered_response_1exp'] = filtered_response_long_1exp.tolist()
        node.results['fit_results'][q.name]['exponential_filter'] = exponential_filter
    else:
        raise ValueError(f"Invalid value for iir_or_fir: {node.parameters.iir_or_fir}")

# %% {plot final results}
if not node.parameters.simulate:
    print('\033[1m\033[32m PLOT FINAL RESULTS \033[0m')
    fig, ax = plt.subplots()
    ax.plot(ds.time, da.values / da.values[-10:].mean(), label = 'data')
    if node.parameters.iir_or_fir == 'iir':
        ax.plot(ds.time, filtered_response_long_1exp / filtered_response_long_1exp[-10:].mean(), '--', label = 'predicted corrected response')
    elif node.parameters.iir_or_fir == 'fir':
        ax.plot(ds.time, corrected_response / corrected_response[-10:].mean(), '--', label = 'predicted corrected response')
    else:
        raise ValueError(f"Invalid value for iir_or_fir: {node.parameters.iir_or_fir}")
    ax.axhline(1.001, color = 'k')
    ax.axhline(0.999, color = 'k')
    ax.set_ylim([0.95, 1.05])
    ax.legend()
    ax.set_xlabel('time (ns)')
    ax.set_ylabel('normalized amplitude')
    ax.set_title(f'Final Results - {qubit.name} \n {node.date_time} GMT+{node.time_zone} #{node.node_id} \n reset type = {node.parameters.reset_type_active_or_thermal}')
    node.results['figure7'] = fig  

# %% {Update state}
if not node.parameters.simulate:
    with node.record_state_updates():
        for qubit in qubits:
            if node.parameters.iir_or_fir == 'fir':
                #check if the filter is already set
                if qubit.z.opx_output.feedforward_filter is None:
                    fir_list = inv_fir.tolist()
                else:
                    inv_fir_old = np.array(qubit.z.opx_output.feedforward_filter)
                    # Preserve the length of the existing filter when convolving
                    # This ensures we don't lose taps when the optimization finds a shorter filter
                    fir_list = conv_causal(inv_fir, inv_fir_old, N=len(inv_fir_old)).tolist()
                
                qubit.z.opx_output.feedforward_filter = fir_list
            elif node.parameters.iir_or_fir == 'iir':
                qubit.z.opx_output.exponential_filter = [*qubit.z.opx_output.exponential_filter, *exponential_filter]
            else:
                raise ValueError(f"Invalid value for iir_or_fir: {node.parameters.iir_or_fir}")

# %% {save node}
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()
# %%
