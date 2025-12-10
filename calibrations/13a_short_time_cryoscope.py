# %%
"""
        SHORT-TIME CRYOSCOPE

This node is designed for performing a short-time cryoscope calibration experiment on a specified qubit. 
The cryoscope protocol is used to characterize and analyze the flux distortion at short time scales 
(typically less than 100 ns).
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
from scipy.signal import lfilter
from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode, NodeParameters
from typing import Optional, Literal, List
from iqcc_calibration_tools.analysis.cryoscope_tools import (
    resample_to_target_rate, 
    conv_causal,
    analyze_and_plot_inverse_fir
)
import time
start = time.time()

# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ['qC2']    
    num_averages: int = 2500
    frequency_offset_in_mhz: float = 700
    cryoscope_len: int = 64
    num_frames: int = 17
    reset_type_active_or_thermal: Literal['active', 'thermal'] = 'active'
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    simulate: bool = False
    timeout: int = 100
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

def baked_waveform(waveform_amp, qubit):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    waveform = [waveform_amp] * node.parameters.cryoscope_len

    for i in range(1, node.parameters.cryoscope_len + 1):  # from first item up to pulse_duration (16)
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
        with for_(idx, 0, idx<cryoscope_len, idx+1):
            
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
                    for j in range(cryoscope_len):
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

        # with for_(t, 4, t < cryoscope_len // 4, t + 4):

        #     with for_(idx, 0, idx<16, idx+1):
        #         assign(
        #         virtual_detuning_phases[i],
        #         Cast.mul_fixed_by_int(node.parameters.frequency_offset_in_mhz * 1e-3, idx + t * 4),
        #         )
                
        #         with for_(*from_array(frame, frames)):
        #             # Initialize the qubits
        #             if reset_type == "active":
        #                 for qubit in qubits:
        #                     active_reset(qubit)
        #             else:
        #                 wait(qubit.thermalization_time * u.ns)
        #             align()
        #             # Play first X/2
        #             for qubit in qubits:
        #                 qubit.xy.play("x90")
        #             align()
        #             # Delay between x90 and the flux pulse
        #             wait(4)
        #             align()
        #             with switch_(idx):
        #                 for j in range(16):
        #                     with case_(j):
        #                         baked_signals[j].run() 
        #                         qubits[0].z.play('const', duration=t, amplitude_scale=flux_amplitudes[qubits[0].name] / qubits[0].z.operations["const"].amplitude)

        #             # Wait for the idle time set slightly above the maximum flux pulse duration to ensure that the 2nd x90
        #             # pulse arrives after the longest flux pulse
        #             for qubit in qubits:
        #                 qubit.xy.wait((cryoscope_len + 160) // 4)
        #                 # Play second X/2
        #                 qubit.xy.frame_rotation_2pi(-1*virtual_detuning_phases[i])
        #                 qubit.xy.frame_rotation_2pi(frame)
        #                 qubit.xy.play("x90")

        #             # Measure resonator state after the sequence
        #             align()
        #             qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
        #             assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
        #             save(state[i], state_st[i])

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

# %% {data analysis - Calculate FIR filter}
if not node.parameters.simulate:
    print('\033[1m\033[32m CALCULATE FILTERED RESPONSE USING FIR FILTER \033[0m')
    
    response_raw = ds.flux.sel(qubit = qubit.name).values
    normalized_response_raw = response_raw / response_raw[-10:].mean()
    normalized_response_2gsps = resample_to_target_rate(normalized_response_raw, 1, 0.5)
    time_2gsps = np.arange(len(normalized_response_2gsps)) * 0.5 + 1
    h_fir, inv_fir, fig_fir_fit, fig_inv_fir_fit = analyze_and_plot_inverse_fir(
        response=normalized_response_2gsps,
        time=time_2gsps,
        Ts=0.5,
        L_values=node.parameters.num_forward_firs_values,
        lam1_values=node.parameters.lam1_values,
        lam2_values=node.parameters.lam2_values,
        sigma_ns=node.parameters.sigma_ns,
        lam_smooth=node.parameters.lam_smooth,
        method=node.parameters.method,
        verbose=True
        ) 
    ideal_response = np.ones(len(response_raw))
    predistorted_response = lfilter(inv_fir, 1, ideal_response)
    corrected_response = lfilter(h_fir, 1, predistorted_response)
    node.results['figure5'] = fig_fir_fit
    node.results['figure6'] = fig_inv_fir_fit
    
    node.results['fit_results'] = {}
    for q in qubits:
        node.results['fit_results'][q.name] = {}
        node.results['fit_results'][q.name]['inverse_fir'] = inv_fir.tolist()
        node.results['fit_results'][q.name]['forward_fir'] = h_fir.tolist()
        node.results['fit_results'][q.name]['corrected_response'] = corrected_response

# %% {plot final results}
if not node.parameters.simulate:
    print('\033[1m\033[32m PLOT FINAL RESULTS \033[0m')
    fig, ax = plt.subplots()
    ax.plot(ds.time, normalized_response_raw, label = 'data')
    ax.plot(ds.time, corrected_response / corrected_response[-10:].mean(), '--', label = 'expected corrected response')
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
            #check if the filter is already set
            if qubit.z.opx_output.feedforward_filter is None:
                fir_list = inv_fir.tolist()
            else:
                inv_fir_old = qubit.z.opx_output.feedforward_filter
                fir_list = conv_causal(inv_fir, inv_fir_old).tolist()
                
            qubit.z.opx_output.feedforward_filter = fir_list

# %% {save node}
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()
# %%
