"""
"""

# %% {Imports}
from datetime import datetime, timezone, timedelta
from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode, NodeParameters
from iqcc_calibration_tools.quam_config.components import Quam
from iqcc_calibration_tools.quam_config.macros import qua_declaration
from iqcc_calibration_tools.quam_config.lib.qua_datasets import convert_IQ_to_V
from iqcc_calibration_tools.analysis.fit_utils import fit_resonator
from iqcc_calibration_tools.storage.save_utils import fetch_results_as_xarray, load_dataset
from iqcc_calibration_tools.analysis.twpa_utils import  * 
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import math


# %% {Node_parameters}
class Parameters(NodeParameters):
    twpas: Optional[List[str]] = ['twpa1']
    num_averages: int = 10
    amp_min: float =  0.2
    amp_max: float =  0.6
    amp_step: float = 0.01
    frequency_span_in_mhz: float = 7
    frequency_step_in_mhz: float =0.1
    p_frequency_span_in_mhz: float = 100
    p_frequency_step_in_mhz: float = 0.5
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    simulation_duration_ns: int = 4000
    timeout: int = 300
    load_data_id: Optional[int] = None
    pumpline_attenuation: int = -50-10-5 #(-50: fridge atten+directional coupler, -10: room temp line(8m), -5: fridge line)
    
node = QualibrationNode(name="twpa_calibration", parameters=Parameters())
# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = Quam.load()

# Get the relevant QuAM components
twpas = [machine.twpas[t] for t in node.parameters.twpas]
qubits = [machine.qubits[machine.twpas['twpa1'].qubits[i]] for i in range(len(machine.twpas['twpa1'].qubits))]
resonators = [machine.qubits[machine.twpas['twpa1'].qubits[i]].resonator for i in range(len(machine.twpas['twpa1'].qubits))]

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
#%% # readout pulse information
from iqcc_calibration_tools.quam_config.lib.qua_datasets import dBm, mV
signal_line_atten=-60-6-5 #-60dB : fridge atten, -6dB : cryogenic wiring, -5dB : room temp wiring
readout_power=[np.round(dBm(qubits[i].resonator.opx_output.full_scale_power_dbm,qubits[i].resonator.operations["readout"].amplitude)+signal_line_atten,2) for i in range(len(qubits))]
readout_length=[qubits[i].resonator.operations["readout"].length for i in range(len(qubits))]
readout_voltage=[np.round(mV(qubits[i].resonator.opx_output.full_scale_power_dbm,qubits[i].resonator.operations["readout"].amplitude),2) for i in range(len(qubits))]
for i in range(len(readout_power)):
    print(f"{qubits[i].name}: readout power @ resonator={readout_power[i]}dBm, opx output={readout_voltage[i]}mV, readout length={readout_length[i]}")
# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)

amp_max = node.parameters.amp_max
amp_min = node.parameters.amp_min
amp_step = node.parameters.amp_step
daps = np.arange(amp_min, amp_max, amp_step)
daps = np.insert(daps,0,0)

span_p = node.parameters.p_frequency_span_in_mhz * u.MHz
step_p = node.parameters.p_frequency_step_in_mhz * u.MHz
dfps = np.arange(-span_p / 2, +span_p / 2, step_p)
# pump duration should be able to cover the resonator spectroscopy which takes #(dfs) (as we are multiplexing qubit number doesnt matter) 
pump_duration = (10*len(dfs)*(machine.qubits[twpas[0].qubits[0]].resonator.operations["readout"].length+machine.qubits[twpas[0].qubits[0]].resonator.depletion_time))/4#(n_avg*len(dfs)*(machine.qubits[twpas[0].qubits[0]].resonator.operations["readout"].length+machine.qubits[twpas[0].qubits[0]].resonator.depletion_time))/4
with program() as twpa_calibration:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=len(qubits))
    dp = declare(int)  # QUA variable for the pump frequency
    da = declare(float)# QUA variable for the pump amplitude
    df = declare(int)  # QUA variable for the readout frequency

#### test for checking pump with SA
    # with infinite_loop_():
    #     update_frequency(twpas[0].pump.name,  17e6+ twpas[0].pump.intermediate_frequency)
    #     twpas[0].pump.play('pump', amplitude_scale=0.42, duration=pump_duration)
# ####
    # turn on twpa pump   
    with for_(n, 0, n < n_avg, n + 1):  
        save(n, n_st)
        with for_(*from_array(dp, dfps)):  
            update_frequency(twpas[0].pump.name, dp + twpas[0].pump.intermediate_frequency)
            with for_each_(da, daps):  
                twpas[0].pump.play('pump', amplitude_scale=da, duration=pump_duration)#+250)
                wait(250) #1000/4 wait 1us for pump to settle before readout
    # measure amplified readout responses around readout resonators with pump
                with for_(*from_array(df, dfs)):
                    for i, rr in enumerate(resonators):
                        # Update the resonator frequencies for all resonators
                        update_frequency(rr.name, df + rr.intermediate_frequency)
                        # Measure the resonator
                        rr.measure("readout", qua_vars=(I[i], Q[i]))
                        # wait for the resonator to relax
                        rr.wait(rr.depletion_time * u.ns)
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i]) 
                align()    
    with stream_processing():
        n_st.save("n")
        for i in range(len(qubits)):
            I_st[i].buffer(len(dfs)).buffer(len(daps)).buffer(len(dfps)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).buffer(len(daps)).buffer(len(dfps)).average().save(f"Q{i + 1}")
 
# ## kill
# qm=qmm.open_qm(config,close_other_machines=False)
# job=qm.execute(twpa_calibration)        
# #qm.close()            
# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, twpa_calibration, simulation_config)
    
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(f'{con}-Pump, Readout pulse simulation')
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(twpa_calibration)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            # progress_counter(n, n_avg, start_time=results.start_time)
# #%% control the window
# res = job.get_simulated_samples().con1
# t_min, t_max = 0,5000     # ns

# def t_axis(n_samples, sr):
#     return np.arange(n_samples) / sr * 1e9       # ns
# def label_from(port, typ):
#     port = str(port)
#     addr = port.split("-")
#     if len(addr) == 2:
#         return f"FEM{addr[0]}-{typ}O{addr[1]}"
#     if len(addr) == 3:
#         return f"FEM{addr[0]}-{typ}O{addr[1]}-UP{addr[2]}"
#     return port
# # --- analog ---
# for port, samples in res.analog.items():
#     sr     = res._analog_sampling_rate[str(port)]
#     t      = t_axis(len(samples), sr)
#     window = (t >= t_min) & (t <= t_max)

#     if np.iscomplexobj(samples):
#         I = samples.real
#         Q = samples.imag
#         if np.any(I[window]) or np.any(Q[window]):
#             plt.plot(t[window], I[window], label=f"{label_from(port,'A')} I")
#             plt.plot(t[window], Q[window], label=f"{label_from(port,'A')} Q")
#     else:
#         if np.any(samples[window]):
#             plt.plot(t[window], samples[window], label=label_from(port,'A'))
# # --- digital (optional, rarely useful in this window) ---
# for port, dig in res.digital.items():
#     if not np.any(dig):
#         continue
#     t  = t_axis(len(dig), 1e9)   # digital always at 1 GS/s
#     window = (t >= t_min) & (t <= t_max)
#     plt.plot(t[window], dig[window], label=label_from(port,'D'))
# plt.xlabel("Time [ns]")
# plt.ylabel("Output")
# # plt.legend()
# plt.xlim(t_min, t_max)          # keeps ticks sensible
# plt.tight_layout()
# plt.show()
# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    if node.parameters.load_data_id is not None:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    else:
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": dfs, "pump_amp": daps, "pump_freq" : dfps})
        # Convert IQ data into volts
        ds = convert_IQ_to_V(ds, qubits)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        ds = ds.assign({"IQ_abs": 1e3*np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        # get pump off - resonator spec, signal, snr
        pumpoff_resspec = pumpoff_res_spec_per_qubit(ds.IQ_abs, qubits, dfs, dfps)
        pumpoff_signal_snr = pumpzero_signal_snr(ds.IQ_abs, dfs, qubits, dfps, daps)
        # get pump on - max signal Gain, max DSNR
        pumpon_resspec_maxG = pumpoon_maxgain_res_spec(ds.IQ_abs, qubits,  dfps, daps,dfs)
        pumpon_resspec_maxDsnr = pumpoon_maxdsnr_res_spec(ds.IQ_abs, qubits,  dfps, daps,dfs)
        # get gain & snr improvement
        pumpon_signal_snr = pump_signal_snr(ds.IQ_abs, qubits, dfps, daps,dfs)
        gain_dsnr = pumpon_signal_snr-pumpoff_signal_snr
        # gain_dsnr = pumpon_signal_snr[:,:,1:len(daps)]-pumpoff_signal_snr[:,:,1:len(daps)] #to remove pump da=0
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        RF_freq = np.array([dfs + q.resonator.RF_frequency for q in qubits])
    node.results = {"ds": ds}

    # %% {Data_analysis}
    full_scale_power_dbm=twpas[0].pump.opx_output.full_scale_power_dbm
    print(f'twpa calibration took {pump_duration*n_avg*len(daps)*len(dfps)*1e-9}sec ')
    p_lo=twpas[0].pump.LO_frequency
    p_if=twpas[0].pump.intermediate_frequency
    # get pump point of max G
    pumpATmaxG=pump_maxgain(pumpon_signal_snr, dfps, daps)
    print(f'max Avg Gain at fp={np.round((p_lo+p_if+pumpATmaxG[0])*1e-9,3)}GHz,Pp={node.parameters.pumpline_attenuation+dBm(full_scale_power_dbm,pumpATmaxG[1])}')
    # get pump point of max dSNR
    pumpATmaxDSNR=pump_maxdsnr(pumpon_signal_snr,dfps, daps)
    print(f'max Avg dSNR at fp={np.round((p_lo+p_if+pumpATmaxDSNR[0])*1e-9,3)}GHz,Pp={node.parameters.pumpline_attenuation+dBm(full_scale_power_dbm,pumpATmaxDSNR[1])}')
    
    operation_point={'fp':np.round((p_lo+p_if+pumpATmaxDSNR[0]),3), 'Pp': node.parameters.pumpline_attenuation+dBm(full_scale_power_dbm,pumpATmaxDSNR[1])}
    node.results["pumping point"] = operation_point

    # %% {Plotting}
    # resonator
    ncols = 2
    nrows = math.ceil(len(qubits) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6, 8)) 
    axes = axes.flatten() 
    for i, ax in enumerate(axes):
        if i < len(qubits):  # only plot existing qubits
            ax.plot(RF_freq[i]*1e-9, pumpoff_resspec[i], label='pumpoff',color='black')
            ax.plot(RF_freq[i]*1e-9, pumpon_resspec_maxG[i], label='pump @ maxG',color='red')
            ax.plot(RF_freq[i]*1e-9, pumpon_resspec_maxDsnr[i], label='pump @ maxDsnr',color='blue')

            ax.set_title(f'{qubits[i].name}', fontsize=14)
            ax.set_xlabel('Res.freq [GHz]', fontsize=12)
            ax.set_ylabel('Trans.amp. [mV]', fontsize=12)
            ax.legend(fontsize=4, loc='upper right')
        else:
            ax.axis("off")  # hide empty subplot
    plt.tight_layout()
    plt.show()
    ##       
    pump_frequency=machine.twpas['twpa1'].pump.LO_frequency+machine.twpas['twpa1'].pump.intermediate_frequency+dfps
    pump_power=dBm(full_scale_power_dbm,daps)+node.parameters.pumpline_attenuation
    pump_power[np.isneginf(pump_power)]=0
    indices=np.linspace(0, len(pump_frequency)-1,10, dtype=int)
    selected_frequencies=np.round(pump_frequency[indices]*1e-9,3)
    ytick_pos = np.linspace(0, len(dfps)-1, len(selected_frequencies))
    indices_=np.linspace(0, len(pump_power)-1,10, dtype=int)
    selected_powers=np.round(pump_power[indices_],2)
    xtick_pos = np.linspace(0, len(daps)-1, len(selected_powers))
    # 
    gain_dsnr_avg=np.mean(gain_dsnr,axis=0)
    data_gain = gain_dsnr_avg[:, :, 0] 
    data_dSNR = gain_dsnr_avg[:, :, 1]  
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # plot gain vs pump
    im0 = axs[0].imshow(data_gain, origin='lower', aspect='auto',
                        extent=[0, len(daps)-1, 0, len(dfps)-1])
    axs[0].set_xticks(xtick_pos)
    axs[0].set_xticklabels(selected_powers,rotation=90)
    axs[0].set_yticks(ytick_pos)
    axs[0].set_yticklabels(selected_frequencies)
    axs[0].set_title('pump vs Gain', fontsize=20)
    axs[0].set_xlabel('pump power[dBm]', fontsize=20)
    axs[0].set_ylabel('pump frequency[GHz]', fontsize=20)
    cbar0 = fig.colorbar(im0, ax=axs[0])
    cbar0.set_label('Avg Gain [dB]', fontsize=14)
    # plot dSNR vs pump
    im1 = axs[1].imshow(data_dSNR, origin='lower', aspect='auto',
                        extent=[0, len(daps)-1, 0, len(dfps)-1])
    axs[1].set_xticks(xtick_pos)
    axs[1].set_xticklabels(selected_powers,rotation=90)
    axs[1].set_yticks(ytick_pos)
    axs[1].set_yticklabels(selected_frequencies)
    axs[1].set_title('pump vs dSNR', fontsize=20)
    axs[1].set_xlabel('pump amplitude', fontsize=20)
    axs[1].set_ylabel('pump frequency[GHz]', fontsize=20)
    cbar1 = fig.colorbar(im1, ax=axs[1])
    cbar1.set_label('Avg dSNR [dB]', fontsize=14)

    plt.tight_layout()
    plt.show()

    # %% {Update_state}
    # if not node.parameters.load_data_id:
    #     with node.record_state_updates():
           
    #         machine.twpas['twpa1'].pump.operations.pump.amplitude=pumpamp # need to find out how to update in the state file
    #         machine.twpas['twpa1'].pump.intermediate_frequency=pumpfreq

    #     # %% {Save_results}
    #     node.outcomes = {q.name: "successful" for q in qubits}
    #     node.results["initial_parameters"] = node.parameters.model_dump()
    #     node.machine = machine
    #     node.save()

# %%
