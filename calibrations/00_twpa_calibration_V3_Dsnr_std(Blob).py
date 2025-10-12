""" Jeongwon Kim, Akiva, Omrie, Wei  IQCC, 250924
Sweep pump frequency and amplitude
For each pumping point, calculate Gain, SNR improvement.
Currently optimized pumping point is maximum SNR improvement point(will be upgraded soon)

Prerequisites:
    - Need to the in which frequency dispersive feature of TWPA RPM resonator appears
    - Having calibrated the resonator frequency (nodes 02a, 02b and/or 02c).

* Gain is defined as the increase in the signal level.
    - twpa pump off : measure the signal response within  a 4MHz around the readout resonator
      singal_off= signal[dB] 
    - twpa pump on :  measure signal response within the same 4MHz around the readout resonator
      singal_on= signal[dB]
    => gain=signal_on-signal_off
* SNR improvement is defined using OPX1000 as Spectrum Analyzer.
    - twpa pump off : measure within a 4 MHz window around the readout resonator twice
      measure(amp=0) for noise level
      measure(amp=from state file) for signal level
      snr_off= signal[dB]-noise[dB]
    - twpa pump on :  measure within a 4 MHz window around the readout resonator twice
      measure(amp=0) for noise level
      measure(amp=from state file) for signal level
    => dsnr=snr_on-snr_off
"""

# %% {Imports}
from datetime import datetime, timezone, timedelta
from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode, NodeParameters
from iqcc_calibration_tools.quam_config.components import Quam
from iqcc_calibration_tools.quam_config.macros import qua_declaration
from iqcc_calibration_tools.quam_config.lib.qua_datasets import convert_IQ_to_V
from iqcc_calibration_tools.storage.save_utils import fetch_results_as_xarray
from iqcc_calibration_tools.analysis.twpa_utils import  * 
from qualang_tools.results import  fetching_tool
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
    num_averages: int = 30
    amp_min: float =  0.15
    amp_max: float =  0.26
    frequency_span_in_mhz: float = 4
    frequency_step_in_mhz: float = 0.1
    p_frequency_span_in_mhz: float = 40
    p_frequency_step_in_mhz: float =0.5
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    simulation_duration_ns: int = 4000
    timeout: int = 300
    load_data_id: Optional[int] = None
    pumpline_attenuation: int = -50 #(-50: fridge atten(-30)+directional coupler(-20)/ room temp line(4m)~-5,)  #-5: fridge line # exclude for now
    
node = QualibrationNode(name="twpa_calibration", parameters=Parameters())
date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")

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
n_avg = node.parameters.num_averages  
# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)
# pump amp, frequency sweep
full_scale_power_dbm=twpas[0].pump.opx_output.full_scale_power_dbm
amp_max = node.parameters.amp_max
amp_min = node.parameters.amp_min
amp_step = int((dBm(full_scale_power_dbm, amp_max)-dBm(full_scale_power_dbm, amp_min))/0.2)
daps = np.linspace(amp_min, amp_max, 40)#40)
# daps = np.arange(amp_min, amp_max, 0.01)

span_p = node.parameters.p_frequency_span_in_mhz * u.MHz
step_p = node.parameters.p_frequency_step_in_mhz * u.MHz
dfps = np.arange(-span_p / 2, +span_p / 2, step_p)
# pump duration should be able to cover the resonator spectroscopy which takes #(dfs) (as we are multiplexing qubit number doesnt matter) 
pump_duration = (10*len(dfs)*(machine.qubits[twpas[0].qubits[0]].resonator.operations["readout"].length+machine.qubits[twpas[0].qubits[0]].resonator.depletion_time))/4#(n_avg*len(dfs)*(machine.qubits[twpas[0].qubits[0]].resonator.operations["readout"].length+machine.qubits[twpas[0].qubits[0]].resonator.depletion_time))/4
with program() as twpa_pump_off_n:    
    I, I_st, Q, Q_st,n,n_st = qua_declaration(num_qubits=len(qubits))
    dp = declare(int)  # QUA variable for the pump frequency
    da = declare(float)# QUA variable for the pump amplitude
# ### test for checking pump with SA
#     with infinite_loop_():
#         update_frequency(twpas[0].pump.name,  2e6+ twpas[0].pump.intermediate_frequency)
#         twpas[0].pump.play('pump', amplitude_scale=0.63, duration=pump_duration)
# TWPA off
    with for_(n, 0, n < n_avg, n + 1):  
        save(n, n_st)
        with for_(*from_array(dp, dfps)): 
            with for_each_(da, daps):
# measure readout responses around readout resonators without pump
                for i, rr in enumerate(resonators):
                    # Update the resonator frequencies for all resonators
                    update_frequency(rr.name,  rr.intermediate_frequency)
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
            I_st[i].buffer(len(daps)).buffer(len(dfps)).buffer(n_avg).save(f"I{i + 1}")
            Q_st[i].buffer(len(daps)).buffer(len(dfps)).buffer(n_avg).save(f"Q{i + 1}")
with program() as twpa_pump_on_n:    
    I, I_st, Q, Q_st,n,n_st = qua_declaration(num_qubits=len(qubits))
    dp = declare(int)  # QUA variable for the pump frequency
    da = declare(float)# QUA variable for the pump amplitude
    with for_(n, 0, n < n_avg, n + 1):  
        save(n, n_st)
        with for_(*from_array(dp, dfps)):  
            update_frequency(twpas[0].pump.name, dp + twpas[0].pump.intermediate_frequency)
            with for_each_(da, daps):  
                twpas[0].pump.play('pump', amplitude_scale=da, duration=pump_duration)#+250)
                wait(250) #1000/4 wait 1us for pump to settle before readout
                for i, rr in enumerate(resonators):
                    # Update the resonator frequencies for all resonators
                    update_frequency(rr.name,  rr.intermediate_frequency)
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
            I_st[i].buffer(len(daps)).buffer(len(dfps)).buffer(n_avg).save(f"I{i + 1}")
            Q_st[i].buffer(len(daps)).buffer(len(dfps)).buffer(n_avg).save(f"Q{i + 1}")
with program() as twpa_pump_off_s:    
    I, I_st, Q, Q_st,n,n_st = qua_declaration(num_qubits=len(qubits))
    dp = declare(int)  # QUA variable for the pump frequency
    da = declare(float)# QUA variable for the pump amplitude
    df = declare(int)  # QUA variable for the readout frequency=pump_duration)

    with for_(n, 0, n < n_avg, n + 1):  
        save(n, n_st)
        with for_(*from_array(dp, dfps)):  
            with for_each_(da, daps): 
# measure readout responses around readout resonators without pump
                with for_(*from_array(df, dfs)):
                    for i, rr in enumerate(resonators):
                        # Update the resonator frequencies for all resonators
                        update_frequency(rr.name, df + rr.intermediate_frequency)
                        # Measure the resonator
                        rr.measure("readout",  qua_vars=(I[i], Q[i]))
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
with program() as twpa_pump_on_s:    
    I, I_st, Q, Q_st,n,n_st = qua_declaration(num_qubits=len(qubits))
    dp = declare(int)  # QUA variable for the pump frequency
    da = declare(float)# QUA variable for the pump amplitude
    df = declare(int)  # QUA variable for the readout frequency
    with for_(n, 0, n < n_avg, n + 1):  
        save(n, n_st)
        with for_(*from_array(dp, dfps)):  
            update_frequency(twpas[0].pump.name, dp + twpas[0].pump.intermediate_frequency)
            with for_each_(da, daps):  
                twpas[0].pump.play('pump', amplitude_scale=da, duration=pump_duration)#+250)
                wait(250) #1000/4 wait 1us for pump to settle before readout
                with for_(*from_array(df, dfs)):
                    for i, rr in enumerate(resonators):
                        # Update the resonator frequencies for all resonators
                        update_frequency(rr.name, df + rr.intermediate_frequency)
                        # Measure the resonator
                        rr.measure("readout",  qua_vars=(I[i], Q[i]))
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
# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, twpa_pump_off, simulation_config)
    
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
        job_off_n = qm.execute(twpa_pump_off_n)
        results_off_n = fetching_tool(job_off_n, ["n"], mode="live")
        while results_off_n.is_processing():
            n_off_n = results_off_n.fetch_all()[0]
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job_on_n = qm.execute(twpa_pump_on_n)
        results_on_n = fetching_tool(job_on_n, ["n"], mode="live")
        while results_on_n.is_processing():
            n_on_n = results_on_n.fetch_all()[0]
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job_off_s = qm.execute(twpa_pump_off_s)
        results_off_s = fetching_tool(job_off_s, ["n"], mode="live")
        while results_off_s.is_processing():
            n_off_s = results_off_s.fetch_all()[0]
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job_on_s = qm.execute(twpa_pump_on_s)
        results_on_s = fetching_tool(job_on_s, ["n"], mode="live")
        while results_on_s.is_processing():
            n_on_s = results_on_s.fetch_all()[0]
# %% {Data_fetching_and_dataset_creation}
#data for pump off
ds_off_n = fetch_results_as_xarray(job_off_n.result_handles, qubits, {"pump_amp": daps, "pump_freq" : dfps, "n" : np.arange(n_avg)})
ds_on_n = fetch_results_as_xarray(job_on_n.result_handles, qubits, {"pump_amp": daps, "pump_freq" : dfps, "n" : np.arange(n_avg)})
ds_off_s = fetch_results_as_xarray(job_off_s.result_handles, qubits, {"freq": dfs, "pump_amp": daps, "pump_freq" : dfps})
ds_on_s = fetch_results_as_xarray(job_on_s.result_handles, qubits, {"freq": dfs, "pump_amp": daps, "pump_freq" : dfps})
# Convert IQ data into volts
ds_off_n = convert_IQ_to_V(ds_off_n, qubits)
ds_on_n = convert_IQ_to_V(ds_on_n, qubits)
ds_off_s = convert_IQ_to_V(ds_off_s, qubits)
ds_on_s = convert_IQ_to_V(ds_on_s, qubits)
# Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
ds_off_s = ds_off_s.assign({"IQ_abs_signal": np.sqrt(ds_off_s["I"] ** 2 + ds_off_s["Q"] ** 2)})
ds_on_s = ds_on_s.assign({"IQ_abs_signal": np.sqrt(ds_on_s["I"] ** 2 + ds_on_s["Q"] ** 2)})

# %% {Data Analysis}
# SNR improvement & Gain
RF_freq = np.array([dfs + q.resonator.RF_frequency for q in qubits])
signal_off=signal(ds_off_s)
signal_on=signal(ds_on_s)
noise_on=noise(ds_on_n,qubits, dfps, daps, n_avg)
noise_off=noise(ds_off_n, qubits, dfps, daps, n_avg)
dsnr = (signal_on-signal_off)-(noise_on-noise_off)
Gain = signal_on-signal_off
node.results = {"snr_improvement": dsnr,
                "gain": Gain}
print(f'twpa calibration took {pump_duration*n_avg*len(daps)*len(dfps)*1e-9}sec ')
p_lo=twpas[0].pump.LO_frequency
p_if=twpas[0].pump.intermediate_frequency
# pump at max avg_gain
pumpATmaxG=pump_maxgain(Gain, dfps, daps)
print(f'max Avg Gain({np.round(np.max(np.mean(Gain,axis=0)))}dB) at fp={np.round((p_lo+p_if+pumpATmaxG[0][0])*1e-9,3)}GHz,Pp={np.round(node.parameters.pumpline_attenuation+dBm(full_scale_power_dbm,pumpATmaxG[0][1]),2)},Pamp={np.round(pumpATmaxG[0][1],3)}')
# pump at max dSNR
pumpATmaxDSNR=pump_maxdsnr(dsnr, dfps, daps)
print(f'max Avg dSNR({np.round(np.max(np.mean(dsnr,axis=0)),2)}dB) at fp={np.round((p_lo+p_if+pumpATmaxDSNR[0][0])*1e-9,3)}GHz,Pp={np.round(node.parameters.pumpline_attenuation+dBm(full_scale_power_dbm,pumpATmaxDSNR[0][1]),2)},Pamp={np.round(pumpATmaxG[0][1],3)}')
operation_point={'fp':np.round((p_lo+p_if+pumpATmaxDSNR[0][0]),3), 
                 'Pp': node.parameters.pumpline_attenuation+dBm(full_scale_power_dbm,pumpATmaxDSNR[0][1]),
                 'Pamp': np.round(pumpATmaxG[0][1],3)}
node.results["pumping point"] = operation_point
# %% {Plotting}
# resonator response
cmap = plt.cm.viridis.copy()
cmap.set_under('gray')
ncols = 2
nrows = math.ceil(len(qubits) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(6, 8))
axes = axes.flatten()
for i, ax in enumerate(axes):
    if i < len(qubits):
        for n in range(n_avg):
            ax.scatter(ds_off_n.I.values[i][n][0][0] * 1e3,
                       ds_off_n.Q.values[i][n][0][0] * 1e3,
                       color='black', s=5)
            ax.scatter(ds_on_n.I.values[i][n][pumpATmaxG[1][0]][pumpATmaxG[1][1]] * 1e3,
                       ds_on_n.Q.values[i][n][pumpATmaxG[1][0]][pumpATmaxG[1][1]] * 1e3,
                       color='red', s=5)
            ax.scatter(ds_on_n.I.values[i][n][pumpATmaxDSNR[1][0]][pumpATmaxDSNR[1][1]] * 1e3,
                       ds_on_n.Q.values[i][n][pumpATmaxDSNR[1][0]][pumpATmaxDSNR[1][1]] * 1e3,
                       color='blue', s=5)

        ax.set_title(f'{qubits[i].name}, Noise', fontsize=14)
        ax.set_xlabel('I [mV]', fontsize=12)
        ax.set_ylabel('Q [mV]', fontsize=12)
    else:
        ax.axis("off")

handles = [
    plt.Line2D([], [], color='black', marker='o', linestyle='', label='pump off'),
    plt.Line2D([], [], color='red', marker='o', linestyle='', label='pump @ maxG'),
    plt.Line2D([], [], color='blue', marker='o', linestyle='', label='pump @ maxDSNR')
]
# Place the legend below all subplots
fig.legend(handles=handles,
           loc='lower center',
           ncol=3,
           fontsize=10,
           frameon=False,
           bbox_to_anchor=(0.5, -0.02))  # adjust vertical offset as needed

plt.subplots_adjust(top=0.93, bottom=0.10, hspace=0.5, wspace=0.4)
plt.show()
fig_, axes_ = plt.subplots(nrows, ncols, figsize=(6, 8))
axes_ = axes_.flatten()

for i, ax in enumerate(axes_):
    if i < len(qubits):
        ax.plot(RF_freq[i]*1e-9,
                ds_off_s.IQ_abs_signal.values[i].mean(axis=(0,1))*1e3,
                label='pump off', color='black')
        ax.plot(RF_freq[i]*1e-9,
                ds_on_s.IQ_abs_signal.values[i][pumpATmaxG[1][0]][pumpATmaxG[1][1]]*1e3,
                label='pump @ maxG', color='red')
        ax.plot(RF_freq[i]*1e-9,
                ds_on_s.IQ_abs_signal.values[i][pumpATmaxDSNR[1][0]][pumpATmaxDSNR[1][1]]*1e3,
                label='pump @ maxDSNR', color='blue')

        ax.set_title(f'{qubits[i].name}, Signal', fontsize=14)
        ax.set_xlabel('Res. freq [GHz]', fontsize=12)
        ax.set_ylabel('Trans. amp. [mV]', fontsize=12)
    else:
        ax.axis("off")
handles = [
    plt.Line2D([], [], color='black', label='pump off'),
    plt.Line2D([], [], color='red', label='pump @ maxG'),
    plt.Line2D([], [], color='blue', label='pump @ maxDSNR')
]

fig_.legend(handles=handles,
            loc='lower center',
            ncol=3,
            fontsize=10,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02))
plt.subplots_adjust(top=0.93, bottom=0.10, hspace=0.5, wspace=0.4)
plt.show()

##  plot Gain vs pump      
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
gain_avg=np.mean(Gain,axis=0)
dsnr_avg=np.mean(dsnr,axis=0)
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
print(f'max Avg dSNR({np.round(np.max(np.mean(dsnr,axis=0)),2)}dB) \n at fp={np.round((p_lo+p_if+pumpATmaxDSNR[0][0])*1e-9,3)}GHz,Pp={np.round(node.parameters.pumpline_attenuation+dBm(full_scale_power_dbm,pumpATmaxDSNR[0][1]),2)},Pamp={np.round(pumpATmaxDSNR[0][1],3)} \n {date_time} \n {len(dfs)}*{len(daps)}*{len(dfps)}*{n_avg}')

# plot gain vs pump
im0 = axs[0].imshow(gain_avg, origin='lower', aspect='auto',
                    extent=[0, len(daps)-1, 0, len(dfps)-1],cmap=cmap,vmin=0)  
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
im1 = axs[1].imshow(dsnr_avg, origin='lower', aspect='auto',
                    extent=[0, len(daps)-1, 0, len(dfps)-1],cmap=cmap,vmin=0)
axs[1].set_xticks(xtick_pos)
axs[1].set_xticklabels(selected_powers,rotation=90)
axs[1].set_yticks(ytick_pos)
axs[1].set_yticklabels(selected_frequencies)
axs[1].set_title('pump vs dSNR', fontsize=20)
axs[1].set_xlabel('pump amplitude', fontsize=20)
axs[1].set_ylabel('pump frequency[GHz]', fontsize=20)
cbar1 = fig.colorbar(im1, ax=axs[1])
cbar1.set_label('Avg dSNR [dB]', fontsize=14)
# plot gain, dsnr
axs[2].scatter(gain_avg, dsnr_avg, s=4)
axs[2].set_title('pump vs gain,dsnr', fontsize=20)
axs[2].set_xlabel('Gain Average', fontsize=20)
axs[2].set_ylabel('dSNR Average', fontsize=20)
axs[2].set_xlim(0,20)
axs[2].set_ylim(0,12)
plt.tight_layout()
plt.show()
#%% {Update_state}
if not node.parameters.load_data_id:
    with node.record_state_updates():        
        machine.twpas['twpa1'].pump_frequency=pumpATmaxDSNR[0][0]
        machine.twpas['twpa1'].pump_amplitude=pumpATmaxDSNR[0][1]
        machine.twpas['twpa1'].max_gain=np.round(np.max(np.mean(Gain,axis=0)))
        machine.twpas['twpa1'].max_snr_improvement=np.round(np.max(np.mean(dsnr,axis=0)))

    # %% {Save_results}
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %%
