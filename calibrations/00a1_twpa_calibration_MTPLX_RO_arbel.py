
""" Jeongwon Kim Akiva, Omrie, Wei  250924
        TWPA CALIBRATION FOR OPTIMIAL PUMPING POINT

Sweep pump frequency and amplitude to find the optimal pump frequency and pump amplitude for the TWPA.
For each pumping point, calculate Gain, SNR improvement.
Optimize pumping point for the worst SNR Qubit so that the multiplexed readout could be done faster without losing SNR.
* Gain is defined as the increase in the signal level.
    - twpa pump off : measure the signal response 4MHz around the readout resonator when the pump is off
      singal_off= signal[dB] 
    - twpa pump on :  measure signal response 4MHz around the readout resonator when the pump is on
      singal_on= signal[dB]
    => gain=signal_on-signal_off
* SNR improvement is defined by the difference in the signal to noise ratio between pump on and pump off.
    - twpa pump off : measure the signal response 4 MHz around the readout resonator twice with 
      measure(amp=0) for noise level
      measure(amp=from state file) for signal level
      snr_off= signal[dB]-noise[dB]
    - twpa pump on :  measure the signal response 4 MHz around the readout resonator twice with
      measure(amp=0) for noise level
      measure(amp=from state file) for signal level
    => dsnr=snr_on-snr_off
Prerequisites:
    - Need to know in which frequency dispersive feature of TWPA RPM resonator appears
    - Having calibrated the resonator frequency (nodes 02a, 02b and/or 02c).
    - Having calibrated the worst SNR Qubit 
How to use optimizers : 
    - average optimized pumping point: define mingain and mindsnr, then the function will return the optimized pump frequency and pump amplitude
      which maximizes the average dSNR among the pumping points which satisfies the minimum gain and minimum dSNR conditions for all qubits
    - multiplexed readout optimized pumping point: define mingain, mindsnr and poorqubit index, then the function will return the optimized pump frequency and pump amplitude
      which maximizes the dSNR of the poor qubit among the pumping points which satisfies the minimum gain and minimum dSNR conditions for all qubits
Before proceeding to the next node:
    - Updates the optimal pump frequency and pump amplitude for the TWPA
    (average optimal point & multiplxed readout optimal point) in the state
    - Save the current state
"""
# %% {Imports}
from datetime import datetime, timezone, timedelta
from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode, NodeParameters
from iqcc_calibration_tools.quam_config.components.quam_root import Quam
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
from iqcc_calibration_tools.quam_config.lib.qua_datasets import opxoutput

# %% {Node_parameters}
class Parameters(NodeParameters):
    twpas: Optional[List[str]] = ['twpa1-3']
    num_averages: int =30
    frequency_span_in_mhz: float = 4
    frequency_step_in_mhz: float = 0.1
    amp_min: float =  0.1
    amp_max: float =  0.3
    points : int = 40    
    p_frequency_span_in_mhz: float = 60
    p_frequency_step_in_mhz: float =0.5
    simulate: bool = False
    simulation_duration_ns: int = 4000
    timeout: int = 300
    load_data_id: Optional[int] = None
node = QualibrationNode(name="00a_twpa1_3_calibration_MTPLX_RO", parameters=Parameters())
date_time = datetime.now(timezone(timedelta(hours=2))).strftime("%Y-%m-%d %H:%M:%S")
node.results["date"]={"date":date_time}
# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = Quam.load()
# Get the relevant QuAM components
twpas = [machine.twpas[t] for t in node.parameters.twpas]
twpa_id=node.parameters.twpas[0]
qubits = [machine.qubits[machine.twpas[twpa_id].qubits[i]] for i in range(len(machine.twpas[twpa_id].qubits))]
resonators = [machine.qubits[machine.twpas[twpa_id].qubits[i]].resonator for i in range(len(machine.twpas[twpa_id].qubits))]
pumpline_attenuation=twpas[0].pumpline_attenuation
signalline_attenuation=twpas[0].signalline_attenuation
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
#%% # readout pulse information
readout_power=[np.round(opxoutput(qubits[i].resonator.opx_output.full_scale_power_dbm,qubits[i].resonator.operations["readout"].amplitude)+signalline_attenuation,2) for i in range(len(qubits))]
readout_length=[qubits[i].resonator.operations["readout"].length for i in range(len(qubits))]
for i in range(len(readout_power)):
    print(f"{qubits[i].name}: readout power @ resonator={readout_power[i]}dBm, readout length={readout_length[i]}, Aro={qubits[i].resonator.operations['readout'].amplitude} ")
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
amp_step = int((opxoutput(full_scale_power_dbm, amp_max)-(opxoutput(full_scale_power_dbm, amp_min)))/0.2)
daps = np.logspace(np.log10(amp_min), np.log10(amp_max), node.parameters.points)

span_p = node.parameters.p_frequency_span_in_mhz * u.MHz
step_p = node.parameters.p_frequency_step_in_mhz * u.MHz
dfps = np.arange(-span_p / 2, +span_p / 2, step_p)
pump_duration = (10*len(dfs)*(machine.qubits[twpas[0].qubits[0]].resonator.operations["readout"].length+machine.qubits[twpas[0].qubits[0]].resonator.depletion_time))/4#(n_avg*len(dfs)*(machine.qubits[twpas[0].qubits[0]].resonator.operations["readout"].length+machine.qubits[twpas[0].qubits[0]].resonator.depletion_time))/4
with program() as twpa_pump_off:    
    I, I_st, Q, Q_st,n,n_st = qua_declaration(num_qubits=len(qubits))
    I_, I_st_, Q_, Q_st_,n_,n_st_ = qua_declaration(num_qubits=len(qubits))
    dp = declare(int)  # QUA variable for the pump frequency
    da = declare(float)# QUA variable for the pump amplitude
    df = declare(int)  # QUA variable for the readout frequency
    for qubit in qubits:
        machine.set_all_fluxes(flux_point="joint", target=qubit)
# TWPA off
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
                        rr.measure("readout",amplitude_scale=0, qua_vars=(I[i], Q[i]))
                        # wait for the resonator to relax
                        rr.wait(rr.depletion_time * u.ns)
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i]) 
                #align() why it doesnt work if i add this 10.12
                with for_(*from_array(df, dfs)):
                    for i, rr in enumerate(resonators):
                        # Update the resonator frequencies for all resonators
                        update_frequency(rr.name, df + rr.intermediate_frequency)
                        # Measure the resonator
                        rr.measure("readout",  qua_vars=(I_[i], Q_[i]))
                        # wait for the resonator to relax
                        rr.wait(rr.depletion_time * u.ns)
                        # save data
                        save(I_[i], I_st_[i])
                        save(Q_[i], Q_st_[i])
                align()  
    with stream_processing():
        n_st.save("n")
        for i in range(len(qubits)):
            I_st[i].buffer(len(dfs)).buffer(len(daps)).buffer(len(dfps)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).buffer(len(daps)).buffer(len(dfps)).average().save(f"Q{i + 1}")
            I_st_[i].buffer(len(dfs)).buffer(len(daps)).buffer(len(dfps)).average().save(f"I_{i + 1}")
            Q_st_[i].buffer(len(dfs)).buffer(len(daps)).buffer(len(dfps)).average().save(f"Q_{i + 1}") 
### TWPA on
with program() as twpa_pump_on:    
    I, I_st, Q, Q_st,n,n_st = qua_declaration(num_qubits=len(qubits))
    I_, I_st_, Q_, Q_st_,n_,n_st_ = qua_declaration(num_qubits=len(qubits))
    dp = declare(int)  # QUA variable for the pump frequency
    da = declare(float)# QUA variable for the pump amplitude
    df = declare(int)  # QUA variable for the readout frequency
    for qubit in qubits:
        machine.set_all_fluxes(flux_point="joint", target=qubit)
    with for_(n, 0, n < n_avg, n + 1):  
        save(n, n_st)
        with for_(*from_array(dp, dfps)):  
            update_frequency(twpas[0].pump_.name, dp + twpas[0].pump_.intermediate_frequency)
            with for_each_(da, daps):  
                twpas[0].pump_.play('pump_', amplitude_scale=da, duration=pump_duration)
                wait(25) #100/4 wait 100ns(change on 30/11) 1000/4 wait 1us for pump to settle before readout
# measure readout responses around readout resonators with pump
                with for_(*from_array(df, dfs)):
                    for i, rr in enumerate(resonators):
                        # Update the resonator frequencies for all resonators
                        update_frequency(rr.name, df + rr.intermediate_frequency)
                        # Measure the resonator, amp=0 -> see the noise level
                        rr.measure("readout",amplitude_scale=0, qua_vars=(I[i], Q[i]))
                        # wait for the resonator to relax
                        rr.wait(rr.depletion_time * u.ns)
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i]) 
                #align() doesnt work
                with for_(*from_array(df, dfs)):
                    for i, rr in enumerate(resonators):
                        # Update the resonator frequencies for all resonators
                        update_frequency(rr.name, df + rr.intermediate_frequency)
                        # Measure the resonator, amp, see the signal level
                        rr.measure("readout", qua_vars=(I_[i], Q_[i]))
                        # wait for the resonator to relax
                        rr.wait(rr.depletion_time * u.ns)
                        # save data
                        save(I_[i], I_st_[i])
                        save(Q_[i], Q_st_[i])
                align()  
    with stream_processing():
        n_st.save("n")
        for i in range(len(qubits)):
            I_st[i].buffer(len(dfs)).buffer(len(daps)).buffer(len(dfps)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).buffer(len(daps)).buffer(len(dfps)).average().save(f"Q{i + 1}")
            I_st_[i].buffer(len(dfs)).buffer(len(daps)).buffer(len(dfps)).average().save(f"I_{i + 1}")
            Q_st_[i].buffer(len(dfs)).buffer(len(daps)).buffer(len(dfps)).average().save(f"Q_{i + 1}") 
# %% {Simulate_or_execute}
import requests
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
        job = qm.execute(twpa_pump_off)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job_ = qm.execute(twpa_pump_on)
        results_ = fetching_tool(job_, ["n"], mode="live")
        while results_.is_processing():
            n_ = results_.fetch_all()[0]
# %% {Data_fetching_and_dataset_creation}
#data for pump off
ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": dfs, "pump_amp": daps, "pump_freq" : dfps})
# Convert IQ data into volts
ds = convert_IQ_to_V(ds, qubits)
# Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
ds = ds.assign({"IQ_abs_noise": 1e3*np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
ds = ds.assign({"IQ_abs_signal": 1e3*np.sqrt(ds["I_"] ** 2 + ds["Q_"] ** 2)})
pumpoff_snr = snr(ds, qubits, dfps, daps)

#data for pump on
ds_ = fetch_results_as_xarray(job_.result_handles, qubits, {"freq": dfs, "pump_amp": daps, "pump_freq" : dfps})
# Convert IQ data into volts
ds_ = convert_IQ_to_V(ds_, qubits)
# Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
ds_ = ds_.assign({"IQ_abs_noise": 1e3*np.sqrt(ds_["I"] ** 2 + ds_["Q"] ** 2)})
ds_ = ds_.assign({"IQ_abs_signal": 1e3*np.sqrt(ds_["I_"] ** 2 + ds_["Q_"] ** 2)})
pumpon_snr = snr(ds_, qubits, dfps, daps)
# %% {Data Analysis}
# SNR improvement & Gain
RF_freq = np.array([dfs + q.resonator.RF_frequency for q in qubits])
dsnr = pumpon_snr-pumpoff_snr
Gain = gain(ds, ds_, qubits, dfps, daps)
linear_gain = 10**(Gain/20)
linear_dsnr = 10**(dsnr/20)
average_gain=20*np.log10(np.mean(linear_gain,axis=0))
average_dsnr=20*np.log10(np.mean(linear_dsnr,axis=0))
p_lo=twpas[0].pump.LO_frequency
p_if=twpas[0].pump.intermediate_frequency
# pump at max avg_gain
pumpATmaxG=pump_maxgain(Gain, dfps, daps)
print(f'max Avg Gain({np.round(20*np.log10(np.max(np.mean(linear_gain,axis=0))))}dB) at fp={np.round((p_lo+p_if+pumpATmaxG[0][0])*1e-9,3)}GHz,Pp={np.round(pumpline_attenuation+opxoutput(full_scale_power_dbm,pumpATmaxG[0][1]),2)},Pamp={np.round(pumpATmaxG[0][1],3)}')
# pump at max dSNR
pumpATmaxDSNR=pump_maxdsnr(dsnr, dfps, daps)
maxDSNR_point={'fp':np.round((p_lo+p_if+pumpATmaxDSNR[0][0]),3), 
                 'Pp': pumpline_attenuation+opxoutput(full_scale_power_dbm,pumpATmaxDSNR[0][1]),
                 'Pamp': np.round(pumpATmaxDSNR[0][1],3)}
node.results["maxDSNR point"] = maxDSNR_point
# %% {Plotting}
time_sec = 1e-9 * 12 * n_avg * len(daps) * len(dfps) * len(dfs) * (
    machine.qubits[twpas[0].qubits[0]].resonator.operations["readout"].length +
    machine.qubits[twpas[0].qubits[0]].resonator.depletion_time
)
print(f"calibration time = {np.round(time_sec, 3)} sec")
pump_frequency=machine.twpas[twpa_id].pump.LO_frequency+machine.twpas[twpa_id].pump.intermediate_frequency+dfps
pump_power=opxoutput(full_scale_power_dbm,daps)+pumpline_attenuation
pump_power[np.isneginf(pump_power)]=0
indices=np.linspace(0, len(pump_frequency)-1,10, dtype=int)
selected_frequencies=np.round(pump_frequency[indices]*1e-9,3)
ytick_pos = np.linspace(0, len(dfps)-1, len(selected_frequencies))
indices_=np.linspace(0, len(pump_power)-1,10, dtype=int)
selected_powers=np.round(pump_power[indices_],2)
xtick_pos = np.linspace(0, len(daps)-1, len(selected_powers))
##---------------------------- SIGNAL S21 PLOT -----------------------------------------------
ncols = 2
nrows = math.ceil(len(qubits) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(6, 8)) 
axes = axes.flatten() 
for i, ax in enumerate(axes):
    if i < len(qubits):  # only plot existing qubits
        ax.plot(RF_freq[i]*1e-9, ds.IQ_abs_signal.values[i][0][0], label='pumpoff',color='black') #[0][0] arbitrary n_avg number of averaged S21 data
        ax.plot(RF_freq[i]*1e-9,ds_.IQ_abs_signal.values[i][pumpATmaxG[1][0]][pumpATmaxG[1][1]], label='pump @ max avg G',color='red')
        ax.plot(RF_freq[i]*1e-9, ds_.IQ_abs_signal.values[i][pumpATmaxDSNR[1][0]][pumpATmaxDSNR[1][1]], label='pump @ max avg Dsnr',color='blue')
        ax.set_title(f'{qubits[i].name} Signal S21 \n {date_time}', fontsize=14)
        ax.set_xlabel('Res.freq [GHz]', fontsize=12)
        ax.set_ylabel('Trans.amp. [mV]', fontsize=12)
        ax.legend(fontsize=4, loc='upper right')
    else:
        ax.axis("off")  
plt.tight_layout()
s=plt.gcf()
plt.show()
##-----------------------------------Noise PLOT-----------------------------------------
fig, axes = plt.subplots(nrows, ncols, figsize=(6, 8)) 
axes = axes.flatten() 
for i, ax in enumerate(axes):
    if i < len(qubits):  # only plot existing qubits
        ax.plot(RF_freq[i]*1e-9, ds.IQ_abs_noise.values[i][0][0], label='pumpoff',color='black') #[0][0] arbitrary n_avg number of averaged S21 data
        ax.plot(RF_freq[i]*1e-9,ds_.IQ_abs_noise.values[i][pumpATmaxG[1][0]][pumpATmaxG[1][1]], label='pump @ maxG',color='red')
        ax.plot(RF_freq[i]*1e-9, ds_.IQ_abs_noise.values[i][pumpATmaxDSNR[1][0]][pumpATmaxDSNR[1][1]], label='pump @ maxDsnr',color='blue')
        ax.set_title(f'{qubits[i].name}, Noise\n {date_time}', fontsize=14)
        ax.set_xlabel('Res.freq [GHz]', fontsize=12)
        ax.set_ylabel('Trans.amp. [mV]', fontsize=12)
        ax.set_ylim(0, max(ds_.IQ_abs_signal.values[i][pumpATmaxG[1][0]][pumpATmaxG[1][1]]))
        ax.legend(fontsize=4, loc='upper right')
    else:
        ax.axis("off")  
plt.tight_layout()
n=plt.gcf()
plt.show()

### ------------------------- avggain vs pump ----------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(12,5))
cmap = plt.cm.viridis.copy()
cmap.set_under('gray')
# plot gain vs pump
im0 = axs[0].imshow(average_gain, origin='lower', aspect='auto',
                    extent=[0, len(daps)-1, 0, len(dfps)-1], cmap=cmap, vmin=0)
axs[0].set_xticks(xtick_pos)
axs[0].set_xticklabels(selected_powers,rotation=90)
axs[0].set_yticks(ytick_pos)
axs[0].set_yticklabels(selected_frequencies)
axs[0].set_title(f'{twpas[0].id} pump vs Avg Gain \n {date_time}', fontsize=15)
axs[0].set_xlabel('pump power[dBm]', fontsize=20)
axs[0].set_ylabel('pump frequency[GHz]', fontsize=20)
cbar0 = fig.colorbar(im0, ax=axs[0])
cbar0.set_label('Avg Gain [dB]', fontsize=14)
print(f'max Avg dSNR({np.round(20*np.log10(np.max(np.mean(linear_dsnr,axis=0))),2)}dB) \n at fp={np.round((p_lo+p_if+pumpATmaxDSNR[0][0])*1e-9,3)}GHz,Pp={np.round(pumpline_attenuation+opxoutput(full_scale_power_dbm,pumpATmaxDSNR[0][1]),2)},Pamp={np.round(pumpATmaxDSNR[0][1],3)} \n {date_time} \n {len(dfs)}*{len(daps)}*{len(dfps)}*{n_avg}')

# plot avgDdSNR vs pump
im1 = axs[1].imshow(average_dsnr, origin='lower', aspect='auto',
                    extent=[0, len(daps)-1, 0, len(dfps)-1], cmap=cmap, vmin=0)
axs[1].set_xticks(xtick_pos)
axs[1].set_xticklabels(selected_powers,rotation=90)
axs[1].set_yticks(ytick_pos)
axs[1].set_yticklabels(selected_frequencies)
axs[1].set_title(f'{twpas[0].id} pump vs Avg dSNR\n {date_time} ', fontsize=15)
axs[1].set_xlabel('pump amplitude', fontsize=20)
axs[1].set_ylabel('pump frequency[GHz]', fontsize=20)
cbar1 = fig.colorbar(im1, ax=axs[1])
cbar1.set_label('Avg dSNR [dB]', fontsize=14)
# plot gain, dsnr : TWPA SPEC
axs[2].scatter(average_gain, average_dsnr, s=4)
axs[2].set_title('pump vs gain,dsnr', fontsize=20)
axs[2].set_xlabel('Gain Average', fontsize=20)
axs[2].set_ylabel('dSNR Average', fontsize=20)
axs[2].set_xlim(0,np.max(average_gain)+1)
axs[2].set_ylim(0,np.max(average_dsnr)+1)
plt.tight_layout()
map=plt.gcf()
plt.show()
# %% ############################{Average optimum}##################################
plt.plot(figzise=(4,3))
mingain=16
mindsnr=8
avg_optimized_pump=optimizer(mingain, mindsnr,  Gain, dsnr,  average_dsnr, dfps, daps, p_lo,p_if)
for i in range(len(qubits)):
    print(f"{qubits[i].id}:dSNR:{np.round(dsnr[i][avg_optimized_pump[0]][avg_optimized_pump[1]][0],2)}dB, gain:{np.round(Gain[i][avg_optimized_pump[0]][avg_optimized_pump[1]][0],2)}dB")
plt.scatter(average_gain, average_dsnr, s=4)
plt.scatter(average_gain[avg_optimized_pump],  average_dsnr[avg_optimized_pump], s=10, color='red', label='Average Optimum')
plt.title(f'{node.add_node_info_subtitle()},{twpas[0].id}\n Average Gain & dSNR \n @ Average Optimum pumping point', fontsize=20)
plt.legend(loc='upper left',fontsize=15,framealpha=0.4)#, bbox_to_anchor=(1, 1))
plt.xlabel('Average Gain', fontsize=20)
plt.ylabel('Average dSNR', fontsize=20)
plt.xlim(0,np.max(average_gain)+1)
plt.ylim(0,np.max(average_dsnr)+1)
plt.axvline(mingain, color='red', linestyle='--', linewidth=1)
plt.axhline(mindsnr, color='red', linestyle='--', linewidth=1)
plt.tight_layout()
window=plt.gcf()
plt.show()
#%% ###################{Plot} {Multiplexed Readout spec} #######################################
### plot gain vs pump  
fig, axes = plt.subplots(nrows, ncols, figsize=(6, 8))
axes = axes.flatten()
for i, ax in enumerate(axes[:len(qubits)]):  # only use first num_qubits axes
    im0 = ax.imshow(
        Gain[i], origin='lower', aspect='auto',
        extent=[0, len(daps) - 1, 0, len(dfps) - 1],
        cmap=cmap,vmin=0)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(selected_powers, rotation=90, fontsize=8)
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(selected_frequencies, fontsize=8)
    ax.set_title(f'{twpas[0].id}, {qubits[i].name}, pump vs Gain\n{date_time}', fontsize=10)
    ax.set_xlabel('pump power [dBm]', fontsize=10)
    ax.set_ylabel('pump frequency [GHz]', fontsize=10)
    cbar0 = fig.colorbar(im0, ax=ax)
    cbar0.set_label('Gain [dB]', fontsize=10)
for ax in axes[len(qubits):]:
    ax.set_visible(False)
plt.tight_layout()
gain_indiv=plt.gcf()
plt.show()
# plot dSNR vs pump
fig, axes = plt.subplots(nrows, ncols, figsize=(6, 8))
axes = axes.flatten()
for i, ax in enumerate(axes[:len(qubits)]):  # only use first num_qubits axes
    im0 = ax.imshow(
        dsnr[i], origin='lower', aspect='auto',
        extent=[0, len(daps) - 1, 0, len(dfps) - 1],
        cmap=cmap, vmin=0
    )
    im0.set_clim(vmin=0, vmax=dsnr[i].max())
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(selected_powers, rotation=90, fontsize=8)
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(selected_frequencies, fontsize=8)
    ax.set_title(f'{twpas[0].id}, {qubits[i].name}, pump vs dSNR\n{date_time}', fontsize=10)
    ax.set_xlabel('pump power [dBm]', fontsize=10)
    ax.set_ylabel('pump frequency [GHz]', fontsize=10)
    cbar0 = fig.colorbar(im0, ax=ax)
    cbar0.set_label('dSNR [dB]', fontsize=10)
for ax in axes[len(qubits):]:
    ax.set_visible(False)
plt.tight_layout()
dsnr_indiv=plt.gcf()
plt.show()
#--------------- multiplexed readout optimal point--------------------------------------------
# %% {Multiplexed Readout optimum}
plt.plot(figzise=(4,3))
mingain=10
mindsnr=8
poorqubit=3
mtplx_optimized_pump_idx=multiplexed_optimizer(mingain,mindsnr, qubits, Gain, dsnr,poorqubit, dfps, daps, p_lo,p_if)
colors=[]
for i in range(len(qubits)):
    gain_flat = np.array(Gain[i]).flatten()
    dsnr_flat = np.array(dsnr[i]).flatten()
    sc = plt.scatter(
        gain_flat[::2], dsnr_flat[::2], s=4, # downsample
        alpha=1 - 0.2*i
    )
    colors.append(sc.get_facecolor()[0])
for i in range(len(qubits)):
    # Plot black outline (larger, behind)
    plt.scatter(Gain[i][mtplx_optimized_pump_idx[0]][mtplx_optimized_pump_idx[1]], dsnr[i][mtplx_optimized_pump_idx[0]][mtplx_optimized_pump_idx[1]], s=5,marker='X',color='black',linewidths=7,zorder=9,alpha=1)
    # Plot colored marker (smaller, on top)
    plt.scatter(Gain[i][mtplx_optimized_pump_idx[0]][mtplx_optimized_pump_idx[1]], dsnr[i][mtplx_optimized_pump_idx[0]][mtplx_optimized_pump_idx[1]], s=3,marker='X',color=colors[i],linewidths=3,zorder=10,alpha=1,label=f'qB{i+1}')
plt.title(f'{node.add_node_info_subtitle()}, {twpas[0].id}\n Gain & dSNR\n@ Multiplexed RO Optimum pumping point', fontsize=20)
plt.xlabel('Gain[dB]', fontsize=20)
plt.ylabel('dSNR[dB]', fontsize=20)
plt.xlim(0,np.max(Gain)+1)
plt.ylim(0,np.max(dsnr)+1)
plt.tight_layout()
plt.vlines(mingain, ymin=0, ymax=np.max(dsnr)+1, color='black', linestyle='--', linewidth=1)
plt.hlines(mindsnr, xmin=0, xmax=np.max(Gain)+1, color='black', linestyle='--', linewidth=1)
plt.legend(loc='upper left',fontsize=15,framealpha=0.4)#, bbox_to_anchor=(1, 1))
multiplexed_optimization=plt.gcf()
plt.show()
print(f'MPLX RO Optimized pumping point \n at fp={np.round((p_lo+p_if+dfps[mtplx_optimized_pump_idx[0]])*1e-9,3)}GHz,Pp={np.round(pumpline_attenuation+opxoutput(full_scale_power_dbm,daps[mtplx_optimized_pump_idx[1]]),2)},Pamp={np.round(daps[mtplx_optimized_pump_idx[1]],3)} \n {date_time} \n {len(dfs)}*{len(daps)}*{len(dfps)}*{n_avg}')

# %% {Update_state}
avg_operation_point={'fp':np.round((p_lo+p_if+dfps[avg_optimized_pump[0]]),3), 
                 'Pp': pumpline_attenuation+opxoutput(full_scale_power_dbm,daps[avg_optimized_pump[1]]),
                 'Pamp': np.round(daps[avg_optimized_pump[1]],3)}
node.results["avg_operation_point"] = avg_operation_point
node.results["multiplexed_RO_operation_point"]={'fp':np.round((p_lo+p_if+dfps[mtplx_optimized_pump_idx[0]]),3), 
                 'Pp': pumpline_attenuation+opxoutput(full_scale_power_dbm,daps[mtplx_optimized_pump_idx[1]]),
                 'Pamp': np.round(daps[mtplx_optimized_pump_idx[1]],3)}
node.results["Ps"]={"Ps":readout_power}
node.results["figures"]={"signal": s,
                         "noise": n,
                         "map": map,
                         "operation_window" : window,
                         "gain_indiv":gain_indiv,
                         "dsnr_indiv":dsnr_indiv,
                         "multiplexed_window": multiplexed_optimization}
if not node.parameters.load_data_id:
    with node.record_state_updates():        
        machine.twpas[twpa_id].pump_frequency=dfps[avg_optimized_pump[0]]
        machine.twpas[twpa_id].pump_amplitude=daps[avg_optimized_pump[1]]
        machine.twpas[twpa_id].mltpx_pump_frequency=dfps[mtplx_optimized_pump_idx[0]]
        machine.twpas[twpa_id].mltpx_pump_amplitude=daps[mtplx_optimized_pump_idx[1]]
        machine.twpas[twpa_id].max_avg_gain=np.round(np.max(np.mean(average_gain,axis=0)))
        machine.twpas[twpa_id].max_avg_snr_improvement=np.round(np.max(np.mean(average_dsnr,axis=0)))

    # %% {Save_results}
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %%
