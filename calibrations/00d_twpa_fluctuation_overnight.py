
""" Jeongwon Kim IQCC, 251022
do spectroscopy around the bandwidth ~(7GHz ~ 7.6GHz : our usual readout bandwidth)
for various readout power ~(-120dBm~-95dBm)
with the pump off and on(pumping condition is given through node 001) and get the Gain.
For each signal frequency, get the Gain as a function of readout power
and get the input power at which gain is compressed 1dB(P1dB)

Prerequisites:
    - Having calibrated the optimal twpa pumping point (nodes 001). All the gain compression 
      is measured under the given pumping point which is obatained through node001

* Gain is defined as the increase in the signal level.
    - twpa pump off : measure the signal response within 600MHz around the readout bandwidth
      singal_off= signal[dB] 
    - twpa pump on :  measure signal response within 600MHz around the readoutbandwidth
      singal_on= signal[dB]
    => gain=signal_on-signal_off
* P1dB : measure the gain as a function of readout amplitude and find the point where
        gain starts to get compressed 1dB
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
import os
import pandas as pd
import re
import ast
import requests
import time
# %% {Node_parameters}
class Parameters(NodeParameters):
    twpas: Optional[List[str]] = ['twpa2-1']
    num_averages: int = 100
    measurement : int = 200
    frequency_span_in_mhz: float = 500
    frequency_step_in_mhz: float = 0.5
    simulate: bool = False
    simulation_duration_ns: int = 4000
    timeout: int = 300
    load_data_id: Optional[int] = None
    amplitude_scale: float = 0.03#0.2
    pumpline_attenuation: int = -50-10-4 #(-50: fridge atten(-30)+directional coupler(-20)/
    signalline_attenuation : int = -60-9 #-60dB : fridge atten,  
node = QualibrationNode(name="00d_twpa_fluctuation", parameters=Parameters())
date_time = datetime.now(timezone(timedelta(hours=2))).strftime("%Y-%m-%d %H:%M:%S")
node.results["date"]={"date":date_time}
# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = Quam.load()

# Get the relevant QuAM components
twpas = [machine.twpas[t] for t in node.parameters.twpas]
qubits = [machine.qubits[machine.twpas['twpa2-1'].qubits[1]]]
resonators = [machine.qubits[machine.twpas['twpa2-1'].qubits[1]].resonator]
spectroscopy = [twpas[0].spectroscopy]

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
# if node.parameters.load_data_id is None:
qmm = machine.connect()
#%% # readout pulse information
from iqcc_calibration_tools.quam_config.lib.qua_datasets import opxoutput, mV
readout_power=[np.round(opxoutput(qubits[i].resonator.opx_output.full_scale_power_dbm,qubits[i].resonator.operations["readout"].amplitude)+node.parameters.signalline_attenuation,2) for i in range(len(qubits))]
readout_length=[qubits[i].resonator.operations["readout"].length for i in range(len(qubits))]
readout_voltage=[np.round(mV(qubits[i].resonator.opx_output.full_scale_power_dbm,qubits[i].resonator.operations["readout"].amplitude),2) for i in range(len(qubits))]
for i in range(len(readout_power)):
    print(f"{qubits[i].name}: readout power @ resonator={readout_power[i]}dBm, opx output={readout_voltage[i]}mV, readout length={readout_length[i]}")
# %% {QUA_program}
gain_overnight=[]
dsnr_overnight=[]
number=node.parameters.measurement
n_avg = node.parameters.num_averages  
# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)

# pump amp, frequency sweep
amp_scale=node.parameters.amplitude_scale
full_scale_power_dbm=twpas[0].pump.opx_output.full_scale_power_dbm
# daps = np.logspace(np.log10(node.parameters.amp_min), np.log10(node.parameters.amp_max), node.parameters.points)
f_p = twpas[0].pump_frequency
p_p = twpas[0].pump_amplitude
# pump duration should be able to cover the resonator spectroscopy which takes #(dfs) (as we are multiplexing qubit number doesnt matter) 
pump_duration = (10*20*len(dfs)*(machine.qubits[twpas[0].qubits[0]].resonator.operations["readout"].length+machine.qubits[twpas[0].qubits[0]].resonator.depletion_time))/4#(n_avg*len(dfs)*(machine.qubits[twpas[0].qubits[0]].resonator.operations["readout"].length+machine.qubits[twpas[0].qubits[0]].resonator.depletion_time))/4
with program() as twpa_pump_off:    
    I, I_st, Q, Q_st,n,n_st = qua_declaration(num_qubits=len(qubits))
    I_, I_st_, Q_, Q_st_,n_,n_st_ = qua_declaration(num_qubits=len(qubits))
    df = declare(int)  # QUA variable for the readout frequency
# TWPA off : measure readout responses around readout resonators without pump
    with for_(n, 0, n < n_avg, n + 1):  
        save(n, n_st)
        with for_(*from_array(df, dfs)): 
            for i, spec in enumerate(spectroscopy):
                # Update the resonator frequencies for all resonators
                update_frequency(spec.name, df+spec.intermediate_frequency)
                # Measure the resonator
                spec.measure("readout", amplitude_scale=0, qua_vars=(I[i], Q[i]))
                # wait for the resonator to relax
                spec.wait(spec.depletion_time * u.ns)
                # save data
                save(I[i], I_st[i])
                save(Q[i], Q_st[i])
        with for_(*from_array(df, dfs)): 
            for i, spec in enumerate(spectroscopy):
                # Update the resonator frequencies for all resonators
                update_frequency(spec.name, df+spec.intermediate_frequency)
                # Measure the resonator
                spec.measure("readout", amplitude_scale=amp_scale, qua_vars=(I_[i], Q_[i]))
                # wait for the resonator to relax
                spec.wait(spec.depletion_time * u.ns)
                # save data
                save(I_[i], I_st_[i])
                save(Q_[i], Q_st_[i])
        align()  
    with stream_processing():
        n_st.save("n")
        for i in range(len(qubits)):
            I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")
            I_st_[i].buffer(len(dfs)).average().save(f"I_{i + 1}")
            Q_st_[i].buffer(len(dfs)).average().save(f"Q_{i + 1}")
        
with program() as twpa_pump_on:    
    I, I_st, Q, Q_st,n,n_st = qua_declaration(num_qubits=len(qubits))
    I_, I_st_, Q_, Q_st_,n_,n_st_ = qua_declaration(num_qubits=len(qubits))
    df = declare(int)  # QUA variable for the readout frequency
# TWPA on
    with for_(n, 0, n < n_avg, n + 1):  
        save(n, n_st)
        update_frequency(twpas[0].pump.name, f_p+twpas[0].pump.intermediate_frequency)
        twpas[0].pump.play('pump', amplitude_scale=p_p, duration=pump_duration)#+250)
        wait(250) #1000/4 wait 1us for pump to settle before readout
        with for_(*from_array(df, dfs)):
            for i, spec in enumerate(spectroscopy):
                # Update the resonator frequencies for all resonators
                update_frequency(spec.name, df+spec.intermediate_frequency)
                # Measure the resonator
                spec.measure("readout", amplitude_scale=0, qua_vars=(I[i], Q[i]))
                # wait for the resonator to relax
                spec.wait(spec.depletion_time * u.ns)
                # save data
                save(I[i], I_st[i])
                save(Q[i], Q_st[i])
        with for_(*from_array(df, dfs)): 
            for i, spec in enumerate(spectroscopy):
                # Update the resonator frequencies for all resonators
                update_frequency(spec.name, df+spec.intermediate_frequency)
                # Measure the resonator
                spec.measure("readout", amplitude_scale=amp_scale, qua_vars=(I_[i], Q_[i]))
                # wait for the resonator to relax
                spec.wait(spec.depletion_time * u.ns)
                # save data
                save(I_[i], I_st_[i])
                save(Q_[i], Q_st_[i])
        align()  
    with stream_processing():
        n_st.save("n")
        for i in range(len(qubits)):
            I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")
            I_st_[i].buffer(len(dfs)).average().save(f"I_{i + 1}")
            Q_st_[i].buffer(len(dfs)).average().save(f"Q_{i + 1}")
# %% {execute}
start=datetime.now()
for i in range(number):
    time.sleep(10)
    date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
    requests.get('http://10.2.1.5/PWD=1234;' +':PWR:RF:OFF')
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
# {Data_fetching_and_dataset_creation}
    requests.get('http://10.2.1.5/PWD=1234;' +':PWR:RF:ON')
    #data for pump off
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": dfs})
    ds = convert_IQ_to_V(ds, qubits)
    ds = ds.assign({"IQ_abs_signal": 1e3*np.sqrt(ds["I_"] ** 2 + ds["Q_"] ** 2)})
    ds = ds.assign({"IQ_abs_noise": 1e3*np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
    #data for pump on
    ds_ = fetch_results_as_xarray(job_.result_handles, qubits, {"freq": dfs})
    ds_ = convert_IQ_to_V(ds_, qubits)
    ds_ = ds_.assign({"IQ_abs_signal": 1e3*np.sqrt(ds_["I_"] ** 2 + ds_["Q_"] ** 2)})
    ds_ = ds_.assign({"IQ_abs_noise": 1e3*np.sqrt(ds_["I"] ** 2 + ds_["Q"] ** 2)})
# {Data Analysis}
    Gain = mvTOdbm(ds_.IQ_abs_signal.values[0])-mvTOdbm(ds.IQ_abs_signal.values[0])
    dsnr = Gain - (mvTOdbm(ds_.IQ_abs_noise.values[0])-mvTOdbm(ds.IQ_abs_noise.values[0]))
    gain_overnight.append(Gain)
    dsnr_overnight.append(dsnr)
end=datetime.now()
requests.get('http://10.2.1.5/PWD=1234;' +':PWR:RF:ON')
#%%
d = datetime.now()
dir = f'/Users/wjd__/OneDrive - QM Machines LTD/바탕 화면/Project/Measurement_/overnight/{d.year}.{d.month}.{d.day}/'
dir_ = r"C:\Users\wjd__\OneDrive - QM Machines LTD\바탕 화면\Project\Measurement_\overnight"
try:
    if not os.path.exists(dir):
        os.makedirs(dir)
except OSError:
    print('Error: Can\'t create directory. ' + dir)
overnight_data=pd.DataFrame({'gain_':gain_overnight, 'dsnr_':dsnr_overnight})
overnight_data.to_csv(dir+f"{start.hour}.{start.minute}~{end.hour}.{end.minute}, n={number}"+' Fluctuation.csv', index=False)

#%%{data retrieve and plot}
def parse_array(s):
    s = s.strip("[] \n")
    return [float(x) for x in s.split() if x]
path=f"{dir_}\\{d.year}.{d.month}.{d.day}\\{start.hour}.{start.minute}~{end.hour}.{end.minute}, n={number} Fluctuation.csv"
fs=np.array([dfs+twpas[0].spectroscopy.f_01])[0]
overnightgain=pd.read_csv(path)['gain_'].apply(parse_array)
overnightgain=np.vstack(overnightgain.to_numpy())
ov_gain_std=np.std(overnightgain,axis=0)
overnightdsnr=pd.read_csv(path)['dsnr_'].apply(parse_array)
overnightdsnr=np.vstack(overnightdsnr.to_numpy())
# ov_dsnr_std=np.std(overnightdsnr,axis=0)
ps=np.round(opxoutput(twpas[0].spectroscopy.opx_output.full_scale_power_dbm,
                twpas[0].spectroscopy.operations["readout"].amplitude*amp_scale)#amp_scale=0.03일지도
                +node.parameters.signalline_attenuation,2)
# -------- Profile------------- -------------
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
# -------- Gain subplot --------
ax = axes[0]
for i in range(1, len(overnightgain)-30, 40):
    ax.plot(fs, overnightgain[i], label=f'{np.round((1.2*((i-1)/40+1)),2)}h, avg gain={np.round(np.mean(overnightgain[i]),2)}dB')
ax.set_title(f'{twpas[0].id}:Gain time fluctuation\n {date_time}')
ax.set_xlabel('frequency[GHz]')
ax.set_ylabel('gain[dB]')
ax.legend(fontsize=6, loc='lower left')#, bbox_to_anchor=(1, 1))
# -------- DSNR subplot --------
ax = axes[1]
overnightdsnr_noise_avg=np.vstack([np.mean(overnightdsnr[i*40:(i+1)*40,:],axis=0)for i in range(5)])
ov_dsnr_std=np.std(overnightdsnr_noise_avg,axis=0)
for i in range(len(overnightdsnr_noise_avg)):
    ax.plot(fs, overnightdsnr_noise_avg[i], label=f'{np.round(1.2*(i+1),1)}h, avg dsnr={np.round(np.mean(overnightdsnr_noise_avg[i]),2)}dB')
ax.set_title(f'{twpas[0].id}: DSNR time fluctuation\n {date_time}')
ax.set_xlabel('frequency[GHz]')
ax.set_ylabel('DSNR[dB]')
ax.legend(fontsize=6,loc='lower left')#, bbox_to_anchor=(1, 1))
plt.tight_layout()
profile= plt.gcf()
# --------- std--------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
# -------- Gain subplot --------
ax = axes[0]
ax.plot(fs, ov_gain_std, label=f'avg std={np.round(np.mean(ov_gain_std),2)}dB')
ax.set_title(f'{twpas[0].id}:std(gain)\n {date_time}')
ax.legend()
# -------- DSNR subplot --------
ax = axes[1]
ax.plot(fs, ov_dsnr_std, label=f'avg std={np.round(np.mean(ov_dsnr_std),2)}dB')
ax.set_title(f'{twpas[0].id}:std(DSNR)\n {date_time}')
ax.legend()
plt.tight_layout()
std= plt.gcf()
print(ps)
# %% {Update state}
node.results = {"fp":f_p+twpas[0].pump.intermediate_frequency+twpas[0].pump.LO_frequency,
                 "pp":p_p,
                  "Ps":ps }
node.results["figures"]={"profile": profile,
                         "std": std,}
with node.record_state_updates():        
    machine.twpas['twpa2-1'].avg_std_gain=np.round(np.mean(ov_gain_std),2)
    machine.twpas['twpa2-1'].avg_std_snr_improvement=np.round(np.mean(ov_dsnr_std),2)
# %% {Save_results}
node.outcomes = {q.name: "successful" for q in qubits}
node.results["initial_parameters"] = node.parameters.model_dump()
node.machine = machine
node.save()

# %%
