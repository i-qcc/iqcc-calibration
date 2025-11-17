# %%
from datetime import datetime, timezone, timedelta
from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode, NodeParameters
from iqcc_calibration_tools.quam_config.components import Quam
from iqcc_calibration_tools.quam_config.macros import qua_declaration
from iqcc_calibration_tools.quam_config.lib.qua_datasets import convert_IQ_to_V
from iqcc_calibration_tools.storage.save_utils import fetch_results_as_xarray
from iqcc_calibration_tools.analysis.twpa_utils import  * 
from iqcc_calibration_tools.quam_config.lib.qua_datasets import opxoutput
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
    twpas: Optional[List[str]] = ['twpa2-1']
    num_averages: int = 30
    amp_min: float =  0.25
    amp_max: float =  0.6
    points : int = 40
    frequency_span_in_mhz: float = 4
    frequency_step_in_mhz: float = 0.1
    p_frequency_span_in_mhz: float = 60
    p_frequency_step_in_mhz: float =0.5
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    simulation_duration_ns: int = 4000
    timeout: int = 300
    load_data_id: Optional[int] = None
    pumpline_attenuation: int = -50-14 
    signalline_attenuation : int = -60-9
    
node = QualibrationNode(name="test", parameters=Parameters())
date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = Quam.load()

# Get the relevant QuAM components
twpas = [machine.twpas[t] for t in node.parameters.twpas]
qubits = [machine.qubits[machine.twpas['twpa2-1'].qubits[i]] for i in range(len(machine.twpas['twpa2-1'].qubits))]
resonators = [machine.qubits[machine.twpas['twpa2-1'].qubits[i]].resonator for i in range(len(machine.twpas['twpa2-1'].qubits))]

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
# %% {QUA_program}
n_avg = node.parameters.num_averages  
# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)
# pump amp, frequency sweep
full_scale_power_dbm=twpas[0].pump.opx_output.full_scale_power_dbm

# daps = np.arange(amp_min, amp_max, 0.01)

span_p = node.parameters.p_frequency_span_in_mhz * u.MHz
step_p = node.parameters.p_frequency_step_in_mhz * u.MHz
# pump duration should be able to cover the resonator spectroscopy which takes #(dfs) (as we are multiplexing qubit number doesnt matter) 
pump_duration = (10*len(dfs)*(machine.qubits[twpas[0].qubits[0]].resonator.operations["readout"].length+machine.qubits[twpas[0].qubits[0]].resonator.depletion_time))/4#(n_avg*len(dfs)*(machine.qubits[twpas[0].qubits[0]].resonator.operations["readout"].length+machine.qubits[twpas[0].qubits[0]].resonator.depletion_time))/4
f_p=twpas[0].pump_frequency
p_p=twpas[0].pump_amplitude
full_scale_power_dbm=twpas[0].pump.opx_output.full_scale_power_dbm
print(f'Pp={node.parameters.pumpline_attenuation+opxoutput(full_scale_power_dbm,p_p)}dBm\n opxoutput={opxoutput(full_scale_power_dbm,p_p)}')
with program() as on:    
    I, I_st, Q, Q_st,n,n_st = qua_declaration(num_qubits=len(qubits))
    I_, I_st_, Q_, Q_st_,n_,n_st_ = qua_declaration(num_qubits=len(qubits))
    dp = declare(int)  # QUA variable for the pump frequency
    da = declare(float)# QUA variable for the pump amplitude
    df = declare(int)  # QUA variable for the readout frequency
### test for checking pump with SA
    with infinite_loop_():
        update_frequency(twpas[0].pump.name,  f_p+ twpas[0].pump.intermediate_frequency)
        twpas[0].pump.play('pump', amplitude_scale=0.6, duration=pump_duration)

#%%
with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(on)
        results_off_n = fetching_tool(job, ["n"], mode="live")
        while results_off_n.is_processing():
            n_off_n = results_off_n.fetch_all()[0]
# %%
