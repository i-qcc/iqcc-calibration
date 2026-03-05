
""" Jeongwon Kim Omrie, Wei, Akiva, Ariel  2060226
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
from calibration_utils.T2echo.system import system
# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ['qD1','qD2','qD3','qD4','qD5']
    
# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = Quam.load()
node = QualibrationNode(name=f"000_system_analysis", parameters=Parameters())
date_time = datetime.now(timezone(timedelta(hours=2))).strftime("%Y-%m-%d %H:%M:%S")
node.results["date"]={"date":date_time}
# Get the relevant QuAM components
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()
# %% {Analysis}
t1 = np.array([])
t2echo = np.array([])
chi_list = []
kappa_list = []
for q in node.parameters.qubits:
    qubit = machine.qubits[q]
    t1 = np.append(t1, qubit.T1)
    t2echo = np.append(t2echo, qubit.T2echo)
    chi_list.append(qubit.chi)
    kappa_list.append(qubit.resonator.kappa)
# T1 and T2echo from machine are in seconds; system() expects µs
t1_us = t1 * 1e6
t2echo_us = t2echo * 1e6
carmel_gilboa_D40 = system(opx_temp=75000,
                        attenuation={'300K': 10, 
                                    '50K': 2, 
                                    '4K': 22, 
                                    'ST': 2, 
                                    'CP': 22, 
                                    'MXC': 42},
                        T1=t1_us, T2=t2echo_us,
                        chi=chi_list, kappa=kappa_list) 
carmel_gilboa_D40.T2_analysis(n = np.logspace(-6, -1, 100), title='CarmelGilboaD -40dB@MXC')
# %%
