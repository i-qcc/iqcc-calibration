# %%
import json
import os
from qualang_tools.units import unit
from iqcc_calibration_tools.quam_config.components import Quam
from iqcc_calibration_tools.quam_config.components.transmon import Transmon
from quam_builder_local.machine import save_machine
import numpy as np

# %%
def modify_quam(qubits: list[str], *, rr_LO, xy_LO, rr_if, xy_if, rr_max_power_dBm, xy_max_power_dBm):
    
    # NOTE: be aware of coupled ports for bands
    for i, q in enumerate(qubits):
        
        ## Update qubit rr freq and power
        machine.qubits[q].resonator.opx_output.full_scale_power_dbm = rr_max_power_dBm
        machine.qubits[q].resonator.opx_output.upconverter_frequency = rr_LO[i]
        machine.qubits[q].resonator.opx_input.downconverter_frequency = rr_LO[i]
        machine.qubits[q].resonator.opx_input.band = get_band(rr_LO[i])
        machine.qubits[q].resonator.opx_output.band = get_band(rr_LO[i])
        machine.qubits[q].resonator.intermediate_frequency = rr_if[i]

        ## Update qubit xy freq and power
        machine.qubits[q].xy.opx_output.full_scale_power_dbm = xy_max_power_dBm
        machine.qubits[q].xy.opx_output.upconverter_frequency = xy_LO[i]
        machine.qubits[q].xy.opx_output.band = get_band(xy_LO[i])
        machine.qubits[q].xy.intermediate_frequency = xy_if[i]

        # Update flux channels
        machine.qubits[q].z.opx_output.output_mode = "direct"
        machine.qubits[q].z.opx_output.upsampling_mode = "pulse"

        ## Update pulses
        # readout
        machine.qubits[q].resonator.operations["readout"].length = 1.5 * u.us
        machine.qubits[q].resonator.operations["readout"].amplitude = 1e-2
        # Qubit saturation
        machine.qubits[q].xy.operations["saturation"].length = 20 * u.us
        machine.qubits[q].xy.operations["saturation"].amplitude = 0.25
        # Single qubit gates - DragCosine
        machine.qubits[q].xy.operations["x180_DragCosine"].length = 48
        machine.qubits[q].xy.operations["x180_DragCosine"].amplitude = 0.2
        machine.qubits[q].xy.operations["x90_DragCosine"].amplitude = (
            machine.qubits[q].xy.operations["x180_DragCosine"].amplitude / 2
        )
        # Single qubit gates - Square
        machine.qubits[q].xy.operations["x180_Square"].length = 40
        machine.qubits[q].xy.operations["x180_Square"].amplitude = 0.1
        machine.qubits[q].xy.operations["x90_Square"].amplitude = (
            machine.qubits[q].xy.operations["x180_Square"].amplitude / 2
        )

def get_band(freq):
    if 50e6 <= freq < 4.5e9:
        return 1
    elif 4.5e9 <= freq < 6.5e9:
        return 2
    elif 6.5e9 <= freq <= 10.5e9:
        return 3
    else:
        raise ValueError(f"The specified frequency {freq} HZ is outside of the MW fem bandwidth [50 MHz, 10.5 GHz]")



# path in quam_state in the same directory as this script
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quam_state")

machine = Quam.load(path)
qubitsA = [q for q in machine.qubits.keys() if q[1] == "A"]
qubitsB = [q for q in machine.qubits.keys() if q[1] == "B"]
qubitsC = [q for q in machine.qubits.keys() if q[1] == "C"]
qubitsD = [q for q in machine.qubits.keys() if q[1] == "D"]

u = unit(coerce_to_integer=True)

# Change active qubits
machine.active_qubit_names = qubitsA + qubitsB + qubitsC + qubitsD
# machine.active_qubit_pair_names = ["coupler_qA3_qA4","coupler_qA4_qB3"]
 

# Update frequencies
modified_quam_vars = {}

# %%
# A Qubits
##################################################
# rr
intermediate_freq = np.array([-286813257.0, -21791553.0, -84115041.0, -164638501.0, 81976903.0, 178379634.0])
modified_quam_vars["rr_LO"] = np.array([7420000000.0, 7420000000.0, 7580000000.0, 7420000000.0, 7580000000.0, 7580000000.0])
modified_quam_vars["rr_if"] = intermediate_freq
modified_quam_vars["rr_max_power_dBm"] = -11

# xy
intermediate_frequency = np.array([
    -62522464.883557156,
    -137289105.9915785,
    -98911220.02489416,
    231101703.49702263,
    -83113227.33357999,
    135419723.0674383
])
modified_quam_vars["xy_LO"] = np.array([
    5910000000.0,  # qubitA1
    5800000000.0,  # qubitA2
    5800000000.0,  # qubitA3
    4950000000.0,  # qubitA4
    5750000000.0,  # qubitA5
    6250000000.0   # qubitA6
])
modified_quam_vars["xy_if"] = intermediate_frequency
modified_quam_vars["xy_max_power_dBm"] = 10

modify_quam(qubitsA, **modified_quam_vars)

# %%
# B Qubits
##################################################
# rr
intermediate_freq = np.array([-316157098.0, -50939140.0, 51005232.0, 261694271.0, 186482123.0])
modified_quam_vars["rr_LO"] = np.array([7480000000.0, 7480000000.0, 7480000000.0, 7480000000.0, 7480000000.0])
modified_quam_vars["rr_if"] = intermediate_freq
modified_quam_vars["rr_max_power_dBm"] = -11

# xy
intermediate_frequency = np.array([
    59844292.468822435,
    -95439117.95454636,
    -61370320.53792459,
    262023921.66865432,
    91149088.92091115
])
modified_quam_vars["xy_LO"] = np.array([
    4950000000.0,  # qubitB1
    6000000000.0,  # qubitB2
    5700000000.0,  # qubitB3
    6300000000.0,  # qubitB4
    5750000000.0   # qubitB5
])
modified_quam_vars["xy_if"] = intermediate_frequency
modified_quam_vars["xy_max_power_dBm"] = 10

modify_quam(qubitsB, **modified_quam_vars)

# %%
# C Qubits
##################################################
# rr
intermediate_freq = np.array([-227003056.0, 40477687.0, 143816428.0, -109407647.0, 280750629.0])
modified_quam_vars["rr_LO"] = np.array([7380000000.0, 7380000000.0, 7380000000.0, 7380000000.0, 7380000000.0])
modified_quam_vars["rr_if"] = intermediate_freq
modified_quam_vars["rr_max_power_dBm"] = -11

# xy
intermediate_frequency = np.array([
    151330899.27614138,
    266294084.49049723,
    85165772.6711429,
    318116712.0827206,
    -67772898.71249232
])
modified_quam_vars["xy_LO"] = np.array([
    4800000000.0,  # qubitC1
    5590000000.0,  # qubitC2
    5590000000.0,  # qubitC3
    4800000000.0,  # qubitC4
    6250000000.0   # qubitC5
])
modified_quam_vars["xy_if"] = intermediate_frequency
modified_quam_vars["xy_max_power_dBm"] = 10

modify_quam(qubitsC, **modified_quam_vars)

# %%
# D Qubits
##################################################
# rr
intermediate_freq = np.array([-307525665.0, -53621742.0, 62113601.0, 276047817.0, 199990165.0])
modified_quam_vars["rr_LO"] = np.array([7460000000.0, 7460000000.0, 7460000000.0, 7460000000.0, 7460000000.0])
modified_quam_vars["rr_if"] = intermediate_freq
modified_quam_vars["rr_max_power_dBm"] = -11

# xy
intermediate_frequency = np.array([
    224628283.86491203,
    109191240.8490092,
    128681593.72370648,
    293407098.9574231,
    -134835088.27633926
])
modified_quam_vars["xy_LO"] = np.array([
    5100000000.0,  # qubitD1
    5900000000.0,  # qubitD2
    5900000000.0,  # qubitD3
    6300000000.0,  # qubitD4
    6000000000.0   # qubitD5
])
modified_quam_vars["xy_if"] = intermediate_frequency
modified_quam_vars["xy_max_power_dBm"] = 10

modify_quam(qubitsD, **modified_quam_vars)

# add threading settings:
for name, qubit in machine.qubits.items():
    qubit.xy.thread = name
    qubit.resonator.thread = name

# add explicit decouple_offset
for name, qubit_pair in machine.qubit_pairs.items():
    qubit_pair.coupler.decouple_offset = 0.0

# add grid
grid = {
    "qA1": "2,4",
    "qA2": "3,4",
    "qA3": "2,3",
    "qA4": "3,3",
    "qA5": "4,3",
    "qA6": "2,2",
    "qB1": "4,2",
    "qB2": "4,1",
    "qB3": "3,2",
    "qB4": "3,1",
    "qB5": "3,0",
    "qC1": "2,0",
    "qC2": "1,0",
    "qC3": "2,1",
    "qC4": "1,1",
    "qC5": "0,1",
    "qD1": "0,2",
    "qD2": "0,3",
    "qD3": "1,2",
    "qD4": "1,3",
    "qD5": "1,4"}

for name, qubit in machine.qubits.items():
    qubit.grid_location = grid[name]

# %%
# save into state.json
save_machine(machine, path)

# %%
# View the corresponding "raw-QUA" config
with open("qua_config.json", "w+") as f:
    json.dump(machine.generate_config(), f, indent=4)

# %%
