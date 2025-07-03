# %%
import json
import os
from qualang_tools.wirer.wirer.channel_specs import *
from qualang_tools.wirer import Instruments, Connectivity, allocate_wiring, visualize
from quam_builder_local.machine import build_quam_wiring



# Define static parameters
host_ip = "10.1.1.11"  # QOP IP address
port = None  # QOP Port
cluster_name = "Cluster_1" # carmel-arbel  # Name of the cluster
quantum_computer_backend = "qc_qwtune"# "arbel"
# Desired location of wiring.json and state.json
# The folder must not contain other json files.
path = "/home/omrieoqm/r_and_d/qw_statestore_QCC/quam_state"

# Delete existing state.json and wiring.json files if they exist
state_file = os.path.join(path, "state.json")
wiring_file = os.path.join(path, "wiring.json")
if os.path.exists(state_file):
    os.remove(state_file)
if os.path.exists(wiring_file):
    os.remove(wiring_file)

# Define the available instrument setup
instruments = Instruments()
instruments.add_mw_fem(controller=1, slots=[1,2,3,4])
instruments.add_lf_fem(controller=1, slots=[5,6,7,8])

# Define which qubit indices are present in the system
qubitsA = ["A1", "A2", "A3", "A4", "A5", "A6"]
qubitsB = ["B1", "B2", "B3", "B4", "B5"]
qubitsC = ["C1", "C2", "C3", "C4", "C5"]
qubitsD = ["D1", "D2", "D3", "D4", "D5"]
qubits = qubitsA + qubitsB + qubitsC + qubitsD

# Must be list of tuples, each tuple is a pair of qubits that are coupled
qubit_pairs = [] # no couplers



# Allocate the wiring to the connectivity object based on the available instruments
connectivity = Connectivity()


# Define any custom/hardcoded channel addresses
q_drive_chs = [
    mw_fem_spec(con=1, slot = 4, out_port = 2),  #A1
    mw_fem_spec(con=1, slot = 4, out_port = 3),  #A2
    mw_fem_spec(con=1, slot = 4, out_port = 4),  #A3 
    mw_fem_spec(con=1, slot = 4, out_port = 5),  #A4 
    mw_fem_spec(con=1, slot = 4, out_port = 6),  #A5 
    mw_fem_spec(con=1, slot = 4, out_port = 7),  #A6
    mw_fem_spec(con=1, slot = 1, out_port = 2),  #B1
    mw_fem_spec(con=1, slot = 1, out_port = 3),  #B2
    mw_fem_spec(con=1, slot = 1, out_port = 4),  #B3 
    mw_fem_spec(con=1, slot = 1, out_port = 5),  #B4 
    mw_fem_spec(con=1, slot = 1, out_port = 6),  #B5 
    mw_fem_spec(con=1, slot = 2, out_port = 2),  #C1
    mw_fem_spec(con=1, slot = 2, out_port = 3),  #C2
    mw_fem_spec(con=1, slot = 2, out_port = 4),  #C3 
    mw_fem_spec(con=1, slot = 2, out_port = 5),  #C4 
    mw_fem_spec(con=1, slot = 2, out_port = 6),  #C5 
    mw_fem_spec(con=1, slot = 3, out_port = 2),  #D1
    mw_fem_spec(con=1, slot = 3, out_port = 3),  #D2
    mw_fem_spec(con=1, slot = 3, out_port = 4),  #D3
    mw_fem_spec(con=1, slot = 3, out_port = 5),  #D4
    mw_fem_spec(con=1, slot = 3, out_port = 6),  #D5
    
]

q_flux_chs = [
            lf_fem_spec(con=1, out_slot = 8, out_port = 1),  #A1
            lf_fem_spec(con=1, out_slot = 8, out_port = 2),  #A2
            lf_fem_spec(con=1, out_slot = 8, out_port = 3),  #A3 
            lf_fem_spec(con=1, out_slot = 8, out_port = 4),  #A4 
            lf_fem_spec(con=1, out_slot = 8, out_port = 5),  #A5 
            lf_fem_spec(con=1, out_slot = 8, out_port = 6),  #A6
            lf_fem_spec(con=1, out_slot = 5, out_port = 1),  #B1
            lf_fem_spec(con=1, out_slot = 5, out_port = 2),  #B2
            lf_fem_spec(con=1, out_slot = 5, out_port = 3),  #B3 
            lf_fem_spec(con=1, out_slot = 5, out_port = 4),  #B4 
            lf_fem_spec(con=1, out_slot = 5, out_port = 5),  #B5 
            
            lf_fem_spec(con=1, out_slot = 7, out_port = 1),  #C1
            lf_fem_spec(con=1, out_slot = 7, out_port = 2),  #C2
            lf_fem_spec(con=1, out_slot = 7, out_port = 3),  #C3 
            lf_fem_spec(con=1, out_slot = 7, out_port = 4),  #C4 
            lf_fem_spec(con=1, out_slot = 7, out_port = 5),  #C5 
            lf_fem_spec(con=1, out_slot = 6, out_port = 1),  #D1
            lf_fem_spec(con=1, out_slot = 6, out_port = 2),  #D2
            lf_fem_spec(con=1, out_slot = 6, out_port = 3),  #D3 
            lf_fem_spec(con=1, out_slot = 6, out_port = 4),  #D4 
            lf_fem_spec(con=1, out_slot = 6, out_port = 5),  #D5 
    
]


# Single feed-line for reading the resonators & individual qubit drive lines
# Define any custom/hardcoded channel addresses
qA_res_ch = mw_fem_spec(con=1, slot=4, in_port=1, out_port=1)
qB_res_ch = mw_fem_spec(con=1, slot=1, in_port=1, out_port=1)
qC_res_ch = mw_fem_spec(con=1, slot=2, in_port=1, out_port=1)
qD_res_ch = mw_fem_spec(con=1, slot=3, in_port=1, out_port=1)

connectivity.add_resonator_line(qubits=qubitsA, constraints=qA_res_ch)
connectivity.add_resonator_line(qubits=qubitsB, constraints=qB_res_ch)
connectivity.add_resonator_line(qubits=qubitsC, constraints=qC_res_ch)
connectivity.add_resonator_line(qubits=qubitsD, constraints=qD_res_ch)

for i in range(len(qubits)):
    connectivity.add_qubit_flux_lines(qubits=qubits[i], constraints=q_flux_chs[i])
    connectivity.add_qubit_drive_lines(qubits=qubits[i], constraints=q_drive_chs[i])
# for i in range(len(qubit_pairs)):
#     connectivity.add_qubit_pair_flux_lines(qubit_pairs=qubit_pairs[i], constraints=q_coupler_chs[i])
#     pass
    
allocate_wiring(connectivity, instruments)

# Single feed-line for reading the resonators & driving the qubits + flux on specific fem slot
# Define any custom/hardcoded channel addresses
# q1_res_ch = mw_fem_spec(con=1, slot=1, in_port=1, out_port=1)
# q1_drive_ch = mw_fem_spec(con=1, slot=1, in_port=None, out_port=2)
# q1_flux_fem = lf_fem_spec(con=1, in_slot=None, in_port=None, out_slot=4, out_port=None)
# connectivity.add_resonator_line(qubits=qubits, constraints=q1_res_ch)
# connectivity.add_qubit_flux_lines(qubits=qubits, constraints=q1_flux_fem)
# connectivity.add_qubit_pair_flux_lines(qubit_pairs=[(1,2)])  # Tunable coupler
# for qubit in qubits:
#     connectivity.add_qubit_drive_lines(qubits=qubit, constraints=q1_drive_ch)
#     allocate_wiring(connectivity, instruments, block_used_channels=False)

# Build the wiring and network into a QuAM machine and save it as "wiring.json"
build_quam_wiring(connectivity, host_ip, cluster_name, path, port)

# add quantum_computer_backend and cloud to the wiring.network
with open(path + "/wiring.json", "r") as f:
    wiring = json.load(f)
wiring["network"]["quantum_computer_backend"] = quantum_computer_backend
wiring["network"]["cloud"] = True
with open(path + "/wiring.json", "w") as f:
    json.dump(wiring, f, indent=4)

# View wiring schematic
visualize(connectivity.elements, available_channels=instruments.available_channels)

# %%
