# %%
"""
Two-Qubit Readout Confusion Matrix Measurement

This sequence measures the readout error when simultaneously measuring the state of two qubits. The process involves:

1. Preparing the two qubits in all possible combinations of computational basis states (|00⟩, |01⟩, |10⟩, |11⟩)
2. Performing simultaneous readout on both qubits
3. Calculating the confusion matrix based on the measurement results

For each prepared state, we measure:
1. The readout result of the first qubit
2. The readout result of the second qubit

The measurement process involves:
1. Initializing both qubits to the ground state
2. Applying single-qubit gates to prepare the desired input state
3. Performing simultaneous readout on both qubits
4. Repeating the process multiple times to gather statistics

The outcome of this measurement will be used to:
1. Quantify the readout fidelity for two-qubit states
2. Identify and characterize crosstalk effects in the readout process
3. Provide data for readout error mitigation in two-qubit experiments

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair
- Calibrated readout for both qubits

Outcomes:
- 4x4 confusion matrix representing the probabilities of measuring each two-qubit state given a prepared input state
- Readout fidelity metrics for simultaneous two-qubit measurement
"""

# %% {Imports}

from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode, NodeParameters
from iqcc_calibration_tools.quam_config.components import Quam
from iqcc_calibration_tools.quam_config.macros import active_reset, readout_state
from iqcc_calibration_tools.storage.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = None
    num_shots: int = 2000
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal['active', 'thermal'] = "thermal"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    plot_raw : bool = False
    measure_leak : bool = False
    targets_name: str = "qubit_pairs"
    multiplexed: bool = True


node = QualibrationNode(
    name="34_2Q_confusion_matrix", parameters=Parameters()
)

assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
node.machine = machine = Quam.load()

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs_raw = machine.active_qubit_pairs
    qubit_pairs = node.get_multiplexed_pair_batches(machine.active_qubit_pair_names)
else:
    qubit_pairs_raw = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]
    qubit_pairs = node.get_multiplexed_pair_batches([qp.id for qp in qubit_pairs_raw])
# if any([qp.q1.z is None or qp.q2.z is None for qp in qubit_pairs]):
#     warnings.warn("Found qubit pairs without a flux line. Skipping")

num_qubit_pairs = len(qubit_pairs)

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
# %%

####################
# Helper functions #
####################


# %% {QUA_program}
n_shots = node.parameters.num_shots  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

with program() as CPhase_Oscillations:
    control_initial = declare(int)
    target_initial = declare(int)
    n = declare(int)
    n_st = declare_stream()
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st = [declare_stream() for _ in range(num_qubit_pairs)]
    
    for multiplexed_qubit_pairs in qubit_pairs.batch():
        # Initialize the qubits
        for qp in multiplexed_qubit_pairs.values():
            node.machine.initialize_qpu(target=qp.qubit_control)
            node.machine.initialize_qpu(target=qp.qubit_target)
        # wait(1000)
        align()
        with for_(n, 0, n < n_shots, n + 1):
            save(n, n_st)         
            with for_(*from_array(control_initial, [0,1])):
                with for_(*from_array(target_initial, [0,1])):
                    # reset
                    for qp in multiplexed_qubit_pairs.values():
                        qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                        qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                    align()
                    
                    # setting both qubits to the initial state
                    for qp in multiplexed_qubit_pairs.values():
                        with if_(control_initial==1):
                            qp.qubit_control.xy.play("x180")
                        with if_(target_initial==1):
                            qp.qubit_target.xy.play("x180")
                    
                    align() # qp.align()
                    # readout
                    for ii, qp in multiplexed_qubit_pairs.items():
                        readout_state(qp.qubit_control, state_control[ii])
                        readout_state(qp.qubit_target, state_target[ii])
                        assign(state[ii], state_control[ii]*2 + state_target[ii])
                        save(state_control[ii], state_st_control[ii])
                        save(state_target[ii], state_st_target[ii])
                        save(state[ii], state_st[ii])
                    align()
        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            state_st_control[i].buffer(2).buffer(2).buffer(n_shots).save(f"state_control{i + 1}")
            state_st_target[i].buffer(2).buffer(2).buffer(n_shots).save(f"state_target{i + 1}")
            state_st[i].buffer(2).buffer(2).buffer(n_shots).save(f"state{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, CPhase_Oscillations, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    
    with qm_session(qmm, config, timeout=node.parameters.timeout ) as qm:
        job = qm.execute(CPhase_Oscillations)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_shots, start_time=results.start_time)


node.namespace['job'] = job
# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"init_state_target": [0,1], "init_state_control": [0,1], "N": np.linspace(1, n_shots, n_shots)})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}
    
# %%
if not node.parameters.simulate:
    states = [0,1,2,3]

    confusions = {}
    for qp in qubit_pairs:
        conf = []
        for state in states:
            row = []
            for q1 in [0,1]:
                for q0 in [0,1]:
                    row.append((ds.sel(qubit = qp.name).state.sel(init_state_target = q0,init_state_control = q1) == state).sum().values)
            conf.append(row)
        confusions[qp.name] = np.array(conf)/node.parameters.num_shots

# %% {Plot_results}
if not node.parameters.simulate:
    import matplotlib.patches as mpatches
    
    # Create mapping from pair name to batch index (1-indexed) and sort pairs by batch order
    pair_to_batch = {}
    qubit_pairs_sorted_by_batch = []
    for batch_idx, batch in enumerate(qubit_pairs.batch(), start=1):
        for pair_idx, qp in batch.items():
            pair_to_batch[qp.name] = batch_idx
            qubit_pairs_sorted_by_batch.append(qp)
    
    # Organize plots in 3-column grid
    num_pairs = len(qubit_pairs_sorted_by_batch)
    num_cols = 3
    num_rows = int(np.ceil(num_pairs / num_cols))
    
    fig_confusion = plt.figure(figsize=(5 * num_cols, 4 * num_rows))
    for idx, qp in enumerate(qubit_pairs_sorted_by_batch):
        ax = fig_confusion.add_subplot(num_rows, num_cols, idx + 1)
        print(qp.name)
        conf = confusions[qp.name]
        ax.pcolormesh(['00','01','10','11'],['00','01','10','11'],conf)
        for i in range(4):
            for j in range(4):
                if i==j:
                    ax.text(i, j, f"{100 * conf[i][j]:.1f}%", ha="center", va="center", color="k")
                else:
                    ax.text(i, j, f"{100 * conf[i][j]:.1f}%", ha="center", va="center", color="w")
        ax.set_ylabel('prepared')
        ax.set_xlabel('measured')
        ax.set_title(f"Confusion matrix {qp.name} \n {node.date_time} GMT+{node.time_zone} #{node.node_id} \n reset type = {node.parameters.reset_type}")
        
        # Add batch number indicator at the bottom right
        batch_num = pair_to_batch.get(qp.name, 0)
        ax.text(0.98, -0.08, str(batch_num), transform=ax.transAxes, 
               fontsize=8, ha='right', va='top',
               bbox=dict(boxstyle='circle', facecolor='plum', edgecolor='magenta', linewidth=1.2))
    
    # Add legend explaining the batch number indicator
    legend_circle = mpatches.Circle((0, 0), 0.5, facecolor='plum', edgecolor='magenta', linewidth=1.2)
    fig_confusion.legend([legend_circle], ['Batch number (pairs run in parallel)'], 
                   loc='upper right', fontsize=8, framealpha=0.9)
    
    fig_confusion.tight_layout()
    fig_confusion.show()
    node.results["figure_confusion"] = fig_confusion
# %%

# %% {Update_state}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qp in qubit_pairs:
                qp.confusion = confusions[qp.name].tolist()
# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
        
# %%
