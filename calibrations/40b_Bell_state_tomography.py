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
from datetime import datetime, timezone
from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode, NodeParameters
from iqcc_calibration_tools.quam_config.components import Quam
from iqcc_calibration_tools.quam_config.macros import active_reset, readout_state
from iqcc_calibration_tools.storage.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import xarray as xr
from calibration_utils.bell_state_tomography import (
    plot_3d_hist_with_frame_real,
    plot_3d_hist_with_frame_imag,
    get_pauli_data,
    get_density_matrix,
)

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = None
    num_shots: int = 2000
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal['active', 'thermal'] = "thermal"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    cz_macro_name: str = "cz"
    targets_name = "qubit_pairs"
    multiplexed: bool = True


node = QualibrationNode(
    name="40b_Bell_state_tomography", parameters=Parameters()
)

assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
node.machine = Quam.load()

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs_raw = node.machine.active_qubit_pairs
    qubit_pairs = node.get_multiplexed_pair_batches(node.machine.active_qubit_pair_names)
else:
    qubit_pairs_raw = [node.machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]
    qubit_pairs = node.get_multiplexed_pair_batches([qp.id for qp in qubit_pairs_raw])
    
num_qubit_pairs = len(qubit_pairs)


# Generate the OPX and Octave configurations
config = node.machine.generate_config()
octave_config = node.machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = node.machine.connect()
# %%

# %% {QUA_program}
n_shots = node.parameters.num_shots  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

with program() as Bell_state_tomography:
    
    n_st = declare_stream()
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st = [declare_stream() for _ in range(num_qubit_pairs)]
    
    
    if flux_point == "joint":
        # Bring the active qubits to the desired frequency point
        node.machine.initialize_qpu(target=qubit_pairs[0].qubit_control)
        #node.machine.set_all_fluxes(flux_point=flux_point, target=qubit_pairs[0].qubit_control)
    else:
        raise ValueError("flux_point must be 'joint'")
    
    for multiplexed_qubit_pairs in qubit_pairs.batch():
        n = declare(int)
        tomo_axis_control = declare(int)
        tomo_axis_target = declare(int)
        
        with for_(n, 0, n < n_shots, n + 1):
            save(n, n_st) 
            wait(100) # I think this is needed this otherwise data is not streaming properly. Need to verify. 
            with for_(tomo_axis_control, 0, tomo_axis_control < 3, tomo_axis_control + 1):
                with for_(tomo_axis_target, 0, tomo_axis_target < 3, tomo_axis_target + 1):
                    
                    # reset
                    for qp in multiplexed_qubit_pairs.values():
                        qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                        qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                    
                    align()
                    
                    # Bell state
                    for qp in multiplexed_qubit_pairs.values():
                        qp.qubit_control.xy.play("-y90")
                        qp.qubit_target.xy.play("y90")
                        qp.align()
                        qp.macros[node.parameters.cz_macro_name].apply()
                        qp.qubit_control.xy.play("y90")
                        qp.align()
                        # tomography pulses
                        with if_(tomo_axis_control == 0): #X axis
                            qp.qubit_control.xy.play("y90")
                        with elif_(tomo_axis_control == 1): #Y axis
                            qp.qubit_control.xy.play("x90")
                        with if_(tomo_axis_target == 0): #X axis
                            qp.qubit_target.xy.play("y90")
                        with elif_(tomo_axis_target == 1): #Y axis
                            qp.qubit_target.xy.play("x90")
                    align()
                    
                    # readout
                    for ii, qp in multiplexed_qubit_pairs.items():
                        readout_state(qp.qubit_control, state_control[ii])
                        readout_state(qp.qubit_target, state_target[ii])
                        assign(state[ii], state_control[ii]*2 + state_target[ii])
                        save(state_control[ii], state_st_control[ii])
                        save(state_target[ii], state_st_target[ii])
                        save(state[ii], state_st[ii])
                    
        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            state_st_control[i].buffer(3).buffer(3).buffer(n_shots).save(f"state_control{i + 1}")
            state_st_target[i].buffer(3).buffer(3).buffer(n_shots).save(f"state_target{i + 1}")
            state_st[i].buffer(3).buffer(3).buffer(n_shots).save(f"state{i + 1}")

node.namespace['qua_program'] = Bell_state_tomography

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, Bell_state_tomography, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout ) as qm:
        job = qm.execute(Bell_state_tomography)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_shots, start_time=results.start_time)

print(job.result_handles.__qpu_execution_time_seconds)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"tomo_axis_target": [0,1,2], "tomo_axis_control": [0,1,2], "N": np.linspace(1, n_shots, n_shots)})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}
    
# %% {Data_processing}
if not node.parameters.simulate:
    states = [0,1,2,3]

    results = []
    for state in states:
        results.append((ds.state == state).sum(dim = "N") / node.parameters.num_shots)
        
results_xr = xr.concat(results, dim=xr.DataArray(states, name="state"))
results_xr = results_xr.rename({"dim_0": "state"})
results_xr = results_xr.stack(
        tomo_axis=['tomo_axis_target', 'tomo_axis_control'])

corrected_results = []
for qp in qubit_pairs:
    corrected_results_qp = [] 
    for tomo_axis_control in [0,1,2]:
        corrected_results_control = []
        for tomo_axis_target in [0,1,2]:
            results = results_xr.sel(tomo_axis_control = tomo_axis_control, tomo_axis_target = tomo_axis_target, 
                                     qubit = qp.name)
            results = np.linalg.inv(qp.confusion) @ results.data
            # results = np.linalg.inv(np.diag((1,1,1,1))) @ results.data

            results = results * (results > 0)
            results = results / results.sum()
            corrected_results_control.append(results)
        corrected_results_qp.append(corrected_results_control)
    corrected_results.append(corrected_results_qp)

# Convert corrected_results to an xarray DataArray
corrected_results_xr = xr.DataArray(
    corrected_results,
    dims=['qubit', 'tomo_axis_control', 'tomo_axis_target', 'state'],
    coords={
        'qubit': [qp.name for qp in qubit_pairs],
        'tomo_axis_control': [0, 1, 2],
        'tomo_axis_target': [0, 1, 2],
        'state': ['00', '01', '10', '11']
    }
)
corrected_results_xr = corrected_results_xr.stack(
        tomo_axis=['tomo_axis_target', 'tomo_axis_control'])

# Store the xarray in the node results
paulis_data = {}
rhos = {}
for qp in qubit_pairs:
    paulis_data[qp.name] = get_pauli_data(corrected_results_xr.sel(qubit = qp.name))
    rhos[qp.name] = get_density_matrix(paulis_data[qp.name])
    
# %%
from scipy.linalg import sqrtm
ideal_dat = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
s_ideal = sqrtm(ideal_dat)
for qp in qubit_pairs:
    fidelity = float(np.abs(np.trace(sqrtm(s_ideal @rhos[qp.name] @ s_ideal)))**2)
    print(f"Fidelity of {qp.name}: {fidelity:.3f}")
    purity = np.abs(np.trace(rhos[qp.name] @ rhos[qp.name]))
    print(f"Purity of {qp.name}: {purity:.3f}")
    print()
    node.results[f"{qp.name}_fidelity"] = fidelity
    node.results[f"{qp.name}_purity"] = purity



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
    
    # Create separate 3D city plots for real and imaginary parts
    num_pairs = len(qubit_pairs_sorted_by_batch)
    num_cols = 3
    num_rows_3d = int(np.ceil(num_pairs / num_cols))
    ideal_dat = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
    
    # Real part 3D city plots
    fig_3d_real, axes_3d_real = plt.subplots(num_rows_3d, num_cols, figsize=(6 * num_cols, 4.5 * num_rows_3d), 
                                             subplot_kw={'projection': '3d'}, squeeze=False)
    
    for idx, qp in enumerate(qubit_pairs_sorted_by_batch):
        fidelity = node.results[f"{qp.name}_fidelity"]
        purity = node.results[f"{qp.name}_purity"]
        
        row = idx // num_cols
        col = idx % num_cols
        
        ax_real = axes_3d_real[row, col]
        plot_3d_hist_with_frame_real(rhos[qp.name], ideal_dat, ax_real)
        ax_real.set_title(f"{qp.name}\nFidelity: {fidelity:.3f}, Purity: {purity:.3f}")
        
        # Add batch number indicator at the bottom right
        batch_num = pair_to_batch.get(qp.name, 0)
        ax_real.text2D(0.98, 0.02, str(batch_num), transform=ax_real.transAxes, 
                      fontsize=8, ha='right', va='bottom',
                      bbox=dict(boxstyle='circle', facecolor='plum', edgecolor='magenta', linewidth=1.2))
    
    # Hide unused subplots
    for i in range(num_pairs, num_rows_3d * num_cols):
        row = i // num_cols
        col = i % num_cols
        axes_3d_real[row, col].axis('off')
    
    fig_3d_real.suptitle(f"Bell state tomography - Real part (3D city plots) \n {node.date_time} GMT+3 #{node.node_id} \n reset type = {node.parameters.reset_type}", y=0.98)
    fig_3d_real.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
    # Add legend explaining the batch number indicator
    legend_circle = mpatches.Circle((0, 0), 0.5, facecolor='plum', edgecolor='magenta', linewidth=1.2)
    fig_3d_real.legend([legend_circle], ['Batch number (pairs run in parallel)'], 
                       loc='upper right', fontsize=8, framealpha=0.9)
    fig_3d_real.show()
    node.results["figure_city_real"] = fig_3d_real
    
    # Imaginary part 3D city plots
    fig_3d_imag, axes_3d_imag = plt.subplots(num_rows_3d, num_cols, figsize=(6 * num_cols, 4.5 * num_rows_3d), 
                                              subplot_kw={'projection': '3d'}, squeeze=False)
    
    for idx, qp in enumerate(qubit_pairs_sorted_by_batch):
        row = idx // num_cols
        col = idx % num_cols
        
        ax_imag = axes_3d_imag[row, col]
        plot_3d_hist_with_frame_imag(rhos[qp.name], ideal_dat, ax_imag)
        ax_imag.set_title(f"{qp.name} - Imaginary")
        
        # Add batch number indicator at the bottom right
        batch_num = pair_to_batch.get(qp.name, 0)
        ax_imag.text2D(0.98, 0.02, str(batch_num), transform=ax_imag.transAxes, 
                      fontsize=8, ha='right', va='bottom',
                      bbox=dict(boxstyle='circle', facecolor='plum', edgecolor='magenta', linewidth=1.2))
    
    # Hide unused subplots
    for i in range(num_pairs, num_rows_3d * num_cols):
        row = i // num_cols
        col = i % num_cols
        axes_3d_imag[row, col].axis('off')
    
    fig_3d_imag.suptitle(f"Bell state tomography - Imaginary part (3D city plots) \n {node.date_time} GMT+3 #{node.node_id} \n reset type = {node.parameters.reset_type}", y=0.98)
    fig_3d_imag.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
    # Add legend explaining the batch number indicator
    legend_circle = mpatches.Circle((0, 0), 0.5, facecolor='plum', edgecolor='magenta', linewidth=1.2)
    fig_3d_imag.legend([legend_circle], ['Batch number (pairs run in parallel)'], 
                       loc='upper right', fontsize=8, framealpha=0.9)
    fig_3d_imag.show()
    node.results["figure_city_imag"] = fig_3d_imag
    
    # Organize 2D plots in 3-column grid
    num_cols = 3
    num_rows = int(np.ceil(num_pairs / num_cols))
    
    # Real part density matrix plot
    fig_real = plt.figure(figsize=(5 * num_cols, 4 * num_rows))
    for idx, qp in enumerate(qubit_pairs_sorted_by_batch):
        ax = fig_real.add_subplot(num_rows, num_cols, idx + 1)
        rho = np.real(rhos[qp.name])
        ax.pcolormesh(rho, vmin = -0.5, vmax = 0.5, cmap = "RdBu")
        for i in range(4):
            for j in range(4):
                if np.abs(rho[i][j]) < 0.1:
                    ax.text(i+0.5, j+0.5, f"{ rho[i][j]:.2f}", ha="center", va="center", color="k")
                else:
                    ax.text(i+0.5, j+0.5, f"{ rho[i][j]:.2f}", ha="center", va="center", color="w")
        fidelity = node.results[f"{qp.name}_fidelity"]
        purity = node.results[f"{qp.name}_purity"]
        ax.set_title(f"{qp.name}\nFidelity: {fidelity:.3f}, Purity: {purity:.3f}")
        ax.set_xlabel('Pauli Operators')
        ax.set_ylabel('Pauli Operators')
        ax.set_xticks(range(4), ['00', '01', '10', '11'])
        ax.set_yticks(range(4), ['00', '01', '10', '11'])
        ax.set_xticklabels(['00', '01', '10', '11'], rotation=45, ha='right')
        ax.set_yticklabels(['00', '01', '10', '11'])
        
        # Add batch number indicator at the bottom right
        batch_num = pair_to_batch.get(qp.name, 0)
        ax.text(0.98, -0.08, str(batch_num), transform=ax.transAxes, 
               fontsize=8, ha='right', va='top',
               bbox=dict(boxstyle='circle', facecolor='plum', edgecolor='magenta', linewidth=1.2))
    fig_real.suptitle(f"Bell state tomography (real part) \n {node.date_time} GMT+3 #{node.node_id} \n reset type = {node.parameters.reset_type}")
    fig_real.tight_layout()
    # Add legend explaining the batch number indicator
    legend_circle = mpatches.Circle((0, 0), 0.5, facecolor='plum', edgecolor='magenta', linewidth=1.2)
    fig_real.legend([legend_circle], ['Batch number (pairs run in parallel)'], 
                   loc='upper right', fontsize=8, framealpha=0.9)
    fig_real.show()
    node.results["figure_rho_real"] = fig_real
    
    # Imaginary part density matrix plot
    fig_imag = plt.figure(figsize=(5 * num_cols, 4 * num_rows))
    for idx, qp in enumerate(qubit_pairs_sorted_by_batch):
        ax = fig_imag.add_subplot(num_rows, num_cols, idx + 1)
        rho = np.imag(rhos[qp.name])
        ax.pcolormesh(rho, vmin = -0.1, vmax = 0.1, cmap = "RdBu")
        for i in range(4):
            for j in range(4):
                if np.abs(rho[i][j]) < 0.1:
                    ax.text(i+0.5, j+0.5, f"{ rho[i][j]:.2f}", ha="center", va="center", color="k")
                else:
                    ax.text(i+0.5, j+0.5, f"{ rho[i][j]:.2f}", ha="center", va="center", color="w")
        fidelity = node.results[f"{qp.name}_fidelity"]
        purity = node.results[f"{qp.name}_purity"]
        ax.set_title(f"{qp.name}\nFidelity: {fidelity:.3f}, Purity: {purity:.3f}")
        ax.set_xlabel('Pauli Operators')
        ax.set_ylabel('Pauli Operators')
        ax.set_xticks(range(4), ['00', '01', '10', '11'])
        ax.set_yticks(range(4), ['00', '01', '10', '11'])
        ax.set_xticklabels(['00', '01', '10', '11'], rotation=45, ha='right')
        ax.set_yticklabels(['00', '01', '10', '11'])
        
        # Add batch number indicator at the bottom right
        batch_num = pair_to_batch.get(qp.name, 0)
        ax.text(0.98, -0.08, str(batch_num), transform=ax.transAxes, 
               fontsize=8, ha='right', va='top',
               bbox=dict(boxstyle='circle', facecolor='plum', edgecolor='magenta', linewidth=1.2))
    fig_imag.suptitle(f"Bell state tomography (imaginary part) \n {node.date_time} GMT+3 #{node.node_id} \n reset type = {node.parameters.reset_type}")
    fig_imag.tight_layout()
    # Add legend explaining the batch number indicator
    legend_circle = mpatches.Circle((0, 0), 0.5, facecolor='plum', edgecolor='magenta', linewidth=1.2)
    fig_imag.legend([legend_circle], ['Batch number (pairs run in parallel)'], 
                   loc='upper right', fontsize=8, framealpha=0.9)
    fig_imag.show()
    node.results["figure_rho_imag"] = fig_imag
    
    # Pauli operators plot
    fig_paulis = plt.figure(figsize=(5 * num_cols, 4 * num_rows))
    for idx, qp in enumerate(qubit_pairs_sorted_by_batch):
        ax = fig_paulis.add_subplot(num_rows, num_cols, idx + 1)
        # Extract the values and labels for plotting
        values = paulis_data[qp.name].values
        labels = paulis_data[qp.name].coords['pauli_op'].values

        # Create a bar plot
        bars = ax.bar(range(len(values)), values)

        # Customize the plot
        ax.set_xlabel('Pauli Operators')
        ax.set_ylabel('Value')
        ax.set_title(qp.name)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right')

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # Add batch number indicator at the bottom right
        batch_num = pair_to_batch.get(qp.name, 0)
        ax.text(0.98, -0.08, str(batch_num), transform=ax.transAxes, 
               fontsize=8, ha='right', va='top',
               bbox=dict(boxstyle='circle', facecolor='plum', edgecolor='magenta', linewidth=1.2))
    
    fig_paulis.tight_layout()
    # Add legend explaining the batch number indicator
    legend_circle = mpatches.Circle((0, 0), 0.5, facecolor='plum', edgecolor='magenta', linewidth=1.2)
    fig_paulis.legend([legend_circle], ['Batch number (pairs run in parallel)'], 
                     loc='upper right', fontsize=8, framealpha=0.9)
    fig_paulis.show()
    node.results["figure_paulis"] = fig_paulis


# %% {Update_state}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qp in qubit_pairs:
                node.machine.qubit_pairs[qp.id].macros[node.parameters.cz_macro_name].fidelity["Bell_State"] = {"Fidelity":  node.results[f"{qp.name}_fidelity"], "Purity":  node.results[f"{qp.name}_purity"]}
                

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()
        
# %%
