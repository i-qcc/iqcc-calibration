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

from iqcc_calibration_tools.analysis.fit import extract_dominant_frequencies
from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode, NodeParameters
from iqcc_calibration_tools.quam_config.components import Quam
from iqcc_calibration_tools.quam_config.macros import active_reset, readout_state, readout_state_gef, active_reset_gef, active_reset_simple
from iqcc_calibration_tools.analysis.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
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
import warnings
from qualang_tools.bakery import baking
# from iqcc_calibration_tools.analysis.fit import fit_oscillation, oscillation, fix_oscillation_phi_2pi
from iqcc_calibration_tools.analysis.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from scipy.optimize import curve_fit
from iqcc_calibration_tools.quam_config.components.gates.two_qubit_gates import CZGate
from iqcc_calibration_tools.quam_config.lib.pulses import FluxPulse

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = None
    num_averages: int = 150
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    
    coupler_flux_min : float = -0.02 #relative to the coupler set point
    coupler_flux_max : float = 0.05 #relative to the coupler set point
    coupler_flux_step : float = 0.0005
    
    qubit_flux_min : float = 0.0 # relative to the known/calculated detuning between the qubits
    qubit_flux_max : float = 0.01 # relative to the known/calculated detuning between the qubits
    qubit_flux_step : float = 0.02  
    
    idle_time_max : int = 1500 # in ns
    idle_time_min : int = 16 # in ns
    idle_time_step : int = 4 # in ns
    
    use_state_discrimination: bool = True
    

node = QualibrationNode(
    name="61_Ramsey_ZZ", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."
node_id = 1

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = Quam.load()

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]
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
n_avg = node.parameters.num_averages  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent_or_pairwise  # 'independent' or 'joint' or 'pairwise'
# Loop parameters
fluxes_coupler = np.arange(node.parameters.coupler_flux_min, node.parameters.coupler_flux_max+0.0001, node.parameters.coupler_flux_step)
fluxes_qubit = np.arange(node.parameters.qubit_flux_min, node.parameters.qubit_flux_max+0.0001, node.parameters.qubit_flux_step)

# idle times in clock cycles = 4ns
idle_times = np.arange(node.parameters.idle_time_min // 4, node.parameters.idle_time_max // 4 +0.0001, node.parameters.idle_time_step // 4)
max_duration = 2 * node.parameters.idle_time_max // 4 + 200

with program() as CPhase_Oscillations:
    n = declare(int)
    flux_coupler = declare(fixed)
    flux_qubit = declare(fixed)
    idle_time = declare(int)
    n_st = declare_stream()

    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st = [declare_stream() for _ in range(num_qubit_pairs)]

    
    if flux_point == "joint":
        machine.initialize_qpu(flux_point=flux_point, target=qubit_pairs[0].qubit_control)
    
    for i, qp in enumerate(qubit_pairs):
        # Bring the active qubits to the minimum frequency point
        if flux_point != "joint":
            machine.initialize_qpu(flux_point=flux_point, target=qp.qubit_control)
        wait(1000)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)         
            with for_(*from_array(flux_coupler, fluxes_coupler)):
                with for_(*from_array(flux_qubit, fluxes_qubit)):
                    with for_(*from_array(idle_time, idle_times)):
                        # reset
                        if node.parameters.reset_type == "active":
                                active_reset_simple(qp.qubit_control)
                                active_reset_simple(qp.qubit_target)
                                qp.align()
                        else:
                            wait(qp.qubit_control.thermalization_time * u.ns)
                            wait(qp.qubit_target.thermalization_time * u.ns)
                        qp.align()
      
                        qp.qubit_control.z.play("const", amplitude_scale = flux_qubit / qp.qubit_control.z.operations["const"].amplitude, duration = max_duration)                
                        qp.coupler.play("const", amplitude_scale = flux_coupler / qp.coupler.operations["const"].amplitude, duration = max_duration)
                        qp.qubit_target.xy.wait(100)
                        qp.qubit_target.xy.play("x90")
                        wait(idle_time)
                        qp.qubit_control.xy.wait(100 + idle_time + qp.qubit_target.xy.operations["x90"].length // 4)
                        qp.qubit_control.xy.play("x180")
                        qp.qubit_target.xy.play("x180")
                        wait(idle_time)
                        qp.qubit_target.xy.play("x90")
                        qp.align()
                        # readout
                        readout_state(qp.qubit_control, state_control[i])
                        readout_state(qp.qubit_target, state_target[i])
                        assign(state[i], state_control[i]*2 + state_target[i])
                        save(state_control[i], state_st_control[i])
                        save(state_target[i], state_st_target[i])
                        save(state[i], state_st[i])
        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            state_st_control[i].buffer(len(idle_times)).buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_control{i + 1}")
            state_st_target[i].buffer(len(idle_times)).buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_target{i + 1}")
            state_st[i].buffer(len(idle_times)).buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state{i + 1}")

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
    
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(CPhase_Oscillations)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {  "idle_time": idle_times * 4, "flux_qubit": fluxes_qubit, "flux_coupler": fluxes_coupler})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}
# %%

ds.state_target.plot()
# %%


ds['dominant_frequency'] = extract_dominant_frequencies(ds.state_target)
ds.dominant_frequency.attrs['units'] = 'GHz'

# %%
from qualibration_libs.analysis import fit_oscillation_decay_exp, oscillation_decay_exp, peaks_dips

fit_data = fit_oscillation_decay_exp(ds.state_target, "idle_time")




# %%
detuning_mode = "quadratic" # "cosine" or "quadratic"
if not node.parameters.simulate:
    if reset_coupler_bias:
        flux_coupler_full = np.array([fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs])
    else:
        flux_coupler_full = np.array([fluxes_coupler for qp in qubit_pairs])
    if detuning_mode == "quadratic":
        detuning = np.array([-fluxes_qp[qp.name] ** 2 * qp.qubit_control.freq_vs_flux_01_quad_term  for qp in qubit_pairs])
    elif detuning_mode == "cosine":
        detuning = np.array([oscillation(fluxes_qubit, qp.qubit_control.extras['a'], qp.qubit_control.extras['f'], qp.qubit_control.extras['phi'], qp.qubit_control.extras['offset']) for qp in qubit_pairs])
    ds = ds.assign_coords({"flux_coupler_full": (["qubit", "flux_coupler"], flux_coupler_full)})
    ds = ds.assign_coords({"detuning": (["qubit", "flux_qubit"], detuning)})
    node.results = {"ds": ds}
  
# %%
ds.state_target.plot()

# %%
node.results["results"] = {}

if not node.parameters.simulate:
    if node.parameters.use_state_discrimination:
        res_sum = -ds.state_control + ds.state_target
    else:
        res_sum = -ds.I_control + ds.I_target
    
    for i, qp in enumerate(qubit_pairs):
        coupler_min_arg = res_sum.sel(qubit = qp.name).mean(dim = 'flux_qubit').argmin()
        flux_coupler_min = ds.flux_coupler_full.sel(qubit = qp.name)[coupler_min_arg]
        qubit_max_arg = res_sum.sel(qubit = qp.name).mean(dim = "flux_coupler").argmax()
        flux_qubit_max = fluxes_qp[qp.name][qubit_max_arg]
        node.results["results"][qp.name] = {"flux_coupler_min": float(flux_coupler_min.values), "flux_qubit_max": float(flux_qubit_max)}


# %% {Plotting}
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        if node.parameters.use_state_discrimination:
            values_to_plot = ds.state_control.sel(qubit=qp['qubit'])
        else:
            values_to_plot = ds.I_control.sel(qubit=qp['qubit'])
        
        values_to_plot.assign_coords({"flux_qubit_mV": 1e3*values_to_plot.flux_qubit_full, "flux_coupler_mV": 1e3*values_to_plot.flux_coupler_full}).plot(ax = ax, cmap = 'viridis', x = 'flux_qubit_mV', y = 'flux_coupler_mV')
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize = 10)
        ax.axhline(1e3*node.results["results"][qp["qubit"]]["flux_coupler_min"], color = 'red', lw = 0.5, ls = '--')
        ax.axvline(1e3*node.results["results"][qp["qubit"]]["flux_qubit_max"], color = 'red', lw =0.5, ls = '--')
        # Create a secondary x-axis for detuning
        flux_qubit_data = ds.sel(qubit=qp['qubit']).flux_qubit_full.values*1e3
        detuning_data = ds.sel(qubit=qp['qubit']).detuning.values * 1e-6

        def flux_to_detuning(x):
            return np.interp(x, flux_qubit_data, detuning_data)

        def detuning_to_flux(y):
            return np.interp(y, detuning_data, flux_qubit_data)

        sec_ax = ax.secondary_xaxis('top', functions=(flux_to_detuning, detuning_to_flux))
        sec_ax.set_xlabel('Detuning [MHz]')
        ax.set_xlabel('Qubit flux pulse [mV]')
        ax.set_ylabel('Coupler flux pulse [mV]')
    grid.fig.suptitle(f'Control: {machine.qubit_pairs[qp["qubit"]].qubit_control.name} \n {node.date_time} GMT+{node.time_zone} #{node_id} \n reset type = {node.parameters.reset_type}')
    plt.tight_layout()
    plt.show()
    node.results['figure_control'] = grid.fig
    
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        if node.parameters.use_state_discrimination:
            values_to_plot = ds.state_target.sel(qubit=qp['qubit'])
        else:
            values_to_plot = ds.I_target.sel(qubit=qp['qubit'])
        
        values_to_plot.assign_coords({"flux_qubit_mV": 1e3*values_to_plot.flux_qubit_full, "flux_coupler_mV": 1e3*values_to_plot.flux_coupler_full}).plot(ax = ax, cmap = 'viridis', x = 'flux_qubit_mV', y = 'flux_coupler_mV')
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize = 10)
        ax.axhline(1e3*node.results["results"][qp["qubit"]]["flux_coupler_min"], color = 'red', lw = 0.5, ls = '--')
        ax.axvline(1e3*node.results["results"][qp["qubit"]]["flux_qubit_max"], color = 'red', lw =0.5, ls = '--')
        # Create a secondary x-axis for detuning
        flux_qubit_data = ds.sel(qubit=qp['qubit']).flux_qubit_full.values*1e3
        detuning_data = ds.sel(qubit=qp['qubit']).detuning.values * 1e-6

        def flux_to_detuning(x):
            return np.interp(x, flux_qubit_data, detuning_data)

        def detuning_to_flux(y):
            return np.interp(y, detuning_data, flux_qubit_data)

        sec_ax = ax.secondary_xaxis('top', functions=(flux_to_detuning, detuning_to_flux))
        sec_ax.set_xlabel('Detuning [MHz]')
        ax.set_xlabel('Qubit flux shift [mV]')
        ax.set_ylabel('Coupler flux [mV]')
    grid.fig.suptitle(f'Target: {machine.qubit_pairs[qp["qubit"]].qubit_target.name} \n {node.date_time} GMT+{node.time_zone} #{node_id}')
    plt.tight_layout()
    plt.show()
    node.results['figure_target'] = grid.fig

# %% {Update_state}
if not node.parameters.simulate:
    with node.record_state_updates():
        for qp in qubit_pairs:
            qp.coupler.decouple_offset = node.results["results"][qp.name]["flux_coupler_min"]
            qp.detuning = node.results["results"][qp.name]["flux_qubit_max"]

# %% {Save_results}
if not node.parameters.simulate:    
    node.outcomes = {q.name: "successful" for q in qubit_pairs}
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    # node.save()
# %%
