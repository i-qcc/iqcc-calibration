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
    num_averages: int = 100
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    coupler_flux_min : float = 0.005  # relative to the coupler set point
    coupler_flux_max : float = 0.025 # relative to the coupler set point
    coupler_flux_step : float = 0.0001
    qubit_flux_min : float = 0.015 # relative to the qubit pair detuning
    qubit_flux_max : float = 0.035 # relative to the qubit pair detuning
    qubit_flux_step : float = 0.0001  
    use_state_discrimination: bool = True
    pulse_duration_ns: int = 80   
    pulsed_qubit: Literal['control', 'target'] = "target"
    flux_amp_target: float = 0.0
    coupler_operation: Literal['slepian', 'const'] = "const"

node = QualibrationNode(
    name="64a_coupler_leakage_cal", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

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
pulsed_qubits = [qp.qubit_target if node.parameters.pulsed_qubit == "target" else qp.qubit_control for qp in qubit_pairs]
flux_point = node.parameters.flux_point_joint_or_independent_or_pairwise  # 'independent' or 'joint' or 'pairwise'
# Loop parameters
fluxes_coupler = np.arange(node.parameters.coupler_flux_min, node.parameters.coupler_flux_max+0.0001, node.parameters.coupler_flux_step)
fluxes_qubit = np.arange(node.parameters.qubit_flux_min, node.parameters.qubit_flux_max+0.0001, node.parameters.qubit_flux_step)
fluxes_qp = {}
for qp in qubit_pairs:
    # estimate the flux shift to get the control qubit to the target qubit frequency
    fluxes_qp[qp.name] = fluxes_qubit
    
pulse_duration = node.parameters.pulse_duration_ns // 4
reset_coupler_bias = False

with program() as CPhase_Oscillations:
    n = declare(int)
    flux_coupler = declare(float)
    flux_qubit = declare(float)
    comp_flux_qubit = declare(float)
    n_st = declare_stream()
    qua_pulse_duration = declare(int, value = pulse_duration)
    
    if node.parameters.use_state_discrimination:
        state_control = [declare(int) for _ in range(num_qubit_pairs)]
        state_target = [declare(int) for _ in range(num_qubit_pairs)]
        state_F_control = [declare(fixed) for _ in range(num_qubit_pairs)]
        state_F_target = [declare(fixed) for _ in range(num_qubit_pairs)]
        state_E_control = [declare(fixed) for _ in range(num_qubit_pairs)]
        state_E_target = [declare(fixed) for _ in range(num_qubit_pairs)]
        state_G_control = [declare(fixed) for _ in range(num_qubit_pairs)]
        state_G_target = [declare(fixed) for _ in range(num_qubit_pairs)]
        state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st_F_control = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st_F_target = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st_E_control = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st_E_target = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st_G_control = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st_G_target = [declare_stream() for _ in range(num_qubit_pairs)]
    else:
        I_control = [declare(float) for _ in range(num_qubit_pairs)]
        Q_control = [declare(float) for _ in range(num_qubit_pairs)]
        I_target = [declare(float) for _ in range(num_qubit_pairs)]
        Q_target = [declare(float) for _ in range(num_qubit_pairs)]
        I_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        Q_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        I_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
        Q_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    
    if flux_point == "joint":
        machine.set_all_fluxes(flux_point=flux_point, target=qubit_pairs[0].qubit_control)
    
    for i, qp in enumerate(qubit_pairs):
        # Bring the active qubits to the minimum frequency point
        if flux_point != "joint":
            machine.set_all_fluxes(flux_point=flux_point, target=qp.qubit_control)
        if reset_coupler_bias:
            qp.coupler.set_dc_offset(0.0)
        wait(1000)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)         
            with for_(*from_array(flux_coupler, fluxes_coupler)):
                with for_(*from_array(flux_qubit, fluxes_qp[qp.name])):
                    # reset
                    if node.parameters.reset_type == "active":
                        active_reset_simple(qp.qubit_control)
                        # active_reset_gef(qp.qubit_control)
                        active_reset_simple(qp.qubit_target)
                        qp.align()
                    else:
                        wait(qp.qubit_control.thermalization_time * u.ns)
                        wait(qp.qubit_target.thermalization_time * u.ns)
                    qp.align()
                    
                    if "coupler_qubit_crosstalk" in qp.extras:
                        assign(comp_flux_qubit, flux_qubit  +  qp.extras["coupler_qubit_crosstalk"] * flux_coupler )
                    else:
                        assign(comp_flux_qubit, flux_qubit)                    # setting both qubits ot the initial state
                    qp.qubit_control.xy.play("x180")
                    qp.qubit_target.xy.play("x180")
                    qp.align()
                    coupler_operation = node.parameters.coupler_operation
                    pulsed_qubit = qp.qubit_target if node.parameters.pulsed_qubit == "target" else qp.qubit_control
                    passive_qubit = qp.qubit_control if node.parameters.pulsed_qubit == "target" else qp.qubit_target
                    pulsed_qubit.z.play("const", amplitude_scale = comp_flux_qubit / pulsed_qubit.z.operations["const"].amplitude, duration = qua_pulse_duration)
                    passive_qubit.z.play("const", amplitude_scale = node.parameters.flux_amp_target / passive_qubit.z.operations["const"].amplitude, duration = qua_pulse_duration)
                    qp.coupler.play(coupler_operation, amplitude_scale = flux_coupler / qp.coupler.operations[coupler_operation].amplitude, duration = qua_pulse_duration)
                    qp.align()
                    # readout
                    if node.parameters.use_state_discrimination:
                        readout_state_gef(qp.qubit_control, state_control[i])
                        readout_state_gef(qp.qubit_target, state_target[i])
                        assign(state_F_control[i], Cast.to_fixed( state_control[i] == 2))
                        assign(state_F_target[i], Cast.to_fixed( state_target[i] == 2))
                        assign(state_E_control[i], Cast.to_fixed( state_control[i] == 1))
                        assign(state_E_target[i], Cast.to_fixed( state_target[i] == 1))
                        assign(state_G_control[i], Cast.to_fixed( state_control[i] == 0))
                        assign(state_G_target[i], Cast.to_fixed( state_target[i] == 0))
                        save(state_control[i], state_st_control[i])
                        save(state_target[i], state_st_target[i])
                        save(state_F_control[i], state_st_F_control[i])
                        save(state_F_target[i], state_st_F_target[i])
                        save(state_E_control[i], state_st_E_control[i])
                        save(state_E_target[i], state_st_E_target[i])
                        save(state_G_control[i], state_st_G_control[i])
                        save(state_G_target[i], state_st_G_target[i])
                    else:
                        qp.qubit_control.resonator.measure("readout", qua_vars=(I_control[i], Q_control[i]))
                        qp.qubit_target.resonator.measure("readout", qua_vars=(I_target[i], Q_target[i]))
                        save(I_control[i], I_st_control[i])
                        save(Q_control[i], Q_st_control[i])
                        save(I_target[i], I_st_target[i])
                        save(Q_target[i], Q_st_target[i])
        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            if node.parameters.use_state_discrimination:
                state_st_control[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_control{i + 1}")
                state_st_target[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_target{i + 1}")
                state_st_F_control[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_F_control{i + 1}")
                state_st_F_target[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_F_target{i + 1}")
                state_st_E_control[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_E_control{i + 1}")
                state_st_E_target[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_E_target{i + 1}")
                state_st_G_control[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_G_control{i + 1}")
                state_st_G_target[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_G_target{i + 1}")
            else:
                I_st_control[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"I_control{i + 1}")
                Q_st_control[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"Q_control{i + 1}")
                I_st_target[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"I_target{i + 1}")
                Q_st_target[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"Q_target{i + 1}")

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
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {  "flux_qubit": fluxes_qubit, "flux_coupler": fluxes_coupler})
        flux_qubit_full = np.array([fluxes_qp[qp.name] for qp in qubit_pairs])
        ds = ds.assign_coords({"flux_qubit_full": (["qubit", "flux_qubit"], flux_qubit_full)})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}
# %%
pulsed_qubits = {qp.name: qp.qubit_target if node.parameters.pulsed_qubit == "target" else qp.qubit_control for qp in qubit_pairs}
if not node.parameters.simulate:
    if reset_coupler_bias:
        flux_coupler_full = np.array([fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs])
    else:
        flux_coupler_full = np.array([fluxes_coupler for qp in qubit_pairs])
    detuning = np.array([-fluxes_qp[qp.name] ** 2 * pulsed_qubits[qp.name].freq_vs_flux_01_quad_term  for qp in qubit_pairs])
    ds = ds.assign_coords({"flux_coupler_full": (["qubit", "flux_coupler"], flux_coupler_full)})
    ds = ds.assign_coords({"detuning": (["qubit", "flux_qubit"], detuning)})
    node.results = {"ds": ds}
  
# %%
node.results["results"] = {}


    
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
        quad = qubit_pair.qubit_control.freq_vs_flux_01_quad_term if node.parameters.pulsed_qubit == "control" else qubit_pair.qubit_target.freq_vs_flux_01_quad_term

        def detuning_to_flux(det, quad = quad):
            return 1e3 * np.sqrt(-1e6 * det / quad)

        def flux_to_detuning(flux, quad = quad):
            return -1e-6 * (flux/1e3)**2 * quad

        sec_ax = ax.secondary_xaxis('top', functions=(flux_to_detuning, detuning_to_flux))
        sec_ax.set_xlabel('Detuning [MHz]')
        ax.set_xlabel('Qubit flux shift [V]')
        ax.set_ylabel('Coupler flux [V]')
    grid.fig.suptitle(f'Control: {machine.qubit_pairs[qp["qubit"]].qubit_control.name} \n Pulsed qubit: {node.parameters.pulsed_qubit}')
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
        # Create a secondary x-axis for detuning
        quad = qubit_pair.qubit_control.freq_vs_flux_01_quad_term if node.parameters.pulsed_qubit == "control" else qubit_pair.qubit_target.freq_vs_flux_01_quad_term

        def detuning_to_flux(det, quad = quad):
            return 1e3 * np.sqrt(-1e6 * det / quad)

        def flux_to_detuning(flux, quad = quad):
            return -1e-6 * (flux/1e3)**2 * quad
        
        sec_ax = ax.secondary_xaxis('top', functions=(flux_to_detuning, detuning_to_flux))
        sec_ax.set_xlabel('Detuning [MHz]')
        ax.set_xlabel('Qubit flux shift [V]')
        ax.set_ylabel('Coupler flux [V]')
    grid.fig.suptitle(f'Target: {machine.qubit_pairs[qp["qubit"]].qubit_target.name} \n Pulsed qubit: {node.parameters.pulsed_qubit}')
    plt.tight_layout()
    plt.show()
    node.results['figure_target'] = grid.fig

# %% plotting individual states
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        values_to_plot = ds.state_F_control.sel(qubit=qp['qubit'])
        values_to_plot.assign_coords({"flux_qubit_mV": 1e3*values_to_plot.flux_qubit_full, "flux_coupler_mV": 1e3*values_to_plot.flux_coupler_full}).plot(ax = ax, cmap = 'viridis', x = 'flux_qubit_mV', y = 'flux_coupler_mV')
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize = 10)
        # ax.axhline(1e3*node.results["results"][qp["qubit"]]["flux_coupler_Cz"], color = 'red', lw = 0.5, ls = '--')
        # ax.axvline(1e3*node.results["results"][qp["qubit"]]["flux_qubit_Cz"], color = 'red', lw =0.5, ls = '--')
        quad = qubit_pair.qubit_control.freq_vs_flux_01_quad_term if node.parameters.pulsed_qubit == "control" else qubit_pair.qubit_target.freq_vs_flux_01_quad_term

        def detuning_to_flux(det, quad = quad):
            return 1e3 * np.sqrt(-1e6 * det / quad)

        def flux_to_detuning(flux, quad = quad):
            return -1e-6 * (flux/1e3)**2 * quad

        sec_ax = ax.secondary_xaxis('top', functions=(flux_to_detuning, detuning_to_flux))
        sec_ax.set_xlabel('Detuning [MHz]')
        ax.set_xlabel('Qubit flux shift [V]')
        ax.set_ylabel('Coupler flux [V]')
    grid.fig.suptitle(f'F State Control: {machine.qubit_pairs[qp["qubit"]].qubit_control.name} \n Pulsed qubit: {node.parameters.pulsed_qubit}')
    plt.tight_layout()
    plt.show()
    node.results['figure_F_control'] = grid.fig
    
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        values_to_plot = ds.state_F_target.sel(qubit=qp['qubit'])
        
        values_to_plot.assign_coords({"flux_qubit_mV": 1e3*values_to_plot.flux_qubit_full, "flux_coupler_mV": 1e3*values_to_plot.flux_coupler_full}).plot(ax = ax, cmap = 'viridis', x = 'flux_qubit_mV', y = 'flux_coupler_mV')
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize = 10)
        # ax.axhline(1e3*node.results["results"][qp["qubit"]]["flux_coupler_Cz"], color = 'red', lw = 0.5, ls = '--')
        # ax.axvline(1e3*node.results["results"][qp["qubit"]]["flux_qubit_Cz"], color = 'red', lw =0.5, ls = '--')
        # Create a secondary x-axis for detuning
        quad = qubit_pair.qubit_control.freq_vs_flux_01_quad_term if node.parameters.pulsed_qubit == "control" else qubit_pair.qubit_target.freq_vs_flux_01_quad_term

        def detuning_to_flux(det, quad = quad):
            return 1e3 * np.sqrt(-1e6 * det / quad)

        def flux_to_detuning(flux, quad = quad):
            return -1e-6 * (flux/1e3)**2 * quad
        

        sec_ax = ax.secondary_xaxis('top', functions=(flux_to_detuning, detuning_to_flux))
        sec_ax.set_xlabel('Detuning [MHz]')
        ax.set_xlabel('Qubit flux shift [V]')
        ax.set_ylabel('Coupler flux [V]')
    grid.fig.suptitle(f'F State Target: {machine.qubit_pairs[qp["qubit"]].qubit_target.name} \n Pulsed qubit: {node.parameters.pulsed_qubit}')
    plt.tight_layout()
    plt.show()
    node.results['figure_F_target'] = grid.fig
    
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        values_to_plot = ds.state_E_control.sel(qubit=qp['qubit'])
        values_to_plot.assign_coords({"flux_qubit_mV": 1e3*values_to_plot.flux_qubit_full, "flux_coupler_mV": 1e3*values_to_plot.flux_coupler_full}).plot(ax = ax, cmap = 'viridis', x = 'flux_qubit_mV', y = 'flux_coupler_mV')
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize = 10)
        # ax.axhline(1e3*node.results["results"][qp["qubit"]]["flux_coupler_Cz"], color = 'red', lw = 0.5, ls = '--')
        # ax.axvline(1e3*node.results["results"][qp["qubit"]]["flux_qubit_Cz"], color = 'red', lw =0.5, ls = '--')
        quad = qubit_pair.qubit_control.freq_vs_flux_01_quad_term if node.parameters.pulsed_qubit == "control" else qubit_pair.qubit_target.freq_vs_flux_01_quad_term

        def detuning_to_flux(det, quad = quad):
            return 1e3 * np.sqrt(-1e6 * det / quad)

        def flux_to_detuning(flux, quad = quad):
            return -1e-6 * (flux/1e3)**2 * quad

        sec_ax = ax.secondary_xaxis('top', functions=(flux_to_detuning, detuning_to_flux))
        sec_ax.set_xlabel('Detuning [MHz]')
        ax.set_xlabel('Qubit flux shift [V]')
        ax.set_ylabel('Coupler flux [V]')
    grid.fig.suptitle(f'E State Control: {machine.qubit_pairs[qp["qubit"]].qubit_control.name} \n Pulsed qubit: {node.parameters.pulsed_qubit}')
    plt.tight_layout()
    plt.show()
    node.results['figure_E_control'] = grid.fig
    
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        values_to_plot = ds.state_E_target.sel(qubit=qp['qubit'])
        
        values_to_plot.assign_coords({"flux_qubit_mV": 1e3*values_to_plot.flux_qubit_full, "flux_coupler_mV": 1e3*values_to_plot.flux_coupler_full}).plot(ax = ax, cmap = 'viridis', x = 'flux_qubit_mV', y = 'flux_coupler_mV')
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize = 10)
        # ax.axhline(1e3*node.results["results"][qp["qubit"]]["flux_coupler_Cz"], color = 'red', lw = 0.5, ls = '--')
        # ax.axvline(1e3*node.results["results"][qp["qubit"]]["flux_qubit_Cz"], color = 'red', lw =0.5, ls = '--')
        # Create a secondary x-axis for detuning
        quad = qubit_pair.qubit_control.freq_vs_flux_01_quad_term if node.parameters.pulsed_qubit == "control" else qubit_pair.qubit_target.freq_vs_flux_01_quad_term

        def detuning_to_flux(det, quad = quad):
            return 1e3 * np.sqrt(-1e6 * det / quad)

        def flux_to_detuning(flux, quad = quad):
            return -1e-6 * (flux/1e3)**2 * quad
        

        sec_ax = ax.secondary_xaxis('top', functions=(flux_to_detuning, detuning_to_flux))
        sec_ax.set_xlabel('Detuning [MHz]')
        ax.set_xlabel('Qubit flux shift [V]')
        ax.set_ylabel('Coupler flux [V]')
    grid.fig.suptitle(f'E State Target: {machine.qubit_pairs[qp["qubit"]].qubit_target.name} \n Pulsed qubit: {node.parameters.pulsed_qubit}')
    plt.tight_layout()
    plt.show()
    node.results['figure_E_target'] = grid.fig
    
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        values_to_plot = ds.state_G_control.sel(qubit=qp['qubit'])
        values_to_plot.assign_coords({"flux_qubit_mV": 1e3*values_to_plot.flux_qubit_full, "flux_coupler_mV": 1e3*values_to_plot.flux_coupler_full}).plot(ax = ax, cmap = 'viridis', x = 'flux_qubit_mV', y = 'flux_coupler_mV')
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize = 10)
        # ax.axhline(1e3*node.results["results"][qp["qubit"]]["flux_coupler_Cz"], color = 'red', lw = 0.5, ls = '--')
        # ax.axvline(1e3*node.results["results"][qp["qubit"]]["flux_qubit_Cz"], color = 'red', lw =0.5, ls = '--')
        quad = qubit_pair.qubit_control.freq_vs_flux_01_quad_term if node.parameters.pulsed_qubit == "control" else qubit_pair.qubit_target.freq_vs_flux_01_quad_term

        def detuning_to_flux(det, quad = quad):
            return 1e3 * np.sqrt(-1e6 * det / quad)

        def flux_to_detuning(flux, quad = quad):
            return -1e-6 * (flux/1e3)**2 * quad

        sec_ax = ax.secondary_xaxis('top', functions=(flux_to_detuning, detuning_to_flux))
        sec_ax.set_xlabel('Detuning [MHz]')
        ax.set_xlabel('Qubit flux shift [V]')
        ax.set_ylabel('Coupler flux [V]')
    grid.fig.suptitle(f'G State Control: {machine.qubit_pairs[qp["qubit"]].qubit_control.name} \n Pulsed qubit: {node.parameters.pulsed_qubit}')
    plt.tight_layout()
    plt.show()
    node.results['figure_G_control'] = grid.fig
    
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        values_to_plot = ds.state_G_target.sel(qubit=qp['qubit'])
        
        values_to_plot.assign_coords({"flux_qubit_mV": 1e3*values_to_plot.flux_qubit_full, "flux_coupler_mV": 1e3*values_to_plot.flux_coupler_full}).plot(ax = ax, cmap = 'viridis', x = 'flux_qubit_mV', y = 'flux_coupler_mV')
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, coupler set point: {qubit_pair.coupler.decouple_offset}", fontsize = 10)
        # ax.axhline(1e3*node.results["results"][qp["qubit"]]["flux_coupler_Cz"], color = 'red', lw = 0.5, ls = '--')
        # ax.axvline(1e3*node.results["results"][qp["qubit"]]["flux_qubit_Cz"], color = 'red', lw =0.5, ls = '--')
        # Create a secondary x-axis for detuning
        quad = qubit_pair.qubit_control.freq_vs_flux_01_quad_term if node.parameters.pulsed_qubit == "control" else qubit_pair.qubit_target.freq_vs_flux_01_quad_term

        def detuning_to_flux(det, quad = quad):
            return 1e3 * np.sqrt(-1e6 * det / quad)

        def flux_to_detuning(flux, quad = quad):
            return -1e-6 * (flux/1e3)**2 * quad
        

        sec_ax = ax.secondary_xaxis('top', functions=(flux_to_detuning, detuning_to_flux))
        sec_ax.set_xlabel('Detuning [MHz]')
        ax.set_xlabel('Qubit flux shift [V]')
        ax.set_ylabel('Coupler flux [V]')
    grid.fig.suptitle(f'G State Target: {machine.qubit_pairs[qp["qubit"]].qubit_target.name} \n Pulsed qubit: {node.parameters.pulsed_qubit}')
    plt.tight_layout()
    plt.show()
    node.results['figure_G_target'] = grid.fig

# %% {Update_state}
# if not node.parameters.simulate:
#     if not node.parameters.simulate:
#         with node.record_state_updates():
#             for qp in qubit_pairs:
#                 qp.extras["CZ_coupler_flux"] = node.results["results"][qp.name]["flux_coupler_Cz"]
#                 qp.extras["CZ_qubit_flux"] = node.results["results"][qp.name]["flux_qubit_Cz"]
#                 qp.extras["CZ_time"] = node.parameters.pulse_duration_ns

# %% {Save_results}
if not node.parameters.simulate:    
    node.outcomes = {q.name: "successful" for q in qubit_pairs}
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    # node.save()
# %%
