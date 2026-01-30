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
from qualibration_libs.analysis import fit_oscillation
from iqcc_calibration_tools.analysis.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from iqcc_calibration_tools.quam_config.components.gate_macros import CZMacro
from iqcc_calibration_tools.quam_config.components.pulses import SlepianPulse

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = None
    num_averages: int = 500
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    coupler_flux_min : float = 0.01  # relative to the coupler set point
    coupler_flux_max : float = 0.028 # relative to the coupler set point
    coupler_flux_step : float = 0.0002
    use_state_discrimination: bool = True
    pulse_duration_ns: int = 100
    num_frames : int = 20
    pulsed_qubit: Literal['control', 'target'] = "target"
    flux_amp_target: float = 0.0
    coupler_operation: Literal['slepian', 'const'] = "slepian"    
    

node = QualibrationNode(
    name="66_1D_slepian_coupler_cal", parameters=Parameters()
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

flux_point = node.parameters.flux_point_joint_or_independent_or_pairwise  # 'independent' or 'joint' or 'pairwise'
# Loop parameters
fluxes_coupler = np.arange(node.parameters.coupler_flux_min, node.parameters.coupler_flux_max+0.0001, node.parameters.coupler_flux_step)
fluxes_qp = {}

pulse_duration = node.parameters.pulse_duration_ns // 4
reset_coupler_bias = False
frames = np.arange(0, 1, 1 / node.parameters.num_frames)

with program() as CPhase_Oscillations:
    n = declare(int)
    flux_coupler = declare(float)
    flux_qubit = declare(float)
    comp_flux_qubit = declare(float)
    n_st = declare_stream()
    qua_pulse_duration = declare(int, value = pulse_duration)
    frame = declare(fixed)
    control_initial = declare(int)
    
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
   
    if flux_point == "joint":
        machine.initialize_qpu(flux_point=flux_point, target=qubit_pairs[0].qubit_control)
    
    for i, qp in enumerate(qubit_pairs):
        # Bring the active qubits to the minimum frequency point
        if flux_point != "joint":
            machine.initialize_qpu(flux_point=flux_point, target=qp.qubit_control)
        if reset_coupler_bias:
            qp.coupler.set_dc_offset(0.0)
        wait(1000)

        with for_(n, 0, n < n_avg , n + 1):
            save(n, n_st)         
            with for_(*from_array(flux_coupler, fluxes_coupler)):
                with for_(*from_array(frame, frames)):
                    with for_(*from_array(control_initial, [0, 1])):      
                        # reset
                        if node.parameters.reset_type == "active":
                            active_reset_simple(qp.qubit_control)
                            active_reset_simple(qp.qubit_target)
                        else:
                            wait(qp.qubit_control.thermalization_time * u.ns)
                            wait(qp.qubit_target.thermalization_time * u.ns)
                        align()
                        
                        with if_(control_initial == 1, unsafe = True):
                            qp.qubit_control.xy.play("x180")
                        qp.qubit_target.xy.play("x90")
                        qp.align()
                        qp.coupler.play(node.parameters.coupler_operation, amplitude_scale = flux_coupler / qp.coupler.operations[node.parameters.coupler_operation].amplitude, duration = qua_pulse_duration)
                        qp.align()
                        qp.qubit_control.xy.play("x180", amplitude_scale=0.0, duration=4)
                        qp.qubit_target.xy.play("x180", amplitude_scale=0.0, duration=4)  
                        qp.align()
                        frame_rotation_2pi(frame, qp.qubit_target.xy.name)
                        qp.qubit_target.xy.play("x90")
                        qp.align()
                        # readout
                        readout_state(qp.qubit_control, state_control[i])
                        readout_state(qp.qubit_target, state_target[i])
                        save(state_control[i], state_st_control[i])
                        save(state_target[i], state_st_target[i])

        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            state_st_control[i].buffer(2).buffer(len(frames)).buffer(len(fluxes_coupler)).average().save(f"state_control{i + 1}")
            state_st_target[i].buffer(2).buffer(len(frames)).buffer(len(fluxes_coupler)).average().save(f"state_target{i + 1}")

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
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"control_ax": [0, 1],  "frame": frames, "flux_coupler": fluxes_coupler})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}
# %%
if not node.parameters.simulate:
    if reset_coupler_bias:
        flux_coupler_full = np.array([fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs])
    else:
        flux_coupler_full = np.array([fluxes_coupler for qp in qubit_pairs])
    ds = ds.assign_coords({"flux_coupler_full": (["qubit", "flux_coupler"], flux_coupler_full)})
    node.results = {"ds": ds}

    # %%
    fit_data = fit_oscillation(ds.state_target, "frame") 
    phase_diff = fit_data.sel(fit_vals = "phi").diff(dim="control_ax")
    distance_from_pi_phase = np.abs((np.abs(phase_diff)-np.pi))
    distance_from_pi_phase.argmin(dim = "flux_coupler")
    min_idx = distance_from_pi_phase.argmin(dim="flux_coupler")
    min_flux_coupler = ds.flux_coupler_full.isel(flux_coupler=min_idx)

    
    node.results["results"] = {}
    for q in min_flux_coupler.qubit.values:
        node.results["results"][q] = {}
        node.results["results"][q]["flux_coupler_Cz"] = float(min_flux_coupler.sel(qubit=q).flux_coupler_full.values)



# %% {Plotting}
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        ((((phase_diff + np.pi) % (2*np.pi)) - np.pi)/np.pi*180).assign_coords({"flux_coupler_mV": 1e3*phase_diff.flux_coupler_full}).sel(qubit = qp['qubit']).plot(ax = ax, x = 'flux_coupler_mV')
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, pulse duration: {node.parameters.pulse_duration_ns} ns", fontsize = 10)
        ax.axvline(1e3*node.results["results"][qp["qubit"]]["flux_coupler_Cz"], color = 'k', lw = 0.5, ls = '--')

        ax.set_xlabel('Coupler flux [V]')
        ax.set_ylabel('Conditional phase $\phi$')
    plt.tight_layout()
    plt.show()
    node.results['figure_phase_diff'] = grid.fig

    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        (ds.state_target.sel(control_ax = 0).mean(dim = "frame")).assign_coords({"flux_coupler_mV": 1e3*phase_diff.flux_coupler_full}).sel(qubit = qp['qubit']).plot(ax = ax, x = 'flux_coupler_mV', label = "state_target, control_ax = 0")
        (ds.state_target.sel(control_ax = 1).mean(dim = "frame")).assign_coords({"flux_coupler_mV": 1e3*phase_diff.flux_coupler_full}).sel(qubit = qp['qubit']).plot(ax = ax, x = 'flux_coupler_mV', label = "state_target, control_ax = 1")
        (ds.state_control.sel(control_ax = 0).mean(dim = "frame")).assign_coords({"flux_coupler_mV": 1e3*phase_diff.flux_coupler_full}).sel(qubit = qp['qubit']).plot(ax = ax, x = 'flux_coupler_mV', label = "state_control, control_ax = 0")
        (ds.state_control.sel(control_ax = 1).mean(dim = "frame")).assign_coords({"flux_coupler_mV": 1e3*phase_diff.flux_coupler_full}).sel(qubit = qp['qubit']).plot(ax = ax, x = 'flux_coupler_mV', label = "state_control, control_ax = 1")
        
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, pulse duration: {node.parameters.pulse_duration_ns} ns", fontsize = 10)
        ax.axvline(1e3*node.results["results"][qp["qubit"]]["flux_coupler_Cz"], color = 'k', lw = 0.5, ls = '--')

        ax.set_xlabel('Coupler flux [V]')
        ax.set_ylabel('state')
        ax.grid()
        ax.legend(loc = 3)
    plt.tight_layout()
    plt.show()
    node.results['state_target_and_control'] = grid.fig

# %% {Update_state}
if not node.parameters.simulate:
    if not node.parameters.simulate:
        with node.record_state_updates():
            for qp in qubit_pairs:
                stationary_qubit = qp.qubit_control if node.parameters.pulsed_qubit == "target" else qp.qubit_target
                pulsed_qubit = qp.qubit_control if node.parameters.pulsed_qubit == "control" else qp.qubit_target
                slepian = SlepianPulse(length=node.parameters.pulse_duration_ns, 
                                       amplitude=node.results["results"][qp.name]["flux_coupler_Cz"], 
                                       theta_i=qp.coupler.operations[node.parameters.coupler_operation].theta_i,
                                       theta_f=qp.coupler.operations[node.parameters.coupler_operation].theta_f,
                                       coeffs=qp.coupler.operations[node.parameters.coupler_operation].coeffs,
                                       id = 'coupler_flux_pulse_' + stationary_qubit.name)
                
                CZgate = CZMacro(coupler_flux_pulse = slepian,
                                 pulsed_qubit = node.parameters.pulsed_qubit)
                qp.macros['Cz_unipolar'] = CZgate
                qp.coupler.operations[f"coupler_flux_pulse_{stationary_qubit.name}"] = SlepianPulse(length=f"#/qubit_pairs/{qp.name}/macros/Cz_unipolar/coupler_flux_pulse/length", 
                                                                        amplitude=f"#/qubit_pairs/{qp.name}/macros/Cz_unipolar/coupler_flux_pulse/amplitude", 
                                                                        theta_i=f"#/qubit_pairs/{qp.name}/macros/Cz_unipolar/coupler_flux_pulse/theta_i", 
                                                                        theta_f=f"#/qubit_pairs/{qp.name}/macros/Cz_unipolar/coupler_flux_pulse/theta_f", 
                                                                        coeffs=f"#/qubit_pairs/{qp.name}/macros/Cz_unipolar/coupler_flux_pulse/coeffs", 
                                                                        id = 'coupler_flux_pulse_' + stationary_qubit.name)
                # qp.gates['Cz_unipolar'] = CZgate
                # qp.gates['Cz'] = f"#./Cz_unipolar"


# %% {Save_results}
if not node.parameters.simulate:    
    node.outcomes = {q.name: "successful" for q in qubit_pairs}
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%
