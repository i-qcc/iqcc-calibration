"""
        AXIS ANGLE CALIBRATION
This sequence involves playing a precursor Ramsey sequence (x90-y90-x90-y90) and measuring the state of the resonator
for different rotation angles of the XY frame.
The Ramsey sequence is sensitive to the frame of the pulses.
The purpose of this protocol is to find the angle that maximizes the visibility of the sequence, thus calibrating the
axis of the single qubit gates.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated the IQ mixer connected to the qubit drive line (external mixer or Octave port)
    - Having found the rough qubit frequency and calibrated the pi pulse amplitude (qubit spectroscopy and power_rabi).
    - Set the desired flux bias.

Next steps before going to the next node:
    - Update the qubit axis angle in the state.
    - Save the current state.
"""

# %% {Imports}

from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode, NodeParameters
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam as Quam
from iqcc_calibration_tools.quam_config.macros import qua_declaration, active_reset
from iqcc_calibration_tools.analysis.plot_utils import QubitGrid, grid_iter
from iqcc_calibration_tools.storage.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from qm.qua.lib import Math
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    num_averages: int = 200
    angle_max_shift: float = 0.4
    num_angles: int = 10
    amplification_length: int = 10
    operation: Literal["y90_DragCosine"] = "y90_DragCosine"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    state_discrimination: bool = True
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True


node = QualibrationNode(name="10e_axis_angle_optimization", parameters=Parameters())

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = Quam.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
amplification_lengths = np.arange(1, node.parameters.amplification_length, 2) # np.arange(0, node.parameters.amplification_length, 1)
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
state_discrimination = node.parameters.state_discrimination
operation = node.parameters.operation

# angle sweep
angle_max_shift = node.parameters.angle_max_shift
angles = np.linspace(0, angle_max_shift, node.parameters.num_angles)
angles = np.concatenate((-angles[::-1][:-1], angles))


with program() as axis_angle_cal:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    if state_discrimination:
        state = [declare(bool) for _ in range(num_qubits)]
        state_stream = [declare_stream() for _ in range(num_qubits)]
    angle = declare(fixed)  # QUA variable for the MW frame rotation angle
    amplification = declare(int)  # QUA variable for counting the qubit pulses
    count = declare(int)  # QUA variable for counting the qubit pulses

    if flux_point == "joint":
        # Bring the active qubits to the desired frequency point
        machine.initialize_qpu(flux_point=flux_point, target=qubits[0])
    
    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the minimum frequency point
        if flux_point != "joint":
            machine.initialize_qpu(flux_point=flux_point, target=qubit)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(angle, angles)):
                with for_(*from_array(amplification, amplification_lengths)):
                    # Initialize the qubits
                    if reset_type == "active":
                        active_reset(qubit, "readout")
                    else:
                        qubit.wait(qubit.thermalization_time * u.ns)
                    
                    with for_(count, 0, count < amplification, count + 1):
                        # Apply the precursor sequence
                        # play("x180", qubit.xy.name)
                        # play("y90"*amp(Math.cos(angle), Math.sin(angle), -Math.sin(angle), Math.cos(angle)), qubit.xy.name)
                        # play("x180", qubit.xy.name)
                        # play("-y90"*amp(Math.cos(angle- np.pi), Math.sin(angle- np.pi), -Math.sin(angle- np.pi), Math.cos(angle- np.pi)), qubit.xy.name)
                        # play("x180", qubit.xy.name)
                        # play("y90"*amp(Math.cos(angle), Math.sin(angle), -Math.sin(angle), Math.cos(angle)), qubit.xy.name)
                        # play("x180", qubit.xy.name)
                        # play("-y90"*amp(Math.cos(angle- np.pi), Math.sin(angle- np.pi), -Math.sin(angle- np.pi), Math.cos(angle- np.pi)), qubit.xy.name)
                        
                        play("y90"*amp(Math.cos(angle), Math.sin(angle), -Math.sin(angle), Math.cos(angle)), qubit.xy.name)
                        play("x180", qubit.xy.name)
                        play("-y90"*amp(Math.cos(angle), Math.sin(angle), -Math.sin(angle), Math.cos(angle)), qubit.xy.name)
                        play("x180", qubit.xy.name)
                        
                        # play("x180", qubit.xy.name)
                        # play("-y90"*amp(Math.cos(angle- np.pi), Math.sin(angle- np.pi), -Math.sin(angle- np.pi), Math.cos(angle- np.pi)), qubit.xy.name)
                        # play("x180", qubit.xy.name)
                        
                        
                    # Align the elements to measure after playing the qubit pulse.
                    qubit.align()
                    # Measure the state of the resonator
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    if state_discrimination:
                        assign(state[i], I[i] > qubit.resonator.operations["readout"].threshold)
                        save(state[i], state_stream[i])
                    else:
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            if state_discrimination:
                state_stream[i].boolean_to_int().buffer(len(amplification_lengths)).buffer(len(angles)).average().save(f"state{i + 1}")
            else:
                I_st[i].buffer(len(amplification_lengths)).buffer(len(angles)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(amplification_lengths)).buffer(len(angles)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, axis_angle_cal, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i + 1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(axis_angle_cal)
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
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"amplification_lengths": amplification_lengths, "angle": angles})
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    # Average over amplification length to get a 1D array vs angle
    if state_discrimination:
        signal_vs_angle = ds.state.mean(dim="amplification_lengths")
        signal = ds.state
    else:
        # Calculate the IQ magnitude and add it to the dataset
        ds["IQ_abs"] = np.sqrt(ds.I**2 + ds.Q**2)
        
        # Take the mean with respect to IQ_abs over amplification lengths
        signal_vs_angle = ds.IQ_abs.mean(dim="amplification_lengths")
        signal = ds.IQ_abs
        
    # Find the angle that minimizes the signal
    min_angle_idx = signal_vs_angle.argmax(dim="angle") # signal_vs_angle.argmin(dim="angle")

    # Save fitting results
    fit_results = {}
    for q in qubits:
        fit_results[q.name] = {}
        min_angle = ds.angle[min_angle_idx.sel(qubit=q.name)].data
        print(f"The angle that minimizes the signal for {q.name} is {min_angle:.3f} rad.")
        # The axis angle is defined as angle / (2*pi)
        fit_results[q.name]["axis_angle"] = min_angle / (2 * np.pi)

    node.results["fit_results"] = fit_results

    # %% {Plotting}
    # Create a grid of plots for each qubit
    grid = QubitGrid(ds, [q.grid_location for q in qubits], size=4)
    # Plot a 2D map of the signal vs angle and amplification length
    for ax, qubit in grid_iter(grid):
        if state_discrimination:
            ds.loc[qubit].state.plot.pcolormesh(ax=ax, x="angle", y="amplification_lengths", cmap="viridis")
        else:
            (ds.loc[qubit].I * 1e3).plot.pcolormesh(ax=ax, x="angle", y="amplification_lengths", cmap="viridis")

        ax.set_ylabel("amplification length")
        ax.set_xlabel("relative angle [rad]")
        ax.set_title(qubit["qubit"])
        min_angle = fit_results[qubit["qubit"]]["axis_angle"] * 2 * np.pi
        ax.axvline(min_angle, color="r", linestyle="--", label="min angle")

    plt.tight_layout()
    plt.show()
    
    # Plot measurement data vs amp len for optimal angle
    data_at_optimal_angle = signal.isel(angle=min_angle_idx)
    grid_opt_meas = QubitGrid(ds, [q.grid_location for q in qubits], size=4)
    for ax, qubit in grid_iter(grid_opt_meas):
        if state_discrimination:
            data_at_optimal_angle.loc[qubit].plot(ax=ax, x="amplification_lengths")
            ax.set_ylabel("Qubit state")
        else:
            (data_at_optimal_angle.loc[qubit] * 1e3).plot(ax=ax, x="amplification_lengths")
            ax.set_ylabel("I [mV]")

        ax.set_xlabel("Amplification length")
        min_angle_val = ds.angle[min_angle_idx.sel(qubit=qubit["qubit"])].data
        ax.set_title(f"{qubit['qubit']} @ opt relative angle={min_angle_val:.2f} rad")

    grid.fig.suptitle(
        f"Axis angle calibration - 2D map \n {node.date_time} GMT+{node.time_zone} #{node.node_id} \n multiplexed = {node.parameters.multiplexed} reset Type = {node.parameters.reset_type_thermal_or_active}"
    )
    
    plt.tight_layout()
    plt.show()
    
    
    
    grid_1d = QubitGrid(ds, [q.grid_location for q in qubits], size=4)
    for ax, qubit in grid_iter(grid_1d):
        if state_discrimination:
            signal_vs_angle.loc[qubit].plot(ax=ax, x="angle")
            ax.set_ylabel("Qubit state")
        else:
            (signal_vs_angle.loc[qubit] * 1e3).plot(ax=ax, x="angle")
            ax.set_ylabel("I [mV]")

        ax.set_xlabel("relative angle [rad]")
        ax.set_title(qubit["qubit"])
        min_angle = fit_results[qubit["qubit"]]["axis_angle"] * 2 * np.pi
        ax.axvline(min_angle, color="r", linestyle="--", label="min angle")
        ax.legend(loc="upper right", fontsize=6)

    grid_1d.fig.suptitle(
        f"Axis angle calibration - 1D average \n {node.date_time} GMT+{node.time_zone} #{node.node_id} \n multiplexed = {node.parameters.multiplexed} reset Type = {node.parameters.reset_type_thermal_or_active}"
    )
    grid_opt_meas.fig.suptitle(
        f"Measurement vs amp len @ optimal angle \n {node.date_time} GMT+{node.time_zone} #{node.node_id} \n multiplexed = {node.parameters.multiplexed} reset Type = {node.parameters.reset_type_thermal_or_active}"
    )
    plt.tight_layout()
    plt.show()
    node.results["figure_2d"] = grid.fig
    node.results["figure_1d"] = grid_1d.fig
    node.results["figure_opt_meas"] = grid_opt_meas.fig

    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for q in qubits:
                # TODO : understand why the 5 factor works
                q.xy.operations["y90_DragCosine"].axis_angle -= 5 * fit_results[q.name]["axis_angle"]
                q.xy.operations["-y90_DragCosine"].axis_angle = q.xy.operations["y90_DragCosine"].axis_angle - np.pi
                q.xy.operations["y180_DragCosine"].axis_angle = q.xy.operations["y90_DragCosine"].axis_angle
                

        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()
# %%
