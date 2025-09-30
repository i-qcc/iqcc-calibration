"""
        RESONATOR SPECTROSCOPY VERSUS FLUX
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures. This is done across various readout intermediate dfs and flux biases.
The resonator frequency as a function of flux bias is then extracted and fitted so that the parameters can be stored in the state.

This information can then be used to adjust the readout frequency for the maximum and minimum frequency points.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude and duration.
    - Specification of the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Update the relevant flux biases in the state.
    - Save the current state
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from iqcc_calibration_tools.quam_config.components import Quam
from iqcc_calibration_tools.quam_config.macros import qua_declaration
from qualibration_libs.data.processing import convert_IQ_to_V
from iqcc_calibration_tools.analysis.plot_utils import QubitGrid, grid_iter
from iqcc_calibration_tools.storage.save_utils import fetch_results_as_xarray, load_dataset
from qualibration_libs.analysis.fitting import fit_oscillation
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
from iqcc_calibration_tools.quam_config.macros import active_reset, readout_state, readout_state_gef, active_reset_gef, active_reset_simple


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ["coupler_qA1_qA2", "coupler_qA2_qA3"]
    qubits: Optional[List[str]] = None
    num_averages: int = 50
    min_flux_offset_in_v: float = -0.3
    max_flux_offset_in_v: float = 0.3
    num_flux_points: int = 201
    frequency_span_in_mhz: float = 100
    frequency_step_in_mhz: float = 1
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    input_line_impedance_in_ohm: float = 50
    line_attenuation_in_db: float = 0
    update_flux_min: bool = False
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    reset_type: Literal['active', 'thermal'] = "active"

node = QualibrationNode(name="03b_Qubit_Spectroscopy_vs_Coupler_Flux", parameters=Parameters())


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

num_qubit_pairs = len(qubit_pairs)
qubit_pair_names = [qp.name for qp in qubit_pairs]

if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
    
num_qubits = len(qubits)

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
    


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# Flux bias sweep in V
dcs = np.linspace(
    node.parameters.min_flux_offset_in_v,
    node.parameters.max_flux_offset_in_v,
    node.parameters.num_flux_points,
)
# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
update_flux_min = node.parameters.update_flux_min  # Update the min flux point

with program() as multi_res_spec_vs_flux:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    dc = declare(fixed)  # QUA variable for the flux bias
    df = declare(int)  # QUA variable for the readout frequency

    if flux_point == "joint":
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubits[0])
    
    for i, qubit in enumerate(qubits):

        if flux_point != "joint":
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)   
        align()
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            for j, qp in enumerate(qubit_pairs):
                with for_(*from_array(dc, dcs)):
                    with for_(*from_array(df, dfs)):
                        # Qubit initialization
                        # Update the qubit frequency
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                        if node.parameters.reset_type == "active":
                            active_reset_simple(qubit)

                        else:
                            qubit.reset_qubit_thermal()
                        
                                                
                        qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency)
                        # Wait for the qubits to decay to the ground state
                        
                        # Flux sweeping for a qubit
                        duration =  1000
                        align()

                        # Qubit manipulation
                        # Bring the qubit to the desired point during the saturation pulse
                        qp.coupler.play(
                                "const", amplitude_scale=dc / qp.coupler.operations["const"].amplitude, duration=duration
                            )
                        # Apply saturation pulse to all qubits
                        qubit.xy.play(
                            "x180",
                                amplitude_scale=0.1,
                                duration=duration,
                            )
                        align()
                        wait(1000)

                        # Qubit readout
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                
            # Measure sequentially
            align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_st[i].buffer(len(dfs)).buffer(len(dcs)).buffer(num_qubit_pairs).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).buffer(len(dcs)).buffer(num_qubit_pairs).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_res_spec_vs_flux, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(multi_res_spec_vs_flux)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    if node.parameters.load_data_id is not None:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    else:
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": dfs, "flux": dcs, "qp": qubit_pair_names})
        # Convert IQ data into volts
        # ds = convert_IQ_to_V(ds, qubit_pairs)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        RF_freq = np.array([dfs + qubit.xy.RF_frequency for qubit in qubits])
        ds = ds.assign_coords({"freq_full_control": (["qubit", "freq"], RF_freq)})
        ds.freq_full_control.attrs["long_name"] = "Frequency"
        ds.freq_full_control.attrs["units"] = "GHz"
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    
    # Find the frequency for which ds.IQ_abs is minimum using xarray's reduction methods
    min_idx = ds.IQ_abs.argmin(dim="freq")
    min_freqs = ds.freq_full_control.isel(freq=min_idx)

    # Plot the minimum frequencies
    min_freqs.plot(col = "qubit", row = "qp", sharey = False)


    # %% {Plotting}

    for qp in qubit_pairs:
        grid = QubitGrid(ds, [q.grid_location for q in qubits])
        for ax, q in grid_iter(grid):
            ds.assign_coords(freq_GHz=ds.freq_full_control / 1e9).sel(qubit=q['qubit'], qp=qp.name).IQ_abs.plot(
                ax=ax,
                add_colorbar=False,
                x="flux",
                y="freq_GHz",
                robust=True,
            )
            ax.set_title(q["qubit"] + " " + qp.name)
            ax.set_xlabel("Coupler Flux (V)")
    
    plt.tight_layout()
    plt.show()
    # %%
    


    # %% {Update_state}

    # %% {Save_results}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()



# %%
