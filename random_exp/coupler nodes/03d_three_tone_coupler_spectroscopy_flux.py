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
from iqcc_calibration_tools.analysis.plot_utils import QubitPairGrid, grid_iter, grid_pair_names


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = None
    qubits: Optional[List[str]] = None
    num_averages: int = 700
    frequency_span_in_mhz: float = 1000
    frequency_step_in_mhz: float = 2
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    
    flux_pulse_start : float = 0.00
    flux_pulse_end : float = 0.025
    flux_pulse_step: float = 0.0025
    
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    
    reset_type: Literal['active', 'thermal'] = "active"
    RF_frequency_startpoint: Optional[float] = 6.5e9

node = QualibrationNode(name="03d_Three_Tone_Coupler_Spectroscopy", parameters=Parameters())


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
    
num_qubit_pairs = len(qubit_pairs)

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
    
# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)

fluxes = np.arange(node.parameters.flux_pulse_start, node.parameters.flux_pulse_end, node.parameters.flux_pulse_step)

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
coupler_RFs = {qp.name: node.parameters.RF_frequency_startpoint if node.parameters.RF_frequency_startpoint is not None else qp.coupler.RF_frequency for qp in qubit_pairs}
coupler_IFs = {qp.name: coupler_RFs[qp.name] - qp.qubit_control.xy.opx_output.upconverter_frequency for qp in qubit_pairs}

with program() as multi_res_spec_vs_flux:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubit_pairs)
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_stream_target = [declare_stream() for _ in range(num_qubit_pairs)]
    flux_pulse = declare(fixed)  # QUA variable for the flux bias
    df = declare(int)  # QUA variable for the readout frequency

    if flux_point == "joint":
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit_pairs[0].qubit_target)

    align()
    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        for i, qp in enumerate(qubit_pairs):
            with for_(*from_array(flux_pulse, fluxes)):  # type: ignore
                with for_(*from_array(df, dfs)):  # type: ignore
                    # Qubit initialization
                    qubit_control = qp.qubit_control
                    qubit_target = qp.qubit_target

                    # Update the qubit frequency
                    qubit_control.xy.update_frequency(qubit_control.xy.intermediate_frequency)
                    if node.parameters.reset_type == "active":
                        active_reset_simple(qubit_control)
                        active_reset_simple(qubit_target)
                        qp.align()

                    else:
                        qubit_control.reset_qubit_thermal()
                        qubit_target.reset_qubit_thermal()
                        qp.align()
                    
                    #update the frequency of the control qubit
                    qubit_control.xy.update_frequency(df + coupler_IFs[qp.name])

                    # Qubit manipulation
                    # Apply saturation pulse to all qubits
                    qubit_control.xy.play(
                        "saturation",
                            amplitude_scale=1.0,
                            duration=10000
                        )
                    qp.coupler.play("const", amplitude_scale=flux_pulse / qp.coupler.operations["const"].amplitude, duration = 10000)
                    qp.align()
                    # qubit_target.xy.play("saturation",duration=1000)
                    qubit_target.xy.play("x180")

                    qp.align()
                    # Qubit readout
                    readout_state(qp.qubit_target, state_target[i])
                    # save data
                    save(state_target[i], state_stream_target[i])
                
        # Measure sequentially
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            state_stream_target[i].buffer(len(dfs)).buffer(len(fluxes)).buffer(num_qubit_pairs).average().save(f"state{i + 1}")


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
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"freq": dfs,  "flux": fluxes, "qp": qubit_pair_names})
        # Convert IQ data into volts
        # ds = convert_IQ_to_V(ds, qubit_pairs)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        RF_freq = np.array([dfs + coupler_RFs[qp.name] for qp in qubit_pairs])
        ds = ds.assign_coords({"freq_full_control": (["qp", "freq"], RF_freq)})
        ds.freq_full_control.attrs["long_name"] = "Frequency"
        ds.freq_full_control.attrs["units"] = "GHz"
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    
    # Find the frequency for which ds.IQ_abs is minimum using xarray's reduction methods
    min_idx = ds.state.argmin(dim="freq")
    min_freqs = ds.freq_full_control.isel(freq=min_idx)

    ds.assign_coords(freq_GHz=ds.freq_full_control / 1e9).sel(qubit=qp.name, flux = 0.015).state.plot(
            x="freq_GHz"
        )

    # %% {Plotting}
    
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)    
    for ax, qp in grid_iter(grid):
        ds.assign_coords(freq_GHz=ds.freq_full_control / 1e9).sel(qubit=qp['qubit']).state.plot(
            ax=ax,
            y="freq_GHz",
            x="flux"
        )
        # ax.axvline(1e-9*min_freqs.sel(qubit=qp['qubit']), color='red', linestyle='--', alpha=0.5)    
        ax.set_title(qp["qubit"] )
        ax.set_xlabel("Flux (V)")
        ax.set_ylabel("Frequency (GHz)")
    
    plt.tight_layout()
    plt.show()    
    # %%
    


    # %% {Update_state}

    # %% {Save_results}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()



# %%
