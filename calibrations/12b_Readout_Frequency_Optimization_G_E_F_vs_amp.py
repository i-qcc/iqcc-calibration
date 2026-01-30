# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode
from iqcc_calibration_tools.quam_config.components.quam_root import Quam
from calibration_utils.readout_frequency_optimization_gef_vs_amp import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_distances_with_fit,
    plot_IQ_abs_with_fit,
    plot_optimal_parameters,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from quam.components import pulses



# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_averages: int = 100
    amp_min: float = 0.1
    amp_max: float = 1.5
    amp_step: float = 0.05
    frequency_span_in_mhz: float = 7
    frequency_step_in_mhz: float = 0.2
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    timeout: int = 100


node = QualibrationNode[Parameters, Quam](
    name="12b_Readout_Frequency_Optimization_G_E_F_vs_amp",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["Q3"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots  # The number of averages
    # The frequency sweep around the resonator resonance frequency
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span / 2, +span / 2, step)
    
    # Amplitude sweep for E->F drive
    amps = np.arange(node.parameters.amp_min, node.parameters.amp_max, node.parameters.amp_step)
    
    flux_point = node.parameters.flux_point_joint_or_independent

    # Determine which operation to use for E->F transition
    operation = {}
    for q in qubits:
        if "EF_x180" in q.xy.operations:
            operation[q.name] = "EF_x180"
        else:
            operation[q.name] = "x180"
    node.namespace["operation"] = operation

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "readout frequency detuning", "units": "Hz"}),
        "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "E->F drive amplitude prefactor", "units": ""}),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        I_g = [declare(fixed) for _ in range(num_qubits)]
        Q_g = [declare(fixed) for _ in range(num_qubits)]
        I_e = [declare(fixed) for _ in range(num_qubits)]
        Q_e = [declare(fixed) for _ in range(num_qubits)]
        I_f = [declare(fixed) for _ in range(num_qubits)]
        Q_f = [declare(fixed) for _ in range(num_qubits)]
        df = declare(int)
        amp = declare(float)
        I_g_st = [declare_stream() for _ in range(num_qubits)]
        Q_g_st = [declare_stream() for _ in range(num_qubits)]
        I_e_st = [declare_stream() for _ in range(num_qubits)]
        Q_e_st = [declare_stream() for _ in range(num_qubits)]
        I_f_st = [declare_stream() for _ in range(num_qubits)]
        Q_f_st = [declare_stream() for _ in range(num_qubits)]
        n_st = declare_stream()

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            if flux_point == "independent":
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
            elif flux_point == "joint":
                node.machine.initialize_qpu(target=multiplexed_qubits[0])
            else:
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    with for_(*from_array(amp, amps)):
                        for i, qubit in multiplexed_qubits.items():
                            # Update the resonator frequencies
                            update_frequency(
                                qubit.resonator.name, df + qubit.resonator.intermediate_frequency
                            )
                        align()
                        
                        # Measure |g> state
                        for i, qubit in multiplexed_qubits.items():
                            qubit.resonator.measure("readout", qua_vars=(I_g[i], Q_g[i]))
                            qubit.align()
                            wait(qubit.thermalization_time * u.ns)
                            save(I_g[i], I_g_st[i])
                            save(Q_g[i], Q_g_st[i])
                        align()

                        # Measure |e> state
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.play("x180")
                            qubit.align()
                            qubit.resonator.measure("readout", qua_vars=(I_e[i], Q_e[i]))
                            qubit.align()
                            wait(qubit.thermalization_time * u.ns)
                            save(I_e[i], I_e_st[i])
                            save(Q_e[i], Q_e_st[i])
                        align()

                        # Measure |f> state
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.play("x180")
                            update_frequency(
                                qubit.xy.name, qubit.xy.intermediate_frequency - qubit.anharmonicity
                            )
                            qubit.wait(10)
                            qubit.xy.play(operation[qubit.name], amplitude_scale=amp)
                            update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency)
                            qubit.align()
                            qubit.resonator.measure("readout", qua_vars=(I_f[i], Q_f[i]))
                            wait(qubit.thermalization_time * u.ns)
                            save(I_f[i], I_f_st[i])
                            save(Q_f[i], Q_f_st[i])
                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_g_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"I_g{i + 1}")
                Q_g_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"Q_g{i + 1}")
                I_e_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"I_e{i + 1}")
                Q_e_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"Q_e{i + 1}")
                I_f_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"I_f{i + 1}")
                Q_f_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"Q_f{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, ro_freq_opt, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)
    # Restore operation mapping
    operation = {}
    for q in node.namespace["qubits"]:
        if "EF_x180" in q.xy.operations:
            operation[q.name] = "EF_x180"
        else:
            operation[q.name] = "x180"
    node.namespace["operation"] = operation


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_distances = plot_distances_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    fig_iq_abs = plot_IQ_abs_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    fig_optimal = plot_optimal_parameters(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    node.add_node_info_subtitle(fig_distances)
    node.add_node_info_subtitle(fig_iq_abs)
    node.add_node_info_subtitle(fig_optimal)
    plt.show()
    node.results["figure3"] = grid.fig

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        best_ds = ds.sel(amp = fit_results[qubit["qubit"]]["GEF_amp"])
        (1e3 * best_ds.assign_coords(freq_MHz=best_ds.freq / 1e6).Dge.loc[qubit]).plot(
            ax=ax, x="freq_MHz", label="GE"
        )
        (1e3 * best_ds.assign_coords(freq_MHz=ds.freq / 1e6).Def.loc[qubit]).plot(
            ax=ax, x="freq_MHz", label="EF"
        )
        (1e3 * best_ds.assign_coords(freq_MHz=ds.freq / 1e6).Dgf.loc[qubit]).plot(
            ax=ax, x="freq_MHz", label="GF"
        )
        (1e3 * best_ds.assign_coords(freq_MHz=ds.freq / 1e6).D.loc[qubit]).plot(
            ax=ax, x="freq_MHz"
        )
        # ax.axvline(
        #     fit_results[qubit["qubit"]]["GEF_detuning"] / 1e6,
        #     color="red",
        #     linestyle="--",
        # )
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Distance between IQ blobs [m.v.]")
        ax.legend()
    grid.fig.suptitle(f"{node.date_time} #{node.node_id}")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        best_ds = ds.sel(amp = fit_results[qubit["qubit"]]["GEF_amp"])
        (1e3 * best_ds.assign_coords(freq_MHz=best_ds.freq / 1e6).IQ_abs_g.loc[qubit]).plot(
            ax=ax, x="freq_MHz", label="g.s."
        )
        (1e3 * best_ds.assign_coords(freq_MHz=best_ds.freq / 1e6).IQ_abs_e.loc[qubit]).plot(
            ax=ax, x="freq_MHz", label="e.s."
        )
        (1e3 * best_ds.assign_coords(freq_MHz=best_ds.freq / 1e6).IQ_abs_f.loc[qubit]).plot(
            ax=ax, x="freq_MHz", label="f.s."
        )
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Resonator response [mV]")
        ax.legend()
    grid.fig.suptitle(f"{node.date_time} #{node.node_id}")
    plt.tight_layout()
    plt.show()
    node.results["figure2"] = grid.fig

# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            fit_result = node.results["fit_results"][q.name]
            operation = node.namespace["operation"][q.name]
            
            # Update the GEF frequency shift
            q.resonator.GEF_frequency_shift = int(fit_result["GEF_detuning"])
            
            # Update or create the EF_x180 operation
            EF_x180_amp = q.xy.operations[operation].amplitude * fit_result["GEF_amp"]
            # Update the existing EF_x180 amplitude
            q.xy.operations["EF_x180"].amplitude = EF_x180_amp


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
