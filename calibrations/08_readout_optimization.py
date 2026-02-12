# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict
from types import SimpleNamespace

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam as Quam
from calibration_utils.readout_power_optimization import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.readout_power_optimization.analysis import FitParameters as ReadoutFitParameters
from calibration_utils.iq_blobs import fit_raw_data as fit_iq_blobs
from calibration_utils.iq_blobs.plotting import plot_iq_blobs, plot_confusion_matrices
from calibration_utils.iq_blobs.analysis import fit_snr_with_gaussians
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from iqcc_calibration_tools.quam_config.lib.qua_datasets import opxoutput


# %% {Description}
description = """
        READOUT POWER OPTIMIZATION
The sequence consists in measuring the state of the resonator after thermalization (qubit in |g>) and after
playing a pi pulse to the qubit (qubit in |e>) successively while sweeping the readout amplitude.
The 'I' & 'Q' quadratures when the qubit is in |g> and |e> are extracted. SNR is computed at each amplitude
using fit_snr_with_gaussians (Gaussian fits to ground and excited blobs). The optimal readout amplitude
is chosen to maximize SNR, and a plot of SNR vs readout power is produced.

Prerequisites:
    - Having calibrated the readout parameters (nodes 02a, 02b and/or 02c).
    - Having calibrated the qubit x180 pulse parameters (nodes 03a_qubit_spectroscopy.py and 04b_power_rabi.py).

State update:
    - The readout amplitude: qubit.resonator.operations["readout"].amplitude
    - The integration weight angle: qubit.resonator.operations["readout"].integration_weights_angle
    - the ge discrimination threshold: qubit.resonator.operations["readout"].threshold
    - the Repeat Until Success threshold: qubit.resonator.operations["readout"].rus_exit_threshold
    - The confusion matrix: qubit.resonator.operations["readout"].confusion_matrix
"""


node = QualibrationNode[Parameters, Quam](
    name="08b_readout_power_optimization",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    node.parameters.num_amps = 25
    node.parameters.start_amp=0.2   
    node.parameters.end_amp=1.99
    node.parameters.multiplexed=True
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

    n_runs = node.parameters.num_shots  # Number of runs
    amps = np.linspace(node.parameters.start_amp, node.parameters.end_amp, node.parameters.num_amps)
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_runs": xr.DataArray(np.linspace(1, n_runs, n_runs), attrs={"long_name": "number of shots"}),
        "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "readout amplitude", "units": ""}),
    }
    with program() as node.namespace["qua_program"]:
        Ig, Ig_st, Qg, Qg_st, n, n_st = node.machine.declare_qua_variables()
        Ie, Ie_st, Qe, Qe_st, _, _ = node.machine.declare_qua_variables()
        a = declare(fixed)

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_runs, n + 1):
                # ground iq blobs for all qubits
                save(n, n_st)
                with for_(*from_array(a, amps)):
                    # Qubit initialization
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                    align()
                    # Qubit readout
                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure("readout", qua_vars=(Ig[i], Qg[i]), amplitude_scale=a)
                        qubit.align()
                        # save data
                        save(Ig[i], Ig_st[i])
                        save(Qg[i], Qg_st[i])

                    # Qubit initialization
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                    align()
                    # Qubit readout
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.play("x180")
                        qubit.align()
                        qubit.resonator.measure("readout", qua_vars=(Ie[i], Qe[i]), amplitude_scale=a)
                        save(Ie[i], Ie_st[i])
                        save(Qe[i], Qe_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                Ig_st[i].buffer(len(amps)).buffer(n_runs).save(f"Ig{i + 1}")
                Qg_st[i].buffer(len(amps)).buffer(n_runs).save(f"Qg{i + 1}")
                Ie_st[i].buffer(len(amps)).buffer(n_runs).save(f"Ie{i + 1}")
                Qe_st[i].buffer(len(amps)).buffer(n_runs).save(f"Qe{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
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


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary.
    SNR is computed per readout amplitude via fit_snr_with_gaussians; optimal readout power is chosen to maximize SNR."""
    # Keep raw (counts) snapshot for per-amplitude SNR before process overwrites it
    if np.all([v in node.results["ds_raw"].data_vars for v in ["Ig", "Qg", "Ie", "Qe"]]):
        node.results["ds_raw_counts"] = node.results["ds_raw"].copy()
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], node.results["ds_iq_blobs"], fit_results = fit_raw_data(node.results["ds_raw"], node)

    # Compute SNR per amplitude using fit_snr_with_gaussians and choose optimal by max SNR
    qubits = node.namespace["qubits"]
    ds_fit = node.results["ds_fit"]
    amps = ds_fit.amp_prefactor.values
    num_amps = len(amps)
    num_qubits = len(qubits)
    snr_array = np.full((num_amps, num_qubits), np.nan)
    fit_results_per_amp = []
    ds_iq_per_amp = []

    if "ds_raw_counts" in node.results:
        ds_raw_counts = node.results["ds_raw_counts"]
        for ia, amp in enumerate(amps):
            ds_amp = xr.Dataset(
                {
                    "Ig": ds_raw_counts.Ig.sel(amp_prefactor=amp).drop_vars("amp_prefactor", errors="ignore"),
                    "Qg": ds_raw_counts.Qg.sel(amp_prefactor=amp).drop_vars("amp_prefactor", errors="ignore"),
                    "Ie": ds_raw_counts.Ie.sel(amp_prefactor=amp).drop_vars("amp_prefactor", errors="ignore"),
                    "Qe": ds_raw_counts.Qe.sel(amp_prefactor=amp).drop_vars("amp_prefactor", errors="ignore"),
                }
            )
            ds_iq_amp, fit_res_amp = fit_iq_blobs(ds_amp, node)
            ds_iq_per_amp.append(ds_iq_amp)
            fit_results_per_amp.append(fit_res_amp)
            snr_list, _, _ = fit_snr_with_gaussians(
                ds_iq_amp, qubits, node, fit_res_amp, axes=None, plot=False
            )
            snr_array[ia, :] = snr_list

        # Optimal amplitude per qubit = argmax SNR
        best_amp_idx = np.nanargmax(snr_array, axis=0)
        # Build fit_results and ds_iq_blobs from the best-SNR amplitude per qubit
        fit_results = {}
        for i, q in enumerate(qubits):
            ia = int(best_amp_idx[i])
            params_dict = {**asdict(fit_results_per_amp[ia][q.name])}
            params_dict["optimal_amplitude"] = float(ds_fit.readout_amplitude.isel(qubit=i, amp_prefactor=ia))
            fit_results[q.name] = ReadoutFitParameters(**params_dict)
        # Combined ds_iq_blobs: for each qubit take the blob at its best amplitude
        ds_iq_parts = [
            ds_iq_per_amp[int(best_amp_idx[i])].sel(qubit=qubits[i].name)
            for i in range(num_qubits)
        ]
        node.results["ds_iq_blobs"] = xr.concat(ds_iq_parts, dim="qubit").assign_coords(
            qubit=[q.name for q in qubits]
        )
        node.results["snr_array"] = snr_array
        node.results["amp_prefactor"] = amps
        node.results["readout_amplitude_sweep"] = ds_fit.readout_amplitude
        node.results["best_amp_idx"] = best_amp_idx

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
    """Plot SNR vs readout amplitude first; then IQ blobs and confusion matrix only at the optimal amplitude (max SNR)."""
    qubits = node.namespace["qubits"]
    # Dataset at optimal amplitude only (used for grid + fits): ds_iq_blobs has (qubit, n_runs) at best amplitude per qubit
    ds_optimal = node.results["ds_iq_blobs"]

    figures = {}

    # 1) SNR vs readout amplitude (first)
    if "snr_array" in node.results and "readout_amplitude_sweep" in node.results:
        snr_array = node.results["snr_array"]
        readout_amp = node.results["readout_amplitude_sweep"]  # (qubit, amp_prefactor)
        best_amp_idx = node.results["best_amp_idx"]
        num_qubits = len(qubits)
        cols = 2
        rows = (num_qubits + 1) // cols
        fig_snr, axes_snr = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), constrained_layout=True)
        axes_snr = np.atleast_1d(axes_snr).flatten()
        for i, q in enumerate(qubits):
            ax = axes_snr[i]
            # X-axis in dBm: opxoutput(full_scale_power_dbm of readout port, readout amplitude × amplitude scale)
            fsp_dbm = q.resonator.opx_output.full_scale_power_dbm
            amp_eff = readout_amp.values[i, :]  # effective amplitude (amp_prefactor * base amplitude)
            x = opxoutput(fsp_dbm, amp_eff)
            ax.plot(x, snr_array[:, i], "o-", label="SNR")
            i_opt = int(best_amp_idx[i])
            ax.axvline(x[i_opt], color="k", linestyle="--", label=f"Optimal (SNR={snr_array[i_opt, i]:.2f})")
            ax.set_xlabel("Readout power (dBm)")
            ax.set_ylabel("SNR")
            ax.set_title(f"{q.name}")
            ax.legend()
        for j in range(num_qubits, len(axes_snr)):
            axes_snr[j].axis("off")
        fig_snr.suptitle("SNR vs readout power (dBm)")
        node.add_node_info_subtitle(fig_snr)  # appends node info (date, id, etc.) to suptitle
        plt.show()
        figures["snr_vs_readout_amplitude"] = fig_snr

    # 2) IQ blobs at optimal amplitude only
    fig_iq = plot_iq_blobs(ds_optimal, qubits, ds_optimal)
    fig_iq.suptitle("IQ blobs at optimal readout amplitude (max SNR)")
    node.add_node_info_subtitle(fig_iq)
    plt.show()
    figures["iq_blobs"] = fig_iq

    # 3) Confusion matrix at optimal amplitude only
    fig_confusion = plot_confusion_matrices(ds_optimal, qubits, ds_optimal)
    fig_confusion.suptitle("Confusion matrix at optimal readout amplitude (max SNR)")
    node.add_node_info_subtitle(fig_confusion)
    plt.show()
    figures["confusion_matrix"] = fig_confusion

    # 4) Gaussian distribution fit at optimal readout amplitude (like 07_iq_blobs_SNR)
    num_qubits = len(qubits)
    cols = 2
    rows = (num_qubits + 1) // cols
    fig_gauss, axes_gauss = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), constrained_layout=True)
    axes_gauss = np.atleast_1d(axes_gauss).flatten()
    fit_results_objs = {
        q.name: SimpleNamespace(**node.results["fit_results"][q.name])
        for q in qubits
    }
    fit_snr_with_gaussians(
        fits=ds_optimal,
        qubits=qubits,
        node=node,
        fit_results=fit_results_objs,
        axes=axes_gauss,
        plot=True,
    )
    for j in range(num_qubits, len(axes_gauss)):
        axes_gauss[j].axis("off")
    fig_gauss.suptitle("Gaussian fits at optimal readout amplitude (max SNR)")
    node.add_node_info_subtitle(fig_gauss)
    plt.show()
    figures["snr_gaussians"] = fig_gauss

    node.results["figures"] = figures


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            fit_results = node.results["fit_results"][q.name]
            operation = q.resonator.operations["readout"]
            operation.integration_weights_angle -= float(fit_results["iw_angle"])
            operation.threshold = float(fit_results["ge_threshold"]) * operation.length / 2**12
            operation.rus_exit_threshold = float(fit_results["rus_threshold"]) * operation.length / 2**12
            operation.amplitude = float(fit_results["optimal_amplitude"])
            q.resonator.confusion_matrix = fit_results["confusion_matrix"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
