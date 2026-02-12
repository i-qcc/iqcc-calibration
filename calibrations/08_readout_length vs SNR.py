# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict
from types import SimpleNamespace
from scipy.optimize import curve_fit
from scipy.special import erf

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam as Quam
from calibration_utils.readout_power_optimization import (
    Parameters as ReadoutPowerParameters,
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


class Parameters(ReadoutPowerParameters):
    """Sweep readout length only; amplitude from state file. Fit SNR = system * sqrt(readout_length)."""
    readout_length_start_ns: int = 600
    readout_length_end_ns: int = 5000
    readout_length_step_ns: int = 200


# %% {Description}
description = """
        READOUT LENGTH vs SNR (fixed amplitude)
Measure |g> and |e> IQ at fixed readout amplitude (from state), sweeping readout length.
SNR is computed per length via fit_snr_with_gaussians. Plot SNR vs readout length and fit
SNR = system * sqrt(readout_length) to extract the system parameter per qubit.

Prerequisites:
    - Readout amplitude already set (e.g. from 08 readout power optimization).
    - Having calibrated the qubit x180 pulse parameters.
"""


node = QualibrationNode[Parameters, Quam](
    name="08_readout_length_vs_SNR",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    node.parameters.readout_length_start_ns = 600
    node.parameters.readout_length_end_ns = 3000
    node.parameters.readout_length_step_ns = 200
    node.parameters.multiplexed = True
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

    n_runs = node.parameters.num_shots
    # Fixed amplitude from state; sweep axes = qubit, n_runs (readout length added in execute)
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_runs": xr.DataArray(np.linspace(1, n_runs, n_runs), attrs={"long_name": "number of shots"}),
    }
    with program() as node.namespace["qua_program"]:
        Ig, Ig_st, Qg, Qg_st, n, n_st = node.machine.declare_qua_variables()
        Ie, Ie_st, Qe, Qe_st, _, _ = node.machine.declare_qua_variables()

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_runs, n + 1):
                save(n, n_st)
                # Qubit initialization
                for i, qubit in multiplexed_qubits.items():
                    qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                align()
                # Readout at fixed amplitude (from state)
                for i, qubit in multiplexed_qubits.items():
                    qubit.resonator.measure("readout", qua_vars=(Ig[i], Qg[i]))
                    qubit.align()
                    save(Ig[i], Ig_st[i])
                    save(Qg[i], Qg_st[i])

                for i, qubit in multiplexed_qubits.items():
                    qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                align()
                for i, qubit in multiplexed_qubits.items():
                    qubit.xy.play("x180")
                    qubit.align()
                    qubit.resonator.measure("readout", qua_vars=(Ie[i], Qe[i]))
                    save(Ie[i], Ie_st[i])
                    save(Qe[i], Qe_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                Ig_st[i].buffer(n_runs).save(f"Ig{i + 1}")
                Qg_st[i].buffer(n_runs).save(f"Qg{i + 1}")
                Ie_st[i].buffer(n_runs).save(f"Ie{i + 1}")
                Qe_st[i].buffer(n_runs).save(f"Qe{i + 1}")


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
    """Run the QUA program for each readout length (fixed amplitude from state), then concat into ds_raw."""
    readout_lengths = np.arange(
        node.parameters.readout_length_start_ns,
        node.parameters.readout_length_end_ns + 1,
        node.parameters.readout_length_step_ns,
    ).tolist()
    qubits = node.namespace["qubits"]
    initial_lengths = [q.resonator.operations["readout"].length for q in qubits]
    datasets = []
    qmm = node.machine.connect()
    for ro_len in readout_lengths:
        for q in qubits:
            q.resonator.operations["readout"].length = int(ro_len)
        config = node.machine.generate_config()
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
            data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
            for dataset in data_fetcher:
                progress_counter(
                    data_fetcher["n"],
                    node.parameters.num_shots,
                    start_time=data_fetcher.t_start,
                )
            node.log(job.execution_report())
        ds_at_len = dataset.expand_dims("readout_length").assign_coords(readout_length=[ro_len])
        datasets.append(ds_at_len)
    for i, q in enumerate(qubits):
        q.resonator.operations["readout"].length = initial_lengths[i]
    node.results["ds_raw"] = xr.concat(datasets, dim="readout_length")
    node.namespace["readout_lengths"] = readout_lengths


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
    """Compute SNR per readout length (fixed amplitude). No amplitude sweep."""
    ds_raw = node.results["ds_raw"]
    qubits = node.namespace["qubits"]
    readout_lengths = list(ds_raw.readout_length.values)
    num_lengths = len(readout_lengths)
    num_qubits = len(qubits)
    snr_array = np.full((num_lengths, num_qubits), np.nan)

    for il, ro_len in enumerate(readout_lengths):
        for q in qubits:
            q.resonator.operations["readout"].length = int(ro_len)
        ds_at_len = ds_raw.sel(readout_length=ro_len).drop_vars("readout_length", errors="ignore")
        ds_iq_amp, fit_res_amp = fit_iq_blobs(ds_at_len, node)
        snr_list, _, _ = fit_snr_with_gaussians(
            ds_iq_amp, qubits, node, fit_res_amp, axes=None, plot=False
        )
        snr_array[il, :] = snr_list

    node.results["snr_array"] = snr_array
    node.results["readout_lengths"] = np.array(readout_lengths)
    # Keep ds_iq_blobs and fit_results from last length for optional plots
    node.results["ds_iq_blobs"] = ds_iq_amp
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_res_amp.items()}

    node.outcomes = {
        q.name: "successful" for q in qubits
    }


def _snr_model(readout_length_ns, system):
    """SNR = system * sqrt(readout_length)."""
    return system * np.sqrt(readout_length_ns)


def _fidelity(readout_length_ns, T1_sec, system):
    """Fidelity = exp(-ro/(2*T1)) * erf(system * sqrt(ro) / sqrt(2)); ro in ns, T1 in s."""
    ro_sec = readout_length_ns * 1e-9
    return np.exp(-ro_sec / (2 * T1_sec)) * erf(system * np.sqrt(readout_length_ns) / np.sqrt(2))


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot SNR vs readout lengthtude) and fit SNR = system * sqrt(readout_length); extract system."""
    qubits = node.namespace["qubits"]
    snr_array = node.results["snr_array"]  # (num_lengths, num_qubits)
    readout_lengths = node.results["readout_lengths"]  # in ns
    num_qubits = len(qubits)
    system_per_qubit = {}
    optimal_readout_length_fidelity = {}

    cols = 2
    rows = (num_qubits + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    for i, q in enumerate(qubits):
        ax = axes[i]
        x = readout_lengths.astype(float)
        y = snr_array[:, i]
        valid = np.isfinite(y) & (x > 0)
        if np.sum(valid) < 2:
            ax.plot(x, y, "o-", label="SNR")
            ax.set_xlabel("Readout length (ns)", fontsize=16)
            ax.set_ylabel("SNR", fontsize=16)
            ax.set_title(f"{q.name}", fontsize=16)
            continue
        x_fit = x[valid]
        y_fit = y[valid]
        try:
            popt, _ = curve_fit(_snr_model, x_fit, y_fit, p0=[np.nanmax(y_fit) / np.sqrt(np.nanmax(x_fit))])
            system = float(popt[0])
            system_per_qubit[q.name] = system
            x_curve = np.linspace(x_fit.min(), x_fit.max(), 200)
            ax.plot(x, y, "o", label="SNR")
            ax.plot(
                x_curve,
                _snr_model(x_curve, system),
                "-",
                color="C1",
                label=rf"fit = {system:.4f} $\times \sqrt{{\tau_{{\mathrm{{RO}}}}}}$",
            )
            ax.set_title(f"{q.name}", fontsize=16)
        except Exception:
            ax.plot(x, y, "o-", label="SNR")
            ax.set_title(f"{q.name}", fontsize=16)
        ax.set_xlabel("Readout length (ns)", fontsize=16)
        ax.set_ylabel("SNR", fontsize=16)
        ax.legend(loc="lower right", fontsize=16)

    for j in range(num_qubits, len(axes)):
        axes[j].axis("off")

    node.results["system_per_qubit"] = system_per_qubit
    fig.suptitle("SNR vs readout length", fontsize=16)
    node.add_node_info_subtitle(fig)
    plt.show()
    node.results["figures"] = {"snr_vs_readout_length": fig}

    # Second plot: Fidelity(T1, Tro) vs readout length; T1 from state, system from fit
    fig2, axes2 = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), constrained_layout=True)
    axes2 = np.atleast_1d(axes2).flatten()
    x_ro = np.linspace(float(readout_lengths.min()), float(readout_lengths.max()), 300)

    for i, q in enumerate(qubits):
        ax2 = axes2[i]
        system = system_per_qubit.get(q.name)
        T1_sec = getattr(q, "T1", None)
        if system is None or T1_sec is None or T1_sec <= 0:
            ax2.set_title(f"{q.name}", fontsize=16)
            ax2.set_xlabel("Readout length (ns)", fontsize=16)
            ax2.set_ylabel("Fidelity", fontsize=16)
            continue
        fidelity_curve = _fidelity(x_ro, T1_sec, system)
        ax2.plot(x_ro, fidelity_curve*100, label="Fidelity")
        idx_opt = np.argmax(fidelity_curve)
        ro_opt = float(x_ro[idx_opt])
        optimal_readout_length_fidelity[q.name] = ro_opt
        ax2.axvline(ro_opt, color="C1", linestyle="--", alpha=0.8, label=rf"max Fidelity at $τ_{{\mathrm{{RO}}}}$={ro_opt:.0f} ns")
        ax2.set_title(f"{q.name}", fontsize=16)
        ax2.set_xlabel("Readout length (ns)", fontsize=16)
        ax2.set_ylabel("Fidelity (%)", fontsize=16)
        ax2.set_ylim(fidelity_curve.min()*100, 100)
        ax2.legend(loc="best", fontsize=16)

    for j in range(num_qubits, len(axes2)):
        axes2[j].axis("off")

    node.results["optimal_readout_length_fidelity"] = optimal_readout_length_fidelity
    fig2.suptitle("Fidelity(T1, τ_RO)", fontsize=16)
    node.add_node_info_subtitle(fig2)
    plt.show()
    node.results["figures"]["fidelity_vs_readout_length"] = fig2


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update state file: set each qubit's readout length to the optimal (max-fidelity) value. Log system and optimal τ_RO."""
    if "system_per_qubit" in node.results:
        for qname, system in node.results["system_per_qubit"].items():
            node.log(f"SNR = system × √(readout_length): {qname} system = {system:.4f}")
    if "optimal_readout_length_fidelity" not in node.results:
        return
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            ro_ns = node.results["optimal_readout_length_fidelity"].get(q.name)
            if ro_ns is None:
                continue
            ro_int = int(round(ro_ns / 4) * 4)  # closest multiple of 4 (hardware sample granularity)
            q.resonator.operations["readout"].length = ro_int
            node.log(f"Optimal readout length (max fidelity): {q.name} τ_RO = {ro_int} ns (state updated)")


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
