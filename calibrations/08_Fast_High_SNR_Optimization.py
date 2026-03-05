"""
kim
       
"""


# %% {Imports}

from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode, NodeParameters
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam as Quam
from iqcc_calibration_tools.quam_config.macros import qua_declaration, active_reset
from iqcc_calibration_tools.analysis.plot_utils import QubitGrid, grid_iter
from iqcc_calibration_tools.storage.save_utils import fetch_results_as_xarray, load_dataset
from iqcc_calibration_tools.quam_config.lib.qua_datasets import opxoutput
from calibration_utils.iq_blobs import process_raw_dataset, fit_raw_data, fit_snr_with_gaussians
from qualang_tools.analysis import two_state_discriminator
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List, Dict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_runs: int = 1100
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    start_amp: float = 0.5
    end_amp: float = 1.9
    num_amps: int = 15
    max_readout_length: int = 2000  # in ns
    min_readout_length: int = 400  # in ns; readout lengths start from this
    duration_chunks: int = 80  # in ns
    target_snr_per_qubit: Optional[Dict[str, float]] = {"qB1": 3, 
                                                        "qB2": 3, 
                                                        "qB3": 3, 
                                                        "qB4": 3, 
                                                        "qB5" :3}   # Per-qubit target SNR, e.g. {"qB1": 3.0, "qB2": 2.5}
    plot_raw: bool = False
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True


node = QualibrationNode(name="08_Fast_High_SNR_Readout_Optimization", parameters=Parameters())

# check that duration_chunks is multiple of 4
if node.parameters.duration_chunks % 4 != 0:
    raise ValueError("duration_chunks should be a multiple of 4 so that it represents an integer number of clock cycles")
if node.parameters.min_readout_length % node.parameters.duration_chunks != 0:
    raise ValueError("min_readout_length must be a multiple of duration_chunks")

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = Quam.load()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


initial_readout_lengths = [q.resonator.operations.readout.length for q in qubits]
# set to max readout length
for q in qubits:
    q.resonator.operations.readout.length = node.parameters.max_readout_length
# Generate the OPX and Octave configurations

config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()


# %% {QUA_program}
n_runs = node.parameters.num_runs  # Number of runs
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
amps = np.linspace(node.parameters.start_amp, node.parameters.end_amp, node.parameters.num_amps)


n_of_chunks = node.parameters.max_readout_length // node.parameters.duration_chunks
start_chunk = node.parameters.min_readout_length // node.parameters.duration_chunks
n_chunks_to_save = n_of_chunks - start_chunk + 1
readout_lengths = [k * node.parameters.duration_chunks for k in range(start_chunk, n_of_chunks + 1)]

with program() as iq_blobs:
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)
    
    I_g = [declare(fixed, size=n_of_chunks) for _ in range(num_qubits)]
    Q_g = [declare(fixed, size=n_of_chunks) for _ in range(num_qubits)]
    I_e = [declare(fixed, size=n_of_chunks) for _ in range(num_qubits)]
    Q_e = [declare(fixed, size=n_of_chunks) for _ in range(num_qubits)]
    
    I_g_st = [declare_stream() for _ in range(num_qubits)]
    Q_g_st = [declare_stream() for _ in range(num_qubits)]
    I_e_st = [declare_stream() for _ in range(num_qubits)]
    Q_e_st = [declare_stream() for _ in range(num_qubits)]
    
    a = declare(fixed)
    
    if flux_point == "joint":
        # Bring the active qubits to the desired frequency point
        machine.initialize_qpu(flux_point=flux_point, target=qubits[0])

    for i, qubit in enumerate(qubits):

        # Bring the active qubits to the desired frequency point
        if flux_point != "joint":
            machine.initialize_qpu(flux_point=flux_point, target=qubit)
         

        with for_(n, 0, n < n_runs, n + 1):
            # ground iq blobs for all qubits
            save(n, n_st)
            with for_(*from_array(a, amps)):
                if reset_type == "active":
                    active_reset(qubit, "readout")
                elif reset_type == "thermal":
                    wait(qubit.thermalization_time * u.ns)
                else:
                    raise ValueError(f"Unrecognized reset type {reset_type}.")

                qubit.align()
                # qubit.resonator.measure("readout", qua_vars=(I_g[i], Q_g[i]), amplitude_scale=a)
                
                integration_weight_labels = list(qubit.resonator.operations.readout.integration_weights_mapping)
                
                measure("readout" * amp(a), qubit.resonator.name, None, 
                        demod.accumulated(integration_weight_labels[0],I_g[i],node.parameters.duration_chunks//4, "out1"),
                        demod.accumulated(integration_weight_labels[1],Q_g[i],node.parameters.duration_chunks//4, "out2"),
                        )
                qubit.align()
                # save data (chunks from min_readout_length to max_readout_length)
                for k in range(start_chunk - 1, n_of_chunks):
                    save(I_g[i][k], I_g_st[i])
                    save(Q_g[i][k], Q_g_st[i])

                if reset_type == "active":
                    active_reset(qubit, "readout")
                elif reset_type == "thermal":
                    wait(qubit.thermalization_time * u.ns)
                else:
                    raise ValueError(f"Unrecognized reset type {reset_type}.")
                qubit.align()
                qubit.xy.play("x180")
                qubit.align()
                # qubit.resonator.measure("readout", qua_vars=(I_e[i], Q_e[i]), amplitude_scale=a)
                measure("readout" * amp(a), qubit.resonator.name, None, 
                        demod.accumulated(integration_weight_labels[0],I_e[i],node.parameters.duration_chunks//4, "out1"),
                        demod.accumulated(integration_weight_labels[1],Q_e[i],node.parameters.duration_chunks//4, "out2"),
                        )
                for k in range(start_chunk - 1, n_of_chunks):
                    save(I_e[i][k], I_e_st[i])
                    save(Q_e[i][k], Q_e_st[i])

        # Measure sequentially
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_g_st[i].buffer(n_chunks_to_save).buffer(len(amps)).buffer(n_runs).save(f"I_g{i + 1}")
            Q_g_st[i].buffer(n_chunks_to_save).buffer(len(amps)).buffer(n_runs).save(f"Q_g{i + 1}")
            I_e_st[i].buffer(n_chunks_to_save).buffer(len(amps)).buffer(n_runs).save(f"I_e{i + 1}")
            Q_e_st[i].buffer(n_chunks_to_save).buffer(len(amps)).buffer(n_runs).save(f"Q_e{i + 1}")


if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, iq_blobs, simulation_config)
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
        job = qm.execute(iq_blobs)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_runs, start_time=results.start_time)

# set the readout length to the initial readout length
for i, q in enumerate(qubits):
    q.resonator.operations["readout"].length = initial_readout_lengths[i]

config = machine.generate_config() # TODO: this is not needed ?

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"readout_length": readout_lengths, "amplitude": amps, "N": np.linspace(1, n_runs, n_runs)})
        # Add readout_amp (V) and readout_power_dbm to the dataset
        # amp_eff = amps * base_amplitude (from state); opxoutput(fsp_dbm, amp_eff) gives power in dBm
        base_amps = np.array([amps * q.resonator.operations["readout"].amplitude for q in qubits])
        ds = ds.assign_coords({
            "readout_amp": (["qubit", "amplitude"], base_amps),
            "readout_power_dbm": (["qubit", "amplitude"], np.array([
                opxoutput(q.resonator.opx_output.full_scale_power_dbm, base_amps[i])
                for i, q in enumerate(qubits)
            ])),
        })
        # Rearrange the data to combine I_g and I_e into I, and Q_g and Q_e into Q
        ds_rearranged = xr.Dataset()
        # Combine I_g and I_e into I
        ds_rearranged["I"] = xr.concat([ds.I_g, ds.I_e], dim="state")
        ds_rearranged["I"] = ds_rearranged["I"].assign_coords(state=[0, 1])
        # Combine Q_g and Q_e into Q
        ds_rearranged["Q"] = xr.concat([ds.Q_g, ds.Q_e], dim="state")
        ds_rearranged["Q"] = ds_rearranged["Q"].assign_coords(state=[0, 1])
        # Copy other coordinates and data variables
        for var in ds.coords:
            if var not in ds_rearranged.coords:
                ds_rearranged[var] = ds[var]

        for var in ds.data_vars:
            if var not in ["I_g", "I_e", "Q_g", "Q_e"]:
                ds_rearranged[var] = ds[var]

        # Replace the original dataset with the rearranged one
        ds = ds_rearranged
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]

    # Ensure readout_power_dbm exists (for loaded data that may lack it)
    if "readout_power_dbm" not in ds.coords:
        _amps = ds.amplitude.values if "amplitude" in ds.coords else amps
        _base_amps = np.array([
            _amps * machine.qubits[qname].resonator.operations["readout"].amplitude
            for qname in ds.qubit.values
        ])
        ds = ds.assign_coords({
            "readout_power_dbm": (["qubit", "amplitude"], np.array([
                opxoutput(machine.qubits[qname].resonator.opx_output.full_scale_power_dbm, _base_amps[i])
                for i, qname in enumerate(ds.qubit.values)
            ])),
        })

    node.results = {"ds": ds, "results": {}, "figs": {}}
    node.namespace["qubits"] = qubits

    if node.parameters.plot_raw:
        fig, axes = plt.subplots(
            ncols=num_qubits,
            nrows=len(ds.amplitude),
            sharex=False,
            sharey=False,
            squeeze=False,
            figsize=(5 * num_qubits, 5 * len(ds.amplitude)),
        )
        for amplitude, ax1 in zip(ds.amplitude, axes):
            for q, ax2 in zip(list(qubits), ax1):
                ds_q = ds.sel(qubit=q.name, amplitude=amplitude)
                ax2.plot(ds_q.I.sel(state=0), ds_q.Q.sel(state=0), ".", alpha=0.2, label="Ground", markersize=2)
                ax2.plot(ds_q.I.sel(state=1), ds_q.Q.sel(state=1), ".", alpha=0.2, label="Excited", markersize=2)
                ax2.set_xlabel("I")
                ax2.set_ylabel("Q")
                ax2.set_title(f"{q.name}, {float(amplitude)}")
                ax2.axis("equal")
        plt.show()
        node.results["figure_raw_data"] = fig


    # %% {Data_analysis}
    # Compute SNR using fit_snr_with_gaussians (double Gaussian like 07_iq_blobs) per (readout_length, amplitude)
    snr_array = np.full((len(readout_lengths), len(amps), num_qubits), np.nan)
    for il, ro_len in enumerate(readout_lengths):
        for ia, amp in enumerate(amps):
            for iq, q in enumerate(qubits):
                q.resonator.operations["readout"].length = int(ro_len)
            ds_slice = ds.sel(readout_length=ro_len, amplitude=amp)
            ig = ds_slice.I.sel(state=0).drop_vars(["state", "readout_length", "amplitude"], errors="ignore").rename(N="n_runs")
            qg = ds_slice.Q.sel(state=0).drop_vars(["state", "readout_length", "amplitude"], errors="ignore").rename(N="n_runs")
            ie = ds_slice.I.sel(state=1).drop_vars(["state", "readout_length", "amplitude"], errors="ignore").rename(N="n_runs")
            qe = ds_slice.Q.sel(state=1).drop_vars(["state", "readout_length", "amplitude"], errors="ignore").rename(N="n_runs")
            ds_for_iq = xr.Dataset({"Ig": ig, "Qg": qg, "Ie": ie, "Qe": qe})
            ds_for_iq = process_raw_dataset(ds_for_iq, node)
            ds_fit, fit_res = fit_raw_data(ds_for_iq, node)
            snr_list, _, _ = fit_snr_with_gaussians(ds_fit, qubits, node, fit_res, axes=None, plot=False)
            snr_array[il, ia, :] = snr_list
    # Restore readout lengths
    for i, q in enumerate(qubits):
        q.resonator.operations["readout"].length = initial_readout_lengths[i]
    snr_data = xr.DataArray(
        snr_array,
        dims=["readout_length", "amplitude", "qubit"],
        coords={"readout_length": readout_lengths, "amplitude": amps, "qubit": [q.name for q in qubits]},
    ).transpose("qubit", "readout_length", "amplitude")
    snr_data = snr_data.assign_coords(
        readout_power_dbm=ds.readout_power_dbm,
        readout_power_m90=ds.readout_power_dbm - 90,  # offset for x-axis display
        readout_amp=ds.readout_amp,
    )

    plot_individual = False
    best_data = {}
    best_amp = {}
    best_readout_length = {}
    best_readout_power_dbm = {}
    best_snr = {}  # SNR at the optimum readout point
    target_snr_per_q = {}  # Per-qubit target SNR for plotting

    def get_target_snr(qubit_name: str) -> float:
        t = node.parameters.target_snr_per_qubit
        if not t or qubit_name not in t:
            raise ValueError(f"target_snr_per_qubit must include '{qubit_name}'. Set target_snr_per_qubit={{{qubit_name!r}: <value>, ...}}")
        return t[qubit_name]
    for q in qubits:
        tgt = get_target_snr(q.name)
        target_snr_per_q[q.name] = tgt
        snr_q = snr_data.sel(qubit=q.name)
        max_snr = float(np.nanmax(snr_q))
        # Valid = points where |SNR - target| <= 0.1 (SNR within 0.1 of target)
        valid_region = snr_q.where(np.abs(snr_q - tgt) <= 0.1, drop=True)

        if valid_region.size > 0:
            # Among valid points (|SNR - target| <= 0.1), pick shortest readout length; among ties, pick highest SNR
            min_readout_length = float(valid_region.readout_length.min())
            candidates = valid_region.where(
                valid_region.readout_length == min_readout_length, drop=True
            )
            best_loc = candidates.where(candidates == np.nanmax(candidates), drop=True)
            best_snr[q.name] = float(np.array(best_loc).flat[0])
            if best_loc.size > 1:
                best_amp[q.name] = float(best_loc.readout_amp.values.flat[0])
                best_readout_length[q.name] = float(best_loc.readout_length.values.flat[0])
                selected_amplitude = best_loc.amplitude.values.flat[0]
                selected_readout_length = best_loc.readout_length.values.flat[0]
            else:
                best_amp[q.name] = float(best_loc.readout_amp)
                best_readout_length[q.name] = float(best_loc.readout_length)
                selected_amplitude = float(best_loc.amplitude)
                selected_readout_length = float(best_loc.readout_length)
        else:
            # No point has |SNR - target| <= 0.1: pick the one closest to target, then shortest length
            dist_to_target = np.abs(snr_q.values - tgt)
            min_dist = np.nanmin(dist_to_target)
            closest = snr_q.where(np.abs(snr_q - tgt) <= min_dist + 1e-9, drop=True)
            min_len = float(closest.readout_length.min())
            best_loc = closest.where(closest.readout_length == min_len, drop=True)
            if best_loc.size == 0:
                max_loc = snr_q.where(snr_q == snr_q.max(), drop=True)
            else:
                max_loc = best_loc
            best_snr[q.name] = float(np.array(max_loc).flat[0])
            if max_loc.size > 1:
                best_amp[q.name] = float(max_loc.readout_amp.values.flat[0])
                best_readout_length[q.name] = float(max_loc.readout_length.values.flat[0])
                selected_amplitude = max_loc.amplitude.values.flat[0]
                selected_readout_length = max_loc.readout_length.values.flat[0]
            else:
                best_amp[q.name] = float(max_loc.readout_amp)
                best_readout_length[q.name] = float(max_loc.readout_length)
                selected_amplitude = float(max_loc.amplitude)
                selected_readout_length = float(max_loc.readout_length)

        print(f"{q.name}: max_SNR={max_snr:.3f}, target_SNR={tgt}")
        print(f"  optimal_amp={best_amp[q.name]:.4f}, optimal_readout_length={best_readout_length[q.name]} ns")

        node.results["results"][q.name] = {}
        node.results["results"][q.name]["best_amp"] = best_amp[q.name]
        node.results["results"][q.name]["best_readout_length"] = best_readout_length[q.name]
        best_readout_power_dbm[q.name] = float(
            opxoutput(q.resonator.opx_output.full_scale_power_dbm, best_amp[q.name])
        )
        node.results["results"][q.name]["best_readout_power_dbm"] = best_readout_power_dbm[q.name]
        node.results["results"][q.name]["max_snr"] = max_snr
        node.results["results"][q.name]["target_snr"] = tgt
        node.results["results"][q.name]["best_snr"] = best_snr[q.name]

        best_amp_data = ds.sel(
            qubit=q.name,
            amplitude=selected_amplitude,
            readout_length=selected_readout_length,
        ).squeeze()
        best_data[q.name] = best_amp_data

        I_g = best_amp_data.I.sel(state=0)
        Q_g = best_amp_data.Q.sel(state=0)
        I_e = best_amp_data.I.sel(state=1)
        Q_e = best_amp_data.Q.sel(state=1)
        angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(
            I_g, Q_g, I_e, Q_e, True, b_plot=plot_individual
        )
        I_rot = I_g * np.cos(angle) - Q_g * np.sin(angle)
        hist = np.histogram(I_rot, bins=100)
        RUS_threshold = hist[1][1:][np.argmax(hist[0])]
        if plot_individual:
            fig = plt.gcf()
            plt.show()
            node.results["figs"][q.name] = fig
        node.results["results"][q.name]["angle"] = float(angle)
        node.results["results"][q.name]["threshold"] = float(threshold)
        node.results["results"][q.name]["fidelity"] = float(fidelity)
        node.results["results"][q.name]["confusion_matrix"] = np.array([[gg, ge], [eg, ee]])
        node.results["results"][q.name]["rus_threshold"] = float(RUS_threshold)


    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        qn = qubit["qubit"]
        tgt = target_snr_per_q[qn]
        snr_q = snr_data.sel(qubit=qn)
        snr_q.plot(ax=ax, x="readout_power_m90", y="readout_length",
                   robust=True, add_colorbar=True, norm=mcolors.PowerNorm(gamma=2))
        # Red dot = shortest Tro among points where |target_snr - snr| <= 0.1
        ax.scatter(best_readout_power_dbm[qn] - 90, best_readout_length[qn], c="red", s=17, zorder=30,
                   label=f"Optimal (shortest Tro where |SNR-{tgt}|≤0.1)")
        ax.set_xlabel("Readout power (dBm)")
        ax.set_ylabel("Readout length (ns)")
        max_snr = node.results["results"][qn]["max_snr"]
        ax.set_title(f"{qn} \nSNR: {best_snr[qn]:.2f}, Tro: {best_readout_length[qn]:.0f} ns")
    grid.fig.suptitle(f"Fast High SNR Readout\n shortest Tro where SNR = target (per qubit)\n{node.date_time} GMT+{node.time_zone} #{node.node_id}\nmultiplexed={node.parameters.multiplexed}, reset={node.parameters.reset_type_thermal_or_active}")

    plt.tight_layout()
    plt.show()
    node.results["figure_snr"] = grid.fig

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ds_q = best_data[qubit["qubit"]]
        qn = qubit["qubit"]
        ax.plot(
            1e3
            * (
                ds_q.I.sel(state=0) * np.cos(node.results["results"][qn]["angle"])
                - ds_q.Q.sel(state=0) * np.sin(node.results["results"][qn]["angle"])
            ),
            1e3
            * (
                ds_q.I.sel(state=0) * np.sin(node.results["results"][qn]["angle"])
                + ds_q.Q.sel(state=0) * np.cos(node.results["results"][qn]["angle"])
            ),
            ".",
            alpha=0.1,
            label="Ground",
            markersize=1,
        )
        ax.plot(
            1e3
            * (
                ds_q.I.sel(state=1) * np.cos(node.results["results"][qn]["angle"])
                - ds_q.Q.sel(state=1) * np.sin(node.results["results"][qn]["angle"])
            ),
            1e3
            * (
                ds_q.I.sel(state=1) * np.sin(node.results["results"][qn]["angle"])
                + ds_q.Q.sel(state=1) * np.cos(node.results["results"][qn]["angle"])
            ),
            ".",
            alpha=0.1,
            label="Excited",
            markersize=1,
        )
        ax.axvline(1e3 * node.results["results"][qn]["threshold"], color="r", linestyle="--", lw=0.5, label="Threshold")
        ax.axis("equal")
        ax.set_xlabel("I [mV]")
        ax.set_ylabel("Q [mV]")
        fid = node.results["results"][qn]["fidelity"]
        ax.set_title(f"{qubit['qubit']}\nF={fid:.1f}%")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    grid.fig.suptitle(f"g.s. and e.s. discriminators (rotated) at optimum readout\n{node.date_time} GMT+{node.time_zone} #{node.node_id}\nmultiplexed={node.parameters.multiplexed}, reset={node.parameters.reset_type_thermal_or_active}")
    plt.tight_layout()
    node.results["figure_IQ_blobs"] = grid.fig

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        qn = qubit["qubit"]
        confusion = node.results["results"][qn]["confusion_matrix"]
        ax.imshow(confusion)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels=["|g>", "|e>"])
        ax.set_yticklabels(labels=["|g>", "|e>"])
        ax.set_ylabel("Prepared")
        ax.set_xlabel("Measured")
        ax.text(0, 0, f"{100 * confusion[0][0]:.1f}%", ha="center", va="center", color="k")
        ax.text(1, 0, f"{100 * confusion[0][1]:.1f}%", ha="center", va="center", color="w")
        ax.text(0, 1, f"{100 * confusion[1][0]:.1f}%", ha="center", va="center", color="w")
        ax.text(1, 1, f"{100 * confusion[1][1]:.1f}%", ha="center", va="center", color="k")
        ax.set_title(f"{qn}\nPro={best_readout_power_dbm[qn]-100:.0f} dBm, Tro={best_readout_length[qn]:.0f} ns")
    grid.fig.suptitle(f"Confusion matrix at optimum readout\n{node.date_time} GMT+{node.time_zone} #{node.node_id}\nmultiplexed={node.parameters.multiplexed}, reset={node.parameters.reset_type_thermal_or_active}")
    plt.tight_layout()
    plt.show()
    node.results["figure_fidelities"] = grid.fig

    # Double Gaussian fit at optimized readout pulse
    ds_opt_parts = []
    for q in qubits:
        qn = q.name
        bd = best_data[qn]
        ig = bd.I.sel(state=0).drop_vars(["state", "readout_length", "amplitude"], errors="ignore")
        qg = bd.Q.sel(state=0).drop_vars(["state", "readout_length", "amplitude"], errors="ignore")
        ie = bd.I.sel(state=1).drop_vars(["state", "readout_length", "amplitude"], errors="ignore")
        qe = bd.Q.sel(state=1).drop_vars(["state", "readout_length", "amplitude"], errors="ignore")
        ig = ig.rename(N="n_runs") if "N" in ig.dims else ig
        qg = qg.rename(N="n_runs") if "N" in qg.dims else qg
        ie = ie.rename(N="n_runs") if "N" in ie.dims else ie
        qe = qe.rename(N="n_runs") if "N" in qe.dims else qe
        ds_opt_q = xr.Dataset({"Ig": ig.expand_dims(qubit=[qn]), "Qg": qg.expand_dims(qubit=[qn]),
                               "Ie": ie.expand_dims(qubit=[qn]), "Qe": qe.expand_dims(qubit=[qn])})
        ds_opt_parts.append(ds_opt_q)
    ds_opt_combined = xr.concat(ds_opt_parts, dim="qubit")
    ds_opt_combined = process_raw_dataset(ds_opt_combined, node)
    ds_opt_combined, fit_results_opt = fit_raw_data(ds_opt_combined, node)
    for q in qubits:
        q.resonator.operations["readout"].length = int(best_readout_length[q.name])
        q.resonator.operations["readout"].amplitude = float(best_amp[q.name])
    cols = 2
    rows = (len(qubits) + 1) // cols
    fig_gauss, axes_gauss = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), constrained_layout=True)
    axes_gauss = np.atleast_1d(axes_gauss).flatten()
    fit_snr_with_gaussians(
        fits=ds_opt_combined,
        qubits=qubits,
        node=node,
        fit_results=fit_results_opt,
        axes=axes_gauss,
        plot=True,
    )
    for j in range(len(qubits), len(axes_gauss)):
        axes_gauss[j].axis("off")
    fig_gauss.suptitle(f"Double Gaussian fit at optimized readout\n{node.date_time} GMT+{node.time_zone} #{node.node_id}\nmultiplexed={node.parameters.multiplexed}, reset={node.parameters.reset_type_thermal_or_active}")
    plt.tight_layout()
    plt.show()
    node.results["figure_gaussian_fit"] = fig_gauss

    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qubit in qubits:
                qubit.resonator.operations["readout"].integration_weights_angle -= float(
                    node.results["results"][qubit.name]["angle"]
                )
                qubit.resonator.operations["readout"].threshold = float(node.results["results"][qubit.name]["threshold"])
                qubit.resonator.operations["readout"].rus_exit_threshold = float(
                    node.results["results"][qubit.name]["rus_threshold"]
                )
                qubit.resonator.operations["readout"].amplitude = float(node.results["results"][qubit.name]["best_amp"])
                qubit.resonator.confusion_matrix = node.results["results"][qubit.name]["confusion_matrix"].tolist()
                
                qubit.resonator.operations["readout"].length = int(node.results["results"][qubit.name]["best_readout_length"])


        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()

# %%
