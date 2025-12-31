"""
        IQ BLOBS WITH DEPLETION TIME OPTIMIZATION
This sequence extends the IQ blobs measurement by adding a scan over depletion time to optimize both readout power and
depletion time simultaneously. The sequence measures the state of the resonator 'N' times, first after active reset
(with the qubit in the |g> state) and then after applying a pi pulse to the qubit (bringing the qubit to the |e> state)
successively.

The resulting IQ blobs are displayed, and the data is processed to determine:
    - The optimal readout power and depletion time combination for maximum fidelity
    - The rotation angle required for the integration weights
    - The threshold along the 'I' quadrature for effective qubit state discrimination
    - The readout fidelity matrix

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy)
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the rotation angle (rotation_angle) in the state
    - Update the g -> e threshold (ge_threshold) in the state
    - Update the optimal depletion time in the state
    - Save the current state by calling machine.save("quam")
"""

# %% {Imports}
from datetime import datetime, timezone, timedelta
from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode, NodeParameters
from iqcc_calibration_tools.quam_config.components import Quam
from iqcc_calibration_tools.quam_config.macros import qua_declaration, active_reset
from iqcc_calibration_tools.analysis.plot_utils import QubitGrid, grid_iter
from iqcc_calibration_tools.storage.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.analysis import two_state_discriminator
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.mixture import GaussianMixture


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["qD2"]
    num_runs: int = 2
    start_amp: float = 0.5
    end_amp: float = 1.99
    num_amps: int = 2
    start_depletion_time: int = 2500  # ns
    end_depletion_time: int = 3500  # ns
    num_depletion_times: int = 2
    outliers_threshold: float = 0.98
    plot_raw: bool = False
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True


node = QualibrationNode(name="08d_Readout_Power_And_Depletion_Time_Optimization", parameters=Parameters())

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions
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

# Define the parameter space
n_runs = node.parameters.num_runs
amps = np.linspace(node.parameters.start_amp, node.parameters.end_amp, node.parameters.num_amps)
depletion_times = np.linspace(node.parameters.start_depletion_time, node.parameters.end_depletion_time, node.parameters.num_depletion_times).astype(int) // 4

# %% {QUA_program}
with program() as iq_blobs:
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)
    a = declare(fixed)
    depletion_times_qua = declare(int, value=depletion_times)
    j = declare(int)
    
    # Set flux point
    machine.initialize_qpu(flux_point="joint", target=qubits[0])

    for i, qubit in enumerate(qubits):

        with for_(j,0,j<len(depletion_times),j+1):
            with for_(*from_array(a, amps)):
                with for_(n, 0, n < n_runs, n + 1):
                    save(n, n_st)
                    # Ground state measurement
                    active_reset(qubit, "readout", depletion_time=depletion_times_qua[j])
                    qubit.align()
                    qubit.resonator.measure("readout", qua_vars=(I_g[i], Q_g[i]), amplitude_scale=a)
                    qubit.align()
                    save(I_g[i], I_g_st[i])
                    save(Q_g[i], Q_g_st[i])

                    # Excited state measurement
                    active_reset(qubit, "readout", depletion_time=depletion_times_qua[j])
                    qubit.align()
                    qubit.xy.play("x180")
                    qubit.align()
                    qubit.resonator.measure("readout", qua_vars=(I_e[i], Q_e[i]), amplitude_scale=a)
                    save(I_e[i], I_e_st[i])
                    save(Q_e[i], Q_e_st[i])

            if not node.parameters.multiplexed:
                align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_g_st[i].buffer(n_runs).buffer(len(amps)).buffer(len(depletion_times)).save(f"I_g{i + 1}")
            Q_g_st[i].buffer(n_runs).buffer(len(amps)).buffer(len(depletion_times)).save(f"Q_g{i + 1}")
            I_e_st[i].buffer(n_runs).buffer(len(amps)).buffer(len(depletion_times)).save(f"I_e{i + 1}")
            Q_e_st[i].buffer(n_runs).buffer(len(amps)).buffer(len(depletion_times)).save(f"Q_e{i + 1}")

# %% {Program_Execution}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)
    job = qmm.simulate(config, iq_blobs, simulation_config)
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(iq_blobs)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_runs, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data and create dataset
        ds = fetch_results_as_xarray(
            job.result_handles, 
            qubits, 
            {
                "N": np.linspace(1, n_runs, n_runs),
                "amplitude": amps,
                "depletion_time": depletion_times
                
            }
        )
        # Add the absolute readout power to the dataset
        ds = ds.assign_coords({
            "readout_amp": (["qubit", "amplitude"], 
            np.array([amps * q.resonator.operations["readout"].amplitude for q in qubits]))
        })
        
        # Rearrange data to combine I_g and I_e into I, and Q_g and Q_e into Q
        ds_rearranged = xr.Dataset()
        ds_rearranged["I"] = xr.concat([ds.I_g, ds.I_e], dim="state")
        ds_rearranged["I"] = ds_rearranged["I"].assign_coords(state=[0, 1])
        ds_rearranged["Q"] = xr.concat([ds.Q_g, ds.Q_e], dim="state")
        ds_rearranged["Q"] = ds_rearranged["Q"].assign_coords(state=[0, 1])
        
        for var in ds.coords:
            if var not in ds_rearranged.coords:
                ds_rearranged[var] = ds[var]

        for var in ds.data_vars:
            if var not in ["I_g", "I_e", "Q_g", "Q_e"]:
                ds_rearranged[var] = ds[var]

        ds = ds_rearranged
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]

    node.results = {"ds": ds, "results": {}, "figs": {}}

# %% {Data_analysis}
def apply_fit_gmm(I, Q):
    I_mean = np.mean(I, axis=1)
    Q_mean = np.mean(Q, axis=1)
    means_init = [[I_mean[0], Q_mean[0]], [I_mean[1], Q_mean[1]]]
    precisions_init = [1 / ((np.mean(np.var(I, axis=1)) + np.mean(np.var(Q, axis=1))) / 2)] * 2
    clf = GaussianMixture(
        n_components=2,
        covariance_type="spherical",
        means_init=means_init,
        precisions_init=precisions_init,
        tol=1e-5,
        reg_covar=1e-12,
    )
    X = np.array([np.array(I).flatten(), np.array(Q).flatten()]).T
    clf.fit(X)
    meas_fidelity = (
        np.sum(clf.predict(np.array([I[0], Q[0]]).T) == 0) / len(I[0])
        + np.sum(clf.predict(np.array([I[1], Q[1]]).T) == 1) / len(I[1])
    ) / 2
    loglikelihood = clf.score_samples(X)
    max_ll = np.max(loglikelihood)
    outliers = np.sum(loglikelihood > np.log(0.01) + max_ll) / len(X)
    return np.array([meas_fidelity, outliers])

# Apply analysis over both amplitude and depletion time dimensions
fit_res = xr.apply_ufunc(
    apply_fit_gmm,
    ds.I,
    ds.Q,
    input_core_dims=[["state", "N"], ["state", "N"]],
    output_core_dims=[["result"]],
    vectorize=True,
)

fit_res = fit_res.assign_coords(result=["meas_fidelity", "outliers"])

# %% {Results_Analysis_and_Plotting}
# Find optimal parameters and analyze results
best_params = {}
for q in qubits:
    fit_res_q = fit_res.sel(qubit=q.name)
    valid_points = fit_res_q.sel(result="outliers") >= node.parameters.outliers_threshold
    fidelities = fit_res_q.sel(result="meas_fidelity").where(valid_points)
    
    # Find the indices of the maximum fidelity
    max_fidelity_coords = fidelities.argmax(dim=["amplitude", "depletion_time"])
    best_amp = float(ds.readout_amp.sel(qubit=q.name, amplitude=max_fidelity_coords["amplitude"]))
    best_depletion = float(depletion_times[max_fidelity_coords["depletion_time"]])
    
    print(f"Optimal parameters for {q.name}:")
    print(f"  Readout amplitude: {best_amp}")
    print(f"  Depletion time: {best_depletion} ns")
    
    # Store best parameters
    best_params[q.name] = {
        "best_amp": best_amp,
        "best_depletion_time": best_depletion
    }
    
    # Select data for the best parameters
    best_data = ds.sel(
        qubit=q.name,
        amplitude=float(max_fidelity_coords["amplitude"]),
        depletion_time=float(depletion_times[max_fidelity_coords["depletion_time"]])
    )
    
    # Calculate optimal angle and threshold
    I_g = best_data.I.sel(state=0)
    Q_g = best_data.Q.sel(state=0)
    I_e = best_data.I.sel(state=1)
    Q_e = best_data.Q.sel(state=1)
    angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(
        I_g, Q_g, I_e, Q_e, True, b_plot=False
    )
    
    # Calculate RUS threshold
    I_rot = I_g * np.cos(angle) - Q_g * np.sin(angle)
    hist = np.histogram(I_rot, bins=100)
    RUS_threshold = hist[1][1:][np.argmax(hist[0])]
    
    # Store results
    node.results["results"][q.name] = {
        "best_amp": best_amp,
        "best_depletion_time": best_depletion,
        "angle": float(angle),
        "threshold": float(threshold),
        "fidelity": float(fidelity),
        "confusion_matrix": np.array([[gg, ge], [eg, ee]]),
        "rus_threshold": float(RUS_threshold)
    }

# %% {Visualization}
# 1. Fidelity heatmaps
grid = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid):
    fidelities = fit_res.sel(qubit=qubit["qubit"], result="meas_fidelity")
    im = ax.pcolormesh(
        depletion_times,
        ds.readout_amp.sel(qubit=qubit["qubit"]),
        fidelities,
        shading='auto'
    )
    best_params_q = best_params[qubit["qubit"]]
    ax.plot(best_params_q["best_depletion_time"], best_params_q["best_amp"], 'r*', markersize=10)
    ax.set_xlabel("Depletion time (ns)")
    ax.set_ylabel("Readout amplitude")
    ax.set_title(f"{qubit['qubit']}\nBest fidelity: {fidelities.max().values:.3f}")
    plt.colorbar(im, ax=ax)

grid.fig.suptitle(f"Measurement Fidelity vs Readout Amplitude and Depletion Time\n{node.date_time} GMT+{node.time_zone} #{node.node_id}")
plt.tight_layout()
node.results["figure_fidelity_heatmap"] = grid.fig

# 2. IQ blobs at optimal points
grid = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid):
    qn = qubit["qubit"]
    best_data = ds.sel(
        qubit=qn,
        amplitude=float(ds.amplitude.sel(qubit=qn)[
            np.argmin(np.abs(ds.readout_amp.sel(qubit=qn) - node.results["results"][qn]["best_amp"]))
        ]),
        depletion_time=float(depletion_times[
            np.argmin(np.abs(depletion_times - node.results["results"][qn]["best_depletion_time"]))
        ])
    )
    
    angle = node.results["results"][qn]["angle"]
    
    # Plot rotated IQ data
    ax.plot(
        1e3 * (best_data.I.sel(state=0) * np.cos(angle) - best_data.Q.sel(state=0) * np.sin(angle)),
        1e3 * (best_data.I.sel(state=0) * np.sin(angle) + best_data.Q.sel(state=0) * np.cos(angle)),
        ".", alpha=0.1, label="Ground", markersize=1
    )
    ax.plot(
        1e3 * (best_data.I.sel(state=1) * np.cos(angle) - best_data.Q.sel(state=1) * np.sin(angle)),
        1e3 * (best_data.I.sel(state=1) * np.sin(angle) + best_data.Q.sel(state=1) * np.cos(angle)),
        ".", alpha=0.1, label="Excited", markersize=1
    )
    
    # Add threshold lines
    ax.axvline(
        1e3 * node.results["results"][qn]["rus_threshold"],
        color="k",
        linestyle="--",
        lw=0.5,
        label="RUS Threshold"
    )
    ax.axvline(
        1e3 * node.results["results"][qn]["threshold"],
        color="r",
        linestyle="--",
        lw=0.5,
        label="Threshold"
    )
    
    ax.axis("equal")
    ax.set_xlabel("I [mV]")
    ax.set_ylabel("Q [mV]")
    ax.set_title(f"{qn}\nFidelity: {node.results['results'][qn]['fidelity']:.3f}")

ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
grid.fig.suptitle(f"IQ Blobs at Optimal Parameters\n{node.date_time} GMT+{node.time_zone} #{node.node_id}")
plt.tight_layout()
node.results["figure_optimal_blobs"] = grid.fig

# %% {Update_state}
if not node.parameters.simulate and node.parameters.load_data_id is None:
    with node.record_state_updates():
        for qubit in qubits:
            qubit.resonator.operations["readout"].integration_weights_angle -= float(
                node.results["results"][qubit.name]["angle"]
            )
            qubit.resonator.operations["readout"].threshold = float(
                node.results["results"][qubit.name]["threshold"]
            )
            qubit.resonator.operations["readout"].rus_exit_threshold = float(
                node.results["results"][qubit.name]["rus_threshold"]
            )
            qubit.resonator.operations["readout"].amplitude = float(
                node.results["results"][qubit.name]["best_amp"]
            )
            qubit.resonator.depletion_time = int(
                node.results["results"][qubit.name]["best_depletion_time"]
            )
            qubit.resonator.confusion_matrix = node.results["results"][qubit.name]["confusion_matrix"].tolist()

    # Save results
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save() 