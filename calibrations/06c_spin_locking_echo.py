# %% {Imports}
import matplotlib.pyplot as plt
import xarray as xr
from dataclasses import asdict
from datetime import datetime
import os
import json
import numpy as np
from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from iqcc_calibration_tools.qualibrate_config.qualibrate.node import QualibrationNode
from iqcc_calibration_tools.quam_config.components.quam_root import Quam
from calibration_utils.spin_echo_sl import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.parameters import get_qubits, get_sl_times_in_clock_cycles
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Description}
description = """ T2 SL MEASUREMENT
The sequence consists in playing an SL sequence (y90 - SL(x,t) - -y90 - measurement) for 
different sl times (spin locking times).
"""
QUBIT = "Q6"
node = QualibrationNode[Parameters, Quam](name="06c_spin_locking", description=description, parameters=Parameters())
# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = [QUBIT]
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
    # Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
    raw_cycles = get_sl_times_in_clock_cycles(node.parameters)
    pulse_counts = np.unique(raw_cycles // 10).astype(int) #    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "spin_locking_time": xr.DataArray(40 * pulse_counts, attrs={"long_name": "spin_locking_time", "units": "ns"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        n_pulses = declare(int)
        shot = declare(int)
        t_sl = declare(int)

        for multiplexed_qubits in qubits.batch():
            
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()
        
            with for_(shot, 0, shot < n_avg, shot + 1):
                save(shot, n_st)
                with for_each_(t_sl, pulse_counts):
                    # Qubit initialization
                    for i, qubit in multiplexed_qubits.items():
                        reset_frame(qubit.xy.name)
                        qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                    align()
                    
                    # Qubit manipulation
                    for i, qubit in multiplexed_qubits.items():
                            qubit.xy.play("-y90")
                            qubit.xy.play("x180_BlackmanIntegralPulse_Rise")
                            with for_(n_pulses, 0, n_pulses<t_sl, n_pulses+1):
                                qubit.xy.play("x180_DetunedSquare")
                            qubit.xy.play("x180_BlackmanIntegralPulse_Fall")
                            qubit.xy.play("-y90")
                    align()
                    # Qubit readout
                    for i, qubit in multiplexed_qubits.items():
                        # Measure the state of the resonators
                        if node.parameters.use_state_discrimination:
                            qubit.readout_state(state[i])
                            save(state[i], state_st[i])
                        else:
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            # save data
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(pulse_counts)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(pulse_counts)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(pulse_counts)).average().save(f"Q{i + 1}")


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
    # Here I can change the config...
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
    # Rename for compatibility with process_raw_dataset and fit_raw_data
    ds_raw_compat = node.results["ds_raw"].rename({"spin_locking_time": "idle_time"})
    
    node.results["ds_raw"] = process_raw_dataset(ds_raw_compat, node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    
    # Revert to your custom naming for state saving and future reference
    node.results["ds_raw"] = node.results["ds_raw"].rename({"idle_time": "spin_locking_time"})
    if node.results["ds_fit"] is not None:
         node.results["ds_fit"] = node.results["ds_fit"].rename({"idle_time": "spin_locking_time"})
         
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)


#%%
def _get_current_amplitude():
    """Reads the current x180_Square amplitude from the state.json file."""
    # Define the path to the state.json file (Hardcoded path from your setup)
    STATE_FILE_PATH = r"C:\Users\gilads\VisualStudioProjects\iqcc_cloud\quam_state_path\state.json"
    
    try:
        with open(STATE_FILE_PATH, 'r') as f:
            data = json.load(f)
        
        # Navigate to the specific key path
        return data["qubits"][QUBIT]["xy"]["operations"]["x180_DetunedSquare"]["amplitude"]
        
    except FileNotFoundError:
        print(f"Error: State file not found at {STATE_FILE_PATH}")
        return None
    except KeyError as e:
        print(f"Error: Could not find key {e} in state.json structure.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode state.json. {e}")
        return None

# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data with temporary name compatibility for idle_time."""
    
    # 1. Temporarily rename 'spin_locking_time' back to 'idle_time' for the plotting utility
    # This avoids the AttributeError: 'Dataset' object has no attribute 'idle_time'
    ds_raw_compat = node.results["ds_raw"].rename({"spin_locking_time": "idle_time"})
    
    # Ensure ds_fit also has the compatible name if it exists
    ds_fit_compat = None
    if node.results.get("ds_fit") is not None:
        ds_fit_compat = node.results["ds_fit"].rename({"spin_locking_time": "idle_time"})

    # 2. Call the plotting utility with compatible datasets
    fig_raw_fit = plot_raw_data_with_fit(
        ds_raw_compat, 
        node.namespace["qubits"], 
        ds_fit_compat
    )
    
    node.add_node_info_subtitle(fig_raw_fit)
    
    # --- Resolve Amplitude String Reference ---
    # Convert string references like "#../amplitude" to float to prevent ValueError in formatting
    current_amp_scale = _get_current_amplitude()
    try:
        amp_scale_val = float(current_amp_scale)
    except (ValueError, TypeError):
        # Fallback if the value is a non-numeric string reference
        amp_scale_val = 0.0
        
    figure_key = f"raw_fit_amp_{amp_scale_val}"
    
    # Store the generated figures
    node.results["figures"] = {
        figure_key: fig_raw_fit,
    }
    plt.show()


# %% {Save_data}
@node.run_action(skip_if=node.parameters.simulate)
def save_data(node: QualibrationNode[Parameters, Quam]):
    """Saves raw data (with spin_locking_time), fit results, and figures."""
    
    # --- Robust Amplitude Retrieval ---
    current_amp_scale = _get_current_amplitude()
    if current_amp_scale is None:
        node.log("CRITICAL: Failed to determine current amplitude scale. Aborting save.")
        return

    # Convert amplitude to float to safely use the :.2f format
    try:
        amp_scale_val = float(current_amp_scale)
    except (ValueError, TypeError):
        # If current_amp_scale is a reference string (e.g. '#../amplitude'), use 0.0 for the filename
        amp_scale_val = 0.0

    base_path = r"C:\\Users\\gilads\\VisualStudioProjects\\iqcc_cloud\\experiment_data"
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")

    ds_raw = node.results["ds_raw"]
    ds_fit = node.results.get("ds_fit", xr.Dataset())
    fit_results = node.results["fit_results"]
    figures = node.results["figures"]
    
    raw_var_name = "state" if node.parameters.use_state_discrimination else "I"
    fit_var_names = [v for v in ds_fit.data_vars if v not in ds_fit.coords]
    fit_var_name = fit_var_names[0] if fit_var_names else None

    os.makedirs(base_path, exist_ok=True)
    qubit_names = ds_raw.coords["qubit"].values
    
    amp_scale_str = f"{amp_scale_val:.2f}".replace('.', 'p')
    raw_var_name = "state" if node.parameters.use_state_discrimination else "I"
    for qubit_name in qubit_names:
        filename_prefix = f"{date_time_str}_{qubit_name}_amp_{amp_scale_str}"
        result_key = f"{qubit_name}_amp_{amp_scale_val}"

        # --- 1. Save JSON results ---
        qubit_fit_result = fit_results.get(qubit_name)
        if qubit_fit_result:
            json_path = os.path.join(base_path, f"{filename_prefix}_results.json")
            with open(json_path, 'w') as f:
                json.dump({qubit_name: qubit_fit_result}, f, indent=4)

        # --- 2. Save NetCDF data (Preserves 'spin_locking_time') ---
        data_path = os.path.join(base_path, f"{filename_prefix}_data.nc")
        try:
            # Slicing the data for the specific qubit
            raw_slice = ds_raw.sel(qubit=qubit_name)[raw_var_name]
            data_to_save_dict = {"raw_data": raw_slice}
            
            if fit_var_name and qubit_name in ds_fit.qubit:
                fit_slice = ds_fit.sel(qubit=qubit_name)[fit_var_name]
                data_to_save_dict["fitted_curve"] = fit_slice
            
            xr.Dataset(data_to_save_dict).to_netcdf(data_path)
            node.log(f"Saved NetCDF data to {data_path}")
        except Exception as e:
            node.log(f"Error saving NetCDF for {qubit_name}: {e}")

        # --- 3. Save Figure ---
        figure_key = f"raw_fit_amp_{amp_scale_val}"
        fig = figures.get(figure_key)
        if fig:
            figure_path = os.path.join(base_path, f"{filename_prefix}_plot.png")
            fig.savefig(figure_path, bbox_inches="tight")

    node.log("Data saving complete.")
# %%
plt.show()
# %%
