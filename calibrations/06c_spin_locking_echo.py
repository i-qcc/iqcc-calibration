# %% {Imports}
import matplotlib.pyplot as plt
import xarray as xr
from dataclasses import asdict
from datetime import datetime
import os
import json

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
from qualibration_libs.parameters import get_qubits, get_idle_times_in_clock_cycles
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Description}
description = """ T2 SL MEASUREMENT
The sequence consists in playing an SL sequence (y90 - SL(x,t) - -y90 - measurement) for 
different idle times.
"""
node = QualibrationNode[Parameters, Quam](name="06c_spin_locking", description=description, parameters=Parameters())

# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["Q5"]
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
    idle_times = get_idle_times_in_clock_cycles(node.parameters)
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(4 * idle_times, attrs={"long_name": "idle time", "units": "ns"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        shot = declare(int)
        t = declare(int)

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()
            for i, qubit in multiplexed_qubits.items():
                with for_(shot, 0, shot < n_avg, shot + 1):
                    save(shot, n_st)
                    with for_each_(t, idle_times):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            reset_frame(qubit.xy.name)
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()
                        # Qubit manipulation
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.play("-y90")
                            qubit.xy.play("x180_BlackmanIntegralPulse_Rise")
                            qubit.xy.play("x180_Square",duration = 2*t)
                            qubit.xy.play("x180_BlackmanIntegralPulse_Fall")
                            qubit.xy.play("-y90")
                            qubit.align()
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
                    state_st[i].buffer(len(idle_times)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(idle_times)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(idle_times)).average().save(f"Q{i + 1}")


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

#%%
def _get_current_amplitude():
    """Reads the current x180_Square amplitude from the state.json file."""
    # Define the path to the state.json file (Hardcoded path from your setup)
    STATE_FILE_PATH = r"C:\Users\gilads\VisualStudioProjects\iqcc_cloud\quam_state_path\state.json"
    
    try:
        with open(STATE_FILE_PATH, 'r') as f:
            data = json.load(f)
        
        # Navigate to the specific key path
        return data["qubits"]["Q5"]["xy"]["operations"]["x180_Square"]["amplitude"]
        
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
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    node.add_node_info_subtitle(fig_raw_fit)
    
    # --- FIX 2: Generate the key needed by the save function ---
    current_amp_scale = _get_current_amplitude()
    amp_scale_val = current_amp_scale if current_amp_scale is not None else 0.0 
    figure_key = f"raw_fit_amp_{amp_scale_val}"
    
    # Store the generated figures using the correct key
    node.results["figures"] = {
        figure_key: fig_raw_fit,
    }
    plt.show()


# %% {Save_data}
@node.run_action(skip_if=node.parameters.simulate)
def save_data(node: QualibrationNode[Parameters, Quam]):
    """Saves the raw data, fitted data, fit results (JSON), and figures for each qubit using the current amplitude scale."""
    # --- Configuration and Initial Data Retrieval ---
    current_amp_scale = _get_current_amplitude()
    if current_amp_scale is None:
        node.log("CRITICAL: Failed to determine current amplitude scale. Aborting save.")
        return

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
    
    node.log(f"Starting data save to: {base_path} for amplitude: {current_amp_scale}")

    # --- AMPLITUDE VARIABLE SETUP ---
    amp_scale_val = current_amp_scale
    amp_scale_str = f"{amp_scale_val:.2f}".replace('.', 'p') # e.g., '0p10'

    for qubit_name in qubit_names:
        
        # --- 1. Define the unique file name prefix ---
        filename_prefix = f"{date_time_str}_{qubit_name}_amp_{amp_scale_str}"
        
        # --- 2. Extract specific fit results for this (qubit, amp_scale) ---
        # FIX 1: Look up fit results by qubit_name only, as the analysis step uses this key.
        fit_lookup_key = qubit_name 
        result_key = f"{qubit_name}_amp_{amp_scale_val}" # Key used for logging/file naming

        qubit_fit_result = fit_results.get(fit_lookup_key) 

        # --- 3. Save the JSON results ---
        if qubit_fit_result:
            json_path = os.path.join(base_path, f"{filename_prefix}_results.json")
            try:
                save_json = {qubit_name: qubit_fit_result}
                with open(json_path, 'w') as f:
                    json.dump(save_json, f, indent=4)
                node.log(f"Saved JSON results for {result_key} to {json_path}")
            except Exception as e:
                node.log(f"Error saving JSON for {result_key}: {e}")
        else:
             node.log(f"Skipping JSON save for {result_key}: Fit result not found under key '{fit_lookup_key}' in fit_results.")


        # --- 4. Save the Experiment Data (NetCDF) ---
        data_path = os.path.join(base_path, f"{filename_prefix}_data.nc")
        
        if raw_var_name:
            try:
                raw_slice_full = ds_raw.sel(qubit=qubit_name).drop_vars('amp_scale', errors='ignore')
                raw_slice = raw_slice_full[raw_var_name]
                
                data_to_save_dict = {"raw_data": raw_slice}
                
                if fit_var_name:
                    try:
                        fit_slice_full = ds_fit.sel(qubit=qubit_name).drop_vars('amp_scale', errors='ignore')
                        data_to_save_dict["fitted_curve"] = fit_slice_full[fit_var_name]
                    except KeyError:
                        node.log(f"Note: No fitted data for {result_key}. Saving raw data only.")
                
                data_to_save = xr.Dataset(data_to_save_dict) 

                data_to_save.to_netcdf(data_path)
                node.log(f"Saved data for {result_key} to {data_path}")
                
            except Exception as e:
                node.log(f"Error saving NetCDF data for {result_key}: {e}")
        else:
             node.log(f"Skipping NetCDF save for {qubit_name}: Raw variable ('{raw_var_name}') not found.")
            
        # --- 5. Save the Figure (Graph Image) ---
        # NOTE: This key must match the key used in the Plot_data section (Fix 2)
        figure_key = f"raw_fit_amp_{amp_scale_val}" 
        fig = figures.get(figure_key)
        
        if fig:
            figure_path = os.path.join(base_path, f"{filename_prefix}_plot.png")
            try:
                fig.savefig(figure_path, bbox_inches="tight")
                node.log(f"Saved figure for {result_key} to {figure_path}")
            except Exception as e:
                node.log(f"Error saving figure for {result_key}: {e}")
        else:
             node.log(f"Skipping Figure save for {result_key}: Figure not found under key '{figure_key}'.")


    node.log("Data saving complete.")
# %%
plt.show()