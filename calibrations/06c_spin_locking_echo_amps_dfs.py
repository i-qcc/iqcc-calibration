# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

import os
import json
from datetime import datetime

from qm.qua import *
from qualang_tools.loops import from_array
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
    node.parameters.qubits = ["Q3"]
    pass

# Instantiate the QUAM class from the state file
node.machine = Quam.load()

# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots
    idle_times = get_idle_times_in_clock_cycles(node.parameters)
    amp_scales = node.parameters.amplitude_scales
    
    # --- dfs definition ---
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    # Ensure dfs is a python list or numpy array for stream_processing
    dfs = np.arange(0, +span, step) # NOTE: you mentioned also doing np.arange(-span, +span, step)
    node.namespace["dfs"] = dfs # Store for use in other cells
    # ----------------------

    # --- UPDATED SWEEP AXES (V3) ---
    # The order MUST match the QPU stream shape reported in the error:
    # ref_shape: (6, 223, 10) --> (amp_scale, idle_time, df)
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "amp_scale": xr.DataArray(amp_scales, attrs={"long_name": "amplitude scale", "units": "a.u."}),
        "idle_time": xr.DataArray(8 * idle_times, attrs={"long_name": "idle time", "units": "ns"}),
        "df": xr.DataArray(dfs, attrs={"long_name": "frequency detuning", "units": "Hz"}),
    }
    # Expected final shape: (1, len(amp_scales), len(idle_times), len(dfs))
    # This will match the stream shape (6, 223, 10) from your error log.
    
    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        df = declare(int)  # QUA variable for the qubit frequency
        shot = declare(int)
        t = declare(int)
        amp = declare(fixed) # QUA variable for amplitude scale
        
        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()
            for i, qubit in multiplexed_qubits.items():
                with for_(shot, 0, shot < n_avg, shot + 1):
                    save(shot, n_st)
                    with for_(*from_array(df,dfs)):
                        qubit.xy.update_frequency(df + qubit.xy_SL.intermediate_frequency)
                        with for_each_(amp, amp_scales): 
                            with for_each_(t, idle_times):
                                # Qubit initialization
                                for i, qubit in multiplexed_qubits.items():
                                    reset_frame(qubit.xy.name)
                                    qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                                align()
                                # Qubit manipulation
                                for i, qubit in multiplexed_qubits.items():
                                    qubit.xy.play("-y90")
                                    qubit.xy_SL.play("x180_FlatTopTanhPulse", duration=12 + 2 * t, amplitude_scale=amp)
                                    qubit.xy.play("-y90")
                                    qubit.align()
                                align()
                                # Qubit readout
                                for i, qubit in multiplexed_qubits.items():
                                    if node.parameters.use_state_discrimination:
                                        qubit.readout_state(state[i])
                                        save(state[i], state_st[i])
                                    else:
                                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                        save(I[i], I_st[i])
                                        save(Q[i], Q_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # The .buffer() order (dfs, amps, times) is determined by your QUA loops
                # This does NOT need to match the sweep_axes order, which is confusing
                # but this is how the fetcher works. Leave this as is.
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(dfs)).buffer(len(amp_scales), len(idle_times)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(dfs)).buffer(len(amp_scales), len(idle_times)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(dfs)).buffer(len(amp_scales), len(idle_times)).average().save(f"Q{i + 1}")
                
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
    
    # Execute the QUA program only if the quantum machine is available.
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        
        # --- FIX 3: Correctly instantiate XarrayDataFetcher with job and sweep axes ---
        # The XarrayDataFetcher will infer the shape (1, 3, 100) from the sweep_axes
        # and match the stream buffer (3, 100) against the inner two dimensions.
        data_fetcher = XarrayDataFetcher(
            job, 
            node.namespace["sweep_axes"]
        )
        # -----------------------------------------------------------------------------
        
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
    
    # 1. Process the raw dataset
    ds_raw_processed = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_raw"] = ds_raw_processed
    
    all_fit_results = {}
    all_ds_fit = []

    # --- MODIFICATION: Loop over 'df' and 'amp_scale' coordinates ---
    df_coords = ds_raw_processed.coords["df"].values
    amp_coords = ds_raw_processed.coords["amp_scale"].values

    for df_val in df_coords:
        for amp_scale_val in amp_coords:
            
            # Select the 1D slice corresponding to the current (df, amp_scale)
            ds_raw_slice = ds_raw_processed.sel(df=df_val, amp_scale=amp_scale_val)
            
            try:
                # --- This is the line that can fail ---
                ds_fit_slice, fit_results_slice = fit_raw_data(ds_raw_slice, node)
                # -------------------------------------

            except Exception as e:
                # --- UPDATED EXCEPTION HANDLING ---
                node.log(f"CRITICAL: Fit failed for df={df_val / 1e6:.2f}MHz, amp_scale={amp_scale_val}. Skipping. Error: {e}")
                
                # Add a "failed" outcome for all qubits for this combo
                for qubit_name in ds_raw_slice.coords["qubit"].values:
                    key = f"{qubit_name}_df_{df_val}_amp_{amp_scale_val}"
                    all_fit_results[key] = {"success": False, "error_message": str(e)}
                
                continue # Continue to the next amp_scale/df combo
                # ----------------------------------

            # Re-introduce the 'df' and 'amp_scale' coordinate to the fitted data and results
            for qubit_name, fit_result in fit_results_slice.items():
                # Create a unique key for the results dictionary
                key = f"{qubit_name}_df_{df_val}_amp_{amp_scale_val}"
                all_fit_results[key] = asdict(fit_result)
            
            # --- THIS IS THE FIX ---
            # Instead of assign_coords, explicitly expand the dimensions.
            # This creates a dataset with (df: 1, amp_scale: 1, ...) dimensions
            # which xr.combine_by_coords can correctly merge.
            ds_fit_slice = ds_fit_slice.expand_dims(dim={"df": [df_val], "amp_scale": [amp_scale_val]})
            # -----------------------

            all_ds_fit.append(ds_fit_slice)
        
    # Combine all fitted datasets back into a single xarray.Dataset
    if all_ds_fit:
        # combine_by_coords will now work, re-assembling the full (df, amp_scale, ...) dataset
        node.results["ds_fit"] = xr.combine_by_coords(all_ds_fit)
    else:
        node.results["ds_fit"] = xr.Dataset() # Handle case with no data
        
    node.results["fit_results"] = all_fit_results

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    
    # Update outcomes based on the fit results for all combinations
    node.outcomes = {
        key: ("successful" if fit_result.get("success", False) else "failed") # Use .get() for safety
        for key, fit_result in node.results["fit_results"].items()
    }

# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    
    import numpy as np

    figures = {}
    qubits_to_plot = node.namespace["qubits"]
    
    raw_var_name = "state" if node.parameters.use_state_discrimination else "I"
    
    ds_fit = node.results.get("ds_fit", xr.Dataset())
    fit_var_names = [v for v in ds_fit.data_vars if v not in ds_fit.coords and "fit_data" not in str(v)]
    fit_var_name = fit_var_names[0] if fit_var_names else "fitted_curve" 
    
    # --- MODIFICATION: Loop over df and amplitude scales ---
    df_coords = node.results["ds_raw"].coords["df"].values
    amp_coords = node.results["ds_raw"].coords["amp_scale"].values

    for df_val in df_coords:
        for amp_scale_val in amp_coords:
            
            # Select the 1D slice for the current (df, amp_scale)
            ds_raw_slice = node.results["ds_raw"].sel(df=df_val, amp_scale=amp_scale_val)
            
            try:
                # This line will fail if the fit was skipped for this combo
                ds_fit_slice = node.results["ds_fit"].sel(df=df_val, amp_scale=amp_scale_val)
                
            except KeyError:
                # --- This is the robust placeholder logic (V4) ---
                node.log(f"Note: No fitted data found for df={df_val / 1e6:.2f}MHz, amp_scale={amp_scale_val}. Plotting raw data only.")
                
                ds_fit_slice = xr.Dataset(coords=ds_raw_slice.coords)
                
                raw_data_array = ds_raw_slice[raw_var_name]
                ds_fit_slice[fit_var_name] = xr.DataArray(
                    np.nan, coords=raw_data_array.coords, dims=raw_data_array.dims
                )
                ds_fit_slice["fit_data"] = xr.DataArray(
                    np.nan, 
                    coords={"qubit": ds_raw_slice.coords["qubit"], "fit_vals": ["a", "offset", "decay"]}, 
                    dims=["qubit", "fit_vals"]
                )
                ds_fit_slice["T2_SL"] = xr.DataArray(
                    np.nan, coords={"qubit": ds_raw_slice.coords["qubit"]}, dims=["qubit"]
                )
                ds_fit_slice["T2_SL_error"] = xr.DataArray(
                    np.nan, coords={"qubit": ds_raw_slice.coords["qubit"]}, dims=["qubit"]
                )
                # ----------------------------------------------------
            
            # This call is now safe
            fig_raw_fit = plot_raw_data_with_fit(ds_raw_slice, qubits_to_plot, ds_fit_slice)
            
            # Update plot title/subtitle to reflect both parameters
            title = f"Spin-Locking T2 | Amp: {amp_scale_val:.2f} | Freq Detuning: {df_val / 1e6:.2f} MHz"
            fig_raw_fit.suptitle(title, fontsize=14, y=1.02)
            node.add_node_info_subtitle(fig_raw_fit)
            plt.show()
            
            # Store the generated figure with a descriptive key
            figures[f"raw_fit_df_{df_val}_amp_{amp_scale_val}"] = fig_raw_fit

    # Store all generated figures
    node.results["figures"] = figures

# %% {Save_data}
@node.run_action(skip_if=node.parameters.simulate)
def save_data(node: QualibrationNode[Parameters, Quam]):
    """Saves the raw data, fitted data, fit results (JSON), and figures for each qubit, df, and amplitude scale."""

    # --- Define the base save path ---
    base_path = r"C:\\Users\\gilads\\VisualStudioProjects\\iqcc_cloud\\experiment_data"
    
    # --- Get current date and time for unique filenames ---
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")

    # --- Data and Coordinates ---
    ds_raw = node.results["ds_raw"]
    ds_fit = node.results.get("ds_fit", xr.Dataset())
    fit_results = node.results["fit_results"]
    figures = node.results["figures"]
    
    raw_var_name = "state" if node.parameters.use_state_discrimination else "I"
    
    fit_var_names = [v for v in ds_fit.data_vars if v not in ds_fit.coords]
    fit_var_name = fit_var_names[0] if fit_var_names else None 

    os.makedirs(base_path, exist_ok=True)

    # --- Get all coordinates for loops ---
    qubit_names = ds_raw.coords["qubit"].values
    df_coords = ds_raw.coords["df"].values
    amp_scales = ds_raw.coords["amp_scale"].values
    
    node.log(f"Starting data save to: {base_path}")

    for qubit_name in qubit_names:
        for df_val in df_coords: # <-- New outer loop
            # Format df value for filename
            df_val_mhz_str = f"{(df_val / 1e6):.2f}MHz".replace('.', 'p').replace('-', 'm')
            
            for amp_scale_val in amp_scales: # <-- Inner loop
                amp_scale_str = f"{amp_scale_val:.2f}".replace('.', 'p')
                
                # --- 1. Define the unique file name prefix ---
                filename_prefix = f"{date_time_str}_{qubit_name}_df_{df_val_mhz_str}_amp_{amp_scale_str}"
                
                # --- 2. Extract specific fit results ---
                result_key = f"{qubit_name}_df_{df_val}_amp_{amp_scale_val}"
                qubit_fit_result = fit_results.get(result_key)

                if qubit_fit_result:
                    # --- 3. Save the JSON results ---
                    json_path = os.path.join(base_path, f"{filename_prefix}_results.json")
                    try:
                        save_json = {qubit_name: qubit_fit_result}
                        with open(json_path, 'w') as f:
                            json.dump(save_json, f, indent=4)
                        node.log(f"Saved JSON results for {result_key} to {json_path}")
                    except Exception as e:
                        node.log(f"Error saving JSON for {result_key}: {e}")

                # --- 4. Save the Experiment Data (ds_raw and ds_fit) ---
                data_path = os.path.join(base_path, f"{filename_prefix}_data.nc")
                
                if raw_var_name:
                    try:
                        # Select data slices for saving
                        raw_slice = ds_raw.sel(qubit=qubit_name, df=df_val, amp_scale=amp_scale_val)
                        
                        data_to_save_dict = {"raw_data": raw_slice[raw_var_name]}

                        if fit_var_name:
                            try:
                                # This sel() will fail if fit was skipped
                                fit_slice = ds_fit.sel(qubit=qubit_name, df=df_val, amp_scale=amp_scale_val)
                                data_to_save_dict["fitted_curve"] = fit_slice[fit_var_name]
                            except KeyError:
                                node.log(f"Note: No fitted data for {result_key}. Saving raw data only.")
                        
                        data_to_save = xr.Dataset(data_to_save_dict, coords=raw_slice.coords)
                        
                        data_to_save.to_netcdf(data_path)
                        node.log(f"Saved data for {result_key} to {data_path}")
                    except Exception as e:
                        node.log(f"Error saving NetCDF data for {result_key}: {e}")
                else:
                    node.log(f"Skipping NetCDF save for {result_key}: Raw variable ('{raw_var_name}') not found.")
                    
                # --- 5. Save the Figure ---
                figure_key = f"raw_fit_df_{df_val}_amp_{amp_scale_val}"
                fig = figures.get(figure_key)
                if fig:
                    figure_path = os.path.join(base_path, f"{filename_prefix}_plot.png")
                    try:
                        fig.savefig(figure_path, bbox_inches="tight")
                        node.log(f"Saved figure for {result_key} to {figure_path}")
                    except Exception as e:
                        node.log(f"Error saving figure for {result_key}: {e}")

    node.log("Data saving complete.")