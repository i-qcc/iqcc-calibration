from typing import List, Dict, Any
import xarray as xr
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import json
import os

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)

def plot_raw_data_with_fit(
    ds_raw: xr.Dataset, 
    qubits: List[AnyTransmon], 
    ds_fit: xr.Dataset,
    fit_results: Dict[str, Any] # <-- Pass in fit results (as dicts)
) -> plt.Figure:
    """
    Plot the raw Time Rabi data and the fitted cosine function.
    
    Also displays the results from both Duration and Amplitude analysis
    in the title of each subplot.
    """
    
    grid = QubitGrid(ds_raw, [q.grid_location for q in qubits])
    
    # Check if we're using state discrimination or I/Q
    if "state" in ds_raw:
        signal_name = "state"
        y_label = "State"
    else:
        signal_name = "V" if "V" in ds_raw else "I"
        y_label = f"{signal_name} (V)"
        
    for ax, qubit_info in grid_iter(grid):
        q_name = qubit_info["qubit"]
        
        # Plot raw data
        ds_raw.sel(qubit=q_name)[signal_name].plot(
            ax=ax, 
            marker=".", 
            linestyle="None", 
            label="Raw data"
        )
        
        # Plot fit
        ds_fit.sel(qubit=q_name).plot(ax=ax, label="Fit")
        
        # --- Create the title string with fit results ---
        fit_result = fit_results[q_name]
        
        if fit_result["success"]:
            # Format results for display
            f_mhz = fit_result['f'] * 1e3
            pi_dur = fit_result['opt_dur_pi']
            pi_half_dur = fit_result['opt_dur_pi_half']
            
            # Get amplitude from state.json for the current qubit's xy_sl.x180_Square.amplitude
            amp_from_state = None
            try:
                # Try to get path from environment variable first
                quam_state_path = os.environ.get("QUAM_STATE_PATH")
                if quam_state_path:
                    state_file_path = os.path.join(quam_state_path, "state.json")
                else:
                    # Fall back to relative path from project root
                    # Go up from calibration_utils/time_rabi/plotting.py to project root
                    current_file = os.path.abspath(__file__)
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
                    state_file_path = os.path.join(project_root, "quam_state_path", "state.json")
                
                with open(state_file_path, 'r') as f:
                    state_data = json.load(f)
                    amp_from_state = state_data["qubits"][q_name]["xy_sl"]["operations"]["x180_Square"]["amplitude"]
            except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError) as e:
                # If we can't read the state file or the path doesn't exist, just skip showing amplitude
                pass
            
            # Create a multi-line title string
            if amp_from_state is not None:
                title_str = (
                    f"{q_name}\n"
                    f"Freq: {f_mhz:.2f} MHz | π-pulse: {pi_dur:.1f} ns\n"
                    f"Amp: {amp_from_state:.3f}"
                )
            else:
                title_str = (
                    f"{q_name}\n"
                    f"Freq: {f_mhz:.2f} MHz | π-pulse: {pi_dur:.1f} ns"
                )
        else:
            title_str = f"{q_name}\n(Fit Failed)"
        
        # --- End title logic ---
        
        ax.set_ylabel(y_label)
        ax.set_xlabel("Pulse duration [ns]")
        # Set the multi-line title, slightly smaller
        ax.set_title(title_str, fontsize=10) 
        
        # Keep legend in a non-overlapping spot
        ax.legend(loc='lower right')

    grid.fig.suptitle("Time Rabi (Duration & Amplitude Analysis)")
    plt.tight_layout()
    # Do not call plt.show() here, let the node script handle it
    
    return grid.fig

def plot_raw_data(
    ds_raw: xr.Dataset, 
    qubits: List[AnyTransmon]
) -> plt.Figure:
    """
    Plot only the raw Time Rabi data.
    """
    
    grid = QubitGrid(ds_raw, [q.grid_location for q in qubits])
    
    # Check if we're using state discrimination or I/Q
    if "state" in ds_raw:
        signal_name = "state"
        y_label = "State"
    else:
        signal_name = "V" if "V" in ds_raw else "I"
        y_label = f"{signal_name} (V)"
        
    for ax, qubit_info in grid_iter(grid):
        q_name = qubit_info["qubit"]
        
        # Plot raw data
        ds_raw.sel(qubit=q_name)[signal_name].plot(
            ax=ax, 
            marker=".", 
            linestyle="None",
            label="Raw data"
        )
        
        ax.set_ylabel(y_label)
        ax.set_xlabel("Pulse duration [ns]")
        ax.set_title(q_name) # Simple title
        ax.legend(loc='lower right')

    grid.fig.suptitle("Time Rabi (Raw Data Only)")
    plt.tight_layout()
    # Do not call plt.show()
    
    return grid.fig

