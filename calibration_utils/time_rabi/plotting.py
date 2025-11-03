from typing import List
import xarray as xr
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)

def plot_raw_data_with_fit(
    ds_raw: xr.Dataset, 
    qubits: List[AnyTransmon], 
    ds_fit: xr.Dataset
) -> plt.Figure:
    """
    Plot the raw Time Rabi data and the fitted cosine function.
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
        
        ax.set_ylabel(y_label)
        ax.set_xlabel("Pulse duration [ns]")
        ax.set_title(q_name)
        ax.legend()

    grid.fig.suptitle("Time Rabi")
    plt.tight_layout()
    # Do not call plt.show() here, let the node script handle it
    
    return grid.fig