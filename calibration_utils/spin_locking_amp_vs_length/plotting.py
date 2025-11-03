from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
# Removed: from qualibration_libs.analysis import oscillation
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)
DURATION_COORD = "duration_spin_locking"


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the Spin-Locking Amplitude vs. Length map (2D) for the given qubits.

    (All fit-related logic has been removed from the call structure.)

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters (unused, but kept for interface consistency).

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        # We only need the 2D plotting function since this experiment is a map.
        plot_individual_data_2D(ax, ds, qubit) # Calling simplified function

    # --- SIMPLIFICATION: Update Title ---
    grid.fig.suptitle("Spin-Locking Amplitude vs. Length Map")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


# --- plot_individual_data_with_fit_1D is REMOVED (no 1D fit/oscillation logic needed) ---
# --- Renamed and simplified the 2D plotting function ---

def plot_individual_data_2D(ax: Axes, ds: xr.Dataset, qubit: dict[str, str]):
    """
    Plots the 2D Spin-Locking map (Amplitude vs. Length) without any fit overlay.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    """
    y_coord = DURATION_COORD

    if hasattr(ds, "I"):
        data = "I"
    elif hasattr(ds, "state"):
        data = "state"
    else:
        raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")
        
    # Plot the data using absolute amplitude [mV]
    (ds.assign_coords(amp_mV=ds.full_amp * 1e3).loc[qubit])[data].plot(
        ax=ax, add_colorbar=True, x="amp_mV", y=y_coord, robust=True
    )
    
    # Update Y-axis label to reflect length/duration
    ax.set_ylabel(f"SL Pulse Length [{ds[y_coord].attrs.get('units', 'ns')}]") 
    ax.set_xlabel("Pulse amplitude [mV]")
    
    # Create twin axis for amplitude prefactor
    ax2 = ax.twiny()
    (ds.assign_coords(amp_mV=ds.amp_prefactor).loc[qubit])[data].plot(
        ax=ax2, add_colorbar=False, x="amp_mV", y=y_coord, robust=True
    )
    ax2.set_xlabel("amplitude prefactor")

    # --- SIMPLIFICATION: Removed all logic related to plotting axvline, opt_amp, and fit.success ---
    