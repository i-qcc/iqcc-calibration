from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import oscillation
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)
# Define the new duration coordinate name for cleaner code
DURATION_COORD = "duration_spin_locking"


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the Spin-Locking Amplitude vs. Length map (2D) or an Amplitude sweep (1D).

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        # Check if the experiment is 1D (Amplitude sweep, Duration is constant) or 2D
        if DURATION_COORD in ds.coords and len(ds[DURATION_COORD]) == 1:
            plot_individual_data_with_fit_1D(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))
        else:
            plot_individual_data_with_fit_2D(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    # --- IMPROVEMENT: Update Title ---
    grid.fig.suptitle("Spin-Locking Amplitude vs. Length Map")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit_1D(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis with optional fit (for 1D Amplitude sweep).

    NOTE: The core logic (oscillation fit) is kept but the dimension check is updated.
    """
    # --- IMPROVEMENT: Check new coordinate DURATION_COORD for 1D logic ---
    if DURATION_COORD in ds.coords and len(ds[DURATION_COORD].data) == 1: 
        if fit:
            # NOTE: This oscillation fit function is designed for Rabi, not necessarily SL-T2.
            fitted_data = oscillation(
                fit.amp_prefactor.data,
                fit.fit.sel(fit_vals="a").data,
                fit.fit.sel(fit_vals="f").data,
                fit.fit.sel(fit_vals="phi").data,
                fit.fit.sel(fit_vals="offset").data,
            )
        else:
            fitted_data = None

        if hasattr(ds, "I"):
            data = "I"
            label = "Rotated I quadrature [mV]"
        elif hasattr(ds, "state"):
            data = "state"
            label = "Qubit state"
        else:
            raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")

        (ds.assign_coords(amp_mV=ds.full_amp * 1e3).loc[qubit] * 1e3)[data].plot(ax=ax, x="amp_mV")
        
        if fitted_data is not None:
             ax.plot(fit.full_amp * 1e3, 1e3 * fitted_data, label="Rabi Fit", color='orange')
        
        # Plot optimal amplitude
        if fit is not None and "opt_amp" in fit and fit.success.values.__bool__():
             ax.axvline(x=fit.opt_amp * 1e3, color="g", linestyle="--", label="Optimal Amp")
             ax.legend()
             
        ax.set_ylabel(label)
        ax.set_xlabel("Pulse amplitude [mV]")
        ax2 = ax.twiny()
        (ds.assign_coords(amp_mV=ds.amp_prefactor).loc[qubit] * 1e3)[data].plot(ax=ax2, x="amp_mV")
        ax2.set_xlabel("amplitude prefactor")


def plot_individual_data_with_fit_2D(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots the 2D Spin-Locking map (Amplitude vs. Length).
    """
    y_coord = DURATION_COORD

    if hasattr(ds, "I"):
        data = "I"
    elif hasattr(ds, "state"):
        data = "state"
    else:
        raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")
        
    # --- IMPROVEMENT: Use the new duration coordinate for the y-axis ---
    (ds.assign_coords(amp_mV=ds.full_amp * 1e3).loc[qubit])[data].plot(
        ax=ax, add_colorbar=False, x="amp_mV", y=y_coord, robust=True
    )
    
    # --- IMPROVEMENT: Update Y-axis label to reflect length/duration ---
    # Get units from the xarray coordinate metadata if available, otherwise default to 'ns'
    ax.set_ylabel(f"SL Pulse Length [{ds[y_coord].attrs.get('units', 'ns')}]") 
    ax.set_xlabel("Pulse amplitude [mV]")
    
    ax2 = ax.twiny()
    # --- IMPROVEMENT: Use the new duration coordinate for the y-axis ---
    (ds.assign_coords(amp_mV=ds.amp_prefactor).loc[qubit])[data].plot(
        ax=ax2, add_colorbar=False, x="amp_mV", y=y_coord, robust=True
    )
    ax2.set_xlabel("amplitude prefactor")
    
    # Plot the optimal amplitude found in the fit (vertical line)
    if fit is not None and "success" in fit and fit.success.values.__bool__():
        opt_amp_prefactor = fit.opt_amp_prefactor.values
        ax2.axvline(
            x=opt_amp_prefactor,
            color="g",
            linestyle="-",
            label=f"Optimal Amp: {opt_amp_prefactor.__float__():.3f}"
        )
        ax2.legend(loc='lower right')