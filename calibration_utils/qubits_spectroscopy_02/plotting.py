from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import lorentzian_peak
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the qubit spectroscopy 0->2 transition amplitude I_rot with fitted curves for the given qubits.

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

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot contains the raw data and the fitted curve.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Qubit spectroscopy 0->2 transition (rotated 'I' quadrature + fit)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis with optional fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).
        Note: fit is already selected for a single qubit via fits.sel(qubit=qubit["qubit"])

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """
    if fit is not None and not xr.ufuncs.isnan(fit.position.values):
        fitted_data = lorentzian_peak(
            fit.detuning,
            float(fit.amplitude.values),
            float(fit.position.values),
            float(fit.width.values) / 2,
            float(fit.base_line.mean().values),
        )
    else:
        fitted_data = None

    # Create a first x-axis for full_freq_GHz
    # Use fit dataset which contains I_rot (created during fitting)
    # fit is already a single qubit dataset after .sel(qubit=...)
    (fit.assign_coords(full_freq_GHz=fit.full_freq / u.GHz).I_rot / u.mV).plot(ax=ax, x="full_freq_GHz")
    ax.set_xlabel("Qubit freq [GHz]")
    ax.set_ylabel("Trans. amp. [mV]")
    ax.set_title(f"{qubit['qubit']}")
    
    # Create a second x-axis for detuning_MHz
    ax2 = ax.twiny()
    (fit.assign_coords(detuning_MHz=fit.detuning / u.MHz).I_rot / u.mV).plot(ax=ax2, x="detuning_MHz", label="")
    ax2.set_xlabel("Detuning [MHz]")
    
    # Plot the fitted data
    if fitted_data is not None:
        ax2.plot(fit.detuning / u.MHz, fitted_data / u.mV, "r--", linewidth=0.5)
        # Mark the peak position
        peak_pos = float(fit.position.values)
        peak_freq = fit.full_freq.sel(detuning=peak_pos, method="nearest").values
        peak_amp = fit.sel(detuning=peak_pos, method="nearest").I_rot.values
        ax.plot(peak_freq / 1e9, peak_amp / 1e-3, ".r", markersize=10)

