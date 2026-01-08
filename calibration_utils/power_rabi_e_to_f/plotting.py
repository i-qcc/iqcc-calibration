from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the power Rabi E->F oscillation with fitted curves for the given qubits.

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

    grid.fig.suptitle("EF Power Rabi: sqrt(I^2 + Q^2) vs. amplitude")
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
    # Plot the IQ_abs data
    (ds.assign_coords(amp_mV=ds.abs_amp * 1e3).loc[qubit].IQ_abs * 1e3).plot(
        ax=ax, x="amp_mV"
    )
    
    # Plot the fitted oscillation if available
    # fit is already selected for a single qubit via fits.sel(qubit=qubit["qubit"])
    if fit is not None and "fit_evals" in fit:
        # fit_evals should have the same amp_prefactor dimension as ds
        # After fits.sel(qubit=...), fit_evals should already be for that qubit
        # But it might still have qubit as a dimension (not coordinate), so we need to handle it
        fit_evals_data = fit.fit_evals
        # If fit_evals still has qubit dimension, use isel to select the first (and only) qubit
        if "qubit" in fit_evals_data.dims:
            # After selection, there should only be one qubit left, so use index 0
            fit_evals_data = fit_evals_data.isel(qubit=0)
        ax.plot(ds.abs_amp.loc[qubit] * 1e3, 1e3 * fit_evals_data.values, label="Fit")
    
    ax.set_ylabel("Trans. amp. [mV]")
    ax.set_xlabel("Amplitude [mV]")
    ax.set_title(qubit["qubit"])

