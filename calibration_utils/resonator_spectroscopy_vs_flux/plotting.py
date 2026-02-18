from typing import List
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the raw data with fitted curves for the given qubits.

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
        plot_individual_raw_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    # Add a single shared legend for the whole figure
    # Get handles and labels from the first axis that has them
    handles, labels = [], []
    for ax in grid.fig.axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        grid.fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.99))

    grid.fig.suptitle("Resonator spectroscopy vs flux")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_raw_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
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

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """

    ax2 = ax.twiny()
    # Plot using the attenuated current x-axis
    ds.assign_coords(freq_GHz=ds.full_freq / 1e9).loc[qubit].IQ_abs.plot(
        ax=ax2,
        add_colorbar=False,
        x="attenuated_current",
        y="freq_GHz",
        robust=True,
    )
    ax2.set_xlabel("Current (A)")
    ax2.set_ylabel("Freq (GHz)")
    ax2.set_title("")
    # Move ax2 behind ax
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    # Plot using the flux x-axis
    qubit_ds = ds.loc[qubit]
    ds.assign_coords(freq_GHz=ds.full_freq / 1e9).loc[qubit].IQ_abs.plot(
        ax=ax, add_colorbar=False, x="flux_bias", y="freq_GHz", robust=True
    )
    
    # Get RF frequency (full_freq = detuning + RF_frequency)
    RF_frequency = qubit_ds.full_freq.isel(detuning=0).values - ds.detuning.values[0]
    
    # Plot peak frequency data points
    peak_freq = qubit_ds.IQ_abs.idxmin(dim="detuning").dropna(dim="flux_bias")
    ax.plot(
        peak_freq.flux_bias.values,
        (peak_freq + RF_frequency).values * 1e-9,
        "o",
        color="yellow",
        markersize=4,
        label="peak freq data",
        zorder=6,
        alpha=0.7,
    )
    
    if fit.fit_results.success.values:
        # Extract fit parameters and calculate cosine fit
        fit_params = {k: fit.fit_results.sel(fit_vals=k).values for k in ["a", "f", "phi", "offset"]}
        flux_bias_smooth = np.linspace(ds.flux_bias.values.min(), ds.flux_bias.values.max(), 200)
        fit_detuning = fit_params["a"] * np.cos(2 * np.pi * fit_params["f"] * flux_bias_smooth + fit_params["phi"]) + fit_params["offset"]
        
        # Plot cosine fit curve
        ax.plot(
            flux_bias_smooth,
            (fit_detuning + RF_frequency) * 1e-9,
            color="cyan",
            linestyle="-",
            linewidth=2,
            label="cos fit",
            zorder=5,
        )
        
        ax.axvline(
            fit.fit_results.idle_offset,
            linestyle="dashed",
            linewidth=2,
            color="r",
            label="idle offset",
        )
        ax.axvline(
            fit.fit_results.flux_min,
            linestyle="dashed",
            linewidth=2,
            color="orange",
            label="min offset",
        )
        # Location of the current resonator frequency
        ax.plot(
            fit.fit_results.idle_offset.values,
            fit.fit_results.sweet_spot_frequency.values * 1e-9,
            "r*",
            markersize=10,
        )
    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Flux (V)")
