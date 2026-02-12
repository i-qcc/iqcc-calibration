from typing import List
import xarray as xr
import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import decay_exp
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset, multiplexed: bool = None, reset_type: str = None, sweep_type: str = None):
    """
    Plots the T2 echo raw data with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.
    multiplexed : bool, optional
        Whether multiplexed readout was used.
    reset_type : str, optional
        The reset type used (e.g., "active" or "thermal").
    sweep_type : str, optional
        The sweep type used (e.g., "log" or "linear").

    Returns
    -------
    Figure
        The matplotlib figure object containing the plots.

    Notes
    -----
    - The function creates a grid of subplots, one for each qubit.
    - Each subplot shows the raw data as a 2D plot (idle_time vs flux).
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    title = "T2 echo vs flux"
    grid.fig.suptitle(title)
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_T2_vs_flux(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots T2 echo vs flux for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the data.
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
        plot_individual_T2_vs_flux(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    title = "T2 echo vs flux"
    grid.fig.suptitle(title)
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis.

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
    """
    qubit_name = qubit["qubit"]
    qubit_data = ds.sel(qubit=qubit_name)
    
    # Determine which data variable to plot
    if hasattr(qubit_data, "state"):
        data_to_plot = qubit_data.state
    elif hasattr(qubit_data, "I"):
        data_to_plot = qubit_data.I * 1e3  # Convert to mV
    else:
        raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")
    
    # --- pcolormesh expects C shaped as (len(y), len(x)) ---
    # Here we want x=idle_time and y=flux, so C must be (flux, idle_time).
    if "flux" in data_to_plot.dims and "idle_time" in data_to_plot.dims:
        da_plot = data_to_plot.transpose("flux", "idle_time")
        flux_values = da_plot.flux.values
        idle_time_values = da_plot.idle_time.values
        c = da_plot.values
    elif "flux_bias" in data_to_plot.dims and "idle_time" in data_to_plot.dims:
        da_plot = data_to_plot.transpose("flux_bias", "idle_time")
        flux_values = da_plot.flux_bias.values
        idle_time_values = da_plot.idle_time.values
        c = da_plot.values
    else:
        raise RuntimeError("Expected dimensions ('idle_time', 'flux') or ('idle_time', 'flux_bias') in dataset.")

    # Use 'nearest' to avoid edge-array size requirements of 'flat'
    im = ax.pcolormesh(idle_time_values, flux_values, c, shading="nearest")
    plt.colorbar(im, ax=ax)
    
    ax.set_title(qubit_name)
    ax.set_xlabel("Idle time (ns)")
    ax.set_ylabel("Flux (V)")


def plot_individual_T2_vs_flux(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots T2 echo vs flux for individual qubit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    ds : xr.Dataset
        The dataset containing the data.
    qubit : dict[str, str]
        mapping to the qubit to plot.
    fit : xr.Dataset, optional
        The dataset containing the fit parameters (default is None).
    """
    qubit_name = qubit["qubit"]
    
    # Get T2 echo values
    T2_echo = fit["T2_echo"]
    T2_echo_error = fit["T2_echo_error"]
    
    # Get flux values
    if "flux" in T2_echo.dims:
        flux_values = T2_echo.flux.values
    elif "flux_bias" in T2_echo.dims:
        flux_values = T2_echo.flux_bias.values
    else:
        raise RuntimeError("The dataset must contain 'flux' or 'flux_bias' dimension.")
    
    # Convert to microseconds for plotting
    # Default in this repo: T2_echo is stored in ns in the dataset.
    units = (getattr(T2_echo, "attrs", {}) or {}).get("units", "ns")
    if units == "ns":
        scale = 1e-3  # ns -> µs
    elif units in ("s", "sec", "seconds"):
        scale = 1e6  # s -> µs
    else:
        # Fallback assume ns
        scale = 1e-3

    T2_echo_us = T2_echo.values * scale
    T2_echo_error_us = T2_echo_error.values * scale
    
    # Plot with error bars
    ax.errorbar(flux_values, T2_echo_us, yerr=T2_echo_error_us, fmt='o-', capsize=5, markersize=4)
    
    ax.set_title(qubit_name)
    ax.set_xlabel("Flux (V)")
    ax.set_ylabel("T2 echo (µs)")
    ax.grid(True, alpha=0.3)
