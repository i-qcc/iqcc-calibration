from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from qualibration_libs.analysis import lorentzian_peak

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.

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

    grid.fig.suptitle("Ramsey vs flux")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_parabolas_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset, outcomes: dict = None):
    """
    Plots the resonator spectroscopy amplitude IQ_abs with fitted curves for the given qubits.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the quadrature data.
    qubits : list of AnyTransmon
        A list of qubits to plot.
    fits : xr.Dataset
        The dataset containing the fit parameters.
    outcomes : dict, optional
        Dictionary containing outcomes ("successful" or "failed") for each qubit.

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
    
    # Get the flux_bias range from the actual data (not from fitted line)
    frequency = fits.sel(fit_vals="f").fit_results
    flux_bias_all = frequency.flux_bias
    flux_min = float(flux_bias_all.min().values)
    flux_max = float(flux_bias_all.max().values)
    
    for ax, qubit in grid_iter(grid):
        plot_individual_parabolas_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]), outcomes)
        # Set consistent x-axis range for all subplots based on actual data range
        ax.set_xlim(flux_min, flux_max)

    grid.fig.suptitle("Ramsey vs flux frequency ")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    
    # Create legend entries manually (xarray plot doesn't create proper handles)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='blue', marker='.', linestyle='', markersize=8, label='Data'),
        Line2D([0], [0], color='orange', linestyle='-', linewidth=2, label='Parabolic fit'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='Flux offset'),
        Line2D([0], [0], color='green', linestyle='--', linewidth=1, label='Detuning from SweetSpot'),
        Line2D([0], [0], color='red', marker='x', linestyle='', markersize=6, markeredgewidth=2, label='Fit failed'),
    ]
    grid.fig.legend(legend_handles, [h.get_label() for h in legend_handles], 
                    loc='upper right', bbox_to_anchor=(0.98, 0.98),
                    frameon=True, fancybox=True, shadow=True, fontsize=9)
    
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

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """

    ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax)
    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Idle_time (uS)")
    ax.set_ylabel(" Flux (V)")

    flux_offset = fit.flux_offset

    ax.axhline(flux_offset, color="red", linestyle="--", label="Flux offset")


def plot_individual_parabolas_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None, outcomes: dict = None):
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
    outcomes : dict, optional
        Dictionary containing outcomes ("successful" or "failed") for each qubit.

    Notes
    -----
    - If the fit dataset is provided, the fitted curve is plotted along with the raw data.
    """
    detuning = float(fit.artifitial_detuning.values)  # Extract scalar value

    # Get the frequency data and flux bias values
    frequency = fit.sel(fit_vals="f").fit_results
    flux_bias = frequency.flux_bias
    
    # Plot the data points
    (frequency * 1e3 - detuning).plot(ax=ax, linestyle="", marker=".", label="Data", color="blue")
    
    # Reconstruct polynomial coefficients from stored fit parameters
    # From analysis.py: quad_term = -1e6 * c2, so c2 = -quad_term / 1e6 (in MHz/V^2)
    # flux_offset = -0.5 * c1 / c2, so c1 = -2 * flux_offset * c2 (in MHz/V)
    # At flux_offset, freq = freq_offset, so we can solve for c0
    quad_term = float(fit.quad_term.values)  # in MHz/V^2
    flux_offset_val = float(fit.flux_offset.values)  # in V
    freq_offset_val = float(fit.freq_offset.values)  # in Hz
    
    c2 = -quad_term / 1e6  # Convert to MHz/V^2
    c1 = -2 * flux_offset_val * c2  # in MHz/V
    # At flux_offset: freq_offset = c2*flux_offset^2 + c1*flux_offset + c0
    # So: c0 = freq_offset - c2*flux_offset^2 - c1*flux_offset
    c0 = freq_offset_val / 1e6 - c2 * flux_offset_val**2 - c1 * flux_offset_val  # in MHz
    
    # Create smooth flux range for plotting
    flux_min = float(flux_bias.min().values)
    flux_max = float(flux_bias.max().values)
    flux_smooth = np.linspace(flux_min, flux_max, 200)
    
    # Evaluate the parabola: y = c2*x^2 + c1*x + c0 (all in MHz)
    freq_smooth = c2 * flux_smooth**2 + c1 * flux_smooth + c0
    
    # Plot the smooth parabola (convert to MHz and subtract detuning)
    ax.plot(flux_smooth, (freq_smooth * 1e3 - detuning), 
            linestyle="-", color="orange", linewidth=2, label="Parabolic fit")
    
    ax.set_title(qubit["qubit"])
    ax.set_xlabel("Flux offset (V)")
    ax.set_ylabel("detuning (MHz)")

    flux_offset = float(fit.flux_offset.values)

    ax.axvline(flux_offset, color="red", linestyle="--", label="Flux offset")

    freq_offset = float(fit.freq_offset.values) * 1e-3 - detuning

    ax.axhline(freq_offset, color="green", linestyle="--", label="Detuning from SweetSpot")
    
    # Add red x in top right if fit failed
    if outcomes is not None and qubit["qubit"] in outcomes:
        if outcomes[qubit["qubit"]] == "failed":
            # Place x marker in top right corner using axes coordinates (0,0 = bottom left, 1,1 = top right)
            ax.scatter([1.0], [1.0], marker='x', s=100, c='red', linewidths=2, transform=ax.transAxes, clip_on=False, zorder=10)
