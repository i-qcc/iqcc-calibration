from typing import List
import xarray as xr
import matplotlib.lines as mlines
from matplotlib.axes import Axes

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from qualibration_libs.analysis import oscillation_decay_exp
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset, fit_results: dict = None):
    """
    Plots the Ramsey oscillation data with fitted curves for the given qubits.

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
        plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]), fit_results[qubit["qubit"]])

    # Create a single global legend for the entire figure
    handles, labels = [], []
    for ax in grid.fig.axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if not handles:
        # Fallback: proxy artists when get_legend_handles_labels is empty (e.g. xarray)
        handles = [
            mlines.Line2D([], [], color="C0", marker=".", ms=5.0, ls="", label=r"$\Delta$ = +"),
            mlines.Line2D([], [], color="C1", marker=".", ms=5.0, ls="", label=r"$\Delta$ = -"),
        ]
        labels = [h.get_label() for h in handles]

    grid.fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.04),
                    ncol=2, frameon=True, fontsize=12)

    grid.fig.suptitle("Ramsey (rotated 'I' quadrature + fit)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout(rect=[0, 0.10, 1, 0.98])
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None, fit_results: dict = None):
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
    if fit:
        fitted_ramsey_data = oscillation_decay_exp(
            ds.idle_time,
            fit.sel(fit_vals="a"),
            fit.sel(fit_vals="f"),
            fit.sel(fit_vals="phi"),
            fit.sel(fit_vals="offset"),
            fit.sel(fit_vals="decay"),
        )
    else:
        fitted_ramsey_data = None

    if hasattr(fit, "state"):
        plot_state(ax, fit, fitted_ramsey_data)
        ax.set_ylabel("State Population")
    elif hasattr(fit, "I"):
        plot_transmission_amplitude(ax, fit, fitted_ramsey_data)
        ax.set_ylabel("Trans. amp. I [mV]")
    else:
        raise RuntimeError("The dataset must contain either 'I' or 'state' for the plotting function to work.")

    ax.set_xlabel("Idle time [ns]")
    ax.set_title(qubit["qubit"])
    if fit is not None:
        add_fit_text(ax, fit_results)


def plot_state(ax, ds, fitted=None):
    """Plot state data for a qubit."""
    ds.sel(detuning_signs=1).state.plot(ax=ax, x="idle_time", c="C0", marker=".", ms=5.0, ls="", label=r"$\Delta$ = +")
    ds.sel(detuning_signs=-1).state.plot(ax=ax, x="idle_time", c="C1", marker=".", ms=5.0, ls="", label=r"$\Delta$ = -")
    if fitted is not None:
        ax.plot(
            ds.idle_time,
            fitted.fit.sel(detuning_signs=1),
            c="C0",
            ls="-",
            lw=1,
        )
        ax.plot(
            ds.idle_time,
            fitted.fit.sel(detuning_signs=-1),
            c="C1",
            ls="-",
            lw=1,
        )


def plot_transmission_amplitude(ax, ds, fitted=None):
    """Plot transmission amplitude for a qubit."""
    (ds.sel(detuning_signs=1).I * 1e3).plot(
        ax=ax, x="idle_time", c="C0", marker=".", ms=5.0, ls="", label=r"$\Delta$ = +"
    )
    (ds.sel(detuning_signs=-1).I * 1e3).plot(
        ax=ax, x="idle_time", c="C1", marker=".", ms=5.0, ls="", label=r"$\Delta$ = -"
    )
    if fitted is not None:
        ax.plot(ds.idle_time, 1e3 * fitted.fit.sel(detuning_signs=1), c="C0", ls="-", lw=1)
        ax.plot(ds.idle_time, 1e3 * fitted.fit.sel(detuning_signs=-1), c="C1", ls="-", lw=1)


def add_fit_text(ax, fit_results):
    """Add fit results text to the axis."""
    ax.text(
        0.9,
        0.9,
        f"T2* = {1e6 * fit_results['decay']:.1f} ± {1e6 * fit_results['decay_error']:.1f} µs",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.5),
    )
