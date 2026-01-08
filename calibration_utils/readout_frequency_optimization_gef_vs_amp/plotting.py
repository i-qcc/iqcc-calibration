from typing import List
import xarray as xr
from matplotlib.figure import Figure

from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def plot_distances_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the distance D between G, E, F states as a function of frequency and amplitude.

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
        plot_individual_distances(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Maximal difference between g, e, f resonance")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_distances(ax, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """Plot individual qubit distance data."""
    (ds.assign_coords(freq_MHz=ds.detuning / 1e6).loc[qubit].D).plot(
        ax=ax, x="freq_MHz", y="amp_prefactor", label="D"
    )
    if fit is not None:
        ax.plot(
            fit.optimal_detuning.values / 1e6,
            fit.optimal_amp.values,
            "ro",
            label="Optimal",
            markersize=10,
        )
    ax.set_xlabel("R/O Freq. [MHz]")
    ax.set_ylabel("relative Drive Amp.")
    ax.set_title(qubit["qubit"])


def plot_IQ_abs_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the IQ amplitude for G, E, F states at optimal amplitude.

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
        plot_individual_IQ_abs(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Resonator response for g, e, f states")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_IQ_abs(ax, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """Plot individual qubit IQ amplitude data at optimal amplitude."""
    if fit is not None:
        best_ds = ds.sel(amp_prefactor=fit.optimal_amp.values, method="nearest")
        (1e3 * best_ds.assign_coords(freq_MHz=best_ds.detuning / 1e6).loc[qubit].IQ_abs_g).plot(
            ax=ax, x="freq_MHz", label="g.s."
        )
        (1e3 * best_ds.assign_coords(freq_MHz=best_ds.detuning / 1e6).loc[qubit].IQ_abs_e).plot(
            ax=ax, x="freq_MHz", label="e.s."
        )
        (1e3 * best_ds.assign_coords(freq_MHz=best_ds.detuning / 1e6).loc[qubit].IQ_abs_f).plot(
            ax=ax, x="freq_MHz", label="f.s."
        )
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Resonator response [mV]")
    ax.set_title(qubit["qubit"])
    ax.legend()


def plot_optimal_parameters(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the distances Dge, Def, Dgf at optimal amplitude.

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
        plot_individual_optimal_distances(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Distances between g, e, f states at optimal parameters")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_optimal_distances(ax, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """Plot individual qubit distance data at optimal amplitude."""
    if fit is not None:
        best_ds = ds.sel(amp_prefactor=fit.optimal_amp.values, method="nearest")
        (1e3 * best_ds.assign_coords(freq_MHz=best_ds.detuning / 1e6).loc[qubit].Dge).plot(
            ax=ax, x="freq_MHz", label="GE"
        )
        (1e3 * best_ds.assign_coords(freq_MHz=best_ds.detuning / 1e6).loc[qubit].Def).plot(
            ax=ax, x="freq_MHz", label="EF"
        )
        (1e3 * best_ds.assign_coords(freq_MHz=best_ds.detuning / 1e6).loc[qubit].Dgf).plot(
            ax=ax, x="freq_MHz", label="GF"
        )
        (1e3 * best_ds.assign_coords(freq_MHz=best_ds.detuning / 1e6).loc[qubit].D).plot(
            ax=ax, x="freq_MHz", label="D (min)"
        )
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Distance between IQ blobs [mV]")
    ax.set_title(qubit["qubit"])
    ax.legend()

