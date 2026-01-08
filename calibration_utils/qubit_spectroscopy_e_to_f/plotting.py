from typing import List
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qualang_tools.units import unit
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def plot_raw_data_with_fit(ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset):
    """
    Plots the qubit spectroscopy E->F transition amplitude I with fitted curves for the given qubits.

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
    - Each subplot contains the raw data and a vertical line marking the anharmonicity.
    """
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        plot_individual_data_with_fit(ax, ds, qubit, fits.sel(qubit=qubit["qubit"]))

    grid.fig.suptitle("Qubit spectroscopy (E->F transition)")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig


def plot_individual_data_with_fit(ax: Axes, ds: xr.Dataset, qubit: dict[str, str], fit: xr.Dataset = None):
    """
    Plots individual qubit data on a given axis with optional fit marker.

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
    - If the fit dataset is provided, a vertical line is plotted at the anharmonicity position.
    """
    # Plot the I quadrature as a function of detuning_ef (E->F detuning)
    (ds.assign_coords(detuning_MHz=ds.detuning_ef / u.MHz).loc[qubit].I * 1e3).plot(
        ax=ax, x="detuning_MHz"
    )
    ax.set_xlabel("Detuning [MHz]")
    ax.set_ylabel("Trans. amp. [mV]")
    ax.set_title(qubit["qubit"])
    
    # Add a vertical line where the E->F transition is found (anharmonicity)
    if fit is not None and not xr.ufuncs.isnan(fit.position.values):
        # Get the anharmonicity from the detuning_ef at the peak position
        peak_pos = float(fit.position.values)
        anharmonicity = fit.detuning_ef.sel(detuning=peak_pos).values
        ax.axvline(
            anharmonicity / 1e6,
            color="r",
            linestyle="--",
            label="E->F transition"
        )

