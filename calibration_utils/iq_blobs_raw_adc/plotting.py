"""Plotting raw ADC traces - no demodulation."""

from typing import List
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from quam_builder.architecture.superconducting.qubit import AnyTransmon


def _get_raw_array(ds: xr.Dataset, prefix: str, num_qubits: int, qubit_names: list) -> xr.DataArray:
    """Get raw ADC array (Igraw, Qgraw, Ieraw, Qeraw) - supports consolidated or per-qubit naming."""
    if prefix in ds:
        return ds[prefix]
    arrays = [ds[f"{prefix}{i + 1}"] for i in range(num_qubits) if f"{prefix}{i + 1}" in ds]
    if len(arrays) == num_qubits:
        return xr.concat(arrays, dim=xr.DataArray(qubit_names, dims=["qubit"]))
    raise KeyError(f"Cannot find '{prefix}'. Available: {list(ds.data_vars)}")


def _fix_tuples(ds: xr.Dataset) -> xr.Dataset:
    """Extract values from tuples if present (from fetcher)."""
    def extract(e):
        return e[0] if isinstance(e, tuple) else e
    return xr.apply_ufunc(extract, ds, vectorize=True, dask="parallelized", output_dtypes=[float])


def plot_raw_adc_traces(ds: xr.Dataset, qubits: List[AnyTransmon]) -> plt.Figure:
    """
    Plot Igraw, Qgraw, Ieraw, Qeraw as a function of readout_time.
    No demodulation - raw ADC traces. Converts to V (-/2**12) for axis labels.
    Two rows per qubit: ground (Igraw, Qgraw vs t) and excited (Ieraw, Qeraw vs t).
    """
    ds = _fix_tuples(ds)
    num_qubits = len(qubits)
    qubit_names = [q.name for q in qubits]

    Igraw = _get_raw_array(ds, "Igraw", num_qubits, qubit_names)
    Qgraw = _get_raw_array(ds, "Qgraw", num_qubits, qubit_names)
    Ieraw = _get_raw_array(ds, "Ieraw", num_qubits, qubit_names)
    Qeraw = _get_raw_array(ds, "Qeraw", num_qubits, qubit_names)

    # Convert to V for display (OPX raw units -> V)
    scale = -1.0 / 2**12
    Igraw_v, Qgraw_v = Igraw * scale, Qgraw * scale
    Ieraw_v, Qeraw_v = Ieraw * scale, Qeraw * scale

    time_dim = [d for d in Igraw_v.dims if d not in ("qubit", "n_runs")][0]
    rows = num_qubits * 2
    cols = 1
    fig, axes = plt.subplots(rows, cols, sharex=True, figsize=(12, 3 * rows), squeeze=False)
    axes = axes.flatten()

    for i, q in enumerate(qubits):
        qname = q.name
        ig = Igraw_v.sel(qubit=qname).mean(dim="n_runs")
        qg = Qgraw_v.sel(qubit=qname).mean(dim="n_runs")
        ie = Ieraw_v.sel(qubit=qname).mean(dim="n_runs")
        qe = Qeraw_v.sel(qubit=qname).mean(dim="n_runs")
        t = ig[time_dim].values

        ax_g = axes[2 * i]
        ax_g.plot(t, 1e3 * ig, label="Igraw", color="C0")
        ax_g.plot(t, 1e3 * qg, label="Qgraw", color="C1")
        ax_g.set_ylabel("ADC [mV]")
        ax_g.set_title(f"{qname} – Ground (Igraw, Qgraw vs time)")
        ax_g.legend(loc="upper right", fontsize=8)
        ax_g.grid(True, alpha=0.3)

        ax_e = axes[2 * i + 1]
        ax_e.plot(t, 1e3 * ie, label="Ieraw", color="C2")
        ax_e.plot(t, 1e3 * qe, label="Qeraw", color="C3")
        ax_e.set_ylabel("ADC [mV]")
        ax_e.set_title(f"{qname} – Excited (Ieraw, Qeraw vs time)")
        ax_e.legend(loc="upper right", fontsize=8)
        ax_e.grid(True, alpha=0.3)

    axes[-1].set_xlabel(f"{time_dim} [ns]")
    fig.suptitle("Raw ADC traces vs readout time (mean over shots, no demodulation)")
    fig.tight_layout()
    return fig


def plot_iq_trajectories(
    ds: xr.Dataset,
    qubits: List[AnyTransmon],
    time_resolution_ns: int = 200,
) -> plt.Figure:
    """
    Plot I,Q trajectories on the IQ plane. At each readout time t, (I(t), Q(t)) is the
    mean over shots. The trajectory traces how the averaged IQ point evolves during readout.
    Uses time_resolution_ns to downsample (default 200 ns).
    """
    ds = _fix_tuples(ds)
    num_qubits = len(qubits)
    qubit_names = [q.name for q in qubits]

    Igraw = _get_raw_array(ds, "Igraw", num_qubits, qubit_names)
    Qgraw = _get_raw_array(ds, "Qgraw", num_qubits, qubit_names)
    Ieraw = _get_raw_array(ds, "Ieraw", num_qubits, qubit_names)
    Qeraw = _get_raw_array(ds, "Qeraw", num_qubits, qubit_names)

    # Convert to V for display
    scale = -1.0 / 2**12
    Igraw_v, Qgraw_v = Igraw * scale, Qgraw * scale
    Ieraw_v, Qeraw_v = Ieraw * scale, Qeraw * scale

    time_dim = [d for d in Igraw_v.dims if d not in ("qubit", "n_runs")][0]

    def _downsample(da: xr.DataArray) -> xr.DataArray:
        """Downsample to time_resolution_ns by averaging over time bins."""
        if time_resolution_ns <= 1:
            return da.mean(dim="n_runs")
        return (
            da.coarsen({time_dim: time_resolution_ns}, boundary="trim")
            .mean()
            .mean(dim="n_runs")
        )
    rows = num_qubits
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()

    for i, q in enumerate(qubits):
        qname = q.name

        # Downsample to time_resolution_ns and average over n_runs -> (I(t), Q(t)) trajectory
        ig = _downsample(Igraw_v.sel(qubit=qname))
        qg = _downsample(Qgraw_v.sel(qubit=qname))
        ie = _downsample(Ieraw_v.sel(qubit=qname))
        qe = _downsample(Qeraw_v.sel(qubit=qname))

        t = ig[time_dim].values

        # Ground trajectory subplot
        ax_g = axes[2 * i]
        sc_g = ax_g.scatter(1e3 * ig, 1e3 * qg, c=t, cmap="viridis", s=15)
        ax_g.plot(1e3 * ig, 1e3 * qg, "C0-", alpha=0.5, lw=1)
        ax_g.set_xlabel("I [mV]")
        ax_g.set_ylabel("Q [mV]")
        ax_g.set_title(f"{qname} – Ground IQ trajectory")
        ax_g.axis("equal")
        ax_g.grid(True, alpha=0.3)
        cbar_g = plt.colorbar(sc_g, ax=ax_g, shrink=0.6)
        cbar_g.set_label(f"{time_dim} [ns] ({time_resolution_ns} ns bins)")

        # Excited trajectory subplot
        ax_e = axes[2 * i + 1]
        sc_e = ax_e.scatter(1e3 * ie, 1e3 * qe, c=t, cmap="plasma", s=15)
        ax_e.plot(1e3 * ie, 1e3 * qe, "C1-", alpha=0.5, lw=1)
        ax_e.set_xlabel("I [mV]")
        ax_e.set_ylabel("Q [mV]")
        ax_e.set_title(f"{qname} – Excited IQ trajectory")
        ax_e.axis("equal")
        ax_e.grid(True, alpha=0.3)
        cbar_e = plt.colorbar(sc_e, ax=ax_e, shrink=0.6)
        cbar_e.set_label(f"{time_dim} [ns] ({time_resolution_ns} ns bins)")

    fig.suptitle(f"IQ trajectories: (I(t), Q(t)) averaged over shots, {time_resolution_ns} ns resolution")
    fig.tight_layout()
    return fig


__all__ = ["plot_raw_adc_traces", "plot_iq_trajectories"]
