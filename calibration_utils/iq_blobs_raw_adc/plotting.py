"""Plotting raw ADC traces."""

from typing import List
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from quam_builder.architecture.superconducting.qubit import AnyTransmon


def _get_raw_array(ds: xr.Dataset, prefix: str, num_qubits: int, qubit_names: list) -> xr.DataArray:
    """Get raw ADC array - supports consolidated or per-qubit naming."""
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
    Plot Igraw, Qgraw, Ieraw, Qeraw vs readout_time.
    Raw ADC traces from input1 real and image (ground and excited states).
    """
    num_qubits = len(qubits)
    qubit_names = [q.name for q in qubits]
    ds = _fix_tuples(ds)

    Igraw = _get_raw_array(ds, "Igraw", num_qubits, qubit_names)
    Qgraw = _get_raw_array(ds, "Qgraw", num_qubits, qubit_names)
    Ieraw = _get_raw_array(ds, "Ieraw", num_qubits, qubit_names)
    Qeraw = _get_raw_array(ds, "Qeraw", num_qubits, qubit_names)
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
        ax_g.plot(t, 1e3 * ig, label="Igraw (real)", color="C0")
        ax_g.plot(t, 1e3 * qg, label="Qgraw (imag)", color="C1")
        ax_g.set_ylabel("ADC [mV]")
        ax_g.set_title(f"{qname} – Ground (Igraw, Qgraw vs time)")
        ax_g.legend(loc="upper right", fontsize=8)
        ax_g.grid(True, alpha=0.3)
        ax_e = axes[2 * i + 1]
        ax_e.plot(t, 1e3 * ie, label="Ieraw (real)", color="C2")
        ax_e.plot(t, 1e3 * qe, label="Qeraw (imag)", color="C3")
        ax_e.set_ylabel("ADC [mV]")
        ax_e.set_title(f"{qname} – Excited (Ieraw, Qeraw vs time)")
        ax_e.legend(loc="upper right", fontsize=8)
        ax_e.grid(True, alpha=0.3)

    axes[-1].set_xlabel(f"{time_dim} [ns]")
    fig.suptitle("Raw ADC traces vs readout time (mean over shots)")
    fig.tight_layout()
    return fig


__all__ = ["plot_raw_adc_traces"]
