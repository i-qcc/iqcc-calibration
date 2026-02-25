"""
Analysis for IQ blobs from raw ADC traces.

Demodulates raw ADC traces (I and Q components over time) to produce
Ig, Qg, Ie, Qe per shot, then reuses the standard IQ blobs fitting logic.
"""

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode

from calibration_utils.iq_blobs.analysis import fit_raw_data as iq_blobs_fit_raw_data
from qualibration_libs.data import convert_IQ_to_V


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Process raw ADC dataset: convert to Volts, demodulate traces to get I,Q per shot,
    and structure as Ig, Qg, Ie, Qe for compatibility with iq_blobs analysis.

    Expects dataset with Ig_raw1, Qg_raw1, Ie_raw1, Qe_raw1, etc. (one per qubit)
    with shape (n_runs, readout_time). Demodulation uses mean over the readout window.
    """
    qubits = node.namespace["qubits"]
    num_qubits = len(qubits)
    qubit_names = [q.name for q in qubits]

    # Fix tuples in data if present
    def extract_value(element):
        if isinstance(element, tuple):
            return element[0]
        return element

    ds = xr.apply_ufunc(
        extract_value,
        ds,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    # XarrayDataFetcher uses extract_string: "Igraw1" -> "Igraw", creating variable "Igraw"
    # Support both Igraw (from Igraw1, Igraw2) and legacy Ig_raw / Ig_raw1
    def get_raw_array(prefixes: tuple) -> xr.DataArray:
        """Get raw ADC array. Tries multiple possible variable names."""
        for prefix in prefixes:
            if prefix in ds:
                return ds[prefix]
            # Per-qubit: Igraw1, Igraw2, ... (in order)
            arrays = [ds[f"{prefix}{i + 1}"] for i in range(num_qubits) if f"{prefix}{i + 1}" in ds]
            if len(arrays) == num_qubits:
                return xr.concat(arrays, dim=xr.DataArray(qubit_names, dims=["qubit"]))
        raise KeyError(
            f"Cannot find raw ADC variables {prefixes}. "
            f"Available data_vars: {list(ds.data_vars)}. "
            f"Ensure QUA saves Igraw1, Qgraw1, Ieraw1, Qeraw1 (or Ig_raw1, etc.)."
        )

    Ig_raw_arr = get_raw_array(("Igraw", "Ig_raw"))
    Qg_raw_arr = get_raw_array(("Qgraw", "Qg_raw"))
    Ie_raw_arr = get_raw_array(("Ieraw", "Ie_raw"))
    Qe_raw_arr = get_raw_array(("Qeraw", "Qe_raw"))

    # Demodulate: average over readout_time dimension
    time_dims = [d for d in Ig_raw_arr.dims if d not in ("qubit", "n_runs")]
    time_dim = time_dims[0] if time_dims else None
    if time_dim:
        Ig_raw = Ig_raw_arr.mean(dim=time_dim)
        Qg_raw = Qg_raw_arr.mean(dim=time_dim)
        Ie_raw = Ie_raw_arr.mean(dim=time_dim)
        Qe_raw = Qe_raw_arr.mean(dim=time_dim)
    else:
        Ig_raw, Qg_raw, Ie_raw, Qe_raw = Ig_raw_arr, Qg_raw_arr, Ie_raw_arr, Qe_raw_arr

    # Convert raw ADC to Volts: OPX outputs in 2**12 units, -/2**12 gives V
    Ig = -Ig_raw / 2**12
    Qg = -Qg_raw / 2**12
    Ie = -Ie_raw / 2**12
    Qe = -Qe_raw / 2**12

    # Scale to match standard IQ blobs (convert_IQ_to_V: val * 2**12/length -> V)
    # Our mean is already in V. To match: we need val such that val * 2**12/length = mean_V
    # So val = mean_V * length / 2**12. Our mean_V = -mean_raw/2**12.
    # So val = (-mean_raw/2**12) * length / 2**12 = -mean_raw * length / 2**24
    # Actually convert_IQ_to_V multiplies by 2**12/length. So input val = V * length / 2**12.
    readout_lengths = xr.DataArray(
        [q.resonator.operations["readout"].length for q in qubits],
        coords=[("qubit", qubit_names)],
    )
    ds = ds.assign(
        Ig=Ig * readout_lengths / 2**12,
        Qg=Qg * readout_lengths / 2**12,
        Ie=Ie * readout_lengths / 2**12,
        Qe=Qe * readout_lengths / 2**12,
    )
    ds = convert_IQ_to_V(ds, qubits, IQ_list=["Ig", "Qg", "Ie", "Qe"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode):
    """Reuse the standard iq_blobs fitting after process_raw_dataset."""
    return iq_blobs_fit_raw_data(ds, node)


def log_fitted_results(fit_results: dict, log_callable=None):
    """Reuse from iq_blobs."""
    from calibration_utils.iq_blobs.analysis import log_fitted_results as _log

    return _log(fit_results, log_callable)
