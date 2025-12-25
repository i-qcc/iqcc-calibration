from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import fit_oscillation_decay_exp, oscillation_decay_exp


@dataclass
class FitParameters:
    """Stores the relevant Ramsey vs flux experiment fit parameters for a single qubit"""

    success: bool
    quad_term: float
    flux_offset: float
    target_offset: float
    freq_offset: float
    t2_star: np.ndarray
    fit_norm: float


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Logs the node-specific fitted results for all qubits from the fit results."""
    pass


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit frequency and FWHM for each qubit in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """
    # # TODO: explain the data analysis
    fit_data = fit_oscillation_decay_exp(ds.state, "idle_times")
    fit_data.attrs = {"long_name": "time", "units": "µs"}
    fitted = oscillation_decay_exp(
        ds.state.idle_times,
        fit_data.sel(fit_vals="a"),
        fit_data.sel(fit_vals="f"),
        fit_data.sel(fit_vals="phi"),
        fit_data.sel(fit_vals="offset"),
        fit_data.sel(fit_vals="decay"),
    )

    frequency = fit_data.sel(fit_vals="f")
    # NOTE: frequency is in GHz (not MHz as attrs might say)
    frequency.attrs = {"long_name": "frequency", "units": "GHz"}

    decay = fit_data.sel(fit_vals="decay")
    decay.attrs = {"long_name": "decay", "units": "nSec"}

    tau = 1 / fit_data.sel(fit_vals="decay")
    tau.attrs = {"long_name": "T2*", "units": "uSec"}

    frequency = frequency.where(frequency > 0, drop=True)

    # Polyfit on frequency (GHz) vs flux_bias (V)
    # Results: c0 in GHz, c1 in GHz/V, c2 in GHz/V^2
    fitvals = frequency.polyfit(dim="flux_bias", deg=2)
    flux = frequency.flux_bias

    # Constants for success determination
    SUCCESS_THRESHOLD = 0.005
    MIN_FREQ_RANGE_MHZ = 500000e-9  # 500 nHz in MHz

    a = {}
    flux_offset = {}
    freq_offset = {}
    target_offset = {}
    t2_star = {}
    fit_norm = {}
    freq_range = {}
    success_dict = {}

    qubits = ds.qubit.values

    for q in qubits:
        # Extract polynomial coefficients for this qubit
        # c0, c1, c2 are in GHz, GHz/V, GHz/V^2 respectively
        coeffs = fitvals.sel(qubit=q)
        c0 = float(coeffs.sel(degree=0).polyfit_coefficients.values)  # GHz
        c1 = float(coeffs.sel(degree=1).polyfit_coefficients.values)  # GHz/V
        c2 = float(coeffs.sel(degree=2).polyfit_coefficients.values)  # GHz/V^2
        
        # Get observed data for this qubit
        freq_q = frequency.sel(qubit=q)
        flux_q = freq_q.flux_bias
        
        # Calculate frequency range (max - min) in GHz
        freq_range[q] = float(freq_q.max() - freq_q.min())
        
        # Calculate fitted values and fit norm (excluding 10% of points with largest residuals)
        freq_fitted = c2 * flux_q**2 + c1 * flux_q + c0
        diff = freq_q - freq_fitted
        abs_diff = np.abs(diff.values)
        n_remove = int(0.1 * len(abs_diff))
        if n_remove > 0:
            keep_indices = np.sort(np.argsort(-abs_diff)[n_remove:])
            trimmed_diff = diff.values[keep_indices]
        else:
            trimmed_diff = diff.values
        fit_norm[q] = float(np.linalg.norm(trimmed_diff))
        
        # Store quad_term in MHz/V^2 (original code)
        a[q] = float(-1e6 * c2)
        
        # Check if fit is successful before calculating offsets
        freq_range_mhz = freq_range[q] * 1e3  # Convert GHz to MHz
        is_successful = fit_norm[q] < SUCCESS_THRESHOLD and freq_range_mhz > MIN_FREQ_RANGE_MHZ
        success_dict[q] = is_successful
        
        if is_successful:
            flux_offset[q] = float(-0.5 * c1 / c2) if c2 != 0 else 0.0
            freq_offset[q] = 1e9 * (
                flux_offset[q] ** 2 * c2
                + flux_offset[q] * c1
                + c0
            )
            
            # Calculate target_offset: solve for flux bias where detuning equals target_detuning_from_sweet_spot
            qubit = node.machine.qubits[q]
            target_detuning_hz = getattr(qubit.xy, 'target_detuning_from_sweet_spot', 0.0) or 0.0
            target_detuning_ghz = target_detuning_hz / 1e9
            
            freq_sweet_spot_ghz = c2 * flux_offset[q]**2 + c1 * flux_offset[q] + c0
            
            if abs(target_detuning_ghz) < 1e-12:
                target_offset[q] = flux_offset[q]
            elif abs(c2) < 1e-20:
                target_offset[q] = np.nan
            else:
                # Solve: target_detuning = freq(V) - freq_sweet_spot
                # Rearranged: c2*V^2 + c1*V + (c0 - freq_sweet_spot - target_detuning) = 0
                target_freq_ghz = freq_sweet_spot_ghz + target_detuning_ghz
                discriminant = c1**2 - 4 * c2 * (c0 - target_freq_ghz)
                
                if discriminant < 0:
                    target_offset[q] = np.nan
                else:
                    sqrt_disc = np.sqrt(discriminant)
                    v1 = (-c1 + sqrt_disc) / (2 * c2)
                    v2 = (-c1 - sqrt_disc) / (2 * c2)
                    
                    # Choose solution closest to target_detuning (prefer positive solutions)
                    freq_v1 = c2 * v1**2 + c1 * v1 + c0
                    freq_v2 = c2 * v2**2 + c1 * v2 + c0
                    detuning_v1 = abs((freq_v1 - freq_sweet_spot_ghz) * 1e9 - target_detuning_hz)
                    detuning_v2 = abs((freq_v2 - freq_sweet_spot_ghz) * 1e9 - target_detuning_hz)
                    
                    if v1 > 0 and v2 > 0:
                        target_offset[q] = v1 if detuning_v1 < detuning_v2 else v2
                    elif v1 > 0:
                        target_offset[q] = v1
                    elif v2 > 0:
                        target_offset[q] = v2
                    else:
                        target_offset[q] = max(v1, v2)
        else:
            # Fit failed - set offsets to NaN
            flux_offset[q] = np.nan
            freq_offset[q] = np.nan
            target_offset[q] = np.nan
        
        t2_star[q] = tau.sel(qubit=q).values

    ds_fit = ds.merge(fit_data.rename("fit_results"))

    # Add a, flux_offset, freq_offset, and target_offset as data variables in the dataset
    ds_fit["quad_term"] = xr.DataArray([a[q] for q in qubits], dims=["qubit"], coords={"qubit": qubits})
    ds_fit["flux_offset"] = xr.DataArray([flux_offset[q] for q in qubits], dims=["qubit"], coords={"qubit": qubits})
    ds_fit["freq_offset"] = xr.DataArray([freq_offset[q] for q in qubits], dims=["qubit"], coords={"qubit": qubits})
    ds_fit["target_offset"] = xr.DataArray([target_offset[q] for q in qubits], dims=["qubit"], coords={"qubit": qubits})
    ds_fit["fit_norm"] = xr.DataArray(
        [fit_norm[q] for q in qubits], dims=["qubit"], coords={"qubit": qubits}
    )
    ds_fit["freq_range"] = xr.DataArray(
        [freq_range[q] for q in qubits], dims=["qubit"], coords={"qubit": qubits}
    )
    ds_fit["artifitial_detuning"] = xr.DataArray(
        node.parameters.frequency_detuning_in_mhz, dims=["qubit"], coords={"qubit": qubits}
    )
    
    # Success was already calculated in the loop above
    ds_fit["success"] = xr.DataArray(
        [success_dict[q] for q in qubits], 
        dims=["qubit"], coords={"qubit": qubits}
    )
    
    fit_results = {
        q: FitParameters(
            success=success_dict[q],
            quad_term=a[q],
            flux_offset=flux_offset[q],
            target_offset=target_offset[q],
            freq_offset=freq_offset[q],
            t2_star=t2_star[q],
            fit_norm=fit_norm[q],
        )
        for q in ds_fit.qubit.values
    }

    return ds_fit, fit_results
