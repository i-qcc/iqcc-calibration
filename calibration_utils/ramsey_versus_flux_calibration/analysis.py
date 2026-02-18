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
    lo_limit_exceeded: bool


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
    # frequency is in GHz 
    frequency.attrs = {"long_name": "frequency", "units": "GHz"}

    decay = fit_data.sel(fit_vals="decay")
    decay.attrs = {"long_name": "decay", "units": "nSec"}

    tau = 1 / fit_data.sel(fit_vals="decay")
    tau.attrs = {"long_name": "T2*", "units": "uSec"}

    frequency = frequency.where(frequency > 0, drop=True)

    # Constants for success determination
    SUCCESS_THRESHOLD = 0.005
    MIN_FREQ_RANGE_MHZ = 500000e-9  # 500 nHz in MHz
    MAX_LO_OFFSET_MHZ = 500  # Maximum allowed offset from LO frequency in MHz

    a = {}
    flux_offset = {}
    freq_offset = {}
    target_offset = {}
    t2_star = {}
    fit_norm = {}
    freq_range = {}
    success_dict = {}
    lo_limit_exceeded_dict = {}
    polyfit_coeffs = {}  # Store per-qubit polyfit coefficients

    qubits = ds.qubit.values
    has_per_qubit_flux = "flux_actual" in ds.data_vars

    for q in qubits:
        freq_q = frequency.sel(qubit=q)
        freq_q_values = freq_q.values
        
        # Get flux values and perform polyfit
        if has_per_qubit_flux:
            flux_q_values = ds.flux_actual.sel(qubit=q).values
            freq_for_fit = xr.DataArray(freq_q_values, dims=["flux"], coords={"flux": flux_q_values})
            fitvals_q = freq_for_fit.polyfit(dim="flux", deg=2)
        else:
            flux_q_values = freq_q.flux_bias.values
            fitvals_q = freq_q.polyfit(dim="flux_bias", deg=2)
        
        # Extract polynomial coefficients (GHz, GHz/V, GHz/V^2)
        c0, c1, c2 = [float(fitvals_q.sel(degree=d).polyfit_coefficients.values) for d in range(3)]
        polyfit_coeffs[q] = {"c0": c0, "c1": c1, "c2": c2}
        
        # Calculate frequency range (max - min) in GHz
        freq_range[q] = float(np.max(freq_q_values) - np.min(freq_q_values))
        
        # Calculate fitted values and fit norm (excluding 10% of points with largest residuals)
        # Use numpy arrays for consistent calculation
        freq_fitted = c2 * flux_q_values**2 + c1 * flux_q_values + c0
        diff = freq_q_values - freq_fitted
        abs_diff = np.abs(diff)
        n_remove = int(0.1 * len(abs_diff))
        if n_remove > 0:
            keep_indices = np.sort(np.argsort(-abs_diff)[n_remove:])
            trimmed_diff = diff[keep_indices]
        else:
            trimmed_diff = diff
        fit_norm[q] = float(np.linalg.norm(trimmed_diff))
        
        # Store quad_term in GHz/V^2 (same units as c2)
        a[q] = float(-c2)
        
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
            target_detuning_hz = qubit.xy.extras.get("target_detuning_from_sweet_spot", 0.0) or 0.0
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
            
            # Check LO frequency limit: verify that the new RF frequency (after update) distance from LO is smaller than 500 MHz
            # Only check if fit succeeded (we're already inside the is_successful block)
            lo_freq_hz = qubit.xy.upconverter_frequency
            current_rf_freq_hz = qubit.xy.RF_frequency
            
            # Calculate freq_offset as it will be calculated in update_state:
            # freq_offset = frequency_detuning_in_mhz * 1e6 - freq_offset[q]
            # Note: freq_offset[q] is stored in Hz
            freq_offset_for_update_hz = (
                node.parameters.frequency_detuning_in_mhz * 1e6
                - freq_offset[q]
            )
            
            # Adjust if target_offset is used (matching update_state logic)
            if not np.isnan(target_offset[q]):
                target_detuning_hz = qubit.xy.extras["target_detuning_from_sweet_spot"]
                freq_offset_for_update_hz -= target_detuning_hz
            
            # Calculate new RF frequency
            new_rf_freq_hz = current_rf_freq_hz + freq_offset_for_update_hz
            
            # Check if distance from LO is smaller than 500 MHz (must be < 500 MHz)
            freq_diff_mhz = abs(new_rf_freq_hz - lo_freq_hz) / 1e6
            lo_limit_exceeded_dict[q] = freq_diff_mhz >= MAX_LO_OFFSET_MHZ  # True if distance >= 500 MHz (i.e., NOT smaller)
            
            # If LO limit exceeded, mark as unsuccessful for parameter updates (but keep fit results)
            if lo_limit_exceeded_dict[q]:
                success_dict[q] = False
        else:
            # Fit failed - set offsets to NaN
            flux_offset[q] = np.nan
            freq_offset[q] = np.nan
            target_offset[q] = np.nan
            lo_limit_exceeded_dict[q] = False
        
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
    ds_fit["lo_limit_exceeded"] = xr.DataArray(
        [lo_limit_exceeded_dict[q] for q in qubits],
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
            lo_limit_exceeded=lo_limit_exceeded_dict[q],
        )
        for q in ds_fit.qubit.values
    }

    return ds_fit, fit_results
