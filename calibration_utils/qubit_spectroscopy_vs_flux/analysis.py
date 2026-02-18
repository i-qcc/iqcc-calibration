import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis import peaks_dips
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from scipy.optimize import curve_fit


@dataclass
class FitParameters:
    """Stores the relevant node-specific fitted parameters used to update the state at the end of the node."""

    success: bool
    qubit_frequency: float
    frequency_shift: float
    idle_offset: float
    target_offset: float
    quad_term: float
    dv_phi0: float
    phi0_current: float
    m_pH: float


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_idle_offset = f"\tidle offset: {fit_results[q]['idle_offset'] * 1e3:.0f} mV | "
        s_freq = f"Qubit frequency: {1e-9 * fit_results[q]['qubit_frequency']:.3f} GHz | "
        s_shift = f"(shift of {1e-6 * fit_results[q]['frequency_shift']:.0f} MHz) | "
        s_quad = f"quad term: {fit_results[q]['quad_term']/1e9:.3e} GHz/V^2\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_idle_offset + s_freq + s_shift + s_quad)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Processes the raw dataset by converting the 'I' and 'Q' quadratures to V, or adding the RF_frequency as a coordinate for instance."""

    # Convert the 'I' and 'Q' quadratures from demodulation units to V.
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    # Add the amplitude and phase to the raw dataset
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    # Add the RF frequency as a coordinate of the raw dataset
    full_freq = np.array([ds.detuning + q.xy.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    # Add the current axis of each qubit to the dataset coordinates for plotting
    current = ds.flux_bias / node.parameters.input_line_impedance_in_ohm
    ds = ds.assign_coords({"current": (["flux_bias"], current.data)})
    ds.current.attrs["long_name"] = "Current"
    ds.current.attrs["units"] = "A"
    # Add attenuated current to dataset
    attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
    attenuated_current = ds.current * attenuation_factor
    ds = ds.assign_coords({"attenuated_current": (["flux_bias"], attenuated_current.values)})
    ds.attenuated_current.attrs["long_name"] = "Attenuated Current"
    ds.attenuated_current.attrs["units"] = "A"
    return ds


def _filter_weak_peaks(peaks: xr.Dataset, threshold_factor: float = 0.4) -> Tuple[xr.DataArray, xr.DataArray]:
    """Filter out weak peaks based on amplitude threshold."""
    filtered_positions_list = []
    peak_mask_list = []
    for q in peaks.qubit.values:
        peak_amps = peaks.amplitude.sel(qubit=q)
        peak_positions = peaks.position.sel(qubit=q)
        valid_amps = peak_amps.dropna(dim="flux_bias")
        if len(valid_amps) > 3:
            sorted_amps = np.sort(valid_amps.values)[::-1]
            threshold = threshold_factor * np.mean(sorted_amps[:3])
            mask = peak_amps.values < threshold
            # Ensure we don't filter out all peaks - keep at least the top 3
            num_kept = np.sum(~mask)
            if num_kept == 0:
                # If all would be filtered, keep at least the top 3
                top_indices = np.argsort(peak_amps.values)[::-1][:min(3, len(peak_amps.values))]
                mask = np.ones(len(peak_amps.values), dtype=bool)
                mask[top_indices] = False  # Don't filter the top 3
            filtered_positions_q = peak_positions.values.copy()
            filtered_positions_q[mask] = np.nan
            filtered_positions_list.append(filtered_positions_q)
            peak_mask_list.append(~mask)
        else:
            filtered_positions_list.append(peak_positions.values)
            peak_mask_list.append(np.ones(len(peak_positions.values), dtype=bool))
    
    filtered_positions = xr.DataArray(
        np.array(filtered_positions_list),
        dims=["qubit", "flux_bias"],
        coords={"qubit": peaks.qubit.values, "flux_bias": peaks.flux_bias.values}
    )
    peak_mask = xr.DataArray(
        np.array(peak_mask_list),
        dims=["qubit", "flux_bias"],
        coords={"qubit": peaks.qubit.values, "flux_bias": peaks.flux_bias.values}
    )
    return filtered_positions, peak_mask


def _remove_outliers(filtered_positions: xr.DataArray, outlier_fraction: float = 0.15) -> xr.DataArray:
    """
    Remove outlier points before fitting by doing an initial fit and removing points with largest residuals.
    
    Parameters:
    -----------
    filtered_positions : xr.DataArray
        DataArray with peak positions (may contain NaN values).
    outlier_fraction : float
        Fraction of points to remove as outliers (default 0.15 = 15%).
    
    Returns:
    --------
    xr.DataArray
        DataArray with outliers set to NaN.
    """
    outlier_removed = filtered_positions.copy()
    
    for q in filtered_positions.qubit.values:
        positions_q = filtered_positions.sel(qubit=q)
        flux_bias_q = positions_q.flux_bias.values
        valid_mask = ~np.isnan(positions_q.values)
        
        if np.sum(valid_mask) < 4:  # Need at least 4 points for outlier removal
            continue
        
        valid_positions = positions_q.values[valid_mask]
        valid_flux = flux_bias_q[valid_mask]
        
        # Initial fit to identify outliers
        try:
            coeffs = np.polyfit(valid_flux, valid_positions, 2)
            fitted = np.polyval(coeffs, valid_flux)
            residuals = np.abs(valid_positions - fitted)
            
            # Remove the top outlier_fraction of points based on residuals
            num_to_remove = max(1, int(len(valid_positions) * outlier_fraction))
            outlier_indices = np.argsort(residuals)[-num_to_remove:]
            
            # Mark outliers as NaN
            outlier_mask = np.zeros(len(positions_q.values), dtype=bool)
            valid_indices = np.where(valid_mask)[0]
            outlier_mask[valid_indices[outlier_indices]] = True
            
            outlier_removed_q = outlier_removed.sel(qubit=q).values.copy()
            outlier_removed_q[outlier_mask] = np.nan
            outlier_removed.loc[dict(qubit=q)] = outlier_removed_q
        except (np.linalg.LinAlgError, ValueError):
            # If initial fit fails, skip outlier removal for this qubit
            continue
    
    return outlier_removed


def _parabola(x, c2, c1, c0):
    """Parabola function: f(x) = c2*x^2 + c1*x + c0"""
    return c2 * x**2 + c1 * x + c0


def _constrained_parabolic_fit(filtered_positions: xr.DataArray) -> xr.Dataset:
    """
    Perform a constrained parabolic fit where the quadratic coefficient must be negative.
    
    This ensures the parabola opens downward (has a maximum), which is physically
    expected for qubit frequency vs flux curves near the sweet spot.
    
    Parameters:
    -----------
    filtered_positions : xr.DataArray
        DataArray with peak positions indexed by qubit and flux_bias.
    
    Returns:
    --------
    xr.Dataset
        Dataset with polyfit_coefficients in the same format as xr.polyfit.
    """
    qubit_values = filtered_positions.qubit.values
    flux_bias_values = filtered_positions.flux_bias.values
    
    # Storage for coefficients: shape (num_qubits, 3) for degree 2, 1, 0
    coefficients = np.full((len(qubit_values), 3), np.nan)
    
    for i, q in enumerate(qubit_values):
        positions_q = filtered_positions.sel(qubit=q).values
        valid_mask = ~np.isnan(positions_q)
        
        if np.sum(valid_mask) < 3:
            # Not enough points for fitting
            continue
        
        valid_flux = flux_bias_values[valid_mask]
        valid_positions = positions_q[valid_mask]
        
        try:
            # Initial guess from unconstrained fit
            initial_coeffs = np.polyfit(valid_flux, valid_positions, 2)
            p0 = [initial_coeffs[0], initial_coeffs[1], initial_coeffs[2]]
            
            # If initial quadratic term is already negative, use it as starting point
            # Otherwise, flip sign for initial guess
            if p0[0] > 0:
                p0[0] = -abs(p0[0])
            
            # Bounds: c2 must be negative (from -inf to 0), c1 and c0 are unbounded
            bounds = (
                [-np.inf, -np.inf, -np.inf],  # lower bounds
                [0, np.inf, np.inf]            # upper bounds (c2 <= 0)
            )
            
            popt, _ = curve_fit(
                _parabola, 
                valid_flux, 
                valid_positions, 
                p0=p0, 
                bounds=bounds,
                maxfev=10000
            )
            
            coefficients[i, :] = popt  # [c2, c1, c0]
            
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            # Fit failed, leave as NaN
            continue
    
    # Create dataset in same format as xr.polyfit output
    # xr.polyfit returns coefficients indexed by 'degree' in descending order
    polyfit_coefficients = xr.DataArray(
        coefficients,
        dims=["qubit", "degree"],
        coords={
            "qubit": qubit_values,
            "degree": [2, 1, 0]  # quadratic, linear, constant
        }
    )
    
    return xr.Dataset({"polyfit_coefficients": polyfit_coefficients})


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit frequency vs flux response using a parabolic (quadratic) fit.
    
    The fit is constrained to have a negative quadratic coefficient (downward-opening
    parabola), which is physically expected for qubit frequency vs flux curves.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node : QualibrationNode
        The node containing parameters and qubit information.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    dict[str, FitParameters]
        Dictionary containing fitted parameters for each qubit.
    """
    # Find peaks with progressively lower prominence until we get enough points
    prominence_factors = [3, 2, 1.5, 1.0]
    peaks = None
    filtered_positions = None
    peak_mask = None
    
    for prominence in prominence_factors:
        peaks = peaks_dips(ds.I, dim="detuning", prominence_factor=prominence)
        filtered_positions, peak_mask = _filter_weak_peaks(peaks)
        
        # Check if we have enough valid points for fitting (at least 3 per qubit)
        has_enough_points = True
        for q in filtered_positions.qubit.values:
            valid_count = np.sum(~np.isnan(filtered_positions.sel(qubit=q).values))
            if valid_count < 3:
                has_enough_points = False
                break
        
        if has_enough_points:
            break
    
    # Remove outliers (30% worst points) before fitting
    filtered_positions = _remove_outliers(filtered_positions, outlier_fraction=0.30)
    
    # Fit with parabola (constrained to have negative quadratic coefficient)
    parabolic_fit_results = _constrained_parabolic_fit(filtered_positions)
    
    # Check if fit failed for any qubit and retry with even more lenient settings
    failed_qubits = []
    for q in parabolic_fit_results.polyfit_coefficients.qubit.values:
        coeffs = parabolic_fit_results.polyfit_coefficients.sel(qubit=q).values
        if np.any(np.isnan(coeffs)):
            failed_qubits.append(q)
    
    if failed_qubits:
        # For failed qubits, try with even lower prominence
        peaks_retry = peaks_dips(ds.I, dim="detuning", prominence_factor=0.8)
        filtered_positions_retry, peak_mask_retry = _filter_weak_peaks(peaks_retry)
        filtered_positions_retry = _remove_outliers(filtered_positions_retry, outlier_fraction=0.30)
        
        # Update only failed qubits
        for q in failed_qubits:
            if q in filtered_positions_retry.qubit.values:
                valid_count = np.sum(~np.isnan(filtered_positions_retry.sel(qubit=q).values))
                if valid_count >= 3:
                    filtered_positions.loc[dict(qubit=q)] = filtered_positions_retry.sel(qubit=q).values
                    peak_mask.loc[dict(qubit=q)] = peak_mask_retry.sel(qubit=q).values
        
        # Refit with updated positions using constrained fit
        parabolic_fit_results = _constrained_parabolic_fit(filtered_positions)
    
    # Merge results
    fit_results_ds = xr.merge([
        parabolic_fit_results,
        {"peak_freq": filtered_positions},
        {"peak_mask": peak_mask}
    ])
    
    # Ensure qubit coordinate is set
    if "qubit" in fit_results_ds.dims and "qubit" not in fit_results_ds.coords:
        qubit_values = (
            fit_results_ds.polyfit_coefficients.qubit.values
            if "polyfit_coefficients" in fit_results_ds.data_vars
            else np.array([q.name for q in node.namespace["qubits"]])
        )
        fit_results_ds = fit_results_ds.assign_coords(qubit=("qubit", qubit_values))
    
    fit_dataset, fit_results = _extract_relevant_fit_parameters(fit_results_ds, node, ds)
    return fit_dataset, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode, ds: xr.Dataset):
    """Add metadata to the fit dataset and fit result dictionary."""
    # Extract relevant fitted parameters from the parabolic fit
    coeff = fit.polyfit_coefficients
    
    # Calculate flux_shift (minimum of parabola): flux_shift = -coeff[1] / (2 * coeff[0])
    # For each qubit, extract coefficients
    flux_shift_values = []
    freq_shift_values = []
    quad_term_values = []
    
    for q in fit.qubit.values:
        coeff_q = coeff.sel(qubit=q)
        # Check if we have valid coefficients
        if np.any(np.isnan(coeff_q.values)):
            flux_shift_values.append(np.nan)
            freq_shift_values.append(np.nan)
            quad_term_values.append(np.nan)
        else:
            # Extract coefficients: degree 0 (constant), degree 1 (linear), degree 2 (quadratic)
            c0 = float(coeff_q.sel(degree=0).values)  # constant term
            c1 = float(coeff_q.sel(degree=1).values)  # linear term
            c2 = float(coeff_q.sel(degree=2).values)   # quadratic term
            
            # Calculate flux_shift (minimum of parabola)
            if abs(c2) > 1e-20:  # Avoid division by zero
                flux_shift = -c1 / (2 * c2)
            else:
                flux_shift = np.nan
            
            # Calculate freq_shift at the flux_shift location
            if not np.isnan(flux_shift):
                freq_shift = c2 * flux_shift**2 + c1 * flux_shift + c0
            else:
                freq_shift = np.nan
            
            flux_shift_values.append(flux_shift)
            freq_shift_values.append(freq_shift)
            quad_term_values.append(c2)
    
    # Create DataArrays for flux_shift, freq_shift, and quad_term
    flux_shift_da = xr.DataArray(
        flux_shift_values,
        dims=["qubit"],
        coords={"qubit": fit.qubit.values}
    )
    freq_shift_da = xr.DataArray(
        freq_shift_values,
        dims=["qubit"],
        coords={"qubit": fit.qubit.values}
    )
    quad_term_da = xr.DataArray(
        quad_term_values,
        dims=["qubit"],
        coords={"qubit": fit.qubit.values}
    )
    
    # Assign coordinates to fit dataset
    fit = fit.assign_coords(idle_offset=("qubit", flux_shift_da.data))
    fit.idle_offset.attrs = {"long_name": "idle flux bias", "units": "V"}
    
    fit = fit.assign_coords(freq_shift=("qubit", freq_shift_da.data))
    fit.freq_shift.attrs = {"long_name": "frequency shift", "units": "Hz"}
    
    fit = fit.assign_coords(quad_term=("qubit", quad_term_da.data))
    fit.quad_term.attrs = {"long_name": "quadratic term", "units": "Hz/V^2"}
    
    # Calculate sweet spot frequency
    full_freq = np.array([q.xy.RF_frequency for q in node.namespace["qubits"]])
    fit = fit.assign_coords(sweet_spot_frequency=("qubit", freq_shift_da.data + full_freq))
    fit.sweet_spot_frequency.attrs = {
        "long_name": "sweet spot frequency",
        "units": "Hz",
    }
    
    # Calculate fitted parabola for plotting
    fitted = (
        coeff.sel(degree=2) * ds.flux_bias**2 
        + coeff.sel(degree=1) * ds.flux_bias 
        + coeff.sel(degree=0)
    )
    fit = fit.assign(fitted_parabola=fitted)
    
    # Calculate target_offset for each qubit using the fitted quad_term
    target_offset_values = []
    for i, q in enumerate(fit.qubit.values):
        qubit = node.machine.qubits[q]
        target_detuning = qubit.xy.extras.get("target_detuning_from_sweet_spot", 0.0) or 0.0
        quad_term = quad_term_values[i]
        idle_offset_val = flux_shift_values[i]
        
        if np.isnan(quad_term) or np.isnan(idle_offset_val) or abs(quad_term) < 1e-20:
            target_offset = np.nan
        elif abs(target_detuning) < 1e-6:
            target_offset = idle_offset_val
        else:
            sqrt_arg = target_detuning / abs(quad_term) + idle_offset_val**2
            if sqrt_arg >= 0:
                sign = np.sign(idle_offset_val) if idle_offset_val != 0 else 1.0
                target_offset = sign * np.sqrt(sqrt_arg)
            else:
                target_offset = np.nan
        
        target_offset_values.append(target_offset)
    
    # Assign target_offset coordinate
    fit = fit.assign_coords(target_offset=("qubit", target_offset_values))
    fit.target_offset.attrs = {"long_name": "target flux bias offset", "units": "V"}
    
    # Calculate m_pH (mutual inductance) - keeping for compatibility but may need adjustment
    attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
    m_pH_values = [
        (1e12 * 2.068e-15 / np.sqrt(abs(qt)) / node.parameters.input_line_impedance_in_ohm * attenuation_factor)
        if not (np.isnan(qt) or abs(qt) < 1e-20) else np.nan
        for qt in quad_term_values
    ]
    
    # Assess whether the fit was successful or not
    freq_success = np.abs(freq_shift_da.data) < node.parameters.frequency_span_in_mhz * 1e6
    nan_success = (
        np.isnan(freq_shift_da.data) 
        | np.isnan(flux_shift_da.data) 
        | np.isnan(quad_term_da.data)
    )
    success_criteria = freq_success & ~nan_success
    fit = fit.assign_coords(success=("qubit", success_criteria))
    
    # Calculate dv_phi0 and phi0_current - keeping for compatibility
    dv_phi0_values = [
        1.0 / np.sqrt(abs(qt)) if not (np.isnan(qt) or abs(qt) < 1e-20) else np.nan
        for qt in quad_term_values
    ]
    phi0_current_values = [
        dv * node.parameters.input_line_impedance_in_ohm * attenuation_factor
        if not np.isnan(dv) else np.nan
        for dv in dv_phi0_values
    ]

    fit_results = {
        q: FitParameters(
            success=bool(fit.sel(qubit=q).success.values),
            qubit_frequency=float(fit.sweet_spot_frequency.sel(qubit=q).values),
            frequency_shift=float(freq_shift_da.sel(qubit=q).values),
            idle_offset=float(flux_shift_da.sel(qubit=q).values),
            target_offset=target_offset_values[i],
            quad_term=float(quad_term_da.sel(qubit=q).values),
            dv_phi0=dv_phi0_values[i],
            phi0_current=phi0_current_values[i],
            m_pH=m_pH_values[i],
        )
        for i, q in enumerate(fit.qubit.values)
    }

    return fit, fit_results
