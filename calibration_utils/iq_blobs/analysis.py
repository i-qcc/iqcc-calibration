import logging
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from scipy.optimize import minimize


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    iw_angle: float
    ge_threshold: float
    rus_threshold: float
    readout_fidelity: float
    confusion_matrix: list
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all qubits.
    log_callable : callable, optional
        Callable for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s = f"IW angle: {fit_results[q]['iw_angle'] * 180 / np.pi:.1f} deg | "
        s += f"ge_threshold: {fit_results[q]['ge_threshold'] * 1e3:.1f} mV | "
        s += f"rus_threshold: {fit_results[q]['rus_threshold'] * 1e3:.1f} mV | "
        s += f"readout fidelity: {fit_results[q]['readout_fidelity']:.1f} % \n "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    # Fix the structure of ds to avoid tuples
    def extract_value(element):
        if isinstance(element, tuple):
            return element[0]
        return element

    ds = xr.apply_ufunc(
        extract_value,
        ds,
        vectorize=True,  # This ensures the function is applied element-wise
        dask="parallelized",  # This allows for parallel processing
        output_dtypes=[float],  # Specify the output data type
    )
    ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["Ig", "Qg", "Ie", "Qe"])
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
    ds_fit = ds
    # Condition to have the Q equal for both states:
    angle = np.arctan2(
        ds_fit.Qe.mean(dim="n_runs") - ds_fit.Qg.mean(dim="n_runs"),
        ds_fit.Ig.mean(dim="n_runs") - ds_fit.Ie.mean(dim="n_runs"),
    )
    ds_fit = ds_fit.assign({"iw_angle": xr.DataArray(angle, coords=dict(qubit=ds_fit.qubit.data))})

    C = np.cos(angle)
    S = np.sin(angle)
    # Condition for having e > Ig
    if np.mean((ds_fit.Ig - ds_fit.Ie) * C - (ds_fit.Qg - ds_fit.Qe) * S) > 0:
        angle += np.pi
        C = np.cos(angle)
        S = np.sin(angle)

    ds_fit = ds_fit.assign({"Ig_rot": ds_fit.Ig * C - ds_fit.Qg * S})
    ds_fit = ds_fit.assign({"Qg_rot": ds_fit.Ig * S + ds_fit.Qg * C})
    ds_fit = ds_fit.assign({"Ie_rot": ds_fit.Ie * C - ds_fit.Qe * S})
    ds_fit = ds_fit.assign({"Qe_rot": ds_fit.Ie * S + ds_fit.Qe * C})

    # Get the blobs histogram along the rotated axis
    hist = [np.histogram(ds_fit.Ig_rot.sel(qubit=q.name), bins=100) for q in node.namespace["qubits"]]
    # Get the discriminating threshold along the rotated axis
    rus_threshold = [
        hist[ii][1][1:][np.argmax(np.histogram(ds_fit.Ig_rot.sel(qubit=q.name), bins=100)[0])]
        for ii, q in enumerate(node.namespace["qubits"])
    ]
    ds_fit = ds_fit.assign({"rus_threshold": xr.DataArray(rus_threshold, coords=dict(qubit=ds_fit.qubit.data))})

    threshold = []
    gg, ge, eg, ee = [], [], [], []
    for q in node.namespace["qubits"]:
        fit = minimize(
            _false_detections,
            0.5 * (np.mean(ds_fit.Ig_rot.sel(qubit=q.name)) + np.mean(ds_fit.Ie_rot.sel(qubit=q.name))),
            (ds_fit.Ig_rot.sel(qubit=q.name), ds_fit.Ie_rot.sel(qubit=q.name)),
            method="Nelder-Mead",
        )
        threshold.append(fit.x[0])
        gg.append(np.sum(ds_fit.Ig_rot.sel(qubit=q.name) < fit.x[0]) / len(ds_fit.Ig_rot.sel(qubit=q.name)))
        ge.append(np.sum(ds_fit.Ig_rot.sel(qubit=q.name) > fit.x[0]) / len(ds_fit.Ig_rot.sel(qubit=q.name)))
        eg.append(np.sum(ds_fit.Ie_rot.sel(qubit=q.name) < fit.x[0]) / len(ds_fit.Ie_rot.sel(qubit=q.name)))
        ee.append(np.sum(ds_fit.Ie_rot.sel(qubit=q.name) > fit.x[0]) / len(ds_fit.Ie_rot.sel(qubit=q.name)))
    ds_fit = ds_fit.assign({"ge_threshold": xr.DataArray(threshold, coords=dict(qubit=ds_fit.qubit.data))})
    ds_fit = ds_fit.assign({"gg": xr.DataArray(gg, coords=dict(qubit=ds_fit.qubit.data))})
    ds_fit = ds_fit.assign({"ge": xr.DataArray(ge, coords=dict(qubit=ds_fit.qubit.data))})
    ds_fit = ds_fit.assign({"eg": xr.DataArray(eg, coords=dict(qubit=ds_fit.qubit.data))})
    ds_fit = ds_fit.assign({"ee": xr.DataArray(ee, coords=dict(qubit=ds_fit.qubit.data))})
    ds_fit = ds_fit.assign(
        {"readout_fidelity": xr.DataArray(100 * (ds_fit.gg + ds_fit.ee) / 2, coords=dict(qubit=ds_fit.qubit.data))}
    )

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _false_detections(threshold, Ig, Ie):
    if np.mean(Ig) < np.mean(Ie):
        false_detections_var = np.sum(Ig > threshold) + np.sum(Ie < threshold)
    else:
        false_detections_var = np.sum(Ig < threshold) + np.sum(Ie > threshold)
    return false_detections_var


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""

    # Assess whether the fit was successful or not
    nan_success = (
        np.isnan(fit.iw_angle)
        | np.isnan(fit.ge_threshold)
        | np.isnan(fit.rus_threshold)
        | np.isnan(fit.readout_fidelity)
    )
    success_criteria = ~nan_success
    fit = fit.assign({"success": success_criteria})

    fit_results = {
        q: FitParameters(
            iw_angle=float(fit.sel(qubit=q).iw_angle),
            ge_threshold=float(fit.sel(qubit=q).ge_threshold),
            rus_threshold=float(fit.sel(qubit=q).rus_threshold),
            readout_fidelity=float(fit.sel(qubit=q).readout_fidelity),
            confusion_matrix=[
                [float(fit.sel(qubit=q).gg), float(fit.sel(qubit=q).ge)],
                [float(fit.sel(qubit=q).eg), float(fit.sel(qubit=q).ee)],
            ],
            success=fit.sel(qubit=q).success.values.__bool__(),
        )
        for q in fit.qubit.values
    }
    return fit, fit_results


def gaussian(x, mu, sigma, A):
    """Single Gaussian function"""
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


def double_gaussian(x, mu1, sigma1, A1, mu2, sigma2, A2):
    """Double Gaussian function: sum of two Gaussians"""
    return A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) + A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))


def fit_snr_with_gaussians(
    fits: xr.Dataset,
    qubits: List,
    node: QualibrationNode,
    fit_results: Dict,
    axes: Optional[np.ndarray] = None,
    plot: bool = True,
) -> Tuple[List[float], List[Dict], List[Dict]]:
    """
    Fit Gaussian distributions to ground and excited state IQ blobs and calculate SNR.
    
    Parameters:
    -----------
    fits : xr.Dataset
        Dataset containing rotated IQ data (Ig_rot, Ie_rot)
    qubits : List
        List of qubit objects
    node : QualibrationNode
        The calibration node
    fit_results : Dict
        Dictionary containing fit results from fit_raw_data
    axes : Optional[np.ndarray]
        Matplotlib axes array for plotting. If None and plot=True, will create new figure.
    plot : bool
        Whether to create plots
        
    Returns:
    --------
    snr : List[float]
        List of SNR values for each qubit
    all_fit_params : List[Dict]
        List of fit parameters dictionaries for each qubit
    all_fit_errors : List[Dict]
        List of fit error dictionaries for each qubit
    """
    num_qubits = len(qubits)
    
    # Create figure and axes if plotting and axes not provided
    if plot and axes is None:
        cols = 2
        rows = (num_qubits + 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), constrained_layout=True)
        axes = axes.flatten()
    
    snr = []
    all_fit_params = []
    all_fit_errors = []
    
    for i in range(num_qubits):
        if plot:
            ax = axes[i]
        else:
            ax = None
        
        qubit_id = qubits[i].id
        
        # Extract and flatten data
        Ig_mv = (fits.Ig_rot.values[i] * 1e3).flatten()
        Ie_mv = (fits.Ie_rot.values[i] * 1e3).flatten()
        
        fit_params = {}
        fit_errors = {}
        
        if plot:
            ax.axvline(1e3 * fits.ge_threshold.values[i], color="r", linestyle="--", lw=1, label="Threshold")
        
        # Process Ground state (single gaussian)
        data = Ig_mv
        label = "Ground"
        color = "tab:blue"
        counts, bins = np.histogram(data, bins=100)
        centers = (bins[:-1] + bins[1:]) / 2
        p0 = [np.mean(data), np.std(data), np.max(counts)]
        
        try:
            param, cov = curve_fit(gaussian, centers, counts, p0=p0)
            mu, sigma, A = param
            # Extract standard errors from covariance matrix diagonal
            errors = np.sqrt(np.diag(cov))
            mu_err, sigma_err, A_err = np.round(errors, 2)
            fit_params[label] = (mu, sigma)
            fit_errors[label] = (mu_err, sigma_err)
            
            if plot:
                x_range = np.linspace(min(data), max(data), 200)
                ax.plot(x_range, gaussian(x_range, *param), color=color, lw=2)
                ax.hist(data, bins=100, alpha=0.3, color=color, label=label)
        
        except RuntimeError:
            fit_params[label] = (np.nan, np.nan)
            fit_errors[label] = (np.nan, np.nan)
        
        # Process Excited state (single or double gaussian based on T1/readout_length ratio)
        data = Ie_mv
        label = "Excited"
        color = "tab:orange"
        counts, bins = np.histogram(data, bins=100)
        centers = (bins[:-1] + bins[1:]) / 2
        
        # Get T1 and readout length to determine fitting method
        readout_length = qubits[i].resonator.operations['readout'].length
        T1 = qubits[i].T1
        
        # Calculate T1/readout_length ratio
        # readout_length is in nanoseconds, T1 is in seconds
        # Convert readout_length to seconds for consistent units
        readout_length_sec = readout_length * 1e-9
        t1_ratio =  readout_length_sec / T1
        
        # Use single gaussian if T1/readout_length < 5% (0.05), otherwise use double gaussian
        if t1_ratio < 0.03:
            # Single gaussian fitting for excited state
            p0 = [np.mean(data), np.std(data), np.max(counts)]
            
            try:
                param, cov = curve_fit(gaussian, centers, counts, p0=p0)
                mu, sigma, A = param
                # Extract standard errors from covariance matrix diagonal
                errors = np.sqrt(np.diag(cov))
                mu_err, sigma_err, A_err = np.round(errors, 2)
                fit_params[label] = (mu, sigma)
                fit_errors[label] = (mu_err, sigma_err)
                
                if plot:
                    x_range = np.linspace(min(data), max(data), 200)
                    ax.plot(x_range, gaussian(x_range, *param), color=color, lw=2, label=label)
                    ax.hist(data, bins=100, alpha=0.3, color=color)
            
            except RuntimeError:
                fit_params[label] = (np.nan, np.nan)
                fit_errors[label] = (np.nan, np.nan)
        else:
            # Double gaussian fitting for excited state (original behavior)
            # Get ground state peak position for second peak search
            mu_g, sig_g = fit_params.get("Ground", (np.nan, np.nan))
            if np.isnan(mu_g):
                mu_g = np.mean(Ig_mv)
            
            mean_data = np.mean(data)
            std_data = np.std(data)
            max_counts = np.max(counts)
            
            # Initialize peaks variable for error handling
            peaks_all = None
            
            # Find the highest peak in excited state for first Gaussian
            prominence_levels = [max_counts * 0.1, max_counts * 0.05, max_counts * 0.02]
            for prominence in prominence_levels:
                peaks_found, properties = find_peaks(counts, prominence=prominence, distance=3)
                if len(peaks_found) >= 1:
                    peaks_all = peaks_found
                    break
            
            if peaks_all is None:
                peaks_all, properties = find_peaks(counts, distance=3)
            
            if len(peaks_all) >= 1:
                # Find highest peak for first Gaussian
                peak_heights = counts[peaks_all]
                highest_peak_idx = np.argmax(peak_heights)
                mu1_init = centers[peaks_all[highest_peak_idx]]
                A1_init = peak_heights[highest_peak_idx]
                
                # Look for second peak near ground state position
                search_radius = 2 * std_data
                peaks_near_ground = []
                peak_heights_near_ground = []
                
                for peak_idx in peaks_all:
                    peak_pos = centers[peak_idx]
                    if abs(peak_pos - mu_g) <= search_radius:
                        peaks_near_ground.append(peak_idx)
                        peak_heights_near_ground.append(counts[peak_idx])
                
                if len(peaks_near_ground) > 0:
                    # Found peak(s) near ground state - use the highest one
                    best_idx = np.argmax(peak_heights_near_ground)
                    mu2_init = centers[peaks_near_ground[best_idx]]
                    A2_init = peak_heights_near_ground[best_idx]
                    
                    peak_separation = abs(mu2_init - mu1_init)
                    sigma_init = max(peak_separation / 4, std_data * 0.3)
                    sigma_init = min(sigma_init, std_data * 0.8)
                else:
                    # No peak found near ground state - place second Gaussian at ground state position
                    mu2_init = mu_g
                    ground_bin_idx = np.argmin(np.abs(centers - mu_g))
                    A2_init = max(counts[ground_bin_idx], max_counts * 0.1)
                    
                    peak_separation = abs(mu2_init - mu1_init)
                    sigma_init = max(peak_separation / 4, std_data * 0.3)
                    sigma_init = min(sigma_init, std_data * 0.8)
            else:
                # No peaks found at all - use mean-based approach
                mu1_init = mean_data
                mu2_init = mu_g
                A1_init = max_counts * 0.7
                A2_init = max_counts * 0.3
                sigma_init = std_data * 0.5
            
            # Ensure first gaussian has higher amplitude (majority peak)
            if A2_init > A1_init:
                mu1_init, mu2_init = mu2_init, mu1_init
                A1_init, A2_init = A2_init, A1_init
            
            p0_double = [mu1_init, sigma_init, A1_init, mu2_init, sigma_init, A2_init]
            
            try:
                # Set bounds to constrain the fit
                data_range = np.max(data) - np.min(data)
                bounds = ([np.min(data) - 0.5*data_range, 0.01*std_data, 0.1*max_counts, 
                           np.min(data) - 0.5*data_range, 0.01*std_data, 0.1*max_counts],
                          [np.max(data) + 0.5*data_range, 5*std_data, 2*max_counts,
                           np.max(data) + 0.5*data_range, 5*std_data, 2*max_counts])
                
                try:
                    param, cov = curve_fit(double_gaussian, centers, counts, p0=p0_double, 
                                          bounds=bounds, maxfev=10000)
                except (RuntimeError, ValueError):
                    param, cov = curve_fit(double_gaussian, centers, counts, p0=p0_double, 
                                          maxfev=10000)
                mu1, sigma1, A1, mu2, sigma2, A2 = param
                
                # Extract standard errors from covariance matrix diagonal
                if cov is None or cov.shape != (6, 6):
                    raise ValueError(f"Invalid covariance matrix shape: {cov.shape if cov is not None else 'None'}")
                
                errors = np.sqrt(np.diag(cov))
                mu1_err_raw, sigma1_err_raw, A1_err_raw, mu2_err_raw, sigma2_err_raw, A2_err_raw = errors
                
                # Ensure first gaussian is the majority (highest amplitude) after fitting
                if A2 > A1:
                    mu1, mu2 = mu2, mu1
                    sigma1, sigma2 = sigma2, sigma1
                    A1, A2 = A2, A1
                    mu1_err_raw, mu2_err_raw = mu2_err_raw, mu1_err_raw
                    sigma1_err_raw, sigma2_err_raw = sigma2_err_raw, sigma1_err_raw
                    A1_err_raw, A2_err_raw = A2_err_raw, A1_err_raw
                
                # Validate and cap unreasonable error values
                max_reasonable_error = 1000  # mV
                errors_list = [mu1_err_raw, sigma1_err_raw, A1_err_raw, mu2_err_raw, sigma2_err_raw, A2_err_raw]
                param_list = [mu1, sigma1, A1, mu2, sigma2, A2]
                errors_validated = []
                param_names = ['mu1', 'sigma1', 'A1', 'mu2', 'sigma2', 'A2']
                
                for j, (err, param_val) in enumerate(zip(errors_list, param_list)):
                    err_float = float(err)
                    if (np.isnan(err_float) or np.isinf(err_float) or 
                        err_float > max_reasonable_error or err_float < 0):
                        capped_err = min(abs(float(param_val)) * 0.1, max_reasonable_error)
                        errors_validated.append(capped_err)
                        print(f"Warning ({qubit_id}): Invalid error for {param_names[j]} "
                              f"({err_float:.2e}), using fallback estimate: {capped_err:.2f}")
                    else:
                        errors_validated.append(err_float)
                
                errors = np.array(errors_validated)
                errors_rounded = np.round(errors, 2)
                errors_rounded = np.clip(errors_rounded, 0, max_reasonable_error)
                mu1_err, sigma1_err, A1_err, mu2_err, sigma2_err, A2_err = errors_rounded
                
                # For SNR calculation, use weighted mean and combined sigma
                total_A = A1 + A2
                mu_combined = (mu1 * A1 + mu2 * A2) / total_A
                sigma_combined = np.sqrt((sigma1**2 * A1 + sigma2**2 * A2) / total_A)
                fit_params[label] = (mu_combined, sigma_combined)
                
                # Store errors for the combined parameters
                mu_err_combined = np.round(np.sqrt((mu1_err**2 * A1**2 + mu2_err**2 * A2**2) / total_A**2), 2)
                sigma_err_combined = np.round(np.sqrt((sigma1_err**2 * A1**2 + sigma2_err**2 * A2**2) / total_A**2), 2)
                fit_errors[label] = (mu_err_combined, sigma_err_combined)
                
                # Store detailed double gaussian parameters and errors
                mu1_err_final = min(mu1_err, max_reasonable_error)
                sigma1_err_final = min(sigma1_err, max_reasonable_error)
                A1_err_final = min(A1_err, max_reasonable_error)
                mu2_err_final = min(mu2_err, max_reasonable_error)
                sigma2_err_final = min(sigma2_err, max_reasonable_error)
                A2_err_final = min(A2_err, max_reasonable_error)
                
                fit_params[f"{label}_double"] = {
                    'mu1': mu1, 'sigma1': sigma1, 'A1': A1,
                    'mu2': mu2, 'sigma2': sigma2, 'A2': A2,
                    'mu1_err': mu1_err_final, 'sigma1_err': sigma1_err_final, 'A1_err': A1_err_final,
                    'mu2_err': mu2_err_final, 'sigma2_err': sigma2_err_final, 'A2_err': A2_err_final
                }
                
                if plot:
                    x_range = np.linspace(min(data), max(data), 200)
                    ax.plot(x_range, double_gaussian(x_range, *param), color=color, lw=2, label=label)
                    ax.hist(data, bins=100, alpha=0.3, color=color)
            
            except (RuntimeError, ValueError, IndexError) as e:
                fit_params[label] = (np.nan, np.nan)
                fit_errors[label] = (np.nan, np.nan)
                print(f"Double gaussian fit failed for {qubit_id}: {e}")
        
        # SNR Calculation
        mu_g, sig_g = fit_params.get("Ground", (np.nan, np.nan))
        mu_e, sig_e = fit_params.get("Excited", (np.nan, np.nan))
        mu_g_err, sig_g_err = fit_errors.get("Ground", (np.nan, np.nan))
        mu_e_err, sig_e_err = fit_errors.get("Excited", (np.nan, np.nan))
        
        if not np.isnan(mu_g) and not np.isnan(mu_e):
            snr_val = abs(mu_e - mu_g) / (abs(sig_g) + abs(sig_e))
            snr.append(snr_val)
            if plot:
                ax.set_title(f"{qubit_id} \n SNR: {snr_val:.3f}, Amp={qubits[i].resonator.operations['readout'].amplitude :.3f},Tro={qubits[i].resonator.operations['readout'].length}, F={fit_results[qubits[i].name].readout_fidelity:.2f}%", fontsize=18)
        else:
            snr.append(np.nan)
            if plot:
                ax.set_title(f"{qubit_id} | Fit Failed")
        
        if plot:
            ax.set_xlabel("I [mV]", fontsize=15)
            ax.set_ylabel("Counts", fontsize=15)
            ax.legend(fontsize=12, loc="upper right")
        
        all_fit_params.append(fit_params)
        all_fit_errors.append(fit_errors)
    
    if plot:
        # Hide unused subplots if num_qubits is odd
        for j in range(num_qubits, len(axes)):
            axes[j].axis('off')
    
    return snr, all_fit_params, all_fit_errors
