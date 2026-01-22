import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from qualibration_libs.analysis import fit_decay_exp


@dataclass
class FitParameters:
    """Stores the relevant spin locking experiment fit parameters for a single qubit at a single amplitude"""

    T2_SL: float
    T2_SL_error: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
        Structure: {qubit_name: {amp_factor: FitParameters}}
    log_callable : callable, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    
    for qubit_name, amp_dict in fit_results.items():
        log_callable(f"Qubit {qubit_name}:")
        successful_fits = sum(1 for fit_result in amp_dict.values() if fit_result.get("success", False))
        total_fits = len(amp_dict)
        log_callable(f"  Successful fits: {successful_fits}/{total_fits}")
        
        # Log a few example fits
        for amp_factor in sorted(amp_dict.keys())[:3]:
            fit_result = amp_dict[amp_factor]
            if fit_result.get("success", False):
                log_callable(f"  Amp {amp_factor:.3f}: T2_SL = {fit_result['T2_SL'] * 1e6:.1f} ns, "
                           f"Error = {fit_result['T2_SL_error'] * 1e6:.1f} ns")


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, Dict[float, FitParameters]]]:
    """
    Fit T2_SL for each amplitude factor.
    Returns a nested dictionary: {qubit_name: {amp_factor: FitParameters}}
    """
    qubits = node.namespace["qubits"]
    fit_results = {}
    
    # Get amplitude factors from the dataset
    amp_factors = ds.amp_prefactor.values if "amp_prefactor" in ds.dims else []
    
    # Rename spin_locking_time to idle_time for compatibility with fit_decay_exp
    ds_renamed = ds.rename({"spin_locking_time": "idle_time"})
    
    fit_evals = []
    
    for q in qubits:
        q_name = q.name
        fit_results[q_name] = {}
        qubit_fit_evals = []
        
        # Get data for this qubit across all amplitudes
        q_data_all_amps = ds_renamed.sel(qubit=q_name)
        
        # Fit the exponential decay for this qubit across all amplitudes at once
        # fit_decay_exp can vectorize over amp_prefactor dimension
        if node.parameters.use_state_discrimination:
            fit_data_all = fit_decay_exp(q_data_all_amps.state, "idle_time")
        else:
            # Only fit I, discard Q
            fit_data_I_all = fit_decay_exp(q_data_all_amps.I, "idle_time")
        
        # Now extract results for each amplitude
        for amp_factor in amp_factors:
            try:
                # Get data for this specific amplitude
                q_data = ds_renamed.sel(qubit=q_name, amp_prefactor=amp_factor)
                
                # Extract fit data for this amplitude
                # Note: fit_data_all doesn't have 'qubit' dimension since we already selected it
                if node.parameters.use_state_discrimination:
                    fit_data = fit_data_all.sel(amp_prefactor=amp_factor)
                    q_data_fit = xr.merge([q_data, fit_data.rename("fit_data")], compat='no_conflicts')
                else:
                    # Only use I fit data, discard Q
                    fit_data_I = fit_data_I_all.sel(amp_prefactor=amp_factor)
                    q_data_fit = xr.merge([q_data, fit_data_I.rename("fit_data_I")], compat='no_conflicts')
                    # For backward compatibility, also add fit_data as fit_data_I
                    q_data_fit = q_data_fit.assign(fit_data=q_data_fit.fit_data_I)
                
                # Extract T2_SL and error
                decay = q_data_fit.fit_data.sel(fit_vals="decay")
                decay_res = q_data_fit.fit_data.sel(fit_vals="decay_decay")
                
                # Convert to scalar values
                decay_val = float(decay.squeeze().item())
                decay_res_val = float(decay_res.squeeze().item())
                
                T2_SL = -1 / decay_val
                T2_SL_error = -T2_SL * (np.sqrt(decay_res_val) / decay_val)
                
                # Check if fit was successful
                success = not (np.isnan(T2_SL) or np.isnan(T2_SL_error))
                
                # Store results
                fit_results[q_name][amp_factor] = FitParameters(
                    T2_SL=1e-9 * float(T2_SL),  # Convert to seconds for consistency
                    T2_SL_error=1e-9 * float(T2_SL_error),  # Convert to seconds
                    success=bool(success),
                )
                
                # Store fit data for plotting (rename back to spin_locking_time)
                q_data_fit = q_data_fit.rename({"idle_time": "spin_locking_time"})
                qubit_fit_evals.append(q_data_fit)
                
            except Exception as e:
                logging.warning(f"Fit failed for qubit {q_name} at amplitude {amp_factor:.3f}: {e}")
                # Store empty results on failure
                fit_results[q_name][amp_factor] = FitParameters(
                    T2_SL=0, T2_SL_error=0, success=False
                )
                # Create empty dataset with same structure
                q_data = ds_renamed.sel(qubit=q_name, amp_prefactor=amp_factor)
                signal_name = "state" if node.parameters.use_state_discrimination else "I"
                empty_ds = q_data.rename({"idle_time": "spin_locking_time"})
                empty_ds = empty_ds.assign(fit_data=xr.full_like(empty_ds[signal_name], np.nan))
                qubit_fit_evals.append(empty_ds)
        
        # Create dataset for this qubit's fits
        if qubit_fit_evals:
            qubit_ds_fit = xr.concat(qubit_fit_evals, dim="amp_prefactor")
            qubit_ds_fit["amp_prefactor"] = amp_factors
            qubit_ds_fit["qubit"] = q_name
            fit_evals.append(qubit_ds_fit)
    
    # Create the dataset for fitted curves
    if fit_evals:
        ds_fit = xr.concat(fit_evals, dim="qubit")
        ds_fit["qubit"] = ds["qubit"]
    else:
        # Create empty dataset with same structure
        ds_fit = xr.Dataset()
    
    return ds_fit, fit_results




def extract_t2_sl_vs_amplitude(
    fit_results: Dict[str, Dict[float, FitParameters]]
) -> Dict[str, xr.Dataset]:
    """
    Extract T2_SL vs amplitude for each qubit.
    Returns a dictionary mapping qubit names to datasets with T2_SL vs amplitude.
    """
    t2_sl_data = {}
    
    for q_name, amp_dict in fit_results.items():
        amp_factors = []
        t2_sl_values = []
        t2_sl_errors = []
        success_flags = []
        
        for amp_factor in sorted(amp_dict.keys()):
            fit_params = amp_dict[amp_factor]
            amp_factors.append(amp_factor)
            t2_sl_values.append(fit_params.T2_SL * 1e9)  # Convert to ns
            t2_sl_errors.append(fit_params.T2_SL_error * 1e9)  # Convert to ns
            success_flags.append(fit_params.success)
        
        # Create dataset
        t2_sl_data[q_name] = xr.Dataset({
            "T2_SL": (["amp_prefactor"], t2_sl_values),
            "T2_SL_error": (["amp_prefactor"], t2_sl_errors),
            "success": (["amp_prefactor"], success_flags)
        }, coords={"amp_prefactor": amp_factors})
    
    return t2_sl_data
