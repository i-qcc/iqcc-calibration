import logging
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode


@dataclass
class FitParameters:
    """Stores the measurement characterization results for a single qubit"""
    confusion_matrix: list
    demolition_error_matrix: list
    success: bool
    optimal_depletion_time: int = None
    """Optimal depletion time in nanoseconds. None if not optimizing."""


def calculate_confusion_matrix(state_g_1, state_e_1):
    """
    Calculate the confusion matrix: P(measured state | prepared state)
    
    Parameters:
    -----------
    state_g_1 : array
        First measurement results when prepared in ground state
    state_e_1 : array
        First measurement results when prepared in excited state
        
    Returns:
    --------
    confusion_matrix : 2x2 array
        [[P(0|g), P(1|g)], [P(0|e), P(1|e)]]
    """
    n_runs = len(state_g_1)
    confusion_matrix = np.zeros((2, 2))
    
    # Ground state preparation
    confusion_matrix[0, 0] = np.sum(state_g_1 == 0) / n_runs  # P(0|g)
    confusion_matrix[0, 1] = np.sum(state_g_1 == 1) / n_runs  # P(1|g)
    
    # Excited state preparation
    confusion_matrix[1, 0] = np.sum(state_e_1 == 0) / n_runs  # P(0|e)
    confusion_matrix[1, 1] = np.sum(state_e_1 == 1) / n_runs  # P(1|e)
    
    return confusion_matrix


def calculate_demolition_error_matrix(state_1, state_2):
    """
    Calculate the demolition error matrix: P(measured state in last measurement | measured state in first measurement)
    
    Parameters:
    -----------
    state_1 : array
        First measurement results
    state_2 : array
        Last measurement results (after consecutive measurements)
        
    Returns:
    --------
    demolition_error_matrix : 2x2 array
        [[P(0|0), P(1|0)], [P(0|1), P(1|1)]]
        where P_ij = P(last=j | first=i) is the probability of measuring j in the last 
        measurement given i was measured in the first measurement
        Rows correspond to first measurement, columns to last measurement
    """
    demolition_error_matrix = np.zeros((2, 2))
    
    # Count transitions: P_ij = P(last=j | first=i)
    # i = first measurement (row), j = last measurement (column)
    for i in range(2):  # i = first measurement
        mask_i = state_1 == i
        n_i = np.sum(mask_i)
        if n_i > 0:
            for j in range(2):  # j = last measurement
                demolition_error_matrix[i, j] = np.sum(mask_i & (state_2 == j)) / n_i
        else:
            demolition_error_matrix[i, :] = np.nan
    
    return demolition_error_matrix


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits.

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing fit results for each qubit.
    log_callable : callable, optional
        Callable for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q_name, fit_result in fit_results.items():
        log_callable(f"Results for qubit {q_name}:")
        log_callable(f"  Confusion matrix: {fit_result['confusion_matrix']}")
        log_callable(f"  Demolition error matrix: {fit_result['demolition_error_matrix']}")
        if fit_result.get('optimal_depletion_time') is not None:
            log_callable(f"  Optimal depletion time: {fit_result['optimal_depletion_time']} ns")
        log_callable(f"  Success: {fit_result['success']}")


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Analyse the raw data and calculate confusion matrices and demolition error matrices.
    If depletion_time dimension exists, find the optimal depletion_time.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node : QualibrationNode
        The calibration node.

    Returns:
    --------
    ds_fit : xr.Dataset
        Dataset containing the fit results with additional dimensions if optimizing.
    fit_results : dict
        Dictionary mapping qubit names to FitParameters objects.
    """
    qubits = node.namespace["qubits"]
    
    # Check if we're optimizing depletion_time
    has_depletion_time_dim = "depletion_time" in ds.dims
    
    fit_results = {}
    fit_data_vars = {}
    
    for q in qubits:
        q_name = q.name
        
        # Extract state measurements using .sel() to select by qubit name
        # Note: XarrayDataFetcher extracts the base name (everything before first digit)
        # So "ag11" extracts to "ag" -> variable "ag"
        # "bg21" extracts to "bg" -> variable "bg"
        # "ce11" extracts to "ce" -> variable "ce"
        # "de21" extracts to "de" -> variable "de"
        state_g_1 = ds["ag"].sel(qubit=q_name)
        state_g_2 = ds["bg"].sel(qubit=q_name)
        state_e_1 = ds["ce"].sel(qubit=q_name)
        state_e_2 = ds["de"].sel(qubit=q_name)
        
        if has_depletion_time_dim:
            # Calculate matrices for each depletion_time
            p00_values = []  # P(0|0) - probability of preserving ground state measurement
            p11_values = []  # P(1|1) - probability of preserving excited state measurement
            confusion_p0g_values = []  # P(0|g) from confusion matrix - baseline
            
            for dt in ds.depletion_time.values:
                state_g_1_dt = state_g_1.sel(depletion_time=dt).values
                state_g_2_dt = state_g_2.sel(depletion_time=dt).values
                state_e_1_dt = state_e_1.sel(depletion_time=dt).values
                state_e_2_dt = state_e_2.sel(depletion_time=dt).values
                
                # Calculate confusion matrix to get baseline P(0|g)
                confusion_matrix_dt = calculate_confusion_matrix(state_g_1_dt, state_e_1_dt)
                confusion_p0g_values.append(confusion_matrix_dt[0, 0])  # P(0|g)
                
                # Calculate demolition error matrices
                demolition_error_matrix_g = calculate_demolition_error_matrix(state_g_1_dt, state_g_2_dt)
                demolition_error_matrix_e = calculate_demolition_error_matrix(state_e_1_dt, state_e_2_dt)
                demolition_error_matrix = (demolition_error_matrix_g + demolition_error_matrix_e) / 2
                
                p00_values.append(demolition_error_matrix[0, 0])
                p11_values.append(demolition_error_matrix[1, 1])
            
            p00_values = np.array(p00_values)
            p11_values = np.array(p11_values)
            confusion_p0g_values = np.array(confusion_p0g_values)
            
            # Optimization strategy:
            # 1. Filter out points where P(0|0) is lower than confusion P(0|g) by more than 10%
            # 2. If no points pass filter, take the one with highest P(0|0)
            # 3. If some points pass filter, optimize on P(1|1) among those
            
            # Filter: keep points where P(0|0) >= P(0|g) - 0.1
            valid_mask = p00_values >= (confusion_p0g_values - 0.1)
            
            if not np.any(valid_mask):
                # No points pass filter - take the one with highest P(0|0)
                optimal_idx = np.argmax(p00_values)
            else:
                # Some points pass filter - optimize on P(1|1) among valid points
                # Set invalid points to -inf so they won't be selected
                p11_masked = np.where(valid_mask, p11_values, -np.inf)
                optimal_idx = np.argmax(p11_masked)
            
            # Keep clock cycles value for dataset selection
            optimal_depletion_time_cycles = int(ds.depletion_time.values[optimal_idx])
            # Convert from clock cycles back to nanoseconds (*4) for storing
            optimal_depletion_time = optimal_depletion_time_cycles * 4
            
            # Use data at optimal depletion_time for final results (use clock cycles for selection)
            state_g_1_opt = state_g_1.sel(depletion_time=optimal_depletion_time_cycles).values
            state_g_2_opt = state_g_2.sel(depletion_time=optimal_depletion_time_cycles).values
            state_e_1_opt = state_e_1.sel(depletion_time=optimal_depletion_time_cycles).values
            state_e_2_opt = state_e_2.sel(depletion_time=optimal_depletion_time_cycles).values
            
            # Store P(0|0), P(1|1), and P(0|g) as data variables for plotting
            fit_data_vars[f"demolition_p00_{q_name}"] = xr.DataArray(
                p00_values,
                dims=["depletion_time"],
                coords={"depletion_time": ds.depletion_time}
            )
            fit_data_vars[f"demolition_p11_{q_name}"] = xr.DataArray(
                p11_values,
                dims=["depletion_time"],
                coords={"depletion_time": ds.depletion_time}
            )
            fit_data_vars[f"confusion_p0g_{q_name}"] = xr.DataArray(
                confusion_p0g_values,
                dims=["depletion_time"],
                coords={"depletion_time": ds.depletion_time}
            )
        else:
            # No optimization, use data as-is
            state_g_1_opt = state_g_1.values
            state_g_2_opt = state_g_2.values
            state_e_1_opt = state_e_1.values
            state_e_2_opt = state_e_2.values
            optimal_depletion_time = None
        
        # Calculate confusion matrix (using first measurements)
        confusion_matrix = calculate_confusion_matrix(state_g_1_opt, state_e_1_opt)
        
        # Calculate demolition error matrices
        demolition_error_matrix_g = calculate_demolition_error_matrix(state_g_1_opt, state_g_2_opt)
        demolition_error_matrix_e = calculate_demolition_error_matrix(state_e_1_opt, state_e_2_opt)
        
        # Average demolition error matrix (or use ground state only, depending on preference)
        # Using average of both prepared states
        demolition_error_matrix = (demolition_error_matrix_g + demolition_error_matrix_e) / 2
        
        # Check if results are valid
        success = (
            not np.any(np.isnan(confusion_matrix)) and
            not np.any(np.isnan(demolition_error_matrix)) and
            np.all(confusion_matrix >= 0) and
            np.all(confusion_matrix <= 1) and
            np.all(demolition_error_matrix >= 0) and
            np.all(demolition_error_matrix <= 1)
        )
        
        fit_results[q_name] = FitParameters(
            confusion_matrix=confusion_matrix.tolist(),
            demolition_error_matrix=demolition_error_matrix.tolist(),
            success=success,
            optimal_depletion_time=optimal_depletion_time,
        )
    
    # Add fit data variables to dataset
    if fit_data_vars:
        ds_fit = xr.merge([ds, xr.Dataset(fit_data_vars)])
    else:
        ds_fit = ds
    
    return ds_fit, fit_results

