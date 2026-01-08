import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


@dataclass
class FitParameters:
    """Stores the relevant readout frequency optimization G-E-F vs amp experiment fit parameters for a single qubit"""

    GEF_detuning: float
    GEF_amp: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    log_callable : callable, optional
        Callable for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_detuning = f"\tGEF readout frequency shift: {1e-6 * fit_results[q]['GEF_detuning']:.1f} MHz | "
        s_amp = f"GEF drive amplitude: {fit_results[q]['GEF_amp']:.3f}\n"
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        log_callable(s_qubit + s_detuning + s_amp)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Process the raw dataset for G-E-F readout frequency optimization."""
    # Convert IQ data into volts - need to specify all IQ variables
    ds = convert_IQ_to_V(ds, node.namespace["qubits"], IQ_list=["I_g", "Q_g", "I_e", "Q_e", "I_f", "Q_f"])
    
    # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2) for |g>, |e> and |f> 
    # as well as the distance between the blobs
    ds = ds.assign(
        {
            "Dge": np.sqrt((ds.I_g - ds.I_e) ** 2 + (ds.Q_g - ds.Q_e) ** 2),
            "Def": np.sqrt((ds.I_e - ds.I_f) ** 2 + (ds.Q_e - ds.Q_f) ** 2),
            "Dgf": np.sqrt((ds.I_g - ds.I_f) ** 2 + (ds.Q_g - ds.Q_f) ** 2),
            "IQ_abs_g": np.sqrt(ds.I_g**2 + ds.Q_g**2),
            "IQ_abs_e": np.sqrt(ds.I_e**2 + ds.Q_e**2),
            "IQ_abs_f": np.sqrt(ds.I_f**2 + ds.Q_f**2),
        }
    )
    
    # Define D as the minimum of Dge, Def, Dgf
    ds["D"] = ds[["Dge", "Def", "Dgf"]].to_array().min("variable")
    
    # Add the absolute frequency to the dataset
    qubit_freqs = {q.name: q.resonator.RF_frequency for q in node.namespace["qubits"]}
    full_freq = np.array([
        ds.detuning + qubit_freqs[q.name]
        for q in node.namespace["qubits"]
    ])
    
    ds = ds.assign_coords({
        "freq_full": (["qubit", "detuning"], full_freq),
    })
    ds.freq_full.attrs = {"long_name": "RF frequency", "units": "Hz"}
    
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the optimal readout frequency and amplitude for G-E-F discrimination.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node : QualibrationNode
        The calibration node with parameters.

    Returns:
    --------
    Tuple[xr.Dataset, dict[str, FitParameters]]
        Dataset containing the fit results and dictionary of fit parameters for each qubit.
    """
    ds_fit = ds
    
    # Get the readout detuning as the index of the maximum of the rolling average of D
    detuning = ds_fit.D.rolling({"detuning": 3}).mean("detuning").rolling({"amp_prefactor": 3}).mean("amp_prefactor")
    
    # Find the indices of maximum detuning for each qubit
    max_indices = detuning.argmax(dim=["detuning", "amp_prefactor"])
    
    # Extract the optimal freq and amp values
    optimal_freq = ds_fit.detuning[max_indices["detuning"]]
    optimal_amp = ds_fit.amp_prefactor[max_indices["amp_prefactor"]]
    
    # Add fit results to dataset
    ds_fit = ds_fit.assign({
        "optimal_detuning": (["qubit"], optimal_freq.values),
        "optimal_amp": (["qubit"], optimal_amp.values),
    })
    
    # Extract the relevant fitted parameters
    fit_results = {}
    for q in node.namespace["qubits"]:
        fit_results[q.name] = FitParameters(
            GEF_detuning=float(optimal_freq.sel(qubit=q.name).values),
            GEF_amp=float(optimal_amp.sel(qubit=q.name).values),
            success=True,
        )
    
    return ds_fit, fit_results

