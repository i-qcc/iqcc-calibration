import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Callable
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq  

# Use the generic QualibrationNode for functions
from qualibrate import QualibrationNode 
from qualibration_libs.data import convert_IQ_to_V
# Import the standard oscillation fitting functions
from qualibration_libs.analysis import fit_oscillation, oscillation
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)
LOG = logging.getLogger(__name__)


@dataclass
class FitParameters:
    """Stores the relevant fit parameters for a single qubit"""
    # Common fit results
    f: float  # Rabi frequency in GHz (since x-axis is in ns)
    success: bool
    fit_params: dict # Raw fit parameters [a, f, phi, offset]
    
    # --- Analysis 1: Duration ---
    opt_dur_pi: float # Optimal pi-pulse duration in ns
    opt_dur_pi_half: float # Optimal pi/2-pulse duration in ns
    
    # --- Analysis 2: Amplitude ---
    target_freq: float # Target frequency in MHz
    drive_amp_scale: float # The amp scale used in the experiment
    amp_fit: float # The new, calculated amplitude scale
    

def process_raw_dataset(
    ds: xr.Dataset, 
    node: QualibrationNode
) -> xr.Dataset:
    """
    Process the raw dataset, e.g., converting I, Q to Voltage.
    """
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    return ds


def fit_raw_data(
    ds: xr.Dataset, 
    node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Fit the raw data to an oscillation and extract parameters
    for both duration and amplitude analysis.
    """
    qubits = node.namespace["qubits"]
    fit_results = {}
    
    # Determine which signal to fit
    if node.parameters.use_state_discrimination:
        signal_name = "state"
    else:
        signal_name = "V" if "V" in ds else "I"
        
    signal_da = ds[signal_name]
        
    fit_evals = []
    
    for q in qubits:
        q_name = q.name
        try:
            # Get data for this qubit and remove NaNs
            q_data = signal_da.sel(qubit=q_name).dropna(dim="duration")
            x_data = q_data.duration.values
            y_data = q_data.values

            # --- 1. Make initial guesses (p0) ---
            guess_offset = y_data.mean()
            guess_a = (y_data.max() - y_data.min()) / 2
            guess_phi = 0
            
            # --- 2. START NEW FFT GUESS ---
            # Check that data is evenly spaced (critical for FFT)
            # The geomspace/unique/astype(int) in the main script creates
            # evenly spaced clock cycles, so x_data *should* be evenly spaced.
            steps = np.diff(x_data)
            if not np.allclose(steps, steps[0]):
                raise ValueError("x_data is not evenly spaced, cannot use FFT.")
            
            dt = steps[0]  # Sample spacing in ns
            N = len(y_data)
            
            # Demean the data (remove DC offset)
            y_data_demeaned = y_data - guess_offset
            
            # Perform FFT
            yf = fft(y_data_demeaned)
            # Get frequencies (in GHz, since dt is in ns)
            xf = fftfreq(N, dt)
            
            # Get positive frequencies only
            xf_pos = xf[:N//2]
            yf_pos = 2.0/N * np.abs(yf[0:N//2]) # Normalize
            
            # Find the strongest peak
            # We skip index 0 to avoid any remaining DC component
            peak_index = np.argmax(yf_pos[1:]) + 1 
            guess_f_ghz = xf_pos[peak_index]
            
            LOG.info(f"FFT guess for {q_name}: {guess_f_ghz * 1e3:.2f} MHz")
            # --- END NEW FFT GUESS ---
            
            p0 = [guess_a, guess_f_ghz, guess_phi, guess_offset]

            # --- 3. Perform the fit using scipy ---
            popt, pcov = curve_fit(oscillation, x_data, y_data, p0=p0)
            
            # Extract fit parameters
            a, f, phi, offset = popt
            
            # Create the dictionary for the dataclass
            fit_params_dict = {
                "a": a, "f": f, "phi": phi, "offset": offset
            }
            
            if f <= 0:
                raise ValueError("Fit frequency is zero or negative, cannot calculate.")
            
            # --- 4. Perform Analysis 1: Duration ---
            opt_dur_pi = 1 / (2 * f)
            opt_dur_pi_half = 1 / (4 * f)

            # --- 5. Perform Analysis 2: Amplitude ---
            target_freq_mhz = node.parameters.target_freq_in_Mhz
            drive_amp_scale = node.parameters.drive_amp_scale
            f_mhz = f * 1e3 
            amp_fit = drive_amp_scale * target_freq_mhz / f_mhz

            # --- 6. Store all results ---
            fit_results[q_name] = FitParameters(
                f=f,
                success=True,
                fit_params=fit_params_dict,
                opt_dur_pi=opt_dur_pi,
                opt_dur_pi_half=opt_dur_pi_half,
                target_freq=target_freq_mhz,
                drive_amp_scale=drive_amp_scale,
                amp_fit=amp_fit
            )
            
            # Generate the fitted curve for plotting
            fit_curve = oscillation(ds.duration, a, f, phi, offset)
            fit_evals.append(fit_curve)
            
        except Exception as e:
            LOG.warning(f"Fit failed for qubit {q_name}: {e}")
            # Store empty results on failure
            fit_results[q_name] = FitParameters(
                f=0, success=False, fit_params={},
                opt_dur_pi=0, opt_dur_pi_half=0,
                target_freq=node.parameters.target_freq_in_Mhz, 
                drive_amp_scale=node.parameters.drive_amp_scale, 
                amp_fit=0
            )
            fit_evals.append(xr.full_like(ds.duration, np.nan))

    # Create the dataset for fitted curves
    ds_fit = xr.concat(fit_evals, dim="qubit")
    ds_fit["qubit"] = ds["qubit"]
    
    return ds_fit, fit_results

def log_fitted_results(
    fit_results: Dict[str, dict], # <-- FIX: Expects dict, not FitParameters
    log_callable: Callable
):
    """Log the fitted results to the node's log."""
    for qubit_name, fit_result in fit_results.items():
        # *** FIX: Use dictionary key access ***
        if fit_result["success"]:
            log_callable(f"Qubit {qubit_name}: [SUCCESS]")
            
            # Analysis 1: Duration
            log_callable(f"  --- Duration Analysis ---")
            log_callable(f"  Rabi Freq. = {fit_result['f'] * 1e3:.2f} MHz")
            log_callable(f"  Pi-pulse   = {fit_result['opt_dur_pi']:.1f} ns")
            log_callable(f"  Pi/2-pulse = {fit_result['opt_dur_pi_half']:.1f} ns")
            
            # Analysis 2: Amplitude
            log_callable(f"  --- Amplitude Analysis ---")
            log_callable(f"  Target Freq: {fit_result['target_freq']:.2f} MHz")
        else:
            log_callable(f"Qubit {qubit_name}: [FAILED]")