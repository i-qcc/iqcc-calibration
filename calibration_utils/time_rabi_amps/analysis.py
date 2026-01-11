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
    """Stores the relevant fit parameters for a single qubit at a single amplitude"""
    # Common fit results
    f: float  # Rabi frequency in GHz (since x-axis is in ns)
    success: bool
    fit_params: dict # Raw fit parameters [a, f, phi, offset]
    opt_dur_pi: float # Optimal pi-pulse duration in ns
    opt_dur_pi_half: float # Optimal pi/2-pulse duration in ns


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
) -> Tuple[xr.Dataset, Dict[str, Dict[float, FitParameters]]]:
    """
    Fit the raw data to an oscillation for each amplitude.
    Returns a nested dictionary: {qubit_name: {amp_factor: FitParameters}}
    """
    qubits = node.namespace["qubits"]
    fit_results = {}
    
    # Determine which signal to fit
    if node.parameters.use_state_discrimination:
        signal_name = "state"
    else:
        signal_name = "V" if "V" in ds else "I"
        
    signal_da = ds[signal_name]
    
    # Get amplitude factors from the dataset
    amp_factors = ds.amp_prefactor.values if "amp_prefactor" in ds.dims else []
    
    fit_evals = []
    
    for q in qubits:
        q_name = q.name
        fit_results[q_name] = {}
        qubit_fit_evals = []
        
        for amp_factor in amp_factors:
            try:
                # Get data for this qubit and amplitude, remove NaNs
                q_data = signal_da.sel(qubit=q_name, amp_prefactor=amp_factor).dropna(dim="duration")
                x_data = q_data.duration.values
                y_data = q_data.values

                if len(x_data) < 3:
                    raise ValueError(f"Insufficient data points for amplitude {amp_factor}")

                # --- 1. Make initial guesses (p0) ---
                guess_offset = y_data.mean()
                guess_a = (y_data.max() - y_data.min()) / 2
                guess_phi = 0
                
                # --- 2. FFT GUESS ---
                # Check that data is evenly spaced (critical for FFT)
                steps = np.diff(x_data)
                if not np.allclose(steps, steps[0], rtol=1e-3):
                    # If not evenly spaced, use a simple guess
                    guess_f_ghz = 1.0 / (x_data[-1] - x_data[0])  # Rough estimate
                else:
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
                    if len(yf_pos) > 1:
                        peak_index = np.argmax(yf_pos[1:]) + 1 
                        guess_f_ghz = xf_pos[peak_index]
                    else:
                        guess_f_ghz = 1.0 / (x_data[-1] - x_data[0])
                
                LOG.info(f"FFT guess for {q_name} at amp {amp_factor:.3f}: {guess_f_ghz * 1e3:.2f} MHz")
                
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
                
                # --- 4. Calculate optimal durations ---
                opt_dur_pi = 1 / (2 * f)
                opt_dur_pi_half = 1 / (4 * f)

                # --- 5. Store results ---
                fit_results[q_name][amp_factor] = FitParameters(
                    f=f,
                    success=True,
                    fit_params=fit_params_dict,
                    opt_dur_pi=opt_dur_pi,
                    opt_dur_pi_half=opt_dur_pi_half
                )
                
                # Generate the fitted curve for plotting
                fit_curve = oscillation(ds.duration, a, f, phi, offset)
                qubit_fit_evals.append(fit_curve)
                
            except Exception as e:
                LOG.warning(f"Fit failed for qubit {q_name} at amplitude {amp_factor:.3f}: {e}")
                # Store empty results on failure
                fit_results[q_name][amp_factor] = FitParameters(
                    f=0, success=False, fit_params={},
                    opt_dur_pi=0, opt_dur_pi_half=0
                )
                qubit_fit_evals.append(xr.full_like(ds.duration, np.nan))
        
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


def extract_rabi_frequencies(
    fit_results: Dict[str, Dict[float, FitParameters]]
) -> Dict[str, xr.Dataset]:
    """
    Extract rabi frequencies vs amplitude for each qubit.
    Returns a dictionary mapping qubit names to datasets with rabi frequency vs amplitude.
    """
    rabi_freqs = {}
    
    for q_name, amp_dict in fit_results.items():
        amp_factors = []
        frequencies = []
        success_flags = []
        
        for amp_factor in sorted(amp_dict.keys()):
            fit_params = amp_dict[amp_factor]
            amp_factors.append(amp_factor)
            frequencies.append(fit_params.f * 1e3)  # Convert to MHz
            success_flags.append(fit_params.success)
        
        # Create dataset
        rabi_freqs[q_name] = xr.Dataset({
            "rabi_frequency": (["amp_prefactor"], frequencies),
            "success": (["amp_prefactor"], success_flags)
        }, coords={"amp_prefactor": amp_factors})
    
    return rabi_freqs


def log_fitted_results(
    fit_results: Dict[str, Dict[float, dict]], # Nested dict: {qubit: {amp: fit_result_dict}}
    log_callable: Callable
):
    """Log the fitted results to the node's log."""
    for qubit_name, amp_dict in fit_results.items():
        log_callable(f"Qubit {qubit_name}:")
        successful_fits = sum(1 for fit_result in amp_dict.values() if fit_result.get("success", False))
        total_fits = len(amp_dict)
        log_callable(f"  Successful fits: {successful_fits}/{total_fits}")
        
        # Log a few example fits
        for amp_factor in sorted(amp_dict.keys())[:3]:
            fit_result = amp_dict[amp_factor]
            if fit_result.get("success", False):
                log_callable(f"  Amp {amp_factor:.3f}: Rabi Freq = {fit_result['f'] * 1e3:.2f} MHz, "
                           f"Pi-pulse = {fit_result['opt_dur_pi']:.1f} ns")
