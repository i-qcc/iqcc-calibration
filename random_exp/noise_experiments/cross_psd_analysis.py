"""
Cross-PSD Analysis Implementation based on Yan et al. 2012
Implements the interleaved cross-PSD method to eliminate statistical white noise floor
while preserving underlying 1/f noise signals in binary time series data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import xarray as xr

def cross_psd_yan_method(data_q, time_stamp_q, dt=None):
    """
    Calculate Cross-PSD using Yan et al. interleaved method
    
    Parameters:
    -----------
    data_q : array-like
        Binary time series data (0s and 1s)
    time_stamp_q : array-like  
        Timestamps corresponding to data_q
    dt : float, optional
        Time step. If None, calculated from time_stamp_q
        
    Returns:
    --------
    dict containing:
        - frequencies : frequency axis
        - cross_psd : cross power spectral density
        - regular_psd : regular PSD for comparison
        - white_noise_floor : estimated white noise floor
        - interleaved_data : the two interleaved series used
    """
    
    # Convert to numpy arrays
    data_q = np.array(data_q)
    time_stamp_q = np.array(time_stamp_q)
    
    # Calculate time step if not provided
    if dt is None:
        dt = np.mean(np.diff(time_stamp_q))
    
    N = len(data_q)
    
    # Ensure even number of points for interleaving
    if N % 2 == 1:
        data_q = data_q[:-1]
        time_stamp_q = time_stamp_q[:-1]
        N = len(data_q)
    
    # Split into interleaved series (Yan et al. method)
    # z'_n = z_{2n-1} and z''_n = z_{2n} (n = 1, ..., N/2)
    z_prime = data_q[::2]  # Even indices (0, 2, 4, ...)
    z_double_prime = data_q[1::2]  # Odd indices (1, 3, 5, ...)
    
    # Calculate FFTs
    Z_prime = np.fft.fft(z_prime)
    Z_double_prime = np.fft.fft(z_double_prime)
    
    # Frequency axis (up to Nyquist frequency)
    freqs = np.fft.fftfreq(N//2, d=dt)
    freqs = freqs[:N//4 + 1]  # Only positive frequencies + DC
    
    # Cross-PSD calculation (Eq. 12 from Yan et al.)
    cross_psd = np.zeros_like(freqs, dtype=complex)
    
    # DC component (k=0)
    cross_psd[0] = (2*np.pi)**2 * 0.5 * Z_prime[0] * np.conj(Z_double_prime[0]) / (N//2 * dt)
    
    # Non-DC components (k≠0)
    for k in range(1, len(freqs)):
        cross_psd[k] = (2*np.pi)**2 * Z_prime[k] * np.conj(Z_double_prime[k]) / (N//2 * dt)
    
    # Regular PSD for comparison (Eq. 6 from Yan et al.)
    Z_full = np.fft.fft(data_q)
    regular_psd = np.zeros_like(freqs)
    
    # DC component
    regular_psd[0] = (2*np.pi)**2 * 0.5 * np.abs(Z_full[0])**2 * dt**2 / (N * dt)
    
    # Non-DC components  
    for k in range(1, len(freqs)):
        regular_psd[k] = (2*np.pi)**2 * np.abs(Z_full[k])**2 * dt**2 / (N * dt)
    
    # White noise floor estimation (Eq. 8 from Yan et al.)
    p = np.mean(data_q)  # Probability of switching
    sigma_b_squared = p * (1 - p)  # Variance of Bernoulli process
    white_noise_floor = (2*np.pi)**2 * sigma_b_squared * dt
    
    return {
        'frequencies': freqs,
        'cross_psd': np.abs(cross_psd),
        'regular_psd': regular_psd,
        'white_noise_floor': white_noise_floor,
        'interleaved_data': {
            'z_prime': z_prime,
            'z_double_prime': z_double_prime,
            'time_prime': time_stamp_q[::2],
            'time_double_prime': time_stamp_q[1::2]
        },
        'statistics': {
            'p': p,
            'sigma_b_squared': sigma_b_squared,
            'N': N,
            'dt': dt
        }
    }

def plot_cross_psd_results(results, qubit_name="Qubit"):
    """
    Plot Cross-PSD results with comparison to regular PSD
    
    Parameters:
    -----------
    results : dict
        Output from cross_psd_yan_method
    qubit_name : str
        Name for plot titles
    """
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    freqs = results['frequencies']
    cross_psd = results['cross_psd']
    regular_psd = results['regular_psd']
    white_noise_floor = results['white_noise_floor']
    
    # Plot 1: Cross-PSD vs Regular PSD
    ax1.loglog(freqs[1:], cross_psd[1:], 'b-', label='Cross-PSD (Yan et al.)', linewidth=2)
    ax1.loglog(freqs[1:], regular_psd[1:], 'r--', label='Regular PSD', alpha=0.7)
    ax1.axhline(y=white_noise_floor, color='k', linestyle=':', 
                label=f'White noise floor = {white_noise_floor:.2e}')
    
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power Spectral Density')
    ax1.set_title(f'{qubit_name}: Cross-PSD vs Regular PSD')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Noise reduction factor
    noise_reduction = regular_psd[1:] / cross_psd[1:]
    ax2.loglog(freqs[1:], noise_reduction, 'g-', linewidth=2)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Noise Reduction Factor')
    ax2.set_title(f'{qubit_name}: Noise Reduction (Regular PSD / Cross-PSD)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_1f_noise(cross_psd_results, freq_range=None):
    """
    Analyze 1/f noise characteristics from Cross-PSD results
    
    Parameters:
    -----------
    cross_psd_results : dict
        Output from cross_psd_yan_method
    freq_range : tuple, optional
        (f_min, f_max) frequency range for 1/f fitting
        
    Returns:
    --------
    dict containing 1/f noise parameters
    """
    
    freqs = cross_psd_results['frequencies']
    psd = cross_psd_results['cross_psd']
    
    # Remove DC component for fitting
    freqs_fit = freqs[1:]
    psd_fit = psd[1:]
    
    # Default frequency range for 1/f fitting
    if freq_range is None:
        # Focus on low frequencies where 1/f noise dominates
        f_min = freqs_fit[0]
        f_max = freqs_fit[len(freqs_fit)//4]  # Use first quarter of frequency range
    else:
        f_min, f_max = freq_range
    
    # Select frequency range for fitting
    mask = (freqs_fit >= f_min) & (freqs_fit <= f_max)
    freqs_fit_range = freqs_fit[mask]
    psd_fit_range = psd_fit[mask]
    
    # Fit 1/f noise: PSD = A / f^α
    # Linear fit in log space: log(PSD) = log(A) - α*log(f)
    log_freqs = np.log(freqs_fit_range)
    log_psd = np.log(psd_fit_range)
    
    # Linear regression
    coeffs = np.polyfit(log_freqs, log_psd, 1)
    alpha = -coeffs[0]  # 1/f exponent
    log_A = coeffs[1]  # Amplitude
    A = np.exp(log_A)
    
    # Calculate R-squared
    log_psd_fit = log_A - alpha * log_freqs
    ss_res = np.sum((log_psd - log_psd_fit)**2)
    ss_tot = np.sum((log_psd - np.mean(log_psd))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        'alpha': alpha,
        'A': A,
        'r_squared': r_squared,
        'freq_range': (f_min, f_max),
        'fitted_psd': A / freqs_fit_range**alpha
    }

# Example usage function
def analyze_ramsey_data_yan_style(data_q, time_stamp_q, qubit_name="Qubit"):
    """
    Complete Yan et al. style analysis of Ramsey data
    
    Parameters:
    -----------
    data_q : array-like
        Binary time series data
    time_stamp_q : array-like
        Timestamps
    qubit_name : str
        Qubit identifier
        
    Returns:
    --------
    dict containing all analysis results
    """
    
    # Calculate Cross-PSD
    cross_psd_results = cross_psd_yan_method(data_q, time_stamp_q)
    
    # Analyze 1/f noise
    noise_analysis = analyze_1f_noise(cross_psd_results)
    
    # Create plots
    fig = plot_cross_psd_results(cross_psd_results, qubit_name)
    
    return {
        'cross_psd_results': cross_psd_results,
        'noise_analysis': noise_analysis,
        'plot': fig,
        'summary': {
            'qubit_name': qubit_name,
            'data_length': len(data_q),
            'sampling_rate': 1/cross_psd_results['statistics']['dt'],
            'switching_probability': cross_psd_results['statistics']['p'],
            'white_noise_floor': cross_psd_results['white_noise_floor'],
            '1f_exponent': noise_analysis['alpha'],
            '1f_amplitude': noise_analysis['A'],
            'fit_quality': noise_analysis['r_squared']
        }
    }
