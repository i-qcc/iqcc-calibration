import matplotlib.pylab as plt
import numpy as np
# from numpy.linalg import solve
from numpy.polynomial import Polynomial as P
from scipy.optimize import minimize, curve_fit
from scipy import linalg
from scipy.interpolate import interp1d
from functools import reduce
from typing import List, Tuple, Sequence, Literal
from typing import Optional
from numpy.typing import ArrayLike


def expdecay(x: np.ndarray, s: float, a: float, t: float) -> np.ndarray:
    """Exponential decay defined as s * (1 + a * np.exp(-x / t)).
    
    Parameters
    ----------
    x: np.ndarray for the time vector in ns
    s: float for the scaling factor
    a: float for the exponential amplitude
    t: float for the exponential decay time in ns
    
    Returns
    -------
    numpy array for the exponential decay
    """
    return s * (1 + a * np.exp(-(x) / t))


# def two_expdecay(x, s, a, t, a2, t2):
#     """Double exponential decay defined as s * (1 + a * np.exp(-x / t) + a2 * np.exp(-x / t2)).
#     :param x: numpy array for the time vector in ns
#     :param s: float for the scaling factor
#     :param a: float for the first exponential amplitude
#     :param t: float for the first exponential decay time in ns
#     :param a2: float for the second exponential amplitude
#     :param t2: float for the second exponential decay time in ns
#     :return: numpy array for the double exponential decay
#     """
#     return s * (1 + a * np.exp(-(x) / t) + a2 * np.exp(-(x) / t2))


def single_exp(da, plot=True):
    first_vals = da.sel(time=slice(0, 1)).mean().values
    final_vals = da.sel(time=slice(20, None)).mean().values
    print(first_vals, final_vals)

    fit = da.curvefit(
        "time", expdecay, p0={"a": 1 - first_vals / final_vals, "t": 50, "s": final_vals}
    ).curvefit_coefficients

    fit_vals = {k: v for k, v in zip(fit.to_dict()["coords"]["param"]["data"], fit.to_dict()["data"])}

    t_s = 1
    alpha = np.exp(-t_s / fit_vals["t"])
    A = fit_vals["a"]
    fir = [1 / (1 + A), -alpha / (1 + A)]
    iir = [(A + alpha) / (1 + A)]

    if plot:
        fig, ax = plt.subplots()
        ax.plot(da.time, da, label="data")
        ax.plot(da.time, expdecay(da.time, **fit_vals), label="fit")
        ax.grid("all")
        ax.legend()
        print(f"Qubit - FIR: {fir}\nIIR: {iir}")
    else:
        fig = None
        ax = None
    return fir, iir, fig, ax, (da.time, expdecay(da.time, **fit_vals))


def conv_causal(v: ArrayLike, h: ArrayLike, N: Optional[int] = None) -> np.ndarray:
    """
    Perform a causal (one-sided) convolution of input signal v with filter h.

    Parameters
    ----------
    v : array-like
        Input sequence (e.g., signal to be filtered)
    h : array-like
        Impulse response (filter coefficients)
    N : int or None, optional
        Number of output points to return. If None, returns the convolution up to the length of v.

    Returns
    -------
    y : ndarray
        The result of the causal convolution of v with h, truncated to N or len(v) samples.
    """
    v = np.asarray(v, dtype=float)
    h = np.asarray(h, dtype=float)
    y = np.convolve(v, h, mode='full')
    
    return y[: (len(v) if N is None else N)]


def build_toeplitz_matrix(v: ArrayLike, L: int) -> np.ndarray:
    """
    Construct a Toeplitz matrix from input sequence v for FIR system identification.

    Parameters
    ----------
    v : array-like
        Input sequence (stimulus to system).
    L : int
        Number of filter coefficients (FIR length).

    Returns
    -------
    V : ndarray, shape (N, L)
        Toeplitz matrix such that each row forms a lagged vector of v,
        suitable for linear convolution: phi ≈ V @ h.
    """
    v = np.asarray(v, dtype=float)
    N = len(v)
    V = np.zeros((N, L))
    for k in range(L):
        V[k:, k] = v[:N-k]
    
    return V


def resample_to_target_rate(
    data: ArrayLike, 
    original_Ts: float, 
    target_Ts: float, 
    kind: str = 'cubic'
    ) -> np.ndarray:
    """
    Resample time-domain data to a specified target sampling rate.
    
    Parameters
    ----------
    data : array-like
        The original time-domain data
    original_Ts : float
        The original sampling time, in nanoseconds
    target_Ts : float
        The target sampling time, in nanoseconds
    kind : str
        Interpolation kind ('linear', 'cubic', etc.)
    
    Returns
    -------
    resampled : numpy.ndarray
        The data interpolated onto the target sampling rate grid
    """
    
    data = np.asarray(data)
    N = len(data)
    t_original = np.arange(N) * original_Ts
    max_time = t_original[-1]
    
    num_samples = int(np.floor(max_time / target_Ts)) + 1
    t_target = np.arange(num_samples) * target_Ts
    t_target = t_target[t_target <= max_time]
    
    interp_fun = interp1d(t_original, data, kind=kind, fill_value="extrapolate", bounds_error=False)
    return interp_fun(t_target)


def fit_fir(
    phi: ArrayLike, 
    v: ArrayLike, 
    L: int, 
    Ts: float = 0.5, 
    lam1: float = 1e-2, 
    lam2: float = 1e-2, 
    tail_ns: Optional[float] = None
    ) -> np.ndarray:
    """
    Fit a finite impulse response (FIR) filter to data. Solves for FIR filter coefficients `h` 
    with length `L` such that `phi ≈ V @ h`, where `V` is the Toeplitz matrix constructed from 
    input `v`.

    Parameters
    ----------
    phi : array_like
        response signal (1D array).
    v : array_like
        Input signal to be filtered (1D array).
    L : int
        Number of FIR filter taps (length of FIR filter).
    Ts : float, optional
        Sampling time (in nanoseconds). Default is 0.5.
    lam1 : float, optional
        Regularization parameter for the identity matrix. Default is 1e-2.
    lam2 : float, optional
        Regularization parameter for exponential tail. Default is 1e-2.
    tail_ns : float, optional
        Time constant (in nanoseconds) for tail regularization. If None, computed as (L*Ts)/3.0.

    Returns
    -------
    h : ndarray
        Fitted FIR filter coefficients (1D array of length L).
    """
    phi = np.asarray(phi, float)
    v = np.asarray(v, float)
    V = build_toeplitz_matrix(v, L)

    if tail_ns is None:
        tail_ns = (L*Ts)/3.0
    idx = np.arange(L)
    x = np.exp(idx * Ts / tail_ns)

    A = V.T @ V + lam1*np.eye(L) + lam2*np.diag(x)
    b = V.T @ phi
    h = linalg.solve(A, b, assume_a='pos')
    
    return h


def optimize_fir_parameters(
    response: ArrayLike, 
    Ts: float = 0.5
    ) -> Tuple[List[dict], float, dict, np.ndarray, np.ndarray]:
    """
    Optimize FIR extraction parameters (L, lam1, lam2) to minimize the reconstruction error
    between the measured response and the reconstructed signal.

    Parameters
    ----------
        response: array_like
            response signal (1D array).
        Ts: float, optional
            Sampling time (in nanoseconds). Default is 0.5.
    
    Returns
    -------
        best_error: minimum NRMS error found
        best_params: dictionary of optimal parameters (L, lam1, lam2, error, h, reconstructed)
        best_h: best FIR filter coefficients found
        best_reconstructed: reconstructed signal using the best FIR filter
        results: list of dictionaries of all results for each parameter combination
    """
    print("="*70)
    print("OPTIMIZING FIR EXTRACTION PARAMETERS")
    print("="*70)
    print("Searching over L, lam1, lam2 to minimize reconstruction error")
    print("Error = ||response - reconstructed|| / ||response||\n")
    print(f"Signal length: {len(response)} samples")
    print(f"Time span: {(len(response) * Ts):.1f} ns")
    print(f"Sampling time: {Ts:.2f} ns")
    print(f"Sampling rate: {1/Ts:.2f} GS/s\n")

    # Parameter ranges to search
    L_values = [16, 20, 24, 28, 32, 40, 48]  # FIR filter lengths
    lam1_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]   # Regularization 1
    lam2_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]   # Regularization 2

    total_combinations = len(L_values) * len(lam1_values) * len(lam2_values)
    print(f"Total parameter combinations to test: {total_combinations}")
    print(f"L: {L_values}")
    print(f"lam1: {lam1_values}")
    print(f"lam2: {lam2_values}")
    print("\nStarting optimization...\n")

    results = []
    best_error = float('inf')
    best_params = None
    best_h = None
    best_reconstructed = None

    # Ideal signal assumed as a step function
    ideal_response = np.ones(len(response))

    iteration = 0
    for L in L_values:
        for lam1 in lam1_values:
            for lam2 in lam2_values:
                iteration += 1
                try:
                    # Extract forward FIR filter
                    h = fit_fir(response, ideal_response, L=L, Ts=Ts, lam1=lam1, lam2=lam2)
                    h /= np.sum(h)  # Normalize
                    
                    # Reconstruct signal using Toeplitz matrix
                    V = build_toeplitz_matrix(ideal_response, L)
                    reconstructed = V @ h

                    # Truncate or pad reconstructed to match length
                    if len(reconstructed) > len(response):
                        reconstructed = reconstructed[:len(response)]
                    elif len(reconstructed) < len(response):
                        reconstructed = np.pad(reconstructed, (0, len(response) - len(reconstructed)), 
                                              mode='edge')

                    # Compute reconstruction error (NRMS)
                    reconstruction_error = np.linalg.norm(response - reconstructed) / np.linalg.norm(response)
                    
                    # Store results
                    result = {
                        'L': L,
                        'lam1': lam1,
                        'lam2': lam2,
                        'error': reconstruction_error,
                        'h': h.copy(),
                        'reconstructed': reconstructed.copy()
                    }
                    results.append(result)
                    
                    # Track best
                    if reconstruction_error < best_error:
                        best_error = reconstruction_error
                        best_params = result
                        best_h = h.copy()
                        best_reconstructed = reconstructed.copy()

                    # Progress update
                    if iteration % 20 == 0:
                        print(f"  Tested {iteration}/{total_combinations} combinations... "
                              f"Best error so far: {best_error:.4e}")
                        
                except Exception as e:
                    print(f"  Warning: Failed for L={L}, lam1={lam1:.0e}, lam2={lam2:.0e}: {e}")
                    continue

    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total combinations tested: {len(results)}")

    return results, best_error, best_params, best_h, best_reconstructed


def analyze_and_plot_fir_fit(
    response: ArrayLike,
    time: ArrayLike,
    Ts: float = 0.5,
    verbose: bool = True
    ) -> Tuple[np.ndarray, dict, np.ndarray, float]:
    """
    For a given response signal, analyze the FIR filter fit that best reconstructs the response 
    from the ideal step response and plot the results.

    Parameters
    ----------
    response : ArrayLike
        The measured (distorted) response signal to analyze, typically normalized amplitude data.
    time : ArrayLike
        The corresponding time array for the response signal (in ns).
    Ts : float, optional
        The sampling interval (in ns). Default is 0.5.
    verbose: bool, optional
        Whether to print verbose output. Default is True.

    Returns
    -------
    best_h : np.ndarray
        The coefficients of the best forward FIR filter found.
    best_params : dict
        Dictionary of the optimal FIR filter hyperparameters.
    best_reconstructed : np.ndarray
        The best reconstructed (filtered) signal from the optimization.
    best_error : float
        Normalized root mean square error (NRMS) for the optimal filter parameters.

    Notes
    -----
    The function also produces a set of diagnostic plots visualizing the reconstruction and residuals.
    """
    results, best_error, best_params, best_h, best_reconstructed = optimize_fir_parameters(response, Ts=Ts)
    if best_params is not None and verbose:
        print(f"\nBEST PARAMETERS:")
        print(f"  L = {best_params['L']}")
        print(f"  lam1 = {best_params['lam1']:.0e}")
        print(f"  lam2 = {best_params['lam2']:.0e}")
        print(f"\nPERFORMANCE:")
        print(f"  Reconstruction error: {best_error:.4e} ({best_error*100:.2f}%)")
        print(f"  Max |h|: {np.max(np.abs(best_h)):.4e}")
        print(f"  h.sum(): {np.sum(best_h):.6f}")
        
        # Visualize best result
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Signal comparison
        ax = axes[0, 0]
        ax.plot(time, response, 'r-', label='Measured (distorted)', linewidth=2, alpha=0.7)
        ax.plot(time, best_reconstructed, 'b--', 
                label=f'Reconstructed (error={best_error:.4e})', linewidth=2)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Best Reconstruction Result')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Residual
        ax = axes[0, 1]
        residual = response - best_reconstructed
        ax.plot(time, residual, 'm-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.fill_between(time, -np.std(residual), np.std(residual), alpha=0.2, color='gray')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Residual')
        ax.set_title(f'Reconstruction Residual (σ={np.std(residual):.4e})')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Best FIR filter
        ax = axes[1, 0]
        ax.plot(best_h, 'b-o', markersize=4, linewidth=2)
        ax.set_xlabel('Tap Index')
        ax.set_ylabel('Coefficient Value')
        ax.set_title(f'Best Forward FIR (L={best_params["L"]})')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Error distribution across parameters
        ax = axes[1, 1]
        errors_by_L = {}
        for r in results:
            if r['L'] not in errors_by_L:
                errors_by_L[r['L']] = []
            errors_by_L[r['L']].append(r['error'])
        
        L_sorted = sorted(errors_by_L.keys())
        error_means = [np.mean(errors_by_L[L]) for L in L_sorted]
        error_stds = [np.std(errors_by_L[L]) for L in L_sorted]
        
        ax.errorbar(L_sorted, error_means, yerr=error_stds, fmt='o-', capsize=5, linewidth=2, markersize=6)
        ax.axhline(y=best_error, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_error:.4e}')
        ax.set_xlabel('Filter Length L')
        ax.set_ylabel('Reconstruction Error')
        ax.set_title('Error vs Filter Length')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Show top 10 results
        print(f"\n{'='*70}")
        print("TOP 10 PARAMETER COMBINATIONS:")
        print(f"{'='*70}")
        sorted_results = sorted(results, key=lambda x: x['error'])[:10]
        print(f"{'Rank':<6} {'L':<4} {'lam1':<8} {'lam2':<8} {'Error':<12}")
        print("-"*70)
        for i, r in enumerate(sorted_results, 1):
            print(f"{i:<6} {r['L']:<4} {r['lam1']:<8.0e} {r['lam2']:<8.0e} {r['error']:<12.4e}")
        print("="*70)
        
    elif verbose:
        print("ERROR: No valid parameter combinations found!")
        print("Try adjusting parameter ranges or check your data.")

    return best_h, best_params, best_reconstructed, best_error


def invert_fir(
    h: ArrayLike, 
    Ts: float = 0.5, 
    M: Optional[int] = None,
    method: Literal['optimization', 'analytical'] = 'optimization',
    sigma_ns: float = 0.75, 
    lam_smooth: float = 5e-2
    ) -> np.ndarray:
    """
    Invert a causal FIR filter by finding an inverse causal FIR (possibly smoothed).

    Solves:
        min_{h_inv} || d - (h * h_inv) ||^2 + lam_smooth * || Δ h_inv ||^2

    where:
        - h is the original causal FIR filter coefficients (array-like, length L)
        - h_inv is the inverse FIR filter coefficients (array-like, length M)
        - d is a desired target response (by default, a normalized Gaussian to approximate a causal impulse)
        - * denotes convolution (implemented via Toeplitz matrix multiplication)
        - Δ is the first difference operator (to penalize roughness in h_inv)
        - lam_smooth is the regularization/smoothing parameter

    Parameters
    ----------
    h : array_like
        Original FIR filter coefficients (causal, length L).
    Ts : float, optional
        Sample spacing (default is 0.5).
    M : int, optional
        Length of the inverse FIR filter coefficients to solve for (default is len(h)).
    method : str, optional
        Method to use for inversion ('optimization' or 'analytical').
    sigma_ns : float, optional
        Standard deviation of target Gaussian for delta approximation (default is 1.0).
    lam_smooth : float, optional
        Regularization parameter for first-difference smoothing of h_inv (default is 5e-2).

    Returns
    -------
    h_inv : ndarray
        FIR inverse coefficients (length M), optionally DC gain-normalized.

    Notes
    -----
    - This routine finds a stable, causal FIR approximate inverse of a given FIR.
    - Regularization improves stability for nearly non-minimum-phase FIRs.
    - The returned inverse may not be exact due to smoothing and length limits.
    """
    h = np.asarray(h, float)
    L = len(h)
    if M is None:
        M = L

    if method == 'optimization':
        # target 'delta'
        t = np.arange(M)*Ts
        d = np.exp(-0.5*(t/(sigma_ns))**2)
        d /= d.sum()  # unit DC gain

        # Build Toeplitz conv matrix H for causal conv(h, h_inv) truncated to M
        H = build_toeplitz_matrix(h, M)[:M, :]

        # First-difference (Sobolev) smoothing
        D = np.eye(M, k=0) - np.eye(M, k=1)
        D = D[:-1, :]  # (M-1) x M

        A = H.T @ H + lam_smooth * (D.T @ D)
        b = H.T @ d
        h_inv = linalg.solve(A, b, assume_a='pos')

        # Optional: normalize composite DC gain to 1
        gain = (h.sum() * h_inv.sum())
        if gain != 0:
            h_inv /= gain

    elif method == 'analytical':
        h_inv = np.zeros(L)
    
        # First condition
        h_inv[0] = 1 / h[0]
        
        # Recursive computation
        for m in range(1, L):
            s = 0
            for i in range(1, m+1):
                s += h_inv[m-i] * h[i]
            h_inv[m] = -s / h[0]
    
    return h_inv


def analyze_and_plot_inverse_fir(
    response: ArrayLike,
    time: ArrayLike,
    Ts: float = 0.5,
    M: Optional[int] = None,
    sigma_ns: float = 0.75, 
    lam_smooth: float = 5e-2, 
    method: Literal['optimization', 'analytical'] = 'optimization',
    verbose: bool = True
    ) -> np.ndarray:
    """
    Analyze and plot the inverse FIR filter for a given FIR filter.

    Parameters
    ----------
    
    response: ArrayLike
        The response signal to analyze.
    time: ArrayLike
        The time array for the response signal.
    Ts: float, optional
        The sampling interval (in ns). Default is 0.5.
    M: int, optional
        The length of the inverse FIR filter coefficients to solve for (default is len(h)).
    sigma_ns: float, optional
        The standard deviation of the target Gaussian for delta approximation (default is 0.75).
    lam_smooth: float, optional
        The regularization parameter for first-difference smoothing of h_inv (default is 5e-2).
    method: Literal['optimization', 'analytical'], optional
        The method to use for inversion ('optimization' or 'analytical'). Default is 'optimization'.
    verbose: bool, optional
        Whether to print verbose output. Default is True.
    """
    best_h, best_params, best_reconstructed, best_error = analyze_and_plot_fir_fit(
        response=response, 
        time=time, 
        Ts=Ts,
        verbose=verbose
        )
    best_h /= np.sum(best_h)
    h_inv = invert_fir(
        h=best_h, 
        Ts=Ts, 
        M=M, 
        method=method, 
        sigma_ns=sigma_ns, 
        lam_smooth=lam_smooth
        )
    delta = conv_causal(best_h, h_inv, N=len(best_h))
    ideal_response = np.ones(len(response))
    
    # ===========================================
    # Compute Corrected Signal
    # ===========================================

    # Method 1: Predistort ideal signal, then apply forward distortion
    # (This simulates what would happen in practice)
    L_guard = len(h_inv)
    guard = np.zeros(L_guard)

    # Pad ideal signal
    ideal_padded = np.concatenate([guard, ideal_response, guard])

    # Apply predistortion
    predistorted_padded = conv_causal(ideal_padded, h_inv)

    # Extract central region
    start = L_guard
    end = start + len(ideal_response)
    predistorted_response = predistorted_padded[start:end]

    # Apply forward distortion (simulate what happens in hardware)
    corrected_response = conv_causal(predistorted_response, best_h, N=len(ideal_response))

    # Compute correction error
    correction_error = np.linalg.norm(corrected_response - ideal_response) / np.linalg.norm(ideal_response)
    print(f"Correction error (NRMS): {correction_error:.3e}")

    # Method 2: Apply inverse directly to measured signal
    # (Alternative approach - corrects the measured signal directly)
    distorted_padded = np.concatenate([guard, response, guard])
    corrected_from_measured_padded = conv_causal(distorted_padded, h_inv)
    corrected_from_measured = corrected_from_measured_padded[start:end]

    correction_error_measured = np.linalg.norm(corrected_from_measured - ideal_response) / np.linalg.norm(ideal_response)
    print(f"Correction error (from measured, NRMS): {correction_error_measured:.3e}")

    # ===========================================
    # Visualization
    # ===========================================
    print("\n" + "="*70)
    print("Step 4: Visualization")
    print("="*70)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # Plot 1: Original signals
    ax = axes[0, 0]
    ax.plot(time, ideal_response, 'g--', label='Ideal Signal', linewidth=2, alpha=0.7)
    ax.plot(time, response, 'r-', label='Distorted Signal', linewidth=2)
    ax.plot(time, best_reconstructed, 'b:', label='Predicted (from FIR)', linewidth=2, alpha=0.7)
    ax.axhline(y=1.001, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=0.999, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_ylim([0.95,1.05])
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Original Signals and FIR Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: FIR Filters
    ax = axes[0, 1]
    ax.plot(best_h, 'b-o', label='Forward FIR (h)', markersize=4, linewidth=2)
    ax.plot(h_inv, 'r-s', label='Inverse FIR (h_inv)', markersize=4, linewidth=2)
    ax.set_xlabel('Tap Index')
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Extracted FIR Filters')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Predistorted and Corrected Signals
    ax = axes[1, 0]
    ax.plot(time, ideal_response, 'g--', label='Ideal Signal', linewidth=2, alpha=0.7)
    ax.plot(time, predistorted_response, 'c-', label='Predistorted Signal', linewidth=2, alpha=0.7)
    ax.plot(time, corrected_response, 'm-', label='Corrected Signal (sim)', linewidth=2)
    ax.axhline(y=1.001, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=0.999, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_ylim([0.95,1.05])
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Predistortion and Correction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Correction Comparison
    ax = axes[1, 1]
    ax.plot(time, ideal_response, 'g--', label='Ideal Signal', linewidth=2, alpha=0.7)
    ax.plot(time, response, 'r-', label='Distorted Signal', linewidth=2, alpha=0.5)
    ax.plot(time, corrected_response, 'm-', label='Corrected (predistort method)', linewidth=2)
    ax.plot(time, corrected_from_measured, 'orange', linestyle='-', 
            label='Corrected (from measured)', linewidth=2, alpha=0.7)
    ax.axhline(y=1.001, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=0.999, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_ylim([0.95,1.05])
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Correction Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Error Analysis
    ax = axes[2, 0]
    residual_fit = response - best_reconstructed
    residual_correction = corrected_response - ideal_response
    ax.plot(time, residual_fit, 'b-', label=f'Fit Residual (σ={np.std(residual_fit):.4e})', linewidth=1.5)
    ax.plot(time, residual_correction, 'm-', label=f'Correction Residual (σ={np.std(residual_correction):.4e})', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Residual')
    ax.set_title('Residual Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: h * h_inv (should be close to delta)
    ax = axes[2, 1]
    t_delta = np.arange(len(delta)) * Ts * 1e9  # Convert Ts (seconds) to ns for plotting
    ax.plot(t_delta, delta, 'g-o', markersize=4, linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'h * h_inv (should approximate δ, peak={np.max(np.abs(delta)):.3e})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    if verbose:
        # ===========================================
        # Summary
        # ===========================================
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Forward FIR (h):")
        print(f"  - Length: {len(best_h)}")
        print(f"  - lam1: {best_params['lam1']:.0e}")
        print(f"  - lam2: {best_params['lam2']:.0e}")
        print(f"  - Sum: {np.sum(best_h):.6f}")
        print(f"  - Max coefficient: {np.max(np.abs(best_h)):.6f}")
        print(f"  - Fit error: {best_error:.3e} ({best_error*100:.2f}%)")

        print(f"\nInverse FIR (h_inv):")
        print(f"  - Length: {len(h_inv)}")
        print(f"  - Sum: {np.sum(h_inv):.6f}")
        print(f"  - Max coefficient: {np.max(np.abs(h_inv)):.6f}")

        print(f"\nCorrection Performance:")
        print(f"  - Correction error (predistort method): {correction_error:.3e} ({correction_error*100:.2f}%)")
        print(f"  - Correction error (from measured): {correction_error_measured:.3e} ({correction_error_measured*100:.2f}%)")

        print(f"\nFilter Coefficients (for use in hardware):")
        print(f"  Forward FIR (h): {best_h[:10]}..." if len(best_h) > 10 else f"  Forward FIR (h): {best_h}")
        print(f"  Inverse FIR (h_inv): {h_inv[:10]}..." if len(h_inv) > 10 else f"  Inverse FIR (h_inv): {h_inv}")
        print("="*70)

    return best_h, h_inv


# def estimate_fir_coefficients(convolved_signal, step_response, num_coefficients):
#     """
#     Estimate the FIR filter coefficients from a convolved signal.

#     :param convolved_signal: The signal after being convolved with the FIR filter.
#     :param step_response: The original step response signal.
#     :param num_coefficients: Number of coefficients of the FIR filter to estimate.
#     :return: Estimated FIR coefficients.
#     """
#     # Deconvolve to estimate the impulse response
#     estimated_impulse_response, _ = deconvolve(convolved_signal, step_response)

#     # Truncate or zero-pad the estimated impulse response to match the desired number of coefficients
#     if len(estimated_impulse_response) > num_coefficients:
#         # Truncate if the estimated response is longer than the desired number of coefficients
#         estimated_coefficients = estimated_impulse_response[:num_coefficients]
#     else:
#         # Zero-pad if shorter
#         estimated_coefficients = np.pad(
#             estimated_impulse_response, (0, num_coefficients - len(estimated_impulse_response)), "constant"
#         )

#     return estimated_coefficients


def gaussian(x, a, x0, sigma, offset):
    """Gaussian function for fitting spectroscopy peaks.
    
    Args:
        x (array): X-axis values
        a (float): Amplitude
        x0 (float): Center position
        sigma (float): Width parameter
        offset (float): Vertical offset
    
    Returns:
        array: Gaussian values
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset


def fit_gaussian(freqs, states):
    """Fit Gaussian to spectroscopy data and return center frequency.
    
    Args:
        freqs (array): Frequency points
        states (array): Measured states
        
    Returns:
        float: Center frequency or np.nan if fit fails
    """
    p0 = [
        np.max(states) - np.min(states),   # amplitude
        freqs[np.argmax(states)],          # center
        (freqs[-1] - freqs[0]) / 10,        # width
        np.min(states)                     # offset
    ]
    try:
        popt, _ = curve_fit(gaussian, freqs, states, p0=p0)
        return popt[1]  # center frequency
    except RuntimeError:
        return np.nan


def single_exp_decay(t, amp, tau):
    """Single exponential decay without offset
    
    Args:
        t (array): Time points
        amp (float): Amplitude of the decay
        tau (float): Time constant of the decay
        
    Returns:
        array: Exponential decay values
    """
    return amp * np.exp(-t/tau)


def sequential_exp_fit(
    t: np.ndarray, 
    y: np.ndarray, 
    start_fractions: List[float], 
    fixed_taus: List[float]=None,
    a_dc: float=None, 
    verbose: bool=True, 
    ) -> Tuple[List[Tuple[float, float]], float, np.ndarray]:
    """
    Fit multiple exponentials sequentially by:
    1. First fit a constant term from the tail of the data
    2. Fit the longest time constant using the latter part of the data
    3. Subtract the fit
    4. Repeat for faster components
    
    Args:
        t (array): Time points in nanoseconds
        y (array): Data points (normalized amplitude)
        start_fractions (list): List of fractions (0 to 1) indicating where to start fitting each component
        fixed_taus (list, optional): Fixed tau values for each exponential component. 
                                   If provided, only amplitudes are fitted, taus are constrained.
                                   Must have same length as start_fractions.
        a_dc (float, optional): Fixed constant term. If provided, the constant term is not fitted.
        verbose (bool): Whether to print detailed fitting information   
        
    Returns:
        tuple: (components, a_dc, residual) where:
            - components: List of (amplitude, tau) pairs for each fitted component
            - a_dc: Fitted constant term or the fixed constant term
            - residual: Residual after subtracting all components
    """
    
    components = []  # List to store (amplitude, tau) pairs
    t_offset = t - t[0]  # Make time start at 0
    
    # Find the flat region in the tail by looking at local variance
    window = max(5, len(y) // 20)  # Window size by dividing signal into 20 equal pieces or at least 5 points
    rolling_var = np.array([np.var(y[i:i+window]) for i in range(len(y)-window)])
    # Find where variance drops below threshold, indicating flat region
    var_threshold = np.mean(rolling_var) * 0.1  # 10% of mean variance
    if a_dc is None:
        try:
            flat_start = np.where(rolling_var < var_threshold)[0][-1]
            # Use the flat region to estimate constant term
            a_dc = np.mean(y[flat_start:])
        except IndexError:
            print("No flat region found, using last point of the signal as constant term")
            a_dc = np.mean(y[-window:])

    if verbose:
        print(f"\nFitted constant term: {a_dc:.3e}")
    
    y_residual = y.copy() - a_dc
    
    for i, start_frac in enumerate(start_fractions):
        # Calculate start index for this component
        start_idx = int(len(t) * start_frac)
        if verbose:
            print(f"\nFitting component {i+1} using data from t = {t[start_idx]:.1f} ns (fraction: {start_frac:.3f})")
        
        # Fit current component
        try:
            # Prepare fitting parameters based on whether tau is fixed
            if fixed_taus is not None:
                # Use fixed tau - only fit amplitude using lambda
                tau_fixed = fixed_taus[i]
                p0 = [y_residual[start_idx]]  # Only amplitude initial guess
                if verbose:
                    print(f"Using fixed tau = {tau_fixed:.3f} ns")
                
                # Perform the fit on the current interval
                t_fit = t_offset[start_idx:]
                y_fit = y_residual[start_idx:]
                popt, _ = curve_fit(lambda t, amp: single_exp_decay(t, amp, tau_fixed), t_fit, y_fit, p0=p0)
                
                # Store the components
                amp = popt[0]
                tau = tau_fixed
            else:
                # Fit both amplitude and tau (original behavior)
                p0 = [
                    y_residual[start_idx],  # amplitude
                    t_offset[start_idx] / 3  # tau
                ]
                
                # Set bounds for the fit
                bounds = (
                    [-np.inf, 0.1],  # lower bounds: amplitude can be negative, tau must be positive (0.1 ns is arbitrary)
                    [np.inf, np.inf]  # upper bounds
                )
                
                # Perform the fit on the current interval
                t_fit = t_offset[start_idx:]
                y_fit = y_residual[start_idx:]
                popt, _ = curve_fit(single_exp_decay, t_fit, y_fit, p0=p0, bounds=bounds)
                
                # Store the components
                amp, tau = popt
            
            components.append((amp, tau))
            if verbose:
                tau_status = "(fixed)" if fixed_taus is not None else ""
                print(f"Found component: amplitude = {amp:.3e}, tau = {tau:.3f} ns {tau_status}")
            
            # Subtract this component from the entire signal
            y_residual -= amp * np.exp(-t_offset/tau)
            
        except (RuntimeError, ValueError) as e:
            if verbose:
                print(f"Warning: Fitting failed for component {i+1}: {e}")
            break
    
    return components, a_dc, y_residual


def optimize_start_fractions(
    t: np.ndarray, 
    y: np.ndarray, 
    base_fractions: List[float], 
    bounds_scale: float=0.5, 
    fixed_taus: List[float]=None, 
    a_dc: float=None, 
    verbose: bool=True
    ) -> Tuple[bool, List[float], List[Tuple[float, float]], float, float]:
    """
    Optimize the start_fractions by minimizing the RMS between the data and the fitted sum 
    of exponentials using scipy.optimize.minimize.
    
    Args:
        t (array): Time points in nanoseconds
        y (array): Data points (normalized amplitude)
        base_fractions (list): Initial guess for start fractions
        bounds_scale (float): Scale factor for bounds around base fractions (0.5 means ±50%)
        fixed_taus (list, optional): Fixed tau values for each exponential component. 
                                   If provided, only amplitudes are fitted, taus are constrained.
                                   Must have same length as base_fractions.
        a_dc (float, optional): Constant term. If not provided, the constant term is fitted from 
                                the tail of the data.
        verbose (bool): Whether to print detailed fitting information
    Returns:
        tuple: (success, best_fractions, best_components, best_dc, best_rms)
    """
    # Validate fixed_taus parameter
    if fixed_taus is not None:
        if len(fixed_taus) != len(base_fractions):
            raise ValueError("fixed_taus must have the same length as base_fractions")
        if any(tau <= 0 for tau in fixed_taus):
            raise ValueError("All fixed_taus values must be positive")
    
    def objective(x):
        """
        Objective function to minimize: RMS between the data and the fitted sum of 
        exponentials.
        """
        # Ensure fractions are ordered in descending order
        if not np.all(np.diff(x) < 0):
            return 1e6  # Return large value if constraint is violated
                
        components, _, residual = sequential_exp_fit(t, y, x, fixed_taus=fixed_taus, a_dc=a_dc, verbose=verbose)
        if len(components) == len(base_fractions):
            current_rms = np.sqrt(np.mean(residual**2))
        else:
            current_rms = 1e6 # Return large value if fitting fails
            
        return current_rms


    # Define bounds for optimization
    bounds = []
    for base in base_fractions:
        min_val = base * (1 - bounds_scale)
        max_val = base * (1 + bounds_scale)
        bounds.append((min_val, max_val))
    
    print("\nOptimizing start_fractions using scipy.optimize.minimize...")
    print(f"Initial values: {[f'{f:.5f}' for f in base_fractions]}")
    print(f"Bounds: ±{bounds_scale*100}% around initial values")
    
    # Run optimization
    result = minimize(
        objective,
        x0=base_fractions,
        bounds=bounds,
        method='Nelder-Mead',  # This method works well for non-smooth functions
        options={'disp': True, 'maxiter': 200}
    )
    
    # Get final results
    if result.success:
        best_fractions = result.x
        components, a_dc, best_residual = sequential_exp_fit(t, y, best_fractions, fixed_taus=fixed_taus, a_dc=a_dc, verbose=False)
        best_rms = np.sqrt(np.mean(best_residual**2))
        print("\nOptimization successful!")
        print(f"Initial fractions: {[f'{f:.5f}' for f in base_fractions]}")
        print(f"Optimized fractions: {[f'{f:.5f}' for f in best_fractions]}")
        if fixed_taus is not None:
            print(f"Fixed taus: {[f'{tau:.3f} ns' for tau in fixed_taus]}")
        print(f"Final RMS: {best_rms:.3e}")
        print(f"Number of iterations: {result.nit}")
    else:
        print("\nOptimization failed. Using initial values.")
        best_fractions = base_fractions
        components, a_dc, best_residual = sequential_exp_fit(t, y, best_fractions, fixed_taus=fixed_taus, a_dc=a_dc, verbose=False)
        best_rms = np.sqrt(np.mean(best_residual**2))
    
    return result.success, best_fractions, components, a_dc, best_rms


# The functions below are used to decompose the sum of exponentials to a cascade of single 
# exponent filters, as implemented in QOP 3.4.    
def add_rational_terms(terms: List[Tuple[np.array, np.array]]) -> Tuple[np.array, np.array]:
    # Convert to Polynomial objects
    rational_terms = [(P(num), P(den)) for num, den in terms]

    # Compute common denominator
    common_den = reduce(lambda acc, t: acc * t[1], rational_terms, P([1]))

    # Adjust numerators to have the common denominator
    adjusted_numerators = []
    for num, den in rational_terms:
        multiplier = common_den // den
        adjusted_numerators.append(num * multiplier)

    # Sum all adjusted numerators
    final_numerator = sum(adjusted_numerators, P([0]))

    # Return as coefficient lists
    return final_numerator.coef, common_den.coef


def decompose_exp_sum_to_cascade(A: Sequence, tau: Sequence, A_dc: float=1.,
                             compensate_v34_fpga_scale: bool=True, Ts: float=0.5) -> \
        tuple[np.ndarray, np.ndarray, float]:
    """decompose_exp_sum_to_cascade
    Translate from filters configuration as defined in QUA for version 3.5 (sum of exponents) to the
    definition of version 3.4.1 (cascade of single exponents filters).
    In v3.5, the analog linear distortion H is characterized by step response:
    s_H(t) = (A_dc + sum(A[i] * exp(-t/tau[i]), for i in 0...(N-1)))*u(t)
    In v3.4.1, it is a cascade of single exponent filters, each with step response:
    s_H_i(t) = (1 + A_c[i] * exp(-t/tau_c[i]))*u(t)
    The parameters [(A_c[0], tau_c[0]), ...] are the definitions of the filters (under "exponents")
    in 3.4.1.
    To make the filters equivalent, the 3.4.1 cascade needs to scaled by the parameter scale.
    This scaling can be done by multiplying the FIR coefficients by scale, or by scaling the waverform
    amp accordingly.
    :return A_c, tau_c, scale
    """

    assert A_dc > 0.2, "HPF mode is currently not supported"

    ba_sum = [get_rational_filter_single_exp_cont_time(A_i, tau_i) for A_i, tau_i in zip(A, tau)]
    ba_sum += [([A_dc], [1])]

    b, a = add_rational_terms(ba_sum)

    zeros = np.sort(np.roots(b))
    poles = np.sort(np.roots(a))

    assert np.all(np.isreal(zeros)) and np.all(np.isreal(poles)), \
        "Got complex zeros; this configuration can't be inverted or decomposed to cascade of single pole stages"

    tau_c = -1 / poles
    A_c = poles/zeros - 1

    scale = 1/A_dc

    if compensate_v34_fpga_scale:
        scale *= get_scaling_of_v34_fpga_filter(A_c, tau_c, Ts)

    return A_c, tau_c, scale


def get_scaling_of_v34_fpga_filter(A_c: np.ndarray, tau_c: np.ndarray, Ts) -> float:
    """get_scaling_of_v34_fpga_filter
    Calculate the scaling factor for the V3.4 FPGA filter implementation.
    This scaling is necessary to make the cascade of single exponent filters equivalent to the sum of exponents.
    :param A_c: Amplitudes of the cascade filters
    :param tau_c: Time constants of the cascade filters
    :param Ts: Sampling period
    :return: scale
    """
    return float(np.prod((Ts + 2*tau_c) / (Ts + 2*tau_c*(1+A_c))))


def get_rational_filter_single_exp_cont_time(A: float, tau: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1, 1/tau])
    b = np.array([A])
    return b, a