import matplotlib.pylab as plt
import xarray as xr
import numpy as np
from iqcc_calibration_tools.quam_config.lib.qua_datasets import apply_angle
from scipy.signal import savgol_filter
from scipy.signal import deconvolve
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial as P
from functools import reduce
from typing import List, Tuple, Sequence

def transform_to_circle(x, y):
    def ellipse_residuals(params, x, y):
        a, b, cx, cy, angle = params
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        x_rot = cos_angle * (x - cx) + sin_angle * (y - cy)
        y_rot = -sin_angle * (x - cx) + cos_angle * (y - cy)
        residuals = (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1
        return np.sum(residuals**2)  # Return the sum of squared residuals

    # Fit ellipse to points
    initial_guess = [0.5, 0.5, 0.5, 0.5, 0.0]  # Initial guess for ellipse parameters including angle
    result = minimize(ellipse_residuals, initial_guess, args=(x, y))
    a_fit, b_fit, cx_fit, cy_fit, angle_fit = result.x

    # Transform ellipse to circle
    scale_factor = max(a_fit, b_fit)
    a_circle = scale_factor
    b_circle = scale_factor
    cx_circle = cx_fit
    cy_circle = cy_fit

    # Generate points for the fitted ellipse
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse_x = a_fit * np.cos(theta)
    ellipse_y = b_fit * np.sin(theta)

    # Rotate the ellipse points
    cos_angle = np.cos(angle_fit)
    sin_angle = np.sin(angle_fit)
    x_ellipse_rot = cos_angle * ellipse_x - sin_angle * ellipse_y + cx_fit
    y_ellipse_rot = sin_angle * ellipse_x + cos_angle * ellipse_y + cy_fit

    # # Plot the original data points
    # plt.scatter(x, y, label='Data Points')

    # # Plot the fitted ellipse
    # plt.plot(x_ellipse_rot, y_ellipse_rot, color='r', label='Fitted Ellipse')
    # plt.show()

    # Apply transform to xy points
    # Step 1: Rotate the original points to align with the ellipse's axes
    x_rot = cos_angle * (x - cx_fit) + sin_angle * (y - cy_fit)
    y_rot = -sin_angle * (x - cx_fit) + cos_angle * (y - cy_fit)

    # Step 2: Scale the rotated points to transform the ellipse into a circle
    x_scaled = x_rot / a_fit * a_circle
    y_scaled = y_rot / b_fit * b_circle

    # Step 3: Rotate the scaled points back to the original orientation
    x_transformed = cos_angle * x_scaled - sin_angle * y_scaled + cx_circle
    y_transformed = sin_angle * x_scaled + cos_angle * y_scaled + cy_circle

    return x_transformed, y_transformed


def savgol(da, dim, range=3, order=2):
    def diff_func(x):
        return savgol_filter(x, range, order, deriv=0, delta=1)

    return xr.apply_ufunc(diff_func, da, input_core_dims=[[dim]], output_core_dims=[[dim]])


def diff_savgol(da, dim, range=3, order=2):
    def diff_func(x):
        return savgol_filter(x / (2 * np.pi), range, order, deriv=1, delta=1)

    return xr.apply_ufunc(diff_func, da, input_core_dims=[[dim]], output_core_dims=[[dim]])


def cryoscope_frequency(da, stable_time_indices, quad_term=-1, sg_range=3, sg_order=2, plot=False):
    da = da.copy()

    da_max = da.sel(time=slice(stable_time_indices[0], stable_time_indices[1])).max(dim="time")
    da_min = da.sel(time=slice(stable_time_indices[0], stable_time_indices[1])).min(dim="time")
    da_offset = (da_max + da_min) / 2
    da -= da_offset

    if plot:
        plt.scatter(da.sel(axis="x"), da.sel(axis="y"))
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("<X>")
        plt.ylabel("<Y>")
        plt.show()

    angle = apply_angle(da.sel(axis="x") + 1j * da.sel(axis="y"), "time").rename("angle")
    if plot:
        angle.plot()
        plt.show()

    freq_cryoscope = diff_savgol(angle, "time", range=sg_range, order=sg_order).rename("freq")
    if plot:
        (-freq_cryoscope).plot()
        plt.title("Frequency")
        plt.show()
    flux_cryoscope = np.sqrt(np.abs(1e9 * freq_cryoscope / quad_term)).fillna(0).rename("flux")
    if plot:
        flux_cryoscope.plot()
        plt.title("Flux")
        plt.show()
    if quad_term == -1:
        flux_cryoscope = flux_cryoscope / flux_cryoscope.sel(time=slice(80, 120)).mean(dim="time")
    return flux_cryoscope


def expdecay(x, s, a, t):
    """Exponential decay defined as 1 + a * np.exp(-x / t).
    :param x: numpy array for the time vector in ns
    :param a: float for the exponential amplitude
    :param t0: time shift
    :param t: float for the exponential decay time in ns
    :return: numpy array for the exponential decay
    """
    return s * (1 + a * np.exp(-(x) / t))


def two_expdecay(x, s, a, t, a2, t2):
    """Double exponential decay defined as s * (1 + a * np.exp(-x / t) + a2 * np.exp(-x / t2)).
    :param x: numpy array for the time vector in ns
    :param s: float for the scaling factor
    :param a: float for the first exponential amplitude
    :param t: float for the first exponential decay time in ns
    :param a2: float for the second exponential amplitude
    :param t2: float for the second exponential decay time in ns
    :return: numpy array for the double exponential decay
    """
    return s * (1 + a * np.exp(-(x) / t) + a2 * np.exp(-(x) / t2))


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


def estimate_fir_coefficients(convolved_signal, step_response, num_coefficients):
    """
    Estimate the FIR filter coefficients from a convolved signal.

    :param convolved_signal: The signal after being convolved with the FIR filter.
    :param step_response: The original step response signal.
    :param num_coefficients: Number of coefficients of the FIR filter to estimate.
    :return: Estimated FIR coefficients.
    """
    # Deconvolve to estimate the impulse response
    estimated_impulse_response, _ = deconvolve(convolved_signal, step_response)

    # Truncate or zero-pad the estimated impulse response to match the desired number of coefficients
    if len(estimated_impulse_response) > num_coefficients:
        # Truncate if the estimated response is longer than the desired number of coefficients
        estimated_coefficients = estimated_impulse_response[:num_coefficients]
    else:
        # Zero-pad if shorter
        estimated_coefficients = np.pad(
            estimated_impulse_response, (0, num_coefficients - len(estimated_impulse_response)), "constant"
        )

    return estimated_coefficients


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
            a_dc = y[-1]

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