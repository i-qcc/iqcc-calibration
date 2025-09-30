from abc import ABC, abstractmethod
from collections.abc import Iterable
import numbers
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Union, Tuple
import numpy as np

from quam.core import QuamComponent, quam_dataclass
from quam.utils import string_reference as str_ref
from quam.components.pulses import Pulse

from qm.qua import (
    AmpValuesType,
    ChirpType,
    StreamType,
)

from scipy.integrate import quad
def slepian_waveform(amplitude, length, theta_i, theta_f, coeffs = 0):
    """
    Generate optimal adiabatic control waveform using Slepian functions.
    
    This function implements the fast adiabatic control protocol described by Martinez and Geller
    (arXiv:1402.5467) for achieving high-fidelity quantum gates using only σ_z control. The 
    Slepian waveform minimizes non-adiabatic errors by optimizing the power spectral density
    of state errors using optimal window functions in the Fourier basis.
    
    Theory:
    The method addresses the challenge of fast adiabatic control where only σ_z control is 
    available (unlike conventional quantum control with σ_x and σ_y components). The optimal
    waveform is derived by mapping state errors to a power spectral density and using 
    Slepian functions as optimal window functions that concentrate energy in both time and
    frequency domains.
    
    The control trajectory θ(t) is parameterized using cosine series:
    θ(τ) = θ_i + (θ_f - θ_i)/2 * Σ[c_n * (1 - cos(2π(2n+1)τ/t_p))]
    
    where τ is normalized time, θ_i/θ_f are initial/final angles, and c_n are Slepian 
    coefficients that determine the waveform shape.
    
    Parameters:
    -----------
    amplitude : float
        Maximum amplitude of the flux control signal. This scales the final waveform
        to the desired control amplitude for the quantum system.
        
    length : int
        Number of time points in the waveform. Determines the temporal resolution
        of the control signal.
        
    theta_i : float
        Initial angle θ_i in the adiabatic control trajectory. Typically small (e.g., 0.1)
        to start near the ground state configuration.
        
    theta_f : float
        Final angle θ_f in the adiabatic control trajectory. Often π/2 for controlled-phase
        gates, representing the target state configuration.
        
    coeffs : float, optional
        Slepian coefficient controlling the waveform shape. Range [0,1] where:
        - 0: Pure cosine waveform (coeffs_list = [1, 0])
        - 1: Pure Slepian waveform (coeffs_list = [0, 1])
        - Intermediate values: Weighted combination
        
    Returns:
    --------
    flux : numpy.ndarray
        Optimized flux control waveform as a 1D array of length 'length'.
        The waveform is normalized to have maximum value 'amplitude' and represents
        the optimal control signal for fast adiabatic gates.
        
    Notes:
    ------
    This implementation follows the theoretical framework from:
    J.M. Martinis and M.R. Geller, "Fast adiabatic qubit gates using only σ_z control"
    arXiv:1402.5467 (2014)
    
    The method achieves:
    - Fast gate times (~40 ns for superconducting qubits)
    - High fidelity (>99.4% demonstrated for CZ gates)
    - Minimal non-adiabatic errors (<10^-4 achievable)
    - Smooth waveforms reducing unwanted transitions
    
    The algorithm:
    1. Constructs parameterized angle trajectory θ(τ) using cosine series
    2. Solves the time mapping t(τ) via numerical integration of sin(θ(τ))
    3. Interpolates to get θ(t) in real time coordinates
    4. Converts to flux control via cotangent relationship
    5. Normalizes and scales to desired amplitude
    
    Examples:
    ---------
    # Generate waveform for CZ gate
    flux_waveform = slepian_waveform(
        amplitude=0.5,      # Half maximum flux
        length=100,         # 100 time points
        theta_i=0.1,        # Start near ground state
        theta_f=np.pi/2,    # Target π/2 rotation
        coeffs=0.5          # Balanced Slepian/cosine
    )
    """
    
    def theta_tau(tau, theta_i, theta_f, t_p, coeffs):
        """
        Parameterized angle trajectory using cosine series expansion.
        
        Implements the Slepian-based parameterization:
        θ(τ) = θ_i + (θ_f - θ_i)/2 * Σ[c_n * (1 - cos(2π(2n+1)τ/t_p))]
        
        This form ensures smooth transitions and optimal spectral properties
        for minimizing non-adiabatic errors.
        """
        return theta_i + (theta_f - theta_i) / 2 * np.sum([coeff * (1 - np.cos(2 * np.pi * (2*n+1) * tau / t_p)) for n, coeff in enumerate(coeffs)])

    def sin_theta_tau(tau, theta_i, theta_f, t_p, coeffs):
        """Sine of the parameterized angle trajectory."""
        return np.sin(theta_tau(tau, theta_i, theta_f, t_p, coeffs))

    def t_tau(tau, theta_i, theta_f, t_p, coeffs):
        """
        Time mapping function via numerical integration.
        
        Solves the differential equation dt/dτ = sin(θ(τ)) to map from
        parameterized time τ to real time t. This accounts for the
        non-uniform time evolution in the adiabatic trajectory.
        """
        return quad(sin_theta_tau, 0, tau, args=(theta_i, theta_f, t_p, coeffs))[0]

    def theta_t(theta_i, theta_f, coeffs, length):
        """
        Generate angle trajectory in real time coordinates.
        
        Converts from parameterized time τ to real time t by:
        1. Computing t(τ) via numerical integration
        2. Normalizing the time mapping
        3. Interpolating θ(τ) to θ(t) in real time
        
        This ensures the final waveform has proper temporal characteristics
        for adiabatic control.
        """
        t_p = 1  # Normalized pulse duration
        ts = np.linspace(0, t_p, length)   
        
        # Compute time mapping t(τ)
        t_taus = np.array([t_tau(t, theta_i, theta_f, t_p, coeffs) for t in ts])
        t_taus = t_taus / np.max(t_taus) * t_p  # Normalize to [0, t_p]

        # Compute angle trajectory θ(τ)
        theta_taus = np.array([theta_tau(t, theta_i, theta_f, t_p, coeffs) for t in ts])
        
        # Interpolate to real time: θ(t)
        theta_ts = np.array([np.interp(t, t_taus, theta_taus) for t in ts])
        
        return theta_ts       

    # Construct Slepian coefficient list for waveform shaping
    coeffs_list = [1-coeffs, coeffs]
    
    # Generate optimized angle trajectory
    theta_ts = theta_t(theta_i, theta_f, coeffs_list, int(length))

    # Convert angle trajectory to flux control signal
    # Using cotangent relationship: flux ∝ √|cot(θ)|
    theths_Z = 1/np.tan(theta_ts)
    theths_Z -= np.max(theths_Z)  # Offset for positive flux values

    flux = np.sqrt(np.abs(theths_Z)) 
    flux = flux/np.max(flux) * amplitude  # Normalize and scale to desired amplitude

    return flux

from quam.utils.qua_types import ScalarInt, ScalarBool

__all__ = [
    "SlepianPulse",
    "CosinePulse",
]


@quam_dataclass
class SlepianPulse(Pulse):

    amplitude: float
    theta_i: float = 0.1
    theta_f: float = np.pi * 0.5
    coeffs: float = 0

    def __post_init__(self) -> None:
        return super().__post_init__()

    def waveform_function(self):

        I= slepian_waveform(
            amplitude=self.amplitude,
            length=self.length,
            theta_i=self.theta_i,
            theta_f=self.theta_f,
            coeffs=self.coeffs,
        )
        I = np.array(I)

        return I
    
@quam_dataclass
class CosinePulse(Pulse):

    axis_angle: float = 0.0
    amplitude: float
    alpha: float = 0.0
    anharmonicity: float = 0.0
    detuning: float = 0.0

    def __post_init__(self) -> None:
        return super().__post_init__()

    def waveform_function(self):
        from qualang_tools.config.waveform_tools import drag_cosine_pulse_waveforms

        I, Q = drag_cosine_pulse_waveforms(
            amplitude=self.amplitude,
            length=self.length,
            alpha=self.alpha,
            anharmonicity=self.anharmonicity,
            detuning=self.detuning,
        )
        I, Q = np.array(I), np.array(Q)

        return I    