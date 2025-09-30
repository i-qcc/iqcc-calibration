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

def sleppain_waveform(amplitude, length, theta_i, theta_f, coeffs = 0):
    
    def theta_tau(tau, theta_i, theta_f,t_p, coeffs):
        return theta_i + (theta_f - theta_i) / 2 * np.sum([ coeff * (1 - np.cos(2 * np.pi * (2*n+1) * tau / t_p)) for n, coeff in enumerate(coeffs)])

    def sin_theta_tau(tau, theta_i, theta_f,t_p, coeffs):
        return np.sin(theta_tau(tau, theta_i, theta_f,t_p, coeffs))

    def t_tau(tau, theta_i, theta_f,t_p, coeffs):
        return quad(sin_theta_tau, 0, tau, args=(theta_i, theta_f,t_p, coeffs))[0]

    def thetha_t(theta_i, theta_f, coeffs, length):
        t_p = 1
        ts = np.linspace(0,t_p,length)   
        
        
        t_taus = np.array([t_tau(t, theta_i, theta_f,t_p, coeffs) for t in ts])
        t_taus = t_taus / np.max(t_taus) * t_p

        theta_taus = np.array([theta_tau(t,theta_i,theta_f,t_p, coeffs) for t in ts])
        theta_ts = np.array([np.interp(t,t_taus,theta_taus) for t in ts])
        
        return theta_ts       

    coeffs_list = [1-coeffs,coeffs]
    theta_ts = thetha_t(theta_i, theta_f, coeffs_list, int(length))

    theths_Z = 1/np.tan(theta_ts)
    theths_Z -= np.max(theths_Z)

    flux = np.sqrt(np.abs(theths_Z)) 
    flux = flux/np.max(flux) * amplitude

    return flux

from quam.utils.qua_types import ScalarInt, ScalarBool

__all__ = [
    "SleppainPulse",
    "CosinePulse",
]


@quam_dataclass
class SleppainPulse(Pulse):

    amplitude: float
    theta_i: float = 0.1
    theta_f: float = np.pi * 0.5
    coeffs: float = 0

    def __post_init__(self) -> None:
        return super().__post_init__()

    def waveform_function(self):

        I= sleppain_waveform(
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