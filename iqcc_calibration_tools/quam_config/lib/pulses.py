from quam.core import quam_dataclass
from quam.components.pulses import Pulse, DragCosinePulse
import numpy as np

@quam_dataclass
class DragPulseCosine(DragCosinePulse):
    """
    Enhanced DragCosinePulse with subtraction option.
    
    This class extends quam's DragCosinePulse with the ability to subtract the final value
    to reduce high-frequency components due to initial and final points offset.

    These DRAG waveforms has been implemented following the next Refs.:
    Chen et al. PRL, 116, 020501 (2016)
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.020501
    and Chen's thesis
    https://web.physics.ucsb.edu/~martinisgroup/theses/Chen2018.pdf

    :param bool subtracted: If true, returns a subtracted waveform to reduce high-frequency components.
    """

    subtracted: bool = True

@quam_dataclass
class FluxPulse(Pulse):
    """Flux pulse QuAM component.

    Args:
        length (int): The total length of the pulse in samples, including zero padding.
        digital_marker (str, list, optional): The digital marker to use for the pulse.
        amplitude (float): The amplitude of the pulse in volts.
        zero_padding (int): Number of samples to zero-pad at the end of the pulse.
    """

    amplitude: float
    zero_padding: int = 0

    def waveform_function(self):
        waveform = self.amplitude * np.ones(self.length)
        if self.zero_padding:
            if self.zero_padding > self.length:
                raise ValueError(
                    f"Flux pulse zero padding ({self.zero_padding} ns) exceeds " f"pulse length ({self.length} ns)."
                )
            waveform[-self.zero_padding :] = 0
        return waveform
    
@quam_dataclass
class SNZPulse(Pulse):
    """Step-Null-Zero (SNZ) pulse for specialized quantum operations.
    
    Args:
        length (int): The total length of the pulse in samples.
        amplitude (float): The main amplitude of the pulse.
        step_amplitude (float): The amplitude of the step sections.
        step_length (int): The length of each step section.
        spacing (int): The spacing between step sections.
    """
    
    amplitude: float
    step_amplitude: float
    step_length: int
    spacing : int
    
    def __post_init__(self):
        self.length -= self.length % 4

    def waveform_function(self):
        rect_duration = (self.length - 4 - 2 * self.step_length - self.spacing) // 2
        waveform = [self.amplitude] * rect_duration
        waveform += [self.step_amplitude] * self.step_length
        waveform += [0] * self.spacing
        waveform += [-self.step_amplitude] * self.step_length
        waveform += [-self.amplitude] * rect_duration
        waveform += [0.0] * (self.length - len(waveform))
        
        return waveform    