from quam.core import quam_dataclass
from quam_builder.architecture.superconducting.qubit.flux_tunable_transmon import (
    FluxTunableTransmon,
)
from quam.components.channels import Pulse
from typing import Dict, Any, Literal, Optional, Callable
from dataclasses import field
import numpy as np
from qm import qua


__all__ = ["Transmon"]


@quam_dataclass
class Transmon(FluxTunableTransmon):
    """
    Optimized QuAM component for a transmon qubit, inheriting from FluxTunableTransmon.
    Only custom methods/fields not present in the base class are defined here.
    """

    anharmonicity: float = 200e6  # default 200 MHz instead of None in base class
    extras: Dict[str, Any] = field(default_factory=dict)

    def get_output_power(self, operation, Z=50) -> float:
        power = self.xy.opx_output.full_scale_power_dbm
        amplitude = self.xy.operations[operation].amplitude
        x_mw = 10 ** (power / 10)
        x_v = amplitude * np.sqrt(2 * Z * x_mw / 1000)
        return 10 * np.log10(((x_v / np.sqrt(2)) ** 2 * 1000) / Z)

    def sigma(self, operation: Pulse):
        return operation.length / self.sigma_time_factor

    @property
    def name(self):
        return self.id if isinstance(self.id, str) else f"q{self.id}"

    def reset(
        self,
        reset_type: Literal[
            "thermal", "active", "active_simple", "active_gef"
        ] = "thermal",
        simulate: bool = False,
        log_callable: Optional[Callable] = None,
        **kwargs,
    ):
        if not simulate and reset_type in ["active_simple", "active"]:
            self.reset_qubit_active_simple()
        else:
            super().reset(reset_type, simulate, log_callable, **kwargs)

    def reset_qubit_active_simple(
        self,
        pi_pulse_name: str = "x180",
        readout_pulse_name: str = "readout",
    ):
        """
        Perform a simple active reset of the qubit.

        This function performs a single measurement and conditional pi pulse to reset the qubit.
        It is simpler but less robust than the full active_reset method.

        Args:
            pi_pulse_name (str): The name of the pi pulse to use for the reset. Default is "x180".
            readout_pulse_name (str): The name of the readout pulse to use for measuring the qubit state. Default is "readout".

        Returns:
            None

        The function:
        1. Measures the qubit state using the specified readout pulse
        2. If the qubit is in the excited state (I > threshold), applies a pi pulse to bring it to ground
        3. Waits for the resonator depletion time
        """
        pulse = self.resonator.operations[readout_pulse_name]

        I = qua.declare(qua.fixed)
        Q = qua.declare(qua.fixed)
        state = qua.declare(bool)
        self.align()
        self.resonator.measure("readout", qua_vars=(I, Q))
        qua.assign(state, I > pulse.threshold)
        qua.wait(self.resonator.depletion_time // 2, self.resonator.name)
        self.xy.play(pi_pulse_name, condition=state)
        self.align()
