import os
from pathlib import Path
from quam.core import quam_dataclass

from quam.components.macro.qubit_macros import PulseMacro

from quam_builder.architecture.superconducting.components.twpa import TWPA
from iqcc_calibration_tools.quam_config.components.gate_macros import (
    VirtualZMacro, DelayMacro, ResetMacro, MeasureMacro
)

from qm import QuantumMachinesManager, QuantumMachine
from qualang_tools.results.data_handler import DataHandler

from dataclasses import field
from typing import Dict, ClassVar, Sequence, Union

from quam_builder.architecture.superconducting.qpu import FluxTunableQuam

__all__ = ["Quam"] # , "FEMQuAM", "OPXPlusQuAM"]


# TODO : to be removed using the FluxTunableQuam class directly

@quam_dataclass
class Quam(FluxTunableQuam):
    """Example Quam root component with enhanced functionality."""

    _data_handler: ClassVar[DataHandler | None] = None
    
    twpas: Dict[str, TWPA] = field(default_factory=dict)

    @classmethod
    def load(cls, *args, **kwargs) -> "Quam":
        if not args:
            if "QUAM_STATE_PATH" in os.environ:
                args = (os.environ["QUAM_STATE_PATH"],)
            else:
                raise ValueError(
                    "No path argument provided to load the QuAM state. "
                    "Please provide a path or set the 'QUAM_STATE_PATH' environment variable. "
                    "See the README for instructions."
                )
        
        # Load the Quam object
        quam_obj = super().load(*args, **kwargs)
        
        # Add macros to the loaded object
        Quam.add_macros_to_qubits(quam_obj)
        
        return quam_obj

    def save(
        self,
        path: Union[Path, str] | None = None,
        content_mapping: Dict[Union[Path, str], Sequence[str]] | None = None,
        include_defaults: bool = False,
        ignore: Sequence[str] | None = None,
    ):
        if path is None and "QUAM_STATE_PATH" in os.environ:
            path = os.environ["QUAM_STATE_PATH"]

        super().save(path, content_mapping, include_defaults, ignore)

    @property
    def data_handler(self) -> DataHandler:
        """Return the existing data handler or open a new one to conveniently handle data saving."""
        if self._data_handler is None:
            self._data_handler = DataHandler(root_data_folder=self.network["data_folder"])
            DataHandler.node_data = {"quam": "./state.json"}
        return self._data_handler

    @classmethod
    def add_macros_to_qubits(cls, quam_obj: "Quam") -> None:
        """Add standard macros to all qubits and qubit pairs."""
        
        # Add macros to each qubit
        for qubit_data in quam_obj.qubits.values():
            qubit_data.macros= {
                "x": PulseMacro(pulse="x180"),
                "rz": VirtualZMacro(),
                "sx": PulseMacro(pulse="x90"),
                "delay": DelayMacro(),
                "reset": ResetMacro(pi_pulse="x180", readout_pulse="readout"),
                "measure": MeasureMacro()
            }
        
    @classmethod
    def remove_macros_from_qubits(cls, quam_obj: "Quam") -> None:
        """Remove all macros from qubits and qubit pairs."""
        
        # Remove macros from each qubit
        for qubit_data in quam_obj.qubits.values():
            if hasattr(qubit_data, 'macros'):
                qubit_data.macros = {}
        
