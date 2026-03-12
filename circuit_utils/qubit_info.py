"""Fetch and expose hardware qubit topology from IQCC cloud."""

from __future__ import annotations

from dataclasses import dataclass, field
from iqcc_cloud_client import IQCC_Cloud


@dataclass
class QubitInfo:
    """Resolved hardware qubit topology for a backend.

    Attributes:
        index2qubit: Ordered list of qubit names (index = physical index).
        qubit2index: Reverse lookup – qubit name to physical index.
        num_qubits: Total number of hardware qubits.
        pair2index: Qubit-pair name to pair index mapping.
        physical_qubits: Raw ``physical_qubits`` dict from cloud.
        transpiler_target: Raw ``transpiler_target`` dict from cloud.
    """
    index2qubit: list[str]
    qubit2index: dict[str, int] = field(repr=False)
    num_qubits: int = field(init=False)
    pair2index: dict[str, int] = field(default_factory=dict, repr=False)
    physical_qubits: dict = field(default_factory=dict, repr=False)
    transpiler_target: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.num_qubits = len(self.index2qubit)

    def resolve_mapping(
        self, qubit_mapping: dict[int, str | int],
    ) -> dict[int, int]:
        """Convert a ``{circuit_idx: qubit_name_or_index}`` mapping to
        ``{circuit_idx: physical_index}`` using the backend topology.

        Strings are looked up in :attr:`qubit2index`; integers pass through.
        """
        resolved: dict[int, int] = {}
        for circuit_idx, target in qubit_mapping.items():
            if isinstance(target, str):
                if target not in self.qubit2index:
                    raise ValueError(
                        f"Qubit name '{target}' not found on backend. "
                        f"Available: {self.index2qubit}"
                    )
                resolved[circuit_idx] = self.qubit2index[target]
            else:
                resolved[circuit_idx] = int(target)
        return resolved


def get_qubit_info(backend: str) -> QubitInfo:
    """Fetch ``physical_qubits`` and ``transpiler_target`` from IQCC cloud.

    Args:
        backend: Quantum computer backend name (e.g. ``"arbel"``).

    Returns:
        A :class:`QubitInfo` instance with the resolved topology.
    """
    qc = IQCC_Cloud(backend)
    physical_qubits = qc.state.get_latest("physical_qubits").data
    transpiler_target = qc.state.get_latest("transpiler_target").data

    index2qubit = physical_qubits["index2qubit"]

    return QubitInfo(
        index2qubit=index2qubit,
        qubit2index={name: i for i, name in enumerate(index2qubit)},
        pair2index=physical_qubits.get("pair2index", {}),
        physical_qubits=physical_qubits,
        transpiler_target=transpiler_target,
    )
