"""High-level Backend runner for circuit execution on IQCC backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from qiskit.circuit import QuantumCircuit
from iqcc_cloud_client import IQCC_Cloud

from .qubit_info import QubitInfo, get_qubit_info
from .combine import combine_circuits
from .results import extract_probabilities

NON_GATE_KEYS = frozenset({"reset", "measure", "characterization"})


@dataclass
class Results:
    """Measurement results from a batch circuit execution.

    Attributes:
        probabilities: P(|1>) for each circuit, in the same order as
            the input circuit list.
        raw: Full result dict returned by ``IQCC_Cloud.execute()``.
        clbit_map: Classical-bit index mapping (from :func:`combine_circuits`).
        qasm3: The combined OpenQASM 3 program that was executed.
        num_shots: Number of shots used.
    """
    probabilities: list[float]
    raw: dict = field(repr=False)
    clbit_map: list[list[int]] = field(repr=False)
    qasm3: str = field(repr=False)
    num_shots: int


class Backend:
    """Connection to an IQCC quantum computer backend.

    Handles connection, qubit topology lookup, and provides a single
    :meth:`run` method that batches circuits, compiles via ``openqasm2qua``,
    executes, and returns parsed results.

    Example::

        backend = Backend("arbel")
        results = backend.run(circuits, qubits=["qA1"], num_shots=1000)
        print(results.probabilities)

    Args:
        name: Quantum computer backend name (e.g. ``"arbel"``).
    """

    def __init__(self, name: str):
        self.name = name
        self._cloud = IQCC_Cloud(name)
        self._qubit_info = get_qubit_info(name)

        self.native_gates = [
            k for k in self._qubit_info.transpiler_target
            if k not in NON_GATE_KEYS
        ]

        print(
            f"Connected to {name}: "
            f"{self._qubit_info.num_qubits} qubits, "
            f"native gates = {self.native_gates}"
        )

    @property
    def qubit_info(self) -> QubitInfo:
        return self._qubit_info

    def run(
        self,
        circuits: list[QuantumCircuit],
        qubits: list[str] | dict[int, str | int] | None = None,
        num_shots: int = 1000,
    ) -> Results:
        """Batch, compile, execute, and return parsed results.

        Circuits should already be transpiled to :attr:`native_gates`.

        Args:
            circuits: List of Qiskit ``QuantumCircuit`` objects.
            qubits: Mapping from circuit qubit indices to physical qubits.

                * **list** – ``["qA1", "qB2"]``: index 0 → qA1, index 1 → qB2.
                * **dict** – ``{0: "qA1", 1: "qB2"}`` for explicit mapping.
                * **None** – uses the first *N* qubits on the backend in
                  alphabetical order (where *N* is the number of circuit
                  qubits) and prints the default mapping.
            num_shots: Number of measurement shots (default 1000).

        Returns:
            A :class:`Results` object with measurement probabilities and
            raw data.
        """
        if qubits is None:
            n_circuit_qubits = max(c.num_qubits for c in circuits)
            default_names = self._qubit_info.index2qubit[:n_circuit_qubits]
            qubit_mapping = {i: name for i, name in enumerate(default_names)}
            print(f"No qubit mapping provided, using default: {default_names}")
        elif isinstance(qubits, list):
            qubit_mapping = {i: name for i, name in enumerate(qubits)}
        else:
            qubit_mapping = qubits

        qasm_str, clbit_map = combine_circuits(
            circuits,
            qubit_mapping=qubit_mapping,
            qubit_info=self._qubit_info,
        )

        r = self._cloud.run("openqasm2qua", {
            "openqasm3": qasm_str,
            "num_shots": num_shots,
        })

        exec_result = self._cloud.execute(
            r["result"]["qua"],
            r["result"]["qua_config"],
        )

        probabilities = extract_probabilities(
            exec_result, clbit_map, num_shots=num_shots,
        )

        return Results(
            probabilities=probabilities,
            raw=exec_result,
            clbit_map=clbit_map,
            qasm3=qasm_str,
            num_shots=num_shots,
        )
