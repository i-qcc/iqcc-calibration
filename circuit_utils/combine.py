"""Utilities for batching multiple Qiskit circuits into a single OpenQASM 3 program."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit import qasm3

from .qubit_info import QubitInfo


def combine_circuits(
    circuits: list[QuantumCircuit],
    qubit_mapping: dict[int, str | int],
    qubit_info: QubitInfo,
    add_measurements: bool = True,
) -> tuple[str, list[list[int]]]:
    """Combine a list of Qiskit circuits into one OpenQASM 3 program.

    Each sub-circuit is preceded by a reset of its active physical qubits.
    Measurements are only applied to active qubits (those with gates on them).

    Args:
        circuits: Qiskit ``QuantumCircuit`` objects.  For example a
            single-qubit experiment can simply use ``QuantumCircuit(1)``.
        qubit_mapping: Mapping from circuit qubit index to physical qubit,
            specified either by name (``{0: "qA1"}``) or by index
            (``{0: 3}``).  Names are resolved via *qubit_info*.
        qubit_info: Backend topology obtained from
            :func:`~circuit_utils.qubit_info.get_qubit_info`.
        add_measurements: When *True* (default) and a circuit has **no**
            measurement instructions, a measurement is automatically appended
            for every active physical qubit.

    Returns:
        qasm3_str: A single OpenQASM 3 program ready for ``openqasm2qua``.
        clbit_map: ``clbit_map[i]`` is the list of classical-bit indices
            (in the combined register) that hold the measurement outcomes of
            ``circuits[i]``.
    """
    resolved_mapping = qubit_info.resolve_mapping(qubit_mapping)
    num_qubits = qubit_info.num_qubits

    def _map(circuit_idx: int) -> int:
        if circuit_idx not in resolved_mapping:
            raise ValueError(
                f"Circuit qubit {circuit_idx} has no entry in qubit_mapping"
            )
        return resolved_mapping[circuit_idx]

    parsed: list[dict] = []
    total_clbits = 0

    for circ in circuits:
        active_physical: set[int] = set()
        gate_ops: list[tuple] = []
        meas_physical: list[int] = []

        for inst in circ.data:
            circuit_indices = [circ.find_bit(q).index for q in inst.qubits]
            physical_indices = [_map(i) for i in circuit_indices]

            if inst.operation.name == "measure":
                meas_physical.append(physical_indices[0])
            elif inst.operation.name != "barrier":
                active_physical.update(physical_indices)
                gate_ops.append((inst.operation, physical_indices))

        if not meas_physical and add_measurements:
            # Measure active qubits, or all mapped qubits if the circuit was
            # optimized to an empty gate set (e.g. H-RZ(0)-H → identity).
            if active_physical:
                meas_physical = sorted(active_physical)
            else:
                all_mapped = sorted(
                    _map(i) for i in range(circ.num_qubits)
                    if i in resolved_mapping
                )
                meas_physical = all_mapped

        for q in active_physical:
            if q >= num_qubits:
                raise ValueError(
                    f"Physical qubit {q} out of range (num_qubits={num_qubits})"
                )

        parsed.append(
            {
                "active_physical": sorted(active_physical),
                "gate_ops": gate_ops,
                "meas_physical": meas_physical,
            }
        )
        total_clbits += len(meas_physical)

    combined = QuantumCircuit(num_qubits, total_clbits)
    clbit_map: list[list[int]] = []
    clbit_offset = 0

    for info in parsed:
        circuit_clbits: list[int] = []

        reset_qubits = info["active_physical"] or info["meas_physical"]
        for q in reset_qubits:
            combined.reset(q)

        for op, physical_indices in info["gate_ops"]:
            combined.append(op, physical_indices)

        for q in info["meas_physical"]:
            combined.measure(q, clbit_offset)
            circuit_clbits.append(clbit_offset)
            clbit_offset += 1

        clbit_map.append(circuit_clbits)

    return qasm3.dumps(combined), clbit_map
