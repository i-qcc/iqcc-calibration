#!/usr/bin/env python3
"""
Generate physical_qubits.json and transpiler_target.json from the QuAM state.

These files describe the hardware topology and gate properties for
quantum circuit compilers / transpilers.

Usage:
    python generate_compiler_data.py
    python generate_compiler_data.py --state-path /path/to/quam_state_folder
    python generate_compiler_data.py --output-dir /path/to/output
"""

import json
import argparse
from pathlib import Path
from typing import Optional

from iqcc_calibration_tools.quam_config.components.quam_root import Quam

OUTPUT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _safe_getattr(obj, *attrs, default=None):
    """Walk a chain of attribute lookups, returning *default* on any failure."""
    for attr in attrs:
        try:
            if isinstance(obj, dict):
                obj = obj[attr]
            else:
                obj = getattr(obj, attr)
        except (AttributeError, KeyError, TypeError, IndexError):
            return default
    return obj


def _get_reset_duration_ns(qubit) -> Optional[int]:
    """Compute active-reset worst-case duration in ns."""
    try:
        x180_len = _get_pulse_length(qubit, "xy", "x180", "x")
        readout_len = _get_pulse_length(qubit, "resonator", "readout")
        if x180_len is not None and readout_len is not None:
            max_attempts = _safe_getattr(qubit, "macros", "reset", "max_attempts", default=5)
            return int((x180_len + readout_len) * max_attempts)
    except Exception:
        pass
    T1 = _safe_getattr(qubit, "T1")
    factor = _safe_getattr(qubit, "thermalization_time_factor", default=5)
    if T1 is not None:
        return int(T1 * 1e9 * factor)
    return None


def _get_measure_error(qubit) -> Optional[float]:
    """1 - average of confusion-matrix diagonal (gg, ee)."""
    cm = _safe_getattr(qubit, "resonator", "confusion_matrix")
    if cm is not None and len(cm) >= 2 and len(cm[0]) >= 1 and len(cm[1]) >= 2:
        gg = cm[0][0]
        ee = cm[1][1]
        return 1.0 - (gg + ee) / 2.0
    return None


def _get_1q_rb_error(qubit) -> Optional[float]:
    """1 - single-qubit averaged RB fidelity."""
    fidelity = _safe_getattr(qubit, "gate_fidelity", "averaged")
    if fidelity is not None:
        return 1.0 - float(fidelity)
    return None


def _get_pulse_length(qubit, channel: str, *operation_names: str) -> Optional[int]:
    """Get pulse length (ns), trying each operation name in order."""
    try:
        ch = getattr(qubit, channel)
        ops = ch.operations
        for op_name in operation_names:
            try:
                op = ops[op_name]
                length = op.length
                if isinstance(length, (int, float)):
                    return int(length)
            except (KeyError, AttributeError, TypeError):
                continue
    except (AttributeError, TypeError):
        pass
    return None


def _get_cz_duration_ns(qp) -> Optional[int]:
    """Get CZ gate duration (ns) from the qubit-pair's CZ macro flux pulse."""
    cz_macro = _safe_getattr(qp, "macros", "cz")
    if cz_macro is None:
        cz_macro = _safe_getattr(qp, "gates", "Cz")
    if cz_macro is None:
        return None

    for pulse_attr in ("flux_pulse_qubit", "flux_pulse_control"):
        pulse = _safe_getattr(cz_macro, pulse_attr)
        if pulse is not None:
            length = _safe_getattr(pulse, "length")
            if isinstance(length, (int, float)):
                return int(length)
    return None


def _get_2q_rb_error(qp) -> Optional[float]:
    """1 - 2Q Standard RB average gate fidelity."""
    cz_macro = _safe_getattr(qp, "macros", "cz")
    if cz_macro is None:
        cz_macro = _safe_getattr(qp, "gates", "Cz")
    if cz_macro is None:
        return None

    fidelity = _safe_getattr(cz_macro, "fidelity", "StandardRB", "average_gate_fidelity")
    if fidelity is not None:
        return 1.0 - float(fidelity)
    return None


# ---------------------------------------------------------------------------
# JSON generators
# ---------------------------------------------------------------------------

def generate_physical_qubits(machine: Quam) -> dict:
    """
    Build physical_qubits mapping:
      index2qubit : sorted list of qubit names
      pair2index  : pair name -> sequential index (alphabetic order)
    """
    qubit_names = sorted(machine.qubits.keys())
    pair_names = sorted(machine.qubit_pairs.keys())
    pair2index = {name: idx for idx, name in enumerate(pair_names)}

    return {
        "index2qubit": qubit_names,
        "pair2index": pair2index,
    }


def generate_transpiler_target(machine: Quam, physical_qubits: dict) -> dict:
    """
    Build transpiler_target with operations:
      reset, measure, x, sx, rz, cz, characterization
    """
    qubit_names = physical_qubits["index2qubit"]
    qubit_to_idx = {name: idx for idx, name in enumerate(qubit_names)}

    reset_entries = []
    measure_entries = []
    x_entries = []
    sx_entries = []
    rz_entries = []
    characterization_entries = []

    for idx, q_name in enumerate(qubit_names):
        qubit = machine.qubits[q_name]

        # Reset
        reset_dur = _get_reset_duration_ns(qubit)
        reset_entries.append([idx, {"duration": reset_dur}])

        # Measure
        meas_err = _get_measure_error(qubit)
        measure_entries.append([idx, {"error": meas_err}])

        # 1Q gate error (shared across X / SX)
        rb_error = _get_1q_rb_error(qubit)

        # X gate
        x_dur = _get_pulse_length(qubit, "xy", "x", "x180")
        x_entries.append([idx, {"duration": x_dur, "error": rb_error}])

        # SX gate
        sx_dur = _get_pulse_length(qubit, "xy", "sx", "x90")
        sx_entries.append([idx, {"duration": sx_dur, "error": rb_error}])

        # RZ (virtual Z — zero cost)
        rz_entries.append([idx, {"duration": 0, "error": 0}])

        # Characterization (T1, T2* in nanoseconds)
        T1 = _safe_getattr(qubit, "T1")
        T2star = _safe_getattr(qubit, "T2ramsey")
        char = {}
        if T1 is not None:
            char["T1"] = T1 * 1e9
        if T2star is not None:
            char["T2star"] = T2star * 1e9
        characterization_entries.append([idx, char])

    # CZ — one entry per qubit pair
    cz_entries = []
    for pair_name in sorted(machine.qubit_pairs.keys()):
        qp = machine.qubit_pairs[pair_name]

        control_name = qp.qubit_control.name
        target_name = qp.qubit_target.name
        if control_name not in qubit_to_idx or target_name not in qubit_to_idx:
            continue

        ctrl_idx = qubit_to_idx[control_name]
        tgt_idx = qubit_to_idx[target_name]

        cz_dur = _get_cz_duration_ns(qp)
        cz_err = _get_2q_rb_error(qp)

        cz_entries.append([[ctrl_idx, tgt_idx], {"duration": cz_dur, "error": cz_err}])

    cz_entries.sort(key=lambda e: (e[0][0], e[0][1]))

    return {
        "reset": reset_entries,
        "measure": measure_entries,
        "x": x_entries,
        "sx": sx_entries,
        "rz": rz_entries,
        "cz": cz_entries,
        "characterization": characterization_entries,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate physical_qubits.json and transpiler_target.json from QuAM state"
    )
    parser.add_argument(
        "--state-path",
        default=None,
        help="Path to QuAM state folder (default: QUAM_STATE_PATH env var)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help=f"Output directory for generated JSON files (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    if args.state_path:
        print(f"Loading QuAM state from: {args.state_path}")
        machine = Quam.load(args.state_path)
    else:
        print("Loading QuAM state from QUAM_STATE_PATH")
        machine = Quam.load()

    physical_qubits = generate_physical_qubits(machine)
    transpiler_target = generate_transpiler_target(machine, physical_qubits)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pq_path = output_dir / "physical_qubits.json"
    with open(pq_path, "w") as f:
        json.dump(physical_qubits, f, indent=2)
    print(f"  -> {pq_path}")

    tt_path = output_dir / "transpiler_target.json"
    with open(tt_path, "w") as f:
        json.dump(transpiler_target, f, indent=2)
    print(f"  -> {tt_path}")

    n_qubits = len(physical_qubits["index2qubit"])
    n_pairs = len(physical_qubits["pair2index"])
    n_cz = len(transpiler_target["cz"])
    print(f"\nSummary: {n_qubits} qubits, {n_pairs} pairs, {n_cz} CZ gates")


if __name__ == "__main__":
    main()
