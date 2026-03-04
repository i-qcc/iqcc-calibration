"""Utilities for extracting measurement probabilities from execution results."""

from __future__ import annotations

import json
import numpy as np


def extract_probabilities(
    result: dict,
    clbit_map: list[list[int]],
    num_shots: int | None = None,
) -> list[float]:
    """Extract P(|1>) for each sub-circuit from a batched execution result.

    Works with the result dict returned by ``IQCC_Cloud.execute()``.

    The typical result shape from ``openqasm2qua`` is a flat list of
    single-element lists (one per classical-bit save), interleaved across
    shots::

        "c": [[True], [False], [True], ...]   # length = num_shots * total_clbits

    Args:
        result: Raw result dict from ``execute()``.
        clbit_map: Classical-bit mapping returned by
            :func:`~circuit_utils.combine.combine_circuits`.
        num_shots: Expected number of shots.  Used to reshape the flat
            result array.  If *None*, inferred from the data length.

    Returns:
        List of P(|1>) values, one per sub-circuit (same order as *clbit_map*).
    """
    res = result.get("result", {})
    total_clbits = sum(len(c) for c in clbit_map)

    for key in res:
        val = res[key]
        if not isinstance(val, list) or len(val) == 0:
            continue

        arr = np.array(val)

        # Squeeze away any trailing size-1 dimensions.
        # e.g. shape (11000, 1) -> (11000,)
        arr = arr.squeeze()
        arr = arr.astype(float)

        if arr.ndim == 1:
            length = len(arr)

            if num_shots is not None and length == num_shots * total_clbits:
                shots = arr.reshape(num_shots, total_clbits)
            elif total_clbits > 0 and length % total_clbits == 0:
                shots = arr.reshape(-1, total_clbits)
            else:
                continue

            return [
                float(np.mean(shots[:, bits[0]])) for bits in clbit_map
            ]

        # Already 2-D with shape (num_shots, total_clbits)
        if arr.ndim == 2 and arr.shape[1] == total_clbits:
            return [
                float(np.mean(arr[:, bits[0]])) for bits in clbit_map
            ]

    print(f"  [WARNING] Could not auto-parse result. Keys: {list(res.keys())}")
    print(f"  Raw result snippet: {json.dumps(res, default=str)[:500]}")
    return [np.nan] * len(clbit_map)
