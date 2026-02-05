"""
Runner script for executing calibration nodes with different multiplexed qubit pair configurations.

This script executes calibrations with the specified qubit pair configurations.

Usage:
    - Modify the configuration parameters below to set your configurations
    - Run this script: python calibrations/run_multiplexed_pairs.py
"""

import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path to import calibration_utils
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from calibration_utils.calibration_runner import run_calibration_with_pairs
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam as Quam

# %% {Configuration}
# Name of the calibration node to run (without .py extension)
# Examples: "70b_two_qubit_standard_rb", "70c_two_qubit_interleaved_rb"
NODE_NAME = "70b_two_qubit_standard_rb"

# List of qubit pair configurations to test
# Each inner list represents one run with those multiplexed qubit pairs
# Example configurations:
#   [["q1-q2", "q3-q4"]] - Run once with pairs q1-q2 and q3-q4 multiplexed together
#   [["q1-q2"], ["q3-q4"]] - Run twice: once with q1-q2, once with q3-q4
#   [["q1-q2", "q3-q4"], ["q5-q6", "q7-q8"]] - Run twice with different multiplexed groups
#   [None] - Use all active qubit pairs from the machine (default behavior)
# If set to None, will be computed as: [[pair.id for pair in batch.values()] for batch in node.get_multiplexed_pair_batches(node.machine.active_qubit_pair_names).batch()]
QUBIT_PAIR_CONFIGS: Optional[List[Optional[List[str]]]] = None

# Additional parameters to override (optional)
# These will be applied to all runs
ADDITIONAL_PARAMS: Dict[str, Any] = {
    "num_circuits_per_length": 15,
    "num_averages": 20,
    "circuit_lengths": [0, 2, 4, 8, 16, 32],
    "reduce_to_1q_cliffords": False,
    "multiplexed": True,
    
    # Example:
    # "num_averages": 20,
    # "circuit_lengths": [0, 2, 4, 8, 16, 32],
    # "simulate": False,
}


# %% {Execute}
if __name__ == "__main__":
    calibrations_dir = Path(__file__).parent
    
    # Compute default configurations if QUBIT_PAIR_CONFIGS is None
    if QUBIT_PAIR_CONFIGS is None:
        print("Computing default qubit pair configurations from multiplexed batches...")
        try:
            script_path = calibrations_dir / f"{NODE_NAME}.py"
            if not script_path.exists():
                raise FileNotFoundError(f"Script file {script_path} not found")
            
            # Read script and execute only up to machine loading (stop before connect/execute)
            with open(script_path, 'r') as f:
                script_lines = f.readlines()
            
            # Find where to stop - before expensive operations
            # Stop before comments or code that indicates expensive operations
            code_lines = []
            stop_index = None
            
            for i, line in enumerate(script_lines):
                stripped = line.strip()
                
                # Stop before comments that indicate expensive operations (like "Open Communication")
                if stripped.startswith('#') and any(keyword in stripped.lower() for keyword in [
                    'open communication',
                    'connect',
                    'execute',
                    'qop',
                ]):
                    stop_index = i
                    break
                
                # Stop before expensive operations themselves
                if any(marker in stripped for marker in [
                    'node.machine.connect()',
                    'qm.execute',
                    'qmm.simulate',
                    'qmm.execute',
                ]):
                    # Check if this is inside an if/for/while block - if so, stop before the block
                    # Look back up to 5 lines to find the control structure start
                    block_start = None
                    for j in range(max(0, i - 5), i):
                        prev_stripped = script_lines[j].strip()
                        if prev_stripped.startswith(('if ', 'for ', 'while ', 'with ')):
                            block_start = j
                            break
                    
                    stop_index = block_start if block_start is not None else i
                    break
                
                code_lines.append(line)
            
            # If we found a stop index, use it; otherwise use all collected lines
            if stop_index is not None:
                code_lines = script_lines[:stop_index]
            
            # Temporarily disable tracing to prevent debugger from stepping into exec()
            # This fixes the hanging issue in debug mode
            original_trace = sys.gettrace()
            sys.settrace(None)
            
            try:
                # Execute only the safe parts
                exec_namespace = {
                    '__name__': '__main__',
                    '__file__': str(script_path),
                }
                
                exec(compile(''.join(code_lines), str(script_path), 'exec'), exec_namespace)
            finally:
                # Restore tracing
                sys.settrace(original_trace)
            
            node = exec_namespace.get('node')
            if node is None:
                raise ValueError("Could not find 'node' in the executed script")
            
            # Ensure machine is loaded
            if node.machine is None:
                node.machine = Quam.load()
            
            # Compute the default configurations
            QUBIT_PAIR_CONFIGS = [
                [pair.id for pair in batch.values()] 
                for batch in node.get_multiplexed_pair_batches(node.machine.active_qubit_pair_names).batch()
            ]
            
            print(f"Computed {len(QUBIT_PAIR_CONFIGS)} batch configuration(s):")
            for idx, batch_config in enumerate(QUBIT_PAIR_CONFIGS, 1):
                print(f"  Batch {idx}: {batch_config}")
            
        except Exception as e:
            print(f"Error computing default configurations: {e}")
            traceback.print_exc()
            print("\nPlease manually set QUBIT_PAIR_CONFIGS in this file")
            sys.exit(1)
    
    # Validate configurations
    if not QUBIT_PAIR_CONFIGS:
        print("No qubit pair configurations specified.")
        print("Please modify QUBIT_PAIR_CONFIGS in this file")
        print("\nExample configurations:")
        print('  QUBIT_PAIR_CONFIGS = [["q1-q2", "q3-q4"], ["q5-q6"]]')
        print("\nOr set to None to use default (multiplexed batches):")
        print('  QUBIT_PAIR_CONFIGS = None')
        sys.exit(1)
    
    run_calibration_with_pairs(
        node_name=NODE_NAME,
        qubit_pair_configs=QUBIT_PAIR_CONFIGS,
        calibrations_dir=calibrations_dir,
        **ADDITIONAL_PARAMS
    )

