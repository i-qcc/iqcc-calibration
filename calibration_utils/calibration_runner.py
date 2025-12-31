"""
Calibration runner utility for executing calibration nodes with different parameter configurations.

This module provides a generic function to run calibration scripts with different parameter
configurations, particularly useful for testing different multiplexed qubit pair combinations.
"""

import re
import sys
from typing import List, Optional, Dict, Any
from pathlib import Path


def run_calibration_with_pairs(
    node_name: str,
    qubit_pair_configs: List[List[str]] = None,
    calibrations_dir: Optional[Path] = None,
    **additional_params
):
    """
    Run a calibration node with different qubit pair configurations.
    
    This function executes the calibration script multiple times, each time with different
    qubit pair parameters. It injects code to modify node parameters right after node creation.
    
    Args:
        node_name: Name of the calibration node script (without .py extension)
        qubit_pair_configs: List of qubit pair configurations. Each configuration is a list of 
                           qubit pair names (strings). For example: [["q1-q2", "q3-q4"], ["q5-q6"]]
                           If None, will use all active qubit pairs from the machine.
        calibrations_dir: Path to the calibrations directory. If None, will try to infer from
                         the calling script's location.
        **additional_params: Additional parameters to override in the node's Parameters class
    """
    if qubit_pair_configs is None:
        qubit_pair_configs = [None]  # Will use active pairs
    
    # Determine calibrations directory
    if calibrations_dir is None:
        # Try to find calibrations directory relative to this file
        this_file = Path(__file__)
        # calibration_utils is sibling to calibrations
        calibrations_dir = this_file.parent.parent / "calibrations"
        if not calibrations_dir.exists():
            # Fallback: try to find it in sys.path
            for path in sys.path:
                potential_dir = Path(path) / "calibrations"
                if potential_dir.exists():
                    calibrations_dir = potential_dir
                    break
            else:
                raise ValueError("Could not find calibrations directory. Please specify calibrations_dir parameter.")
    
    calibrations_dir = Path(calibrations_dir)
    
    print(f"\n{'='*80}")
    print(f"Running {node_name} with {len(qubit_pair_configs)} qubit pair configuration(s)")
    print(f"{'='*80}\n")
    
    script_path = calibrations_dir / f"{node_name}.py"
    if not script_path.exists():
        print(f"Error: Script file {script_path} not found")
        return
    
    # Read the script once
    with open(script_path, 'r') as f:
        script_code = f.read()
    
    # Run calibration for each qubit pair configuration
    for idx, qubit_pairs in enumerate(qubit_pair_configs, 1):
        print(f"\n{'='*80}")
        print(f"Configuration {idx}/{len(qubit_pair_configs)}")
        if qubit_pairs is None:
            print("Using active qubit pairs from machine")
        else:
            print(f"Qubit pairs: {qubit_pairs}")
        if additional_params:
            print("Additional parameters:")
            for key, value in additional_params.items():
                print(f"  {key} = {value}")
        print(f"{'='*80}\n")
        
        try:
            # Create modified script code that injects parameter modification
            # right after node creation
            modified_code = script_code
            
            # Find the line that creates the node and inject parameter modification after it
            # Look for pattern: "node = QualibrationNode[...]"
            # Pattern to match node creation line
            node_pattern = r'(node\s*=\s*QualibrationNode[^\n]*)'
            
            if re.search(node_pattern, modified_code):
                # Create parameter override code
                param_override = f"""
# Parameter override injected by calibration_runner.py
node.parameters.qubit_pairs = {repr(qubit_pairs)}
"""
                for key, value in additional_params.items():
                    param_override += f"node.parameters.{key} = {repr(value)}\n"
                
                # Insert the override right after node creation
                modified_code = re.sub(
                    node_pattern,
                    r'\1\n' + param_override,
                    modified_code,
                    count=1  # Only replace the first occurrence
                )
            else:
                print("Warning: Could not find node creation line in script")
                print("Attempting to modify parameters after node is created...")
                # Fallback: try to modify after the first occurrence of "node ="
                if 'node =' in modified_code:
                    # Find first node assignment and add modification after it
                    lines = modified_code.split('\n')
                    for i, line in enumerate(lines):
                        if 'node = QualibrationNode' in line or ('node =' in line and 'QualibrationNode' in modified_code[max(0, i-5):i+5]):
                            # Insert parameter override after this line
                            param_override = f"node.parameters.qubit_pairs = {repr(qubit_pairs)}\n"
                            for key, value in additional_params.items():
                                param_override += f"node.parameters.{key} = {repr(value)}\n"
                            lines.insert(i + 1, param_override)
                            modified_code = '\n'.join(lines)
                            break
            
            # Create execution namespace
            exec_namespace = {
                '__name__': '__main__',
                '__file__': str(script_path),
            }
            
            # Execute the modified script
            exec(compile(modified_code, str(script_path), 'exec'), exec_namespace)
            
            print(f"\n✓ Completed configuration {idx}/{len(qubit_pair_configs)}")
            
        except KeyboardInterrupt:
            print(f"\n\n⚠ Interrupted by user at configuration {idx}/{len(qubit_pair_configs)}")
            break
        except Exception as e:
            print(f"\n✗ Error in configuration {idx}/{len(qubit_pair_configs)}: {e}")
            import traceback
            traceback.print_exc()
            print("\nContinuing with next configuration...\n")
            continue
    
    print(f"\n{'='*80}")
    print(f"Finished running {node_name} for all configurations")
    print(f"{'='*80}\n")

