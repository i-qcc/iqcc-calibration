"""
QuamTools class for analyzing and manipulating quam objects.

This class provides methods to analyze quam objects that are loaded from state.json
and wiring.json files in the quam state directory.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set

logger = logging.getLogger(__name__)


class QuamTools:
    """
    Tools for analyzing and manipulating quam objects.
    
    This class contains class methods that analyze or change quam objects
    defined in calibration nodes with node.machine. The objects are loaded
    from state.json and wiring.json files.
    """
    
    # Environment variable name for the QUAM state directory
    ENV_VAR_STATE_PATH = "QUAM_STATE_PATH"

    def __init__(self, state_path: Optional[Path] = None) -> None:
        """
        Initialize QuamTools with a path to the QUAM state directory.

        Args:
            state_path: Path to the QUAM state directory. If None, falls back to
                        the QUAM_STATE_PATH environment variable.

        Raises:
            ValueError: If neither state_path is provided nor the environment
                        variable QUAM_STATE_PATH is set.
        """
        resolved_path: Optional[str] = None
        if state_path is not None:
            resolved_path = str(state_path)
        else:
            resolved_path = os.environ.get(self.ENV_VAR_STATE_PATH)

        if not resolved_path:
            raise ValueError(
                "QUAM state path is not provided. Set it when constructing QuamTools "
                "or set the QUAM_STATE_PATH environment variable."
            )

        self.state_path: Path = Path(resolved_path)
    
    # def analyze_quam_state(self) -> Dict[str, Any]:
    #     """
    #     Analyze a quam state and log important information.
        
    #     This method loads the state.json and wiring.json files from the quam state
    #     directory and extracts key information about the quantum system configuration.
        
    #     Returns:
    #         Dictionary containing analyzed information about the quam state.
            
    #     Raises:
    #         FileNotFoundError: If state.json or wiring.json files are not found.
    #         json.JSONDecodeError: If the JSON files are malformed.
    #     """
    #     state_file = self.state_path / "state.json"
    #     wiring_file = self.state_path / "wiring.json"
        
    #     # Check if files exist
    #     if not state_file.exists():
    #         raise FileNotFoundError(f"state.json not found at {state_file}")
    #     if not wiring_file.exists():
    #         raise FileNotFoundError(f"wiring.json not found at {wiring_file}")
            
    #     # Load the JSON files
    #     try:
    #         with open(state_file, 'r') as f:
    #             state_data = json.load(f)
    #         with open(wiring_file, 'r') as f:
    #             wiring_data = json.load(f)
    #     except json.JSONDecodeError as e:
    #         raise json.JSONDecodeError(f"Failed to parse JSON file: {e}")
            
    #     # Analyze the data and extract important information
    #     analysis_result = self._extract_important_info(state_data, wiring_data)
        
    #     # Log the important information
    #     self._log_important_info(analysis_result)
        
    #     return analysis_result

    # @classmethod
    # def analyze_quam_state_from(
    #     cls, quam_state_path: Optional[Path] = None
    # ) -> Dict[str, Any]:
    #     """
    #     Backward-compatible wrapper to analyze a QUAM state by providing a path
    #     or relying on the QUAM_STATE_PATH environment variable.

    #     Args:
    #         quam_state_path: Optional explicit path to the QUAM state directory.

    #     Returns:
    #         Analysis dictionary as returned by analyze_quam_state().
    #     """
    #     instance = cls(state_path=quam_state_path)
    #     return instance.analyze_quam_state()
    
    # @classmethod
    # def _extract_important_info(cls, state_data: Dict[str, Any], wiring_data: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Extract important information from state and wiring data.
        
    #     Args:
    #         state_data: The parsed state.json data
    #         wiring_data: The parsed wiring.json data
            
    #     Returns:
    #         Dictionary containing extracted important information
    #     """
    #     important_info = {
    #         "qubits": {},
    #         "resonators": {},
    #         "mixers": {},
    #         "oscillators": {},
    #         "network": {},
    #         "summary": {}
    #     }
        
    #     # Extract qubit information
    #     if "qubits" in state_data:
    #         for qubit_name, qubit_data in state_data["qubits"].items():
    #             important_info["qubits"][qubit_name] = {
    #                 "xy_frequency": qubit_data.get("xy", {}).get("frequency"),
    #                 "z_frequency": qubit_data.get("z", {}).get("frequency"),
    #                 "readout_frequency": qubit_data.get("readout", {}).get("frequency"),
    #                 "xy_operations": list(qubit_data.get("xy", {}).get("operations", {}).keys()),
    #                 "z_operations": list(qubit_data.get("z", {}).get("operations", {}).keys()),
    #             }
        
    #     # Extract resonator information
    #     if "resonators" in state_data:
    #         for resonator_name, resonator_data in state_data["resonators"].items():
    #             important_info["resonators"][resonator_name] = {
    #                 "frequency": resonator_data.get("frequency"),
    #                 "operations": list(resonator_data.get("operations", {}).keys()),
    #             }
        
    #     # Extract mixer information
    #     if "mixers" in state_data:
    #         for mixer_name, mixer_data in state_data["mixers"].items():
    #             important_info["mixers"][mixer_name] = {
    #                 "intermediate_frequency": mixer_data.get("intermediate_frequency"),
    #                 "lo_frequency": mixer_data.get("lo_frequency"),
    #             }
        
    #     # Extract oscillator information
    #     if "oscillators" in state_data:
    #         for oscillator_name, oscillator_data in state_data["oscillators"].items():
    #             important_info["oscillators"][oscillator_name] = {
    #                 "frequency": oscillator_data.get("frequency"),
    #             }
        
    #     # Extract network information
    #     if "network" in state_data:
    #         important_info["network"] = {
    #             "quantum_computer_backend": state_data["network"].get("quantum_computer_backend"),
    #             "qop_ip": state_data["network"].get("qop_ip"),
    #             "qop_port": state_data["network"].get("qop_port"),
    #         }
        
    #     # Create summary
    #     important_info["summary"] = {
    #         "num_qubits": len(important_info["qubits"]),
    #         "num_resonators": len(important_info["resonators"]),
    #         "num_mixers": len(important_info["mixers"]),
    #         "num_oscillators": len(important_info["oscillators"]),
    #         "has_network_config": bool(important_info["network"]),
    #     }
        
    #     return important_info
    
    # @classmethod
    # def _log_important_info(cls, analysis_result: Dict[str, Any]) -> None:
    #     """
    #     Log the important information extracted from the quam state.
        
    #     Args:
    #         analysis_result: The analysis result from _extract_important_info
    #     """
    #     summary = analysis_result["summary"]
        
    #     logger.info("=== QUAM STATE ANALYSIS ===")
    #     logger.info(f"Number of qubits: {summary['num_qubits']}")
    #     logger.info(f"Number of resonators: {summary['num_resonators']}")
    #     logger.info(f"Number of mixers: {summary['num_mixers']}")
    #     logger.info(f"Number of oscillators: {summary['num_oscillators']}")
        
    #     if summary["has_network_config"]:
    #         network = analysis_result["network"]
    #         logger.info(f"Quantum computer backend: {network.get('quantum_computer_backend', 'Not specified')}")
    #         logger.info(f"QOP IP: {network.get('qop_ip', 'Not specified')}")
    #         logger.info(f"QOP Port: {network.get('qop_port', 'Not specified')}")
        
    #     # Log qubit details
    #     if analysis_result["qubits"]:
    #         logger.info("\n=== QUBIT DETAILS ===")
    #         for qubit_name, qubit_info in analysis_result["qubits"].items():
    #             logger.info(f"Qubit {qubit_name}:")
    #             logger.info(f"  XY Frequency: {qubit_info['xy_frequency']}")
    #             logger.info(f"  Z Frequency: {qubit_info['z_frequency']}")
    #             logger.info(f"  Readout Frequency: {qubit_info['readout_frequency']}")
    #             logger.info(f"  XY Operations: {qubit_info['xy_operations']}")
    #             logger.info(f"  Z Operations: {qubit_info['z_operations']}")
        
    #     # Log resonator details
    #     if analysis_result["resonators"]:
    #         logger.info("\n=== RESONATOR DETAILS ===")
    #         for resonator_name, resonator_info in analysis_result["resonators"].items():
    #             logger.info(f"Resonator {resonator_name}:")
    #             logger.info(f"  Frequency: {resonator_info['frequency']}")
    #             logger.info(f"  Operations: {resonator_info['operations']}")
        
    #     logger.info("=== END ANALYSIS ===\n")

    # ============ Qubit grid and nearest-neighbor utilities ============

    @staticmethod
    def _parse_grid_location(value: str) -> Tuple[int, int]:
        x_str, y_str = value.split(",")
        return int(x_str.strip()), int(y_str.strip())

    def _load_qubit_locations(self) -> Dict[str, Tuple[int, int]]:
        state_file = self.state_path / "state.json"
        if not state_file.exists():
            raise FileNotFoundError(f"state.json not found at {state_file}")

        with state_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        qubits = data.get("qubits", {})
        qubit_to_xy: Dict[str, Tuple[int, int]] = {}
        for qubit_id, qubit_obj in qubits.items():
            grid_loc = qubit_obj.get("grid_location")
            if not grid_loc:
                continue
            try:
                qubit_to_xy[qubit_id] = self._parse_grid_location(grid_loc)
            except Exception:
                # Skip malformed entries silently
                continue

        return qubit_to_xy

    @staticmethod
    def _manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @classmethod
    def _compute_nearest_neighbors(
        cls, qubit_to_xy: Dict[str, Tuple[int, int]]
    ) -> List[Tuple[str, str]]:
        qubit_ids = list(qubit_to_xy.keys())
        nn_pairs: Set[Tuple[str, str]] = set()

        for i in range(len(qubit_ids)):
            q1 = qubit_ids[i]
            p1 = qubit_to_xy[q1]
            for j in range(i + 1, len(qubit_ids)):
                q2 = qubit_ids[j]
                p2 = qubit_to_xy[q2]
                if cls._manhattan_distance(p1, p2) == 1:
                    pair = tuple(sorted((q1, q2)))
                    nn_pairs.add(pair)

        return sorted(nn_pairs)

    def get_qubit_locations(self) -> Dict[str, Tuple[int, int]]:
        """
        Return a mapping from qubit id to its (x, y) grid location as integers.
        """
        return self._load_qubit_locations()

    def get_nearest_neighbor_pairs(self) -> List[Tuple[str, str]]:
        """
        Return a sorted list of nearest-neighbor qubit id pairs using Manhattan distance 1.
        """
        return self._compute_nearest_neighbors(self._load_qubit_locations())
