import logging
import os
import requests
from typing import TypeVar, Generic, List, Dict, Set, Tuple, Union
from datetime import datetime, timezone, timedelta
from qualibrate import QualibrationNode as QualibrationNodeBase
from qualibrate.parameters import NodeParameters
from qualibrate.utils.type_protocols import MachineProtocol
from qualibrate_config.resolvers import get_qualibrate_config_path, get_qualibrate_config
from qualibrate.utils.node.path_solver import get_node_dir_path
from qualibrate.config.resolvers import get_quam_state_path
from qualibrate.storage.local_storage_manager import LocalStorageManager
from iqcc_calibration_tools.quam_config.components.quam_root import Quam
from qm import generate_qua_script
from qualibration_libs.core import BatchableList

# Type variables for generic parameters - using same names and bounds as base class
ParametersType = TypeVar("ParametersType", bound=NodeParameters)
MachineType = TypeVar("MachineType", bound=MachineProtocol)

# ANSI color codes
MAGENTA = '\033[95m'
RESET = '\033[0m'

# Custom formatter for magenta colored logs
class MagentaFormatter(logging.Formatter):
    def format(self, record):
        record.msg = f"{MAGENTA}{record.msg}{RESET}"
        return super().format(record)

# Configure logger with magenta color
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(MagentaFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class QualibrationNode(QualibrationNodeBase, Generic[ParametersType, MachineType]):
    """
    Extended QualibrationNode with cloud upload capabilities.
    
    This class extends the base QualibrationNode to provide automatic cloud upload
    functionality when saving nodes, while maintaining local storage as the primary method.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the QualibrationNode with an automatically generated node_id and timestamp.
        
        Args:
            *args: Arguments passed to the parent class
            **kwargs: Keyword arguments passed to the parent class
        """
        super().__init__(*args, **kwargs)
        self.time_zone = 2
        self.node_id = self.get_node_id()
        self.date_time = datetime.now(timezone(timedelta(hours=self.time_zone))).strftime("%Y-%m-%d %H:%M:%S")
        self._machine = None  # Initialize machine attribute
    
    @property
    def machine(self):
        """
        Property that returns the quam state.
        
        Returns:
            The quam state
        """
        return self._machine
    
    @machine.setter
    def machine(self, machine_config):
        """
        Setter for machine property that automatically processes the quam state.
        
        Args:
            machine_config: Dictionary containing the quam state
        """
        # logger.info("Setting machine configuration and processing...")
        
        if machine_config is None:
            # logger.warning("machine_config is None - cannot process")
            self._machine = None
            return
        
        # Store the machine configuration
        self._machine = machine_config
        # logger.info(f"Machine stored successfully. Type: {type(self._machine)}")
    
    def save(self):
        """
        Save a QualibrationNode both locally and to cloud if possible.
        
        This function first saves the node locally, then attempts to upload to cloud
        if the necessary cloud dependencies are available and the user has proper access rights.
        The cloud upload is optional and will be skipped if:
        1. Cloud dependencies (IQCC_Cloud and QualibrateCloudHandler) are not available
        2. No quantum computer backend is specified
        3. User doesn't have proper IQCC project access rights
        
        Returns:
            None
        """
        logger.info(f"Saving node with snapshot index {self.snapshot_idx}")
        
        # remove macros from quam object
        Quam.remove_macros_from_qubits(self.machine)
        
        # Save locally first (primary operation)
        super().save()
        logger.info("Node saved locally")
        
        # Attempt cloud upload if conditions are met
        self._attempt_cloud_upload()
    
    def _attempt_cloud_upload(self):
        """
        Attempt to upload the node to cloud storage.
        
        This method handles the cloud upload process with proper error handling
        and logging. It will gracefully skip upload if any requirements are not met.
        """
        # Check if cloud dependencies are available
        if not self._check_cloud_dependencies():
            return
        
        # Check if quantum computer backend is specified
        quantum_computer_backend = self._get_quantum_computer_backend()
        if not quantum_computer_backend:
            logger.info("No quantum computer backend specified - skipping cloud upload")
            return
        
        # Check access rights and upload
        self._upload_to_cloud(quantum_computer_backend)
    
    def _check_cloud_dependencies(self):
        """
        Check if required cloud dependencies are available.
        
        Returns:
            bool: True if dependencies are available, False otherwise
        """
        try:
            from iqcc_cloud_client import IQCC_Cloud
            from iqcc_qualibrate2cloud import QualibrateCloudHandler
            return True
        except ImportError:
            logger.info("Cloud dependencies not available - skipping cloud upload")
            return False
    
    def _get_quantum_computer_backend(self):
        """
        Get the quantum computer backend from the node's machine network.
        
        Returns:
            str or None: The quantum computer backend name, or None if not specified
        """
        try:
            return self.machine.network.get("quantum_computer_backend", None)
        except AttributeError:
            logger.warning("Unable to access machine network - skipping cloud upload")
            return None
    
    def _upload_to_cloud(self, quantum_computer_backend):
        """
        Upload the node to cloud storage.
        
        Args:
            quantum_computer_backend (str): The quantum computer backend name
        """
        try:
            from iqcc_cloud_client import IQCC_Cloud
            from iqcc_qualibrate2cloud import QualibrateCloudHandler
            
            logger.info(f"Found quantum computer backend: {quantum_computer_backend}")
            
            # Initialize cloud client
            qc = IQCC_Cloud(quantum_computer_backend=quantum_computer_backend)
            
            # Check access rights
            if not self._has_iqcc_access(qc):
                logger.info("No IQCC project access - skipping cloud upload")
                return
            
            # Perform upload
            self._perform_cloud_upload(quantum_computer_backend)
            
        except Exception as e:
            logger.error(f"Error during cloud upload: {e}")
    
    def _has_iqcc_access(self, qc):
        """
        Check if the user has IQCC project access.
        
        Args:
            qc: IQCC_Cloud instance
            
        Returns:
            bool: True if user has IQCC access, False otherwise
        """
        try:
            return qc.access_rights['projects'] == ['iqcc']
        except (KeyError, AttributeError):
            logger.warning("Unable to verify access rights - skipping cloud upload")
            return False
    
    def _perform_cloud_upload(self, quantum_computer_backend):
        """
        Perform the actual cloud upload operation.
        
        Args:
            quantum_computer_backend (str): The quantum computer backend name
        """
        try:
            from iqcc_qualibrate2cloud import QualibrateCloudHandler
            
            # Get configuration and paths using the actual functions from the script
            q_config_path = get_qualibrate_config_path()
            qs = get_qualibrate_config(q_config_path)
            base_path = qs.storage.location
            node_id = self.snapshot_idx
            node_dir = get_node_dir_path(node_id, base_path)
            
            # Create handler and upload
            handler = QualibrateCloudHandler(str(node_dir))
            handler.upload_to_cloud(quantum_computer_backend)
            
            logger.info("Node successfully uploaded to cloud")
            
        except Exception as e:
            logger.error(f"Error during cloud upload operation: {e}")

    def get_node_id(self) -> int:
        """
        Get the current node ID from the storage manager.
        
        Returns:
            int: The node ID
        """
        q_config_path = get_qualibrate_config_path()
        qs = get_qualibrate_config(q_config_path)
        state_path = get_quam_state_path(qs)
        storage_manager = LocalStorageManager(
                    root_data_folder=qs.storage.location,
                    active_machine_path=state_path,
                )
        
        return storage_manager.data_handler.generate_node_contents()['id']

    def add_node_info_subtitle(self, fig=None, additional_info=None):
        """
        Add a standardized subtitle with node information to a matplotlib figure.
        If a suptitle already exists, the node info will be appended to it.
        
        Args:
            fig: matplotlib figure object. If None, uses plt.gcf()
            additional_info: Optional string with additional information to include
            
        Returns:
            str: The subtitle text that was added
        """
        import matplotlib.pyplot as plt
        
        if fig is None:
            fig = plt.gcf()
        
        # Build the base subtitle
        subtitle_parts = [f"{self.date_time} GMT+{self.time_zone} #{self.node_id}"]
        
        # Add multiplexed info if the parameter exists
        if hasattr(self.parameters, 'multiplexed'):
            subtitle_parts.append(f"multiplexed = {self.parameters.multiplexed}")
        
        # Add reset type info if the parameter exists
        param_name = 'reset_type'
        if hasattr(self.parameters, param_name):
            subtitle_parts.append(f"reset type = {getattr(self.parameters, param_name)}")
        
        # Add any additional info
        if additional_info:
            subtitle_parts.append(additional_info)
        
        # Join all parts with newlines
        node_info_text = "\n".join(subtitle_parts)
        
        # Check if there's an existing suptitle
        existing_suptitle = fig._suptitle
        if existing_suptitle is not None and existing_suptitle.get_text().strip():
            # Append node info to existing suptitle
            combined_text = f"{existing_suptitle.get_text()}\n{node_info_text}"
        else:
            # No existing suptitle, use just the node info
            combined_text = node_info_text
        
        # Add the subtitle to the figure
        fig.suptitle(combined_text, fontsize=10, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to prevent overlap with less spacing
        
        return node_info_text

    def get_node_info_text(self, additional_info=None):
        """
        Get the node information text without adding it to a figure.
        
        Args:
            additional_info: Optional string with additional information to include
            
        Returns:
            str: The formatted node information text
        """
        # Build the base subtitle
        subtitle_parts = [f"{self.date_time} GMT+{self.time_zone} #{self.node_id}"]
        
        # Add multiplexed info if the parameter exists
        if hasattr(self.parameters, 'multiplexed'):
            subtitle_parts.append(f"multiplexed = {self.parameters.multiplexed}")
        
        # Add reset type info if the parameter exists
        param_name = 'reset_type'
        if hasattr(self.parameters, param_name):
            subtitle_parts.append(f"reset type = {getattr(self.parameters, param_name)}")
        
        # Add any additional info
        if additional_info:
            subtitle_parts.append(additional_info)
        
        # Join all parts with newlines
        return "\n".join(subtitle_parts)
    
    def serialize_qua_program(self, path: str | None = None):
        if path is None:
            file_name = f"debug_{self.name}_{self.node_id}.py"
        else:
            file_name = os.path.join(path, f"debug_{self.name}_{self.node_id}.py")
        sourceFile = open(file_name, 'w')
        print(generate_qua_script(self.namespace["qua_program"], self.machine.generate_config()), file=sourceFile) 
        sourceFile.close()
    
    def get_multiplexed_pair_batches(self, qubit_pairs: Union[List[str], List]) -> BatchableList:
        """
        Collect all active qubit pair names and group them into multiplexed batches
        such that pairs in the same batch do not share any qubits and do not have
        qubits that are nearest neighbors. Includes spectator qubits from CZ gates
        in the constraint checking.
        
        Args:
            qubit_pairs: Either a list of pair name strings or a list of pair objects.
                        If strings, they will be converted to pair objects.
        
        Returns:
            BatchableList: A BatchableList containing qubit pair objects, with batch_groups
                          set according to multiplexing constraints.
                          Pairs within the same batch:
                          - Do not share any qubits (including control, target, and spectator qubits)
                          - Do not have qubits that are nearest neighbors (including spectator qubits)
                          Pairs are processed starting with those having the most spectator qubits
                          to optimize the greedy algorithm.
                          If multiplexed is False or not set, each pair is in its own batch.
        
        Raises:
            AttributeError: If machine is None or doesn't have required attributes
        """
        if self.machine is None:
            raise AttributeError("Machine is not set. Cannot collect qubit pairs.")
        
        # Convert input to list of pair objects and pair names
        pair_objects: List = []
        pair_names: List[str] = []
        
        if len(qubit_pairs) > 0:
            # Check if first element is a string (pair name) or an object
            if isinstance(qubit_pairs[0], str):
                # Input is list of pair name strings
                pair_names = qubit_pairs
                for pair_name in pair_names:
                    if pair_name not in self.machine.qubit_pairs:
                        logger.warning(f"Pair '{pair_name}' not found in machine.qubit_pairs, skipping")
                        continue
                    pair_objects.append(self.machine.qubit_pairs[pair_name])
            else:
                # Input is list of pair objects
                pair_objects = qubit_pairs
                for pair_obj in pair_objects:
                    # Get pair name/id
                    if hasattr(pair_obj, 'id'):
                        pair_names.append(pair_obj.id)
                    elif hasattr(pair_obj, 'name'):
                        pair_names.append(pair_obj.name)
                    else:
                        # Try to construct name from qubits
                        if hasattr(pair_obj, 'qubit_control') and hasattr(pair_obj, 'qubit_target'):
                            pair_name = f"{pair_obj.qubit_control.name}-{pair_obj.qubit_target.name}"
                            pair_names.append(pair_name)
                        else:
                            raise ValueError(f"Cannot determine pair name for object {pair_obj}")
        
        # Check if multiplexed is enabled
        is_multiplexed = hasattr(self.parameters, 'multiplexed') and self.parameters.multiplexed
        
        if not is_multiplexed:
            # If not multiplexed, return batches of size 1 (each pair in its own batch)
            batch_groups = [[i] for i in range(len(pair_objects))]
        else:
            # Helper function to parse grid location
            def parse_grid_location(location_str: str) -> Tuple[int, int]:
                """Parse grid location string like '2,4' into (x, y) tuple."""
                x, y = map(int, location_str.split(','))
                return (x, y)
            
            # Helper function to compute Manhattan distance
            def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
                return abs(a[0] - b[0]) + abs(a[1] - b[1])
            
            # Extract grid locations for all qubits
            qubit_grid_locations: Dict[str, Tuple[int, int]] = {}
            for qubit_name, qubit in self.machine.qubits.items():
                if hasattr(qubit, 'grid_location') and qubit.grid_location:
                    try:
                        qubit_grid_locations[qubit_name] = parse_grid_location(qubit.grid_location)
                    except (ValueError, AttributeError):
                        # Skip qubits without valid grid locations
                        pass
            
            # Build set of nearest neighbor pairs (as sets for easy lookup)
            nearest_neighbor_pairs: Set[Tuple[str, str]] = set()
            qubit_names = list(qubit_grid_locations.keys())
            for i in range(len(qubit_names)):
                q1 = qubit_names[i]
                if q1 not in qubit_grid_locations:
                    continue
                p1 = qubit_grid_locations[q1]
                for j in range(i + 1, len(qubit_names)):
                    q2 = qubit_names[j]
                    if q2 not in qubit_grid_locations:
                        continue
                    p2 = qubit_grid_locations[q2]
                    if manhattan_distance(p1, p2) == 1:
                        # Store both orderings for easy lookup
                        nearest_neighbor_pairs.add((q1, q2))
                        nearest_neighbor_pairs.add((q2, q1))
            
            # Helper function to check if two qubits are nearest neighbors
            def are_nearest_neighbors(q1: str, q2: str) -> bool:
                return (q1, q2) in nearest_neighbor_pairs
            
            # Collect pair information and group into batches
            batch_groups: List[List[int]] = []  # Initialize batches list with indices
            # Map pair_index -> set of qubits in that pair (including spectator qubits)
            pair_qubits: Dict[int, Set[str]] = {}
            # Map pair_index -> number of spectator qubits (for sorting)
            pair_spectator_count: Dict[int, int] = {}
            
            for idx, pair_obj in enumerate(pair_objects):
                # Start with control and target qubits
                qubits_set = {
                    pair_obj.qubit_control.name,
                    pair_obj.qubit_target.name
                }
                
                # Add spectator qubits if they exist
                spectator_count = 0
                if hasattr(pair_obj, 'macros') and 'cz' in pair_obj.macros:
                    cz_macro = pair_obj.macros['cz']
                    if hasattr(cz_macro, 'spectator_qubits') and cz_macro.spectator_qubits:
                        # spectator_qubits is a dictionary where keys are qubit names
                        for spectator_qubit_name in cz_macro.spectator_qubits.keys():
                            qubits_set.add(spectator_qubit_name)
                            spectator_count += 1
                
                pair_qubits[idx] = qubits_set
                pair_spectator_count[idx] = spectator_count
            
            # Sort pairs by number of spectator qubits (most first) to optimize greedy algorithm
            sorted_pair_indices = sorted(
                pair_qubits.keys(),
                key=lambda idx: pair_spectator_count[idx],
                reverse=True
            )
            
            # Group pairs into batches using greedy algorithm
            batch_qubits: List[Set[str]] = []  # Track qubits used in each batch
            
            for pair_idx in sorted_pair_indices:
                qubits_in_pair = pair_qubits[pair_idx]
                # Try to find a batch where this pair doesn't conflict
                batch_found = False
                for i, batch_qubit_set in enumerate(batch_qubits):
                    # Check if pairs share any qubits (original constraint)
                    if not qubits_in_pair.isdisjoint(batch_qubit_set):
                        continue
                    
                    # Check if any qubit in this pair is nearest neighbor to any qubit in this batch
                    conflict_with_nn = False
                    for qubit_in_pair in qubits_in_pair:
                        for existing_qubit in batch_qubit_set:
                            if are_nearest_neighbors(qubit_in_pair, existing_qubit):
                                conflict_with_nn = True
                                break
                        if conflict_with_nn:
                            break
                    
                    if conflict_with_nn:
                        continue
                    
                    # No conflict, add to this batch
                    batch_groups[i].append(pair_idx)
                    batch_qubits[i].update(qubits_in_pair)
                    batch_found = True
                    break
                
                # If no existing batch works, create a new one
                if not batch_found:
                    batch_groups.append([pair_idx])
                    batch_qubits.append(qubits_in_pair.copy())
    
        return BatchableList(pair_objects, batch_groups)

    def get_job_billable_cloud_time(self):
        """
        Get the billable cloud execution time from the job's result handles.
        
        Returns:
            float or None: The QPU execution time in seconds if available, None otherwise
        """
        try:
            job = self.namespace.get('job')
            if job is None:
                return None
            
            result_handles = getattr(job, 'result_handles', None)
            if result_handles is None:
                return None
            
            execution_time = getattr(result_handles, '__qpu_execution_time_seconds', None)
            return execution_time
        except (AttributeError, KeyError):
            return None

    