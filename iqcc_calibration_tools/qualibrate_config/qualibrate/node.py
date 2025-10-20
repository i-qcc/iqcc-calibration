import logging
from typing import TypeVar, Generic
from datetime import datetime, timezone, timedelta
from qualibrate import QualibrationNode as QualibrationNodeBase
from qualibrate.parameters import NodeParameters
from qualibrate.utils.type_protocols import MachineProtocol
from qualibrate_config.resolvers import get_qualibrate_config_path, get_qualibrate_config
from qualibrate.utils.node.path_solver import get_node_dir_path
from qualibrate.config.resolvers import get_quam_state_path
from qualibrate.storage.local_storage_manager import LocalStorageManager
from iqcc_calibration_tools.quam_config.components.quam_root import Quam
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
        self.node_id = self.get_node_id()
        self.date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
    
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
            from cloud_qualibrate_link.qualibrate_cloud_handler import QualibrateCloudHandler
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
            from cloud_qualibrate_link.qualibrate_cloud_handler import QualibrateCloudHandler
            
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
            from cloud_qualibrate_link.qualibrate_cloud_handler import QualibrateCloudHandler
            
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
        subtitle_parts = [f"{self.date_time} GMT+3 #{self.node_id}"]
        
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
        subtitle_parts = [f"{self.date_time} GMT+3 #{self.node_id}"]
        
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
