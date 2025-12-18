from iqcc_cloud_client import IQCC_Cloud
import os
import json
import base64
import logging
from pathlib import Path

# Import the plot function
from iqcc_calibration_tools.visualizations.plot_qubit_grid import main as plot_main

# Configure logging with purple color only for our logger
PURPLE = '\033[95m'
RESET = '\033[0m'

# Create a custom formatter for our logger
class PurpleFormatter(logging.Formatter):
    def format(self, record):
        record.msg = f"{PURPLE}{record.msg}{RESET}"
        return super().format(record)

# Configure root logger with default format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configure our specific logger with purple color
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(PurpleFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.propagate = False  # Prevent propagation to root logger

def upload_qubit_grid_plot_to_cloud(quam_state_folder_path: str = None):
    """
    Generate the qubit grid plot and upload it to cloud under the latest parent.
    
    Args:
        quam_state_folder_path: Path to the QUAM state folder. If None, uses QUAM_STATE_PATH env var.
    """
    # Get QUAM state path
    if quam_state_folder_path is None:
        if "QUAM_STATE_PATH" in os.environ:
            quam_state_folder_path = os.environ["QUAM_STATE_PATH"]
            logger.info(f"Using QUAM_STATE_PATH from environment: {quam_state_folder_path}")
        else:
            logger.error("QUAM_STATE_PATH environment variable is not set")
            raise ValueError("QUAM_STATE_PATH is not set")
    
    # Read wiring to get quantum_computer_backend
    wiring_path = quam_state_folder_path + "/wiring.json"
    with open(wiring_path, "r") as f:
        wiring = json.load(f)
    
    quantum_computer_backend = wiring["network"]["quantum_computer_backend"]
    logger.info(f"Initializing IQCC_Cloud with backend: {quantum_computer_backend}")
    qc = IQCC_Cloud(quantum_computer_backend=quantum_computer_backend)
    
    # Get the latest parent (wiring dataset)
    logger.info("Retrieving latest wiring dataset as parent")
    latest_dataset = qc.state.get_latest("wiring")
    logger.info(f"Retrieved latest wiring dataset with ID: {latest_dataset.id}")
    
    # Generate the plot
    logger.info("Generating qubit grid plot...")
    plot_main()
    
    # Find the generated PNG file (same location as plot script)
    import iqcc_calibration_tools.visualizations.plot_qubit_grid as plot_module
    plot_script_dir = Path(plot_module.__file__).parent
    png_path = plot_script_dir / "qubit_grid_plot.png"
    
    if not png_path.exists():
        logger.error(f"PNG file not found at {png_path}")
        raise FileNotFoundError(f"PNG file not found at {png_path}")
    
    logger.info(f"Reading PNG file from: {png_path}")
    
    # Read and encode PNG file
    with open(png_path, "rb") as f:
        png_bytes = f.read()
        base64_str = base64.b64encode(png_bytes).decode("utf-8")
    
    # Prepare PNG data in the same format as qualibrate_cloud_handler
    png_data = {
        "data": base64_str,
        "__type__": "png/base64",
        "file_name": png_path.name,
    }
    
    # Upload PNG to cloud under the latest parent
    logger.info(f"Pushing PNG to cloud with parent ID: {latest_dataset.id}")
    qc.state.push("figure", png_data, comment="", parent_id=latest_dataset.id)
    logger.info("Done!")

if __name__ == "__main__":
    upload_qubit_grid_plot_to_cloud()

