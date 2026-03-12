import os
import json
import logging
from iqcc_cloud_client import IQCC_Cloud

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_compiler_data(quantum_computer_backend: str, folder_path: str | None = None) -> None:
    """
    Download the latest physical_qubits and transpiler_target files from the cloud.

    Args:
        quantum_computer_backend: The name of the quantum computer backend to use.
        folder_path: Optional explicit folder path to save the files.
            If not provided, uses the QUAM_STATE_PATH environment variable.
    """
    try:
        logger.info(f"Connecting to quantum computer backend: {quantum_computer_backend}")
        qc = IQCC_Cloud(quantum_computer_backend=quantum_computer_backend)

        logger.info("Fetching latest physical_qubits and transpiler_target")
        latest_physical_qubits = qc.state.get_latest("physical_qubits")
        latest_transpiler_target = qc.state.get_latest("transpiler_target")

        quam_state_folder_path = folder_path if folder_path is not None else os.environ["QUAM_STATE_PATH"]
        logger.info(f"State folder path: {quam_state_folder_path}")

        os.makedirs(quam_state_folder_path, exist_ok=True)

        pq_path = os.path.join(quam_state_folder_path, "physical_qubits.json")
        tt_path = os.path.join(quam_state_folder_path, "transpiler_target.json")

        with open(pq_path, "w") as f:
            json.dump(latest_physical_qubits.data, f, indent=4)
        logger.info(f"Saved physical_qubits to: {pq_path}")

        with open(tt_path, "w") as f:
            json.dump(latest_transpiler_target.data, f, indent=4)
        logger.info(f"Saved transpiler_target to: {tt_path}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    download_compiler_data("arbel")
