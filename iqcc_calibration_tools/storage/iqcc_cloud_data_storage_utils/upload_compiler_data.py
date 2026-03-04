from iqcc_cloud_client import IQCC_Cloud
import os
import json
import logging

from iqcc_calibration_tools.compiler.generate_compiler_data import (
    generate_physical_qubits,
    generate_transpiler_target,
)
from iqcc_calibration_tools.quam_config.components.quam_root import Quam

PURPLE = '\033[95m'
RESET = '\033[0m'


class PurpleFormatter(logging.Formatter):
    def format(self, record):
        record.msg = f"{PURPLE}{record.msg}{RESET}"
        return super().format(record)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(PurpleFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.propagate = False


def upload_compiler_data_to_cloud(
    quam_state_folder_path: str = None,
    as_new_parent: bool = False,
):
    """
    Generate compiler data from the QuAM state and upload it to IQCC Cloud.

    Pushes ``physical_qubits`` first (optionally as a new root parent), then
    pushes ``transpiler_target`` with the ``physical_qubits`` dataset as its
    parent – mirroring the wiring/state relationship used elsewhere.

    Args:
        quam_state_folder_path: Path to the QuAM state folder.
            Falls back to the ``QUAM_STATE_PATH`` environment variable.
        as_new_parent: If ``True``, push ``physical_qubits`` as a new root
            entry (``parent_id=None``).  Otherwise only the latest existing
            ``physical_qubits`` dataset is used as parent for the transpiler
            target upload.
    """
    if quam_state_folder_path is None:
        if "QUAM_STATE_PATH" in os.environ:
            quam_state_folder_path = os.environ["QUAM_STATE_PATH"]
            logger.info(f"Using QUAM_STATE_PATH from environment: {quam_state_folder_path}")
        else:
            logger.error("QUAM_STATE_PATH environment variable is not set")
            raise ValueError("QUAM_STATE_PATH is not set")

    # Load the QuAM machine and generate compiler JSONs
    logger.info(f"Loading QuAM state from: {quam_state_folder_path}")
    machine = Quam.load(quam_state_folder_path)

    physical_qubits = generate_physical_qubits(machine)
    transpiler_target = generate_transpiler_target(machine, physical_qubits)

    n_qubits = len(physical_qubits["index2qubit"])
    n_pairs = len(physical_qubits["pair2index"])
    n_cz = len(transpiler_target["cz"])
    logger.info(f"Generated compiler data: {n_qubits} qubits, {n_pairs} pairs, {n_cz} CZ gates")

    # Resolve backend from wiring
    wiring_path = os.path.join(quam_state_folder_path, "wiring.json")
    with open(wiring_path, "r") as f:
        wiring = json.load(f)

    quantum_computer_backend = wiring["network"]["quantum_computer_backend"]
    logger.info(f"Initializing IQCC_Cloud with backend: {quantum_computer_backend}")
    qc = IQCC_Cloud(quantum_computer_backend=quantum_computer_backend)

    if as_new_parent:
        logger.info("Pushing physical_qubits as new parent")
        qc.state.push("physical_qubits", physical_qubits, comment="", parent_id=None)

    latest_pq = qc.state.get_latest("physical_qubits")
    logger.info(f"Retrieved latest physical_qubits dataset with ID: {latest_pq.id}")

    logger.info(f"Pushing transpiler_target with parent ID: {latest_pq.id}")
    qc.state.push("transpiler_target", transpiler_target, comment="", parent_id=latest_pq.id)
    logger.info("Done!")


if __name__ == "__main__":
    upload_compiler_data_to_cloud(as_new_parent=True)
