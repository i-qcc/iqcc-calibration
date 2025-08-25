# Make iqcc_calibration_tools a Python package

from iqcc_cloud_client.computers import *  # noqa: F403, F401
from .patches import apply_iqcc_patches


# Apply patches to external libraries
apply_iqcc_patches()


def _get_qubits(machine: AnyQuam, node_parameters: QubitsExperimentNodeParameters) -> List[AnyTransmon]:
    if node_parameters.qubits is None or node_parameters.qubits == "":
        qubits = machine.active_qubits
    else:
        qubits = [machine.qubits[q] for q in node_parameters.qubits]

    return qubits


def get_qubits(node, node_parameters=None):
    if node_parameters is None:
        node_parameters = node.parameters

    qubits = _get_qubits(node.machine, node_parameters)
    from qualibration_libs.parameters.experiment import _make_batchable_list_from_multiplexed

    return _make_batchable_list_from_multiplexed(qubits, node_parameters.multiplexed)
