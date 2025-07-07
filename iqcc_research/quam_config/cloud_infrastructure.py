import warnings
import importlib

if importlib.util.find_spec("iqcc_cloud_client"):
    from iqcc_cloud_client import IQCC_Cloud


class CloudQuantumMachinesManager:
    def __init__(self, backend):
        self.backend = backend

    def open_qm(self, config: dict, close_other_machines: bool,keep_dc_offsets_when_closing=True):
        self._qm = CloudQuantumMachine(self.backend, config)
        return self._qm


class CloudQuantumMachine:
    def __init__(self, backend,config: dict):
        self._qc = IQCC_Cloud(
            quantum_computer_backend=backend)
        self._config = config

    def execute(self, program, terminal_output=False, options = {}):
        run_data = self._qc.execute(program, self._config, terminal_output=terminal_output, options = {"timeout":300, **options})
        self.job = CloudJob(run_data)
        return self.job

    def get_running_job(self):
        if self.job.result_handles.is_processing():
            return self.job
        else:
            return None

    def close(self):
        pass


class CloudJob:
    def __init__(self, run_data: dict):
        self._run_data = run_data
        self.result_handles = CloudResultHandles(self._run_data['result'])
    
    def execution_report(self):
        """
        This is a placeholder for the execution_report method to not break api of qualibration_libs (which does not assumes cloud results object).
        It is used to display the execution report.
        """
        return None


class CloudResultHandles:
    def __init__(self, results_dict: dict):
        self._results_dict = results_dict
        self._is_processing = True
        for result in results_dict:
            setattr(self, result, results_dict[result])

    def is_processing(self):
        is_processing = self._is_processing
        if is_processing:
            self._is_processing = False
        return is_processing

    def wait_for_all_values(self):
        pass

    def keys(self):
        return self._results_dict.keys()

    def get(self, handle: str):
        return CloudResult(self._results_dict[handle])
                   
class CloudResult:
    def __init__(self, data):
        self._data = data

    def fetch_all(self):
        return self._data

    def wait_for_values(self, *args):
        pass
    
    def count_so_far(self):
        """
        This is a placeholder for the count_so_far method to not break api of qualibration_libs (which does not assumes cloud results object).
        It is used to check if the job is processing.
        """
        return None