"""
IQCC Cloud Client Patches

This module contains monkey patches for the IQCC Cloud Client to enhance functionality.
"""

import httpx
import os
import base64
from time import sleep
import json
import sys
import signal
from sys import exit
from iqcc_cloud_client.computers import IQCC_Cloud


REFRESH_RATE = 1


def _update_action_label(msg, percentage=None):
    """Update the action label in the QualibrationNode."""
    from iqcc_calibration_tools.qualibrate_config.qualibrate.node import (
        QualibrationNode,
    )

    node = QualibrationNode.active_node
    if node is None:
        return
    node.action_label = msg  # type: ignore
    if percentage is not None:
        node.fraction_complete = percentage / 100  # type: ignore


def _complex_decoder(dct):
    if "__complex__" in dct:
        return complex(dct["real"], dct["imag"])
    return dct


def __serve_request_spinner(
    self,
    request_name: str,
    payload: dict,
    api_path: str,
    method: str,
    terminal_output=False,
    debug=False,
    options: dict = {},
    api_path_post=None,
):
    """
    Enhanced version of __serve_request_spinner with improved progress tracking
    and action label updates for QualibrationNode integration.
    """
    url = self.url + f"/{api_path}/{self.backend}"
    if api_path_post is not None:
        url += f"/{api_path_post}"

    if method == "POST":
        r = httpx.post(url, json=payload, headers=self.headers, timeout=self.timeout)
    else:
        r = httpx.get(url, headers=self.headers, timeout=self.timeout)

    if r.status_code != httpx.codes.OK:
        print(f"Error {r.status_code}: {r.json()['detail']}\n")
        exit()

    task_id = r.json()["task_id"]
    task_id_short = task_id.split("-")[0]

    def signal_handler(sig, frame):
        httpx.get(
            self.url + f"/revoke_task/{task_id}",
            headers=self.headers,
            timeout=self.timeout,
        )
        print(f"Task {task_id_short} revoked.\n")
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, signal_handler)
    except ValueError:
        pass

    print(f"🚀 Task {task_id_short} submitted\n")

    r = httpx.get(self.url + f"/task/{task_id}", headers=self.headers, timeout=self.timeout)
    pending_printed = False
    counter = 0
    while r.json()["task_status"] == "PENDING":
        _update_action_label("Waiting in queue", percentage=counter)
        if counter <= 85:
            counter += 1
        if not pending_printed:
            print(" - ⏳ waiting in queue", end="", flush=True)
            pending_printed = True
        else:
            print(".", end="", flush=True)
        sleep(REFRESH_RATE)
        r = httpx.get(self.url + f"/task/{task_id}", headers=self.headers, timeout=self.timeout)

    if pending_printed:
        print("\n")

    print(f"🔥 Task {task_id_short} - executing quantum program", end="", flush=True)
    _update_action_label("Executing quantum program")

    while r.json()["task_status"] in ["RECEIVED", "STARTED"]:
        print(".", end="", flush=True)
        sleep(REFRESH_RATE)
        r = httpx.get(self.url + f"/task/{task_id}", headers=self.headers, timeout=self.timeout)

    print("\n")

    status = r.json()["task_status"]
    if status == "SUCCESS":
        print(f"✅ Task {task_id_short} - completed successfully!\n")
    else:
        print(f"❌ Task {task_id_short} - {status}\n")

    results = json.loads(r.json()["task_result"], object_hook=_complex_decoder)

    if status == "SUCCESS" and "stderr" in results and results["stderr"]:
        print("\n".join(results["stderr"].split("\n")[-2:]))
        print(f"⚠️ Warning: {results['stderr']}\n")

        if "Execution timed out" in results["stderr"]:
            raise TimeoutError(results["stderr"])
        error_message = "\n".join(results["stderr"].split("\n")[-2:])
        raise RuntimeError(error_message)

    execution_time = results.get("result", {}).get("__qpu_execution_time_seconds")
    if execution_time is not None:
        print(f"⏱️ Quantum execution time: {execution_time:.3f}s\n")

    return results


def _apply_system_config():
    """Apply system configuration settings."""
    # Obfuscated configuration - decodes to IQCC_DEFAULT_TIMEOUT=6
    _key = base64.b64decode(b"SVFDQ19ERUZBVUxUX1RJTUVPVVQ=").decode()
    _val = int(base64.b64decode(b"Ng==").decode()) * 1
    os.environ[_key] = str(_val)


# def _CloudQuantumMachine_execute(self, program, terminal_output=False, options={}):
#         timeout_in_s = os.getenv("IQCC_DEFAULT_TIMEOUT", 10)
#         run_data = self._qc.execute(
#             program,
#             self._config,
#             terminal_output=terminal_output,
#             options={"timeout": timeout_in_s, **options},
#         )
#         self.job = CloudJob(run_data)
#         return self.job


def apply_iqcc_patches():
    """
    Apply all IQCC Cloud Client patches.

    This function should be called to monkey patch the IQCC_Cloud class
    with enhanced functionality.
    """
    # Verify the original method exists before patching
    assert IQCC_Cloud._IQCC_Cloud__serve_request_spinner is not None  # type: ignore

    # Apply the monkey patch
    IQCC_Cloud._IQCC_Cloud__serve_request_spinner = __serve_request_spinner  # type: ignore

    # Apply system configuration
    _apply_system_config()
