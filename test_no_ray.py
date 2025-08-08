import logging
import os
import subprocess
import tempfile
import traceback
from typing import Any, Dict, Optional
from composer.cli.launcher import _launch_processes, _monitor_processes, _cleanup_processes, _aggregate_process_returncode

def run_distributed_training(
    nproc: int,
    world_size: int,
    base_rank: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
    training_script: str,
    training_script_args: Any=None,
    module_mode: bool=False,
    command_mode: bool=False,
    stdout: Optional[str]=None,
    stderr: Optional[str]=None,
    verbose: bool = False,
) -> int:
    """
    Run distributed training with the given parameters.

    Args:
        nproc (int): Number of processes to launch.
        world_size (int): Total number of processes across all nodes.
        base_rank (int): Base rank of the current node.
        node_rank (int): Rank of the current node.
        master_addr (str): Address of the master node.
        master_port (int): Port of the master node.
        module_mode (bool): Whether to use module mode.
        command_mode (bool): Whether to use command mode.
        stdout (Optional[str]): Stdout file format.
        stderr (Optional[str]): Stderr file format.
        training_script (str): Training script to run.
        training_script_args (Any): Arguments for the training script.
        verbose (bool): Whether to use verbose logging.

    Returns:
        int: Aggregated return code from all processes.
    """
    if training_script_args is None:
        training_script_args = []

    MOSAICML_PLATFORM_ENV_VAR = "MOSAICML_PLATFORM"
    MOSAICML_LOG_DIR_ENV_VAR = "MOSAICML_LOG_DIR"
    MOSAICML_GPU_LOG_FILE_PREFIX_ENV_VAR = "MOSAICML_GPU_LOG_FILE_PREFIX"

    logger = logging.getLogger("distributed_training")
    logging.basicConfig()
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    processes: Dict[Any, Any] = {}

    log_tmpdir = tempfile.TemporaryDirectory()
    if stdout is None:
        stdout = f'{log_tmpdir.name}/rank{{rank}}.stdout.txt'
    if stderr is None:
        stderr = f'{log_tmpdir.name}/rank{{rank}}.stderr.txt'

    # If running on the Mosaic platform, log all gpu ranks' stderr and stdout to Mosaic platform
    if (
        os.environ.get(MOSAICML_PLATFORM_ENV_VAR, 'false').lower() == 'true'
        and str(os.environ.get(MOSAICML_LOG_DIR_ENV_VAR, 'false')).lower() != 'false'
        and os.environ.get(MOSAICML_GPU_LOG_FILE_PREFIX_ENV_VAR, 'false').lower() != 'false'
    ):
        logger.info('Logging all GPU ranks to Mosaic AI Training.')
        log_file_format = (
            f"{os.environ.get(MOSAICML_LOG_DIR_ENV_VAR)}/"
            f"{os.environ.get(MOSAICML_GPU_LOG_FILE_PREFIX_ENV_VAR)}{{local_rank}}.txt"
        )
        stdout = log_file_format
        stderr = None

    try:
        _launch_processes(
            nproc=nproc,
            world_size=world_size,
            base_rank=base_rank,
            node_rank=node_rank,
            master_addr=master_addr,
            master_port=master_port,
            module_mode=module_mode,
            command_mode=command_mode,
            stdout_file_format=stdout,
            stderr_file_format=stderr,
            training_script=training_script,
            training_script_args=training_script_args,
            processes=processes,
        )
        _monitor_processes(processes)
    except Exception:
        # Print the exception first, then kill the training processes, since killing
        # may take up to CLEANUP_TIMEOUT seconds, and the user should know immediately
        # what failed. No need to re-raise the exception, as `aggregate_process_returncode`
        # will return an appropriate error code, which will cause the script to exit.
        logger.error("Exception occurred during distributed training", exc_info=True)
        traceback.print_exc()
        print('Killing training processes')
    finally:
        _cleanup_processes(processes)
        log_tmpdir.cleanup()
        return _aggregate_process_returncode(processes)


if __name__ == "__main__":
    # test on 4 gpus!

    try:
        p1 = subprocess.Popen('CUDA_VISIBLE_DEVICES=0,1 composer -n 2 train.py', shell=True)
        p2 = subprocess.Popen('CUDA_VISIBLE_DEVICES=2,3 python rollout.py', shell=True)
        p1.wait()
        p2.wait()
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        print('Killing training processes')
    finally:
        _cleanup_processes({0: p1, 1: p2})

