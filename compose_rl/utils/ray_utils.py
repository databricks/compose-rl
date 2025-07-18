# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import socket
import subprocess
import time
from contextlib import contextmanager

import ray
import torch.distributed as dist

# Set up logger
logger = logging.getLogger(__name__)


def init_ray_with_torch_distributed(timeout_seconds: int = 30):
    """
    Initialize Ray cluster in a distributed PyTorch environment.
    
    This function sets up a Ray cluster where the master node (rank 0) starts the head node,
    and other nodes connect to it. It handles the coordination between PyTorch distributed
    training and Ray cluster initialization. It assumes torch.distributed
    is already initialized on all ranks and all the associated nodes are joining the ray cluster.
    
    The function:
    1. Starts Ray head node on rank 0
    2. Broadcasts the Ray address to all other ranks
    3. Connects worker nodes to the head node
    4. Waits for all GPUs to be available before proceeding
    
    Args:
        timeout_seconds (int): Maximum time to wait for GPUs to become available (default: 30)
        
    Returns:
        str: The Ray cluster address (GCS address) that can be used by other processes
        
    Raises:
        RuntimeError: If the required number of GPUs are not available within the timeout period
        subprocess.CalledProcessError: If Ray start/stop commands fail
    """
    # init ray on master node, rank 0
    if dist.get_rank() == 0:
        # Start Ray Server on master node
        subprocess.run(['ray', 'start', '--head'], check=True)
        # connect to the ray cluster
        ray.init('auto')
        # get existing ray ip and port
        ctx = ray.get_runtime_context()
        address = ctx.gcs_address
    else:
        address = ''
    address_list = [address]
    # broadcast address to all other ranks
    dist.broadcast_object_list(address_list, src=0)
    if dist.get_rank() != 0 and os.environ.get('LOCAL_RANK', None) == '0':
        address = address_list[0]
        logger.info(f'Rank {dist.get_rank()}: connecting to address {address}')
        subprocess.run(['ray', 'start', f'--address={address}'], check=True)
    dist.barrier()
    if dist.get_rank() == 0:
        # wait until num of gpus reach world_size
        num_gpus = int(ray.cluster_resources()['GPU'])
        start_time = time.time()
        while num_gpus < dist.get_world_size():
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise RuntimeError(
                    f'Timeout after {timeout_seconds}s: Failed to start {dist.get_world_size()} GPUs. Only {num_gpus} GPUs available.'
                )
            
            logger.info(
                f'Waiting for {dist.get_world_size() - num_gpus} GPUs to be available (elapsed: {elapsed_time:.1f}s, timeout: {timeout_seconds}s)'
            )
            num_gpus = int(ray.cluster_resources()['GPU'])
            # sleep ad-hoc 5s to avoid busy waiting
            time.sleep(5)
        
        logger.info(f'Total available GPUs: {ray.available_resources()}')
    return address


@contextmanager
def start_ray_server():
    """
    Context manager for starting and stopping a Ray server in a torch distributed environment.
    
    This context manager handles the complete lifecycle of a Ray cluster:
    - Initializes PyTorch distributed process group if not already initialized
    - Starts the Ray cluster using init_ray_with_torch_distributed()
    - Provides the Ray address to the context
    - Ensures proper cleanup of Ray and distributed resources
    
    The context manager ensures that Ray is properly shut down and the distributed
    process group is destroyed even if an exception occurs.
    
    Yields:
        str: The Ray cluster address (GCS address)
        
    Example:
        >>> with start_ray_server() as ray_address:
        ...     # Use Ray cluster here
        ...     ray.get(some_remote_function.remote())
        ... # Ray is automatically shut down here
    """
    init_torch_dist = False
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo')
        init_torch_dist = True
    address = init_ray_with_torch_distributed()
    try:
        yield address
        # NOTE we have to keep all the MCT orchestrator started processes alive with this barrier
        # until the ray cluster is stopped, otherwise the MCT orchestrator will reclaim the resources
        # once the processes on a node exit
        dist.barrier()
    finally:
        if dist.get_rank() == 0:
            ray.shutdown()
            subprocess.run(['ray', 'stop'], check=True)
        dist.barrier()
        if init_torch_dist:
            dist.destroy_process_group()


def get_node_ip():
    """
    Get the IP address of the current Ray node.
    
    Returns:
        str: The IP address of the current node, with any brackets removed
        
    Example:
        >>> ip = get_node_ip()
        >>> print(f"Current node IP: {ip}")
        Current node IP: 192.168.1.100
    """
    return ray.util.get_node_ip_address().strip('[]')


def get_free_port():
    """
    Get a free port number that can be used for binding a socket.
    
    This function creates a temporary socket, binds it to port 0 (which tells the OS
    to assign any available port), and returns the assigned port number. The socket
    is automatically closed when the context manager exits.

    NOTE there is a low risk that the port is recollected by the system after the context manager exits
    and before current process use it
    
    Returns:
        int: A free port number that can be used for network services
        
    Example:
        >>> port = get_free_port()
        >>> print(f"Available port: {port}")
        Available port: 54321
    """
    with socket.socket() as sock:
        sock.bind(('', 0))
        return sock.getsockname()[1]


def is_cuda_visible_devices_set():
    """
    Check if CUDA_VISIBLE_DEVICES environment variable is being set by Ray.
    
    Ray can automatically set the CUDA_VISIBLE_DEVICES environment variable to
    control which GPUs are visible to processes. This function checks whether
    this behavior is enabled or disabled.
    
    Returns:
        bool: True if Ray is setting CUDA_VISIBLE_DEVICES, False otherwise
    """
    return os.environ.get(
        'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES', '0'
    ) == '0'
