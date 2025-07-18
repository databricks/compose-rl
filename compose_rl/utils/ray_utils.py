import os
import socket
import subprocess
import time
from contextlib import contextmanager

import ray
import torch.distributed as dist


def init_ray():
    # init ray on master node, rank 0
    if dist.get_rank() == 0:
        # Start head node
        subprocess.run(['ray', 'start', '--head'], check=True)
        ray.init('auto')
        # get existing ray ip and port 
        ctx = ray.get_runtime_context()
        address = ctx.gcs_address
        print(f'available gpus: {ray.available_resources()}')
    else:
        address = ''
    address_list = [address]
    # broadcast address to all other ranks
    dist.broadcast_object_list(address_list, src=0)
    if dist.get_rank() != 0 and os.environ.get('LOCAL_RANK', None) == '0':
        address = address_list[0]
        print(f'rank: {dist.get_rank()} connecting to address: {address}')
        subprocess.run(['ray', 'start', f'--address={address}'], check=True)
    dist.barrier()
    if dist.get_rank() == 0:
        # wait until num of gpus reach world_size
        num_gpus = int(ray.cluster_resources()['GPU'])
        counter = 0
        while num_gpus < dist.get_world_size():
            print(f'waiting for {dist.get_world_size() - num_gpus} gpus to be available')
            num_gpus = int(ray.cluster_resources()['GPU'])
            time.sleep(5)
            counter += 1
            if counter > 4:
                raise RuntimeError(f'Failed to start {dist.get_world_size()} gpus')
        print(f'Total available gpus: {ray.available_resources()}')
    return address


@contextmanager
def start_ray_server():
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo')
    address = init_ray()
    try:
        yield address
        dist.barrier()
    finally:
        if dist.get_rank() == 0:
            ray.shutdown()
            subprocess.run(['ray', 'stop'], check=True)
        dist.barrier()
        dist.destroy_process_group()


def get_node_ip():
    return ray.util.get_node_ip_address().strip('[]')


def get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def is_cuda_visible_devices_set():
    return os.environ.get('RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES', '0') == '0'

