import ray
import torch
import torch.distributed as dist
import os
import subprocess
import socket
import time

from datetime import timedelta

def ray_noset_visible_devices():
    return os.environ.get('RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES', '0') == '1'


def get_ranks():
    # get envs set by torchrun
    world_size = os.environ.get('WORLD_SIZE', None)
    rank = os.environ.get('RANK', None)
    local_rank = os.environ.get('LOCAL_RANK', None)
    node_rank = os.environ.get('NODE_RANK', None)
    master_addr = os.environ.get('MASTER_ADDR', None)
    master_port = os.environ.get('MASTER_PORT', None)

    return world_size, rank, local_rank, node_rank, master_addr, master_port


def init_ray(rank: str):
    dist.init_process_group(backend='gloo')

    # init ray on master node, rank 0
    if rank == '0':
        subprocess.run(['ray', 'start', '--head', '--port=6379', '--num-gpus=1'], check=True)
        ray.init(address='auto')
        # get existing ray ip and port
        ctx = ray.get_runtime_context()
        address = ctx.gcs_address
        print(f'available gpus: {ray.available_resources()}')

    else:
        address = None
    address_list = [address]
    # broadcast address to all other ranks
    dist.broadcast_object_list(address_list, src=0)
    if rank != '0':
        address = address_list[0]
        subprocess.run(['ray', 'start', f'--address={address}', '--num-gpus=1'], check=True)
        ray.init(address='auto')
    # wait until num of gpus reach world_size
    while ray.cluster_resources().get('GPU', 0) < dist.get_world_size():
        time.sleep(1)
    if rank == '0':
        print(f'available gpus: {ray.available_resources()}')
    else:
        ray.shutdown()
    dist.destroy_process_group()
    return address


@ray.remote(num_gpus=1)
def simple_gpu_task(master_addr: str, master_port: int, rank: int, node_rank: int, world_size: int):
    """A minimal GPU task that just creates a tensor and does basic operations."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["NODE_RANK"] = str(node_rank)
    local_rank = rank % 8
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    # # NOTE: Ray will automatically set the *_VISIBLE_DEVICES
    # # environment variable for each actor, unless
    # # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set, so
    # # set local rank to 0 when the flag is not applicable.
    # print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    # print(f'ray.get_gpu_ids(): {ray.get_gpu_ids()}')
    # os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0]) if ray_noset_visible_devices() else "0"

    # # number of visible devices
    num_visible_devices = torch.cuda.device_count()
    print(f'num_visible_devices: {num_visible_devices}')
    print('ray run init envs:')
    world_size, rank, local_rank, node_rank, master_addr, master_port = get_ranks()
    print(f'rank: {rank}')
    print(f'node_rank: {node_rank}')
    print(f'world_size: {world_size}')
    print(f'local_rank: {local_rank}')
    print(f'master_addr: {master_addr}')
    print(f'master_port: {master_port}')
    # init_process_group(backend='nccl', init_method=f'tcp://{master_addr}:{master_port}', world_size=int(world_size), rank=int(rank), timeout=timedelta(seconds=10))
    dist.init_process_group(timeout=timedelta(seconds=10))
    print(f'is distributed initialized: {dist.is_initialized()}')

    # Create a tensor on the GPU
    device = torch.device(f"cuda")
    x = torch.randn(1000, 1000, device=device).sum()
    dist.all_reduce(x)
    dist.destroy_process_group()
    return x.item()


if __name__ == '__main__':
    world_size, rank, local_rank, node_rank, master_addr, master_port = get_ranks()
    print('torch run init envs:')
    print(f'world_size: {world_size}')
    print(f'rank: {rank}')
    print(f'local_rank: {local_rank}')
    print(f'node_rank: {node_rank}')
    print(f'master_addr: {master_addr}')
    print(f'master_port: {master_port}')
    address = init_ray(rank)
    if rank == '0':
        try:
            master_addr, _ = address.split(':')
            # if I uncomment this, dist.init_process_group will timeout
            # with socket.socket() as sock:
            #     sock.bind(("", 0))
            #     master_port = sock.getsockname()[1]

            print(f"\n=== STARTING DISTRIBUTED TRAINING ===")
            tasks = [simple_gpu_task.remote(master_addr, master_port, rank, node_rank, world_size) for rank in range(int(world_size))]
            results = ray.get(tasks)
            print(results)
            ray.shutdown()
            subprocess.run(['ray', 'stop'], check=True)
        except Exception as e:
            print(f'Error: {e}')
        finally:
            ray.shutdown()
            subprocess.run(['ray', 'stop'], check=True)
