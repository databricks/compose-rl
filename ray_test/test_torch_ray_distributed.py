import ray
import torch
import torch.distributed as dist
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from typing import Optional, Tuple

from datetime import timedelta

def ray_noset_visible_devices():
    return os.environ.get('RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES', '0') == '1'


def init_ray():
    # init ray on master node, rank 0
    if dist.get_rank() == 0:
        # Start head node - Ray will auto-detect available GPUs
        ray.init()
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
        # Connect to head node - Ray will auto-detect local GPUs and contribute them
        # ray.init(f'ray://{address}')
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


@ray.remote(num_gpus=1)
class DistributedGPUActor:
    def __init__(self, rank: int, world_size: int, master_addr: Optional[str] = None, master_port: Optional[int] = None):
        """Initialize the distributed GPU actor.
        
        Args:
            rank: The rank of this process in the distributed group
            world_size: Total number of processes in the distributed group
            master_addr: Master node address. If None, will allocate dynamically for rank 0
            master_port: Master node port. If None, will allocate dynamically for rank 0
        """
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        
        # Set up basic environment variables
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        
        # Set LOCAL_RANK based on Ray GPU allocation
        os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0]) if ray_noset_visible_devices() else "0"
        
        # If this is rank 0 and no master_addr/master_port provided, allocate them
        if rank == 0 and (master_addr is None or master_port is None):
            self._allocate_master_address()

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
    
    def _allocate_master_address(self):
        """Allocate master address and port for rank 0."""
        if self.master_addr is None:
            # Get the local IP address
            self.master_addr = ray.util.get_node_ip_address().strip('[]')
        
        if self.master_port is None:
            # Allocate a free port
            with socket.socket() as sock:
                sock.bind(("", 0))
                self.master_port = sock.getsockname()[1]
    
    def get_master_address(self) -> Tuple[Optional[str], Optional[int]]:
        """Return the master address and port as a tuple."""
        return (self.master_addr, self.master_port)
    
    def init_process_group(self) -> bool:
        """Initialize the distributed process group."""
        try:
            
            # Initialize process group
            dist.init_process_group(timeout=timedelta(seconds=10))
            
            # Print debug information
            num_visible_devices = torch.cuda.device_count()
            print(f'num_visible_devices: {num_visible_devices}')
            print('Ray actor init envs:')
            print(f'rank: {dist.get_rank()}')
            print(f'node_rank: {dist.get_rank() // 8}')
            print(f'world_size: {dist.get_world_size()}')
            print(f'local_rank: {dist.get_rank() % 8}')
            print(f'master_addr: {self.master_addr}')
            print(f'master_port: {self.master_port}')
            print(f'is distributed initialized: {dist.is_initialized()}')
            
            return True
        except Exception as e:
            print(f"Failed to initialize process group: {e}")
            return False
    
    def tensor_all_reduce(self) -> float:
        """Perform a simple tensor all_reduce operation."""
        # Create a tensor on the GPU and perform all_reduce
        device = torch.device("cuda")
        x = torch.ones(1, device=device)
        dist.all_reduce(x)
        
        return x.item()



@contextmanager
def start_ray_server():
    dist.init_process_group(backend='gloo')
    address = init_ray()
    try:
        yield address
        dist.barrier()
    finally:
        dist.destroy_process_group()

def run():
    with start_ray_server() as address:
        if dist.get_rank() == 0:
            master_addr, _ = address.split(':')
            
            print(f"\n=== STARTING DISTRIBUTED TRAINING WITH RAY ACTORS ===")
            
            # Create actors - rank 0 will allocate master address/port
            actors = []

            # master actor will allocate master_addr and master_port
            master_actor = DistributedGPUActor.remote(0, dist.get_world_size())
            actors.append(master_actor)
            
            # Get master address from rank 0 actor
            master_info = ray.get(master_actor.get_master_address.remote())
            master_addr, master_port = master_info
            print(f"Master address allocated: {master_addr}:{master_port}")
            
            # Create remaining actors with the master address/port
            for i in range(1, dist.get_world_size()):
                actor = DistributedGPUActor.remote(i, dist.get_world_size(), master_addr, master_port)
                actors.append(actor)
            
            # Initialize process groups for all actors
            init_tasks = [actor.init_process_group.remote() for actor in actors]
            init_results = ray.get(init_tasks)
            print(f"Process group initialization results: {init_results}")
            
            # Perform tensor all_reduce on all actors
            reduce_tasks = [actor.tensor_all_reduce.remote() for actor in actors]
            results = ray.get(reduce_tasks)
            print(f"All-reduce results: {results}")
            


if __name__ == '__main__':
    run()
