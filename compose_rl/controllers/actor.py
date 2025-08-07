# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import timedelta
from typing import Any, Callable, Optional

import ray
import torch.distributed as dist

from compose_rl.algorithms.online.generation_utils import init_process_group
from compose_rl.utils.ray_utils import (
    get_free_port,
    get_node_ip,
    is_cuda_visible_devices_set_by_ray,
)


class BaseDistributedGPUActor:

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
    ):
        """Initialize the distributed GPU actor for RAY.

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
        os.environ['WORLD_SIZE'] = str(world_size)
        # FIXME: handle LOCAL_WORLD_SIZE for multiple nodes
        os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)

        # Set LOCAL_RANK based on Ray GPU allocation
        # ray.get_gpu_ids() is empty if ray is not used. 
        print('ray.get_gpu_ids()', ray.get_gpu_ids(), 'setting local rank to', rank)

        os.environ['LOCAL_RANK'] = str(rank) 

        # If this is rank 0 and no master_addr/master_port provided, allocate them
        if rank == 0 and (master_addr is None or master_port is None):
            self._allocate_master_address()

        os.environ['MASTER_ADDR'] = self.master_addr  # type: ignore
        os.environ['MASTER_PORT'] = str(self.master_port)  # type: ignore

        self.model = None
        self.model_update_group = None

    def get_ray_gpu_ids(self) -> list[int]:
        """Get the visible devices for the actor."""
        return ray.get_gpu_ids()

    def set_cuda_visible_devices(self, visible_devices: list[int]):
        """Set the visible devices for the actor."""
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, visible_devices))

    def _allocate_master_address(self):
        """Allocate master address and port for rank 0."""
        if self.master_addr is None:
            # Get the local IP address
            self.master_addr = get_node_ip()

        if self.master_port is None:
            # Allocate a free port
            self.master_port = get_free_port()

    def get_master_address(self) -> tuple[Optional[str], Optional[int]]:
        """Return the master address and port as a tuple."""
        return (self.master_addr, self.master_port)

    def get_free_port(self):
        return get_free_port()

    def init_train_process_group(self):
        """Initialize the distributed process group."""
        # Initialize process group
        dist.init_process_group(timeout=timedelta(seconds=30))

    def add_process_group(
        self,
        backend: str,
        master_addr: str,
        master_port: int,
        world_size: int,
        rank: int,
        group_name: str,
    ):
        """Initialize the process group on trainer rank 0 and vllm engines."""
        # NOTE vLLM seems to have a safer implementation of init_process_group:
        # https://github.com/vllm-project/vllm/blob/v0.9.1/examples/offline_inference/rlhf.py#L105
        # we should look into using that instead
        self.model_update_group = init_process_group(
            backend=backend,
            init_method=f'tcp://{master_addr}:{master_port}',
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
    
    def execute(self, func: Callable[['BaseDistributedGPUActor'], Any]):
        """Dispatch a serializable function to this actor."""
        return func(self)


class SPMDActorGroup:
    """Group managers of SPMD actors."""

    def __init__(self, num_train_actors: int, actor_class: type[BaseDistributedGPUActor], num_gpus_per_actor: int = 1):
        self.num_train_actors = num_train_actors
        self._train_actors: list[BaseDistributedGPUActor] = []
        """Create and initialize all training actors."""
        print(f'\n=== STARTING DISTRIBUTED TRAINING WITH RAY ACTORS ===')

        remote_actor_class = ray.remote(num_gpus=num_gpus_per_actor)(actor_class)
        # Create master actor first
        self._master_actor = remote_actor_class.remote(
            0,
            self.num_train_actors,
        )
        self._train_actors.append(self._master_actor)

        # Get master address from rank 0 actor
        master_addr, master_port = ray.get(
            self._master_actor.get_master_address.remote(),  # type: ignore
        )
        print(f'Master address allocated: {master_addr}:{master_port}')

        # Create remaining actors with the master address/port
        for i in range(1, self.num_train_actors):
            actor = remote_actor_class.remote(
                i,
                self.num_train_actors,
                master_addr,  # type: ignore
                master_port,
            )
            self._train_actors.append(actor)

        # Set the visible devices for all actors
        gpu_ids = []
        for actor in self._train_actors:
            gpu_ids.extend(ray.get(actor.get_ray_gpu_ids.remote()))
            
        for actor in self._train_actors:
            ray.get(actor.set_cuda_visible_devices.remote(gpu_ids))
            print(f'Set visible devices for actor: {gpu_ids}')

    @property
    def train_actors(self):
        return self._train_actors

    @property
    def master_actor(self):
        return self._master_actor

    @property
    def collective_methods(self):
        """Property that provides easy access to method references.
        """
        return _ActorMethodProxy(self)


class _ActorMethodProxy:
    """Proxy class that provides easy access to actor methods.
    """
    
    def __init__(self, actor_group: SPMDActorGroup):
        self._actor_group = actor_group
    
    def __getattr__(self, name: str):
        """Get a method reference that will be called on all actors."""
        if not hasattr(self._actor_group.master_actor, name):
            raise AttributeError(
                f"Method '{name}' not found on actor class: {self._actor_group.master_actor.__class__}"
            )
        
        # Return a callable that will execute the method on all actors
        def method_wrapper(*args: Any, **kwargs: Any):
            # Since all actors are the same class, we can get the same method from each actor
            # and call it remotely. No validation needed since we validated above.
            refs = [getattr(actor, name).remote(*args, **kwargs) for actor in self._actor_group.train_actors]
            return ray.get(refs)
        
        return method_wrapper
