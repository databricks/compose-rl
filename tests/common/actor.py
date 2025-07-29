# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import timedelta
from typing import Optional

import ray
import torch.distributed as dist

from compose_rl.algorithms.online.generation_utils import init_process_group
from compose_rl.utils.ray_utils import (
    get_free_port,
    get_node_ip,
    is_cuda_visible_devices_set,
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
        os.environ['LOCAL_RANK'] = '0' if is_cuda_visible_devices_set(
        ) else str(ray.get_gpu_ids()[0])

        # If this is rank 0 and no master_addr/master_port provided, allocate them
        if rank == 0 and (master_addr is None or master_port is None):
            self._allocate_master_address()

        os.environ['MASTER_ADDR'] = self.master_addr  # type: ignore
        os.environ['MASTER_PORT'] = str(self.master_port)  # type: ignore

        self.model = None
        self.model_update_group = None

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
