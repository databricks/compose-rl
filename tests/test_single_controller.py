# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import pathlib
from datetime import timedelta
from typing import Optional

import pytest
import ray
import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from compose_rl.algorithms.online.generation_utils import (
    create_vllm_engines,
    init_process_group,
)
from compose_rl.utils.ray_utils import (
    get_free_port,
    get_node_ip,
    is_cuda_visible_devices_set,
    start_ray_server,
)
from tests.common import world_size

# Set up logging
logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class DistributedGPUActor:

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
    ):
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
        os.environ['WORLD_SIZE'] = str(world_size)
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
        logger.info(f'is distributed initialized: {dist.is_initialized()}')
        # Print debug information
        num_visible_devices = torch.cuda.device_count()
        logger.info(f'num_visible_devices: {num_visible_devices}')
        logger.info('Ray actor init envs:')
        logger.info(f'rank: {dist.get_rank()}')
        logger.info(f'node_rank: {dist.get_rank() // 8}')
        logger.info(f'world_size: {dist.get_world_size()}')
        logger.info(f'local_rank: {dist.get_rank() % 8}')
        logger.info(f'master_addr: {self.master_addr}')
        logger.info(f'master_port: {self.master_port}')

    def init_model(self, model_name: str):
        """Initialize the model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
        )
        self.model.to('cuda')

    def sync_weights(self, vllm_engines: list):
        """Sync the weights of the model to the vLLM engines."""
        # need to call vllm_engine.remote in Trainer instead of top level controller
        # to ensure name mapping is correct
        for name, p in self.model.named_parameters():
            refs = [
                engine.update_weight.remote(  # type: ignore
                    name,
                    p.dtype,
                    p.shape,
                    empty_cache=False,
                ) for engine in vllm_engines
            ]
            # broadcast by default is blocking, we have to kick it off
            # after the remote calls are initialized above
            dist.broadcast(p, src=0, group=self.model_update_group)
            ray.get(refs)

    def tensor_all_reduce(self) -> float:
        """Perform a simple tensor all_reduce operation."""
        # Create a tensor on the GPU and perform all_reduce
        device = torch.device('cuda')
        x = torch.ones(1, device=device, dtype=torch.int32)
        dist.all_reduce(x)

        return x.item()

    def init_vllm_process_group(
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


@pytest.mark.gpu
@world_size(4)
def test_distributed_ray_actors(
    world_size: int,
    tiny_gpt2_model: PreTrainedModel,
    tiny_gpt2_tokenizer: PreTrainedTokenizerBase,
    tmp_path: pathlib.Path,
):
    """Test basic single contrller with Ray."""
    # Set vLLM attention backend to FLASH_ATTN otherwise FlashInfer backend takes too long to jit compile
    os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'

    prompts = [
        'what is RAY?',
        'what is vLLM?',
    ]

    # Save the model and tokenizer to a temporary directory
    local_save_path = str(tmp_path / 'tiny_gpt2_model')
    tiny_gpt2_model.save_pretrained(local_save_path)
    tiny_gpt2_tokenizer.save_pretrained(local_save_path)

    with start_ray_server() as address:
        if dist.get_rank() == 0:
            # rank 0 is the ray client
            master_addr, _ = address.split(':')

            logger.info(
                f'\n=== STARTING DISTRIBUTED TRAINING WITH RAY ACTORS ===',
            )
            num_train_actors = world_size // 2
            # Create actors - rank 0 will allocate master address/port
            train_actors = []
            # master actor will allocate master_addr and master_port
            master_actor = DistributedGPUActor.remote(0, num_train_actors)
            train_actors.append(master_actor)

            # Get master address from rank 0 actor
            master_info = ray.get(
                master_actor.get_master_address.remote(),  # type: ignore
            )
            master_addr, master_port = master_info
            logger.info(
                f'Master address allocated: {master_addr}:{master_port}',
            )

            # Create remaining actors with the master address/port
            for i in range(1, num_train_actors):
                actor = DistributedGPUActor.remote(
                    i,
                    num_train_actors,
                    master_addr,  # type: ignore
                    master_port,
                )
                train_actors.append(actor)

            # Initialize process groups for all actors
            init_tasks = [
                actor.init_train_process_group.remote()  # type: ignore
                for actor in train_actors
            ]
            ray.get(init_tasks)

            # Perform tensor all_reduce on all actors
            reduce_tasks = [
                actor.tensor_all_reduce.remote()  # type: ignore
                for actor in train_actors
            ]
            results = ray.get(reduce_tasks)
            assert results == [num_train_actors] * num_train_actors

            vllm_tensor_parallel_size = world_size - num_train_actors
            num_vllm_engines = (
                world_size - num_train_actors
            ) // vllm_tensor_parallel_size
            logger.info(f'num_vllm_engines: {num_vllm_engines}')
            vllm_engines = create_vllm_engines(
                num_engines=num_vllm_engines,
                tensor_parallel_size=vllm_tensor_parallel_size,
                enforce_eager=True,
                pretrain=local_save_path,
                revision=None,
                seed=1,
                enable_prefix_caching=False,
                max_model_len=512,
                device_bundle={
                    'GPU': 1,
                    'CPU': 1,
                    'worker_node': 0,
                },
            )

            new_port = ray.get(
                master_actor.get_free_port.remote(),  # type: ignore
            )
            logger.info(f'new_port to init vllm process group: {new_port}')
            # init_process_group of the engine calls vLLM's collective_rpc
            # which calls the init_process_group on every tp rank of the engine
            # so the world_size is the total number of gpus used by all the engines + 1 trainer rank
            # and the rank is head rank of each engine in the world_size and collective_rpc adds proper rank offset
            # to its tp ranks
            refs = [
                engine.init_process_group.remote(  # type: ignore
                    master_addr,
                    new_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size - num_train_actors + 1,
                    'weight-update',
                    backend='nccl',
                ) for i, engine in enumerate(vllm_engines)
            ]
            refs.append(
                master_actor.init_vllm_process_group.remote(  # type: ignore
                    backend='nccl',
                    master_addr=master_addr,
                    master_port=new_port,
                    world_size=world_size - num_train_actors + 1,
                    rank=0,
                    group_name='weight-update',
                ),
            )
            # get refs must be called after all the remote calls are initialized
            # otherwise it hangs
            ray.get(refs)
            logger.info('init vllm process group done')

            refs = [
                actor.init_model.remote(local_save_path)  # type: ignore
                for actor in train_actors
            ]
            ray.get(refs)
            logger.info('Trainer init model done')

            ray.get(
                master_actor.sync_weights.remote(vllm_engines),  # type: ignore
            )
            logger.info('sync weights done')

            ref = vllm_engines[0].generate.remote(prompts)  # type: ignore
            gen_results = ray.get(ref)
            for output in gen_results:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                logger.info(
                    f'Prompt: {prompt!r}, Generated text: {generated_text!r}',
                )
