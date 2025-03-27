# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 The AllenAI Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is a modifed version fom  from https://github.com/OpenRLHF/OpenRLHF
# and The AllenAI Team.

import logging
import time
from datetime import timedelta
from typing import Optional, Union

import ray
import torch
import torch.distributed
import torch.nn as nn
from composer.utils import dist
from ray.exceptions import GetTimeoutError
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed.distributed_c10d import \
    _new_process_group_helper  # type: ignore
from torch.distributed.distributed_c10d import _world  # type: ignore
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    default_pg_timeout,
    rendezvous,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from compose_rl.utils.vllm_actor import LLMRayActor

log = logging.getLogger(__name__)


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: Union[str, Backend],
    init_method: Optional[str],
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: Optional[str] = None,
) -> torch.distributed.ProcessGroup:
    assert (store is None) or (
        init_method is None
    ), 'Cannot specify both init_method and store.'

    if store is not None:
        assert world_size > 0, 'world_size must be positive if using store'
        assert rank >= 0, 'rank must be non-negative if using store'
    elif init_method is None:
        init_method = 'env://'

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend('undefined')

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(
            init_method, # type: ignore
            rank,
            world_size,
            timeout=timeout,
        )
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)  # type: ignore

    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


try:
    # In some cases e.g. CI/CD, vLLM is not installed on cpu
    from vllm.worker.worker import Worker

    class WorkerWrap(Worker):  # type: ignore

        def init_process_group(
            self,
            master_address: str,
            master_port: str,
            rank_offset: int,
            world_size: int,
            group_name: str,
            backend: str,
        ):
            """Init torch process group for model weights update."""
            assert torch.distributed.is_initialized(
            ), 'default torch process group must be initialized'
            assert group_name != '', 'group name must not be empty'

            rank = torch.distributed.get_rank() + rank_offset
            self._model_update_group = init_process_group( # type: ignore
                backend=backend,
                init_method=f'tcp://{master_address}:{master_port}',
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            self.rank = rank
            log.info(f'init process group for: {torch.distributed.get_rank()}')
            log.info(
                f'init_process_group: master_address={master_address}, master_port={master_port}, ',
                f'rank={rank}, world_size={world_size}, group_name={group_name}',
            )

        def update_weight(
            self,
            name: str,
            dtype: torch.dtype,
            shape: Union[tuple[int, ...], list[int], torch.Size],
            empty_cache: bool = False,
        ):
            """Broadcast weights to vllm workers from source rank 0 actor model.

            Args:
                name (str): Name of the weight to be updated
                dtype (torch.dtype): Data type of the weight
                shape (Union[Tuple[int, ...], List[int], torch.Size]): Shape of the weight
                empty_cache (bool): Whether to empty cache after updating weights
            """
            weight = torch.empty(shape, dtype=dtype, device='cuda')
            torch.distributed.broadcast(
                weight,
                0,
                group=self._model_update_group,
            )

            # Because FSDP keeps master weights in FP32 and vLLM typically doesn't do this
            # We will need to cast the weight type to the model_config type
            if weight.dtype != self.model_config.dtype:
                weight = weight.to(self.model_config.dtype)

            self.model_runner.model.load_weights(
                weights=[(name, weight)],
            )  # type: ignore

            del weight

            if empty_cache:
                torch.cuda.empty_cache()

except:
    log.error('vLLM is not installed. WorkerWrap is not available.')
    pass


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    enforce_eager: bool,
    pretrain: str,
    revision: Optional[str],
    seed: int,
    enable_prefix_caching: bool,
    max_model_len: int,
):
    """Creates vllm engines.

    Args:
        num_engines (int): Number of engines to create
        tensor_parallel_size (int): Size of the tensor parallelism
        enforce_eager (bool): Whether to enforce
        pretrain (str): Pretrained model name
        revision (str): Revision of the model
        seed (int): Seed for random number generation
        enable_prefix_caching (bool): Whether to enable prefix caching
        max_model_len (int): Maximum model length
    """
    vllm_engines = []
    for i in range(num_engines):
        # When tensor_parallel_size=1, vLLM init model in LLMEngine directly, assign 1 GPU for it.
        num_gpus = int(tensor_parallel_size == 1)
        scheduling_strategy = None

        if tensor_parallel_size > 1:
            # This code will only allocate resources on worker nodes
            bundles = [{
                'GPU': 1,
                'CPU': 1,
                'worker_node': 1,
            }] * tensor_parallel_size
            pg = placement_group(bundles)  # type: ignore

            try:
                ray.get(pg.ready(), timeout=300)
            except GetTimeoutError as e:
                log.error('Placement group failed')
                log.error(f'error is: {e}')
                raise e

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0,
            )
        log.info(f'vllm: {num_gpus=}, {num_engines=}')
        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=1,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                pretrain, # type: ignore
                revision=revision,
                tokenizer_revision=revision,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,
                enforce_eager=enforce_eager,
                dtype='bfloat16',
                seed=seed + i,
                enable_prefix_caching=enable_prefix_caching,
                max_model_len=max_model_len,
            ),
        )

    return vllm_engines


def build_param_fullnames(top_module: nn.Module) -> dict:
    """Builds a mapping of parameter objects to their fully-qualified names.

    Traverses the entire model from the top level and map each parameter
    object to its fully-qualified name (e.g.,
    "lm_backbone.layer1.mlp.down_proj.weight").

    Args:
        top_module (nn.Module): The top-level module to traverse.
    """
    param2fullname = {}

    def _dfs(current_module: nn.Module, prefix: str = ''):
        # Get local parameters (without recursing into children).
        for local_name, param in current_module.named_parameters(recurse=False):
            full_name = f'{prefix}.{local_name}' if prefix else local_name
            param2fullname[param] = full_name

        # Recurse on child modules.
        for child_name, child_module in current_module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            _dfs(child_module, prefix=child_prefix)

    _dfs(top_module)
    return param2fullname


def simplify_param_path(path: str) -> str:
    """Simplifies the parameter path by removing unnecessary parts.

    Args:
        path (str): The original parameter path.
    """
    # Parts we want to remove
    remove_parts = [
        '_fsdp_wrapped_module',
        '_checkpoint_wrapped_module',
        'lm_backbone',
        'model',
    ]

    # Split the path into parts
    parts = path.split('.')

    # Keep only parts that don't contain any of the remove_parts
    clean_parts = []
    if 'lm_head' not in path:
        clean_parts = ['model']
    for part in parts:
        if not any(remove in part for remove in remove_parts):
            clean_parts.append(part)

    return '.'.join(clean_parts)


def is_fsdp_leaf(module: nn.Module) -> bool:
    """Check if the module is a leaf in the FSDP hierarchy.

    Args:
        module (nn.Module): The torch module to check
    """
    if not isinstance(module, FSDP):
        return False
    for subm in module.modules():
        if subm is not module and isinstance(subm, FSDP):
            return False
    return True


def broadcast_to_vllm(
    model: nn.Module,
    vllm_engines: list,
    model_update_group: Optional[torch.distributed.ProcessGroup],
    batch: dict[str, torch.Tensor],
):
    """Broadcast model weights to all vllm engines.

    Args:
        model (nn.Module): The model to broadcast
        vllm_engines (list): List of vllm engines
        model_update_group (torch.distributed.ProcessGroup): The process group for model updates
        batch (dict[str, torch.Tensor]): The batch to use for the forward pass
    """
    # avoid OOM
    torch.cuda.empty_cache()
    count, num_params = 0, len(
        list(model.model.lm_backbone.named_parameters()),  # type: ignore
    )
    refss = []
    # This is needed to get the correct model device
    cur_device = batch['prompt'].device

    # These apply to llama modules, it might change for other modules
    valid_non_leaf_module_names = [
        'model.embed_tokens.weight',
        'lm_head.weight',
        'model.norm.weight',
    ]
    seen_fsdp_modules = set()
    seen_updated_parsed_names = set()
    count = 0
    param_2_full_name = build_param_fullnames(model)

    with torch.no_grad():
        # Adding a dummy forwards call.
        # We need this otherwise FSDP throws an error during a standard forward pass.
        dummy_batch = {
            'obs':
                torch.tensor([[0]], dtype=torch.long, device=cur_device),
            'right_padded_attn_mask':
                torch.tensor([[1]], dtype=torch.bool, device=cur_device),
            'actions':
                torch.tensor([[0]], dtype=torch.long, device=cur_device),
            'prompt_len':
                torch.tensor([1], device=cur_device),
            'max_gen_len':
                torch.tensor([1], device=cur_device),
            'action_mask':
                torch.tensor([[0]], dtype=torch.long, device=cur_device),
        }
        model(dummy_batch)
    start_time = time.time()
    update_time = 0

    for module_name, module in model.named_modules():
        if isinstance(module, FSDP):
            # This should be the root module, and it's only initialized after we call forwards
            if module_name == 'model':
                # print ("this should be the root module skipping", module)
                continue

            # Only update if we haven't updated this module before
            if module not in seen_fsdp_modules:
                seen_fsdp_modules.add(module)

                # Materializes parameters for this specific FSDP module
                with FSDP.summon_full_params(
                    module,
                    writeback=False,
                    rank0_only=True,
                    recurse=False,
                ):
                    for _, param in module.named_parameters(recurse=True):
                        if dist.get_global_rank() == 0:
                            full_name = param_2_full_name[param]
                            parsed_name = simplify_param_path(full_name)

                            if 'critic_head' in parsed_name:
                                log.info('Critic head found, skipping sending')
                                continue

                            update = False

                            # If we are at a leaf of a FSDP module we should always update it
                            if is_fsdp_leaf(module):
                                update = True
                            elif parsed_name in valid_non_leaf_module_names and 'lm_backbone' in full_name:
                                update = True

                            # We've already updated this module before,
                            if parsed_name in seen_updated_parsed_names:
                                continue

                            # Usually if we have to skip a module, it's because we cannot
                            if update:
                                start_update_time = time.time()
                                seen_updated_parsed_names.add(parsed_name)

                                count += 1
                                shape = param.shape
                                refs = [
                                    engine.update_weight.remote(
                                        parsed_name,
                                        dtype=param.dtype,
                                        shape=shape,
                                        empty_cache=(count == num_params),
                                    ) for engine in vllm_engines
                                ]
                                refss.extend(refs)
                                torch.distributed.broadcast(
                                    param.data,
                                    0,
                                    group=model_update_group,
                                )
                                update_time += time.time() - start_update_time

    log.info(f'for loop took: {time.time() - start_time}')
    start_time = time.time()
    ray.get(refss)
    log.info(f'ray refs took: {time.time() - start_time}')
    log.info(f'update time is: {update_time}')
    log.info(f'number of parameters updated is: {count}')
