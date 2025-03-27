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

# Modified version from https://github.com/OpenRLHF/OpenRLHF and The AllenAI Team.

import logging
from typing import Any, Union

import ray
import torch

try:
    # In some cases e.g. CI/CD, vLLM is not installed on cpu
    from vllm import SamplingParams
except:
    pass

log = logging.getLogger(__name__)

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


@ray.remote
class LLMRayActor:

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        import vllm

        self.__version__ = vllm.__version__
        assert self.__version__ >= '0.4.1', 'Compose RL only supports vLLM >= 0.4.1'

        self.use_gpu_executor = kwargs['tensor_parallel_size'] == 1
        log.info(f'kwargs are: {kwargs}')

        # See https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py
        if self.use_gpu_executor:
            vllm.worker.worker.Worker = WorkerWrap  # type: ignore
        else:

            # RayGPUExecutor
            # See the patch https://github.com/vllm-project/vllm/commit/479d69fad0538f04cb22bf13e76ff91cfeb8a4e5
            kwargs['worker_use_ray'] = True

            if vllm.__version__ > '0.4.1':
                RayWorkerWrapperPath = vllm.executor.ray_utils  # type: ignore
            else:
                RayWorkerWrapperPath = vllm.engine.ray_utils  # type: ignore

            if vllm.__version__ > '0.6.4.post1':
                # https://github.com/vllm-project/vllm/pull/10555
                kwargs['worker_cls'] = 'compose_rl.utils.vllm_actor.WorkerWrap'
            else:
                RayWorkerWrapperPath = vllm.engine.ray_utils  # type: ignore

                class RayWorkerWrapper(RayWorkerWrapperPath.RayWorkerWrapper):

                    def __init__(self, *args: Any, **kwargs: Any) -> None:
                        kwargs['worker_module_name'
                              ] = 'compose_rl.utils.vllm_actor'
                        kwargs['worker_class_name'
                              ] = 'compose_rl.utils.vllm_actor.WorkerWrap'
                        super().__init__(*args, **kwargs)

                RayWorkerWrapperPath.RayWorkerWrapper = RayWorkerWrapper

        self.llm = vllm.LLM(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any):
        log.info(f'Generate kwargs are: {kwargs}')
        sampling_params = None
        if 'sampling_params' in kwargs:
            sampling_params = SamplingParams(**kwargs.pop('sampling_params'))
            log.info(f'sampling_params is: {sampling_params}')

        return self.llm.generate(
            sampling_params=sampling_params,
            *args,
            **kwargs,
        )

    def init_process_group(
        self,
        master_address: str,
        master_port: str,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
    ):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.init_process_group( # type: ignore
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers( # type: ignore
                'init_process_group',
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
            )

    def update_weight(
        self,
        name: str,
        dtype: torch.dtype,
        shape: Union[tuple[int, ...], list[int]],
        empty_cache: bool = False,
    ):
        self.stop_remote_worker_execution_loop()

        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.update_weight( # type: ignore
                name,
                dtype,
                shape,
                empty_cache,
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers( # type: ignore
                'update_weight',
                name,
                dtype,
                shape,
                empty_cache,
            )

    def stop_remote_worker_execution_loop(self):
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if self.__version__ > '0.4.2':
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop(
            )
