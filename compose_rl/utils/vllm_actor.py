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
import os
from typing import Any, Union

import ray
import torch

try:
    # In some cases e.g. CI/CD, vLLM is not installed on cpu
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
except:
    pass

log = logging.getLogger(__name__)


@ray.remote
class LLMRayActor:

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        import vllm

        noset_visible_devices = kwargs.pop('noset_visible_devices')

        if kwargs.get('distributed_executor_backend') == 'ray':
            # a hack to make the script work.
            # stop ray from manipulating *_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            os.environ.pop('ROCR_VISIBLE_DEVICES', None)
        elif noset_visible_devices:
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
            os.environ['CUDA_VISIBLE_DEVICES'] = str(ray.get_gpu_ids()[0])

        num_gpus = kwargs.pop('num_gpus')
        bundle_indices = kwargs.pop('bundle_indices', None)
        if bundle_indices is not None:
            os.environ['VLLM_RAY_PER_WORKER_GPUS'] = str(num_gpus)
            os.environ['VLLM_RAY_BUNDLE_INDICES'] = ','.join(
                map(str, bundle_indices),
            )
            log.info(f'creating LLM with bundle_indices={bundle_indices}')

        self.llm = vllm.LLM(*args, **kwargs)

    def generate(
        self,
        raw_prompts: Union[str, list[str], list[list[int]]],
        *args: Any,
        **kwargs: Any,
    ):
        log.info(f'Generate kwargs are: {kwargs}')
        sampling_params = None
        if 'sampling_params' in kwargs:
            sampling_params = SamplingParams(**kwargs.pop('sampling_params'))
            log.info(f'sampling_params is: {sampling_params}')

        if isinstance(raw_prompts, list):
            tokens_request = [TokensPrompt(token_ids=ids) for ids in raw_prompts]
            return self.llm.generate(
                *tokens_request,
                sampling_params=sampling_params,
            )
        else:
            # raw_prompts is a string
            return self.llm.generate(
                raw_prompts,
                sampling_params=sampling_params,
            )

    def chat(self, *args: Any, **kwargs: Any):
        sampling_params = None
        if 'sampling_params' in kwargs:
            sampling_params = SamplingParams(**kwargs.pop('sampling_params'))
            log.info(f'sampling_params is: {sampling_params}')

        return self.llm.chat(
            *args,
            **kwargs,
            sampling_params=sampling_params,
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
        return self.llm.collective_rpc(
            'init_process_group',
            args=(
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
            ),
        )

    def update_weight(
        self,
        name: str,
        dtype: torch.dtype,
        shape: Union[tuple[int, ...], list[int]],
        empty_cache: bool = False,
    ):
        return self.llm.collective_rpc(
            'update_weight',
            args=(name, dtype, shape, empty_cache),
        )
