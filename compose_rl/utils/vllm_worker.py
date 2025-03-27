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
from typing import Union

import torch

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
