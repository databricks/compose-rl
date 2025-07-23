# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from functools import partial
from typing import Any, Optional

import pytest
import ray
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
)

from composer import Trainer
from composer.optim import DecoupledAdamW
from composer.utils import dist as composer_dist
from llmfoundry.models import ComposerHFCausalLM

from compose_rl.algorithms.online import (
    ComposerHFPolicyLM,
    SingleControllerOnPolicyCallback,
)
from compose_rl.algorithms.online.generation_utils import create_vllm_engines
from compose_rl.data import prompt_dataset_collate_fn
from compose_rl.utils.ray_utils import start_ray_server
# add a PYTHONPATH tot he tests
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.common import (
    BaseDistributedGPUActor,
    VerifiablePromptDataset,
    world_size,
)

# Set up logging
logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class DistributedGPUActor(BaseDistributedGPUActor):
    """Distributed GPU actor for testing."""

    def __init__(self,
        rank: int,
        world_size: int,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None):
        super().__init__(rank, world_size, master_addr, master_port)
        self.model = None
        self.model_update_group = None
        self.pretrain_model_name = None
        self.ref_path = None
        self._dataloader = None
        self._tokenizer = None
        self.ppo_callback = None
        self.ppo_trainer: Trainer = None

    def build_dataloader(self):
        max_seq_len = 32
        prompt_len = 10

        dataset = VerifiablePromptDataset(prompt_len=prompt_len)
        dataloader = DataLoader(
            dataset,
            collate_fn=partial(
                prompt_dataset_collate_fn,
                self.tokenizer,
                max_seq_len,
            ),
            sampler=composer_dist.get_sampler(dataset),
            batch_size=4,
        )
        # We need to mock this method, since our dataset isn't a StreamingDataset
        dataloader.state_dict = lambda: {}
        dataloader.load_state_dict = lambda x: None
        return dataloader

    @property
    def dataloader(self):
        if self._dataloader is None:
            self._dataloader = self.build_dataloader()
        return self._dataloader

    def build_tokenizer(self):
        # tokenizer = assets_tokenizer_helper(self.pretrain_model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.pretrain_model_name)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.build_tokenizer()
        return self._tokenizer

    @property
    def model_config(self):
        return {
            'tokenizer': self.tokenizer,
            'pretrained_model_name_or_path': self.pretrain_model_name,
            'pretrained': True,
            'use_flash_attention_2': True,
            'allow_embedding_resizing': True,
        }

    @property
    def fsdp_config(self):
        return dict()

    def build_ref_model(self, pretrain_model_name: str):
        tmp_ref_path = str('./ref_checkpoints')
        ref_path = os.path.join(tmp_ref_path, 'latest-rank0.pt')
        if os.path.exists(ref_path):
            self.ref_path = ref_path
            return

        self.pretrain_model_name = pretrain_model_name
        composer_dist.initialize_dist('gpu')

        tmp_model = ComposerHFCausalLM(**self.model_config, use_auth_token=True)

        tmp_optimizer = DecoupledAdamW(tmp_model.parameters(), lr=1e-6)

        temp_dataloader = [{
            'input_ids': torch.ones((2, 15)).to(dtype=torch.int64),
            'attention_mask': torch.ones((2, 15)),
            'labels': torch.ones((2, 15)).to(dtype=torch.int64),
        }]

        temp_trainer = Trainer(
            model=tmp_model,
            train_dataloader=temp_dataloader,
            optimizers=tmp_optimizer,
            max_duration='1ba',
            parallelism_config={'fsdp': self.fsdp_config},
            save_folder=tmp_ref_path,
            save_weights_only=True,
            device_train_microbatch_size=2,
        )

        temp_trainer.fit()

        # After making the reference model, we can proceed with the PPO training
        self.ref_path = ref_path

    def build_ppo_trainer(self, pretrain_model_name: str):
        self.pretrain_model_name = pretrain_model_name
        composer_dist.initialize_dist('gpu')
        max_seq_len = 32
        precision = 'amp_bf16'

        model = ComposerHFPolicyLM(**self.model_config, use_auth_token=True)

        optimizer = DecoupledAdamW(model.parameters(), lr=1e-8)

        num_batches_per_update = 2

        # ref_model_config = copy.deepcopy(self.model_config)
        ref_model_config = {**self.model_config, 'name': 'hf_causal_lm'}

        variables = {
            'buffer': {
                'name': 'MinibatchRolloutBuffer',
                'max_buffer_size': num_batches_per_update,
            },
            'max_gen_len': 8,
            'gamma': 0.99,
            'lambda_gae': 0.95,
            'generation_kwargs': {
                'use_cache': True,
                'do_sample': False,
            },
            'kl_controller': {
                'init_kl_coef': 0.2,
                'target': 0.01,
                'horizon': 12800,
                'kl_ctl_type': 'adaptive',
            },
            'reference_model': {
                'model_config': ref_model_config,
                'precision': precision,
                'load_path': self.ref_path,
                'non_train_fsdp_config': self.fsdp_config,
            },
            'device_generate_batch_size': 2,
            'epoch_per_iteration': 1,
            'num_batches_per_update': num_batches_per_update,
            'rewards': {
                'output_length': {
                    'reward_type': 'output_length',
                    'max_gen_len': 10,
                },
            },
        }
        train_config = {
            'model': {**self.model_config, 'kl_estimator': 'k1', 'kl_clip_range': 40.0},
            'fsdp_config': self.fsdp_config,
            'seed': 17,
            'precision': precision,
            'variables': variables,
            'max_seq_len': max_seq_len,
            'global_train_batch_size': 2,
            'device_train_batch_size': 2,
            'device_train_microbatch_size': 1,
        }

        self.ppo_callback = SingleControllerOnPolicyCallback(train_config=train_config)
        self.ppo_trainer = Trainer(
            model=model,
            optimizers=optimizer,
            callbacks=self.ppo_callback,
            train_dataloader=self.dataloader,
            precision=precision,
            parallelism_config={'fsdp': self.fsdp_config},
            max_duration='3iter',
            device_train_microbatch_size=1,
            load_path=self.ref_path,
        )

    def train_1_iter(self):
        self.ppo_trainer.fit(duration='1iter')
        # This is the KL assert that must be true if we are truly loading from the same model.
        # This is only true on the first iteration
        assert torch.allclose(
            self.ppo_trainer.state.loss['kl/ift_kl'], # pyright: ignore
            torch.tensor(0.0),
            atol=5e-5,
        )

    def sync_weight_and_gen(self, vllm_engines: list[Any]):
        self.ppo_callback.round_trip_to_inference_engines(
            device=self.ppo_trainer.state.device,
            vllm_engines=vllm_engines,
            model_update_group=self.model_update_group,
        )


def run():
    prompts = [
        "what is RAY?",
        "what is vLLM?",
    ]
    # pretrain_model_name = 'gpt2'
    pretrain_model_name = 'meta-llama/Llama-3.2-1B-Instruct'
    with start_ray_server() as address:
        if dist.get_rank() == 0:
            master_addr, _ = address.split(':')
            
            print(f"\n=== STARTING DISTRIBUTED TRAINING WITH RAY ACTORS ===")
            num_train_actors = dist.get_world_size() // 2
            # Create actors - rank 0 will allocate master address/port
            train_actors = []

            # master actor will allocate master_addr and master_port
            master_actor = DistributedGPUActor.remote(0, num_train_actors)
            train_actors.append(master_actor)
            
            # Get master address from rank 0 actor
            master_info = ray.get(master_actor.get_master_address.remote())
            master_addr, master_port = master_info
            print(f"Master address allocated: {master_addr}:{master_port}")
            
            # Create remaining actors with the master address/port
            for i in range(1, num_train_actors):
                actor = DistributedGPUActor.remote(i, num_train_actors, master_addr, master_port)
                train_actors.append(actor)

            # composer will initialize the process group for each actor, so no need to initialize them explicitly
            build_ref_model_tasks = [actor.build_ref_model.remote(pretrain_model_name) for actor in train_actors]
            ray.get(build_ref_model_tasks)
            print('build ref model done')

            build_ppo_trainer_tasks = [actor.build_ppo_trainer.remote(pretrain_model_name) for actor in train_actors]
            ray.get(build_ppo_trainer_tasks)
            print('build ppo trainer done')

            world_size = dist.get_world_size()
            vllm_tensor_parallel_size = world_size - num_train_actors
            num_vllm_engines = (
                world_size - num_train_actors
            ) // vllm_tensor_parallel_size
            print(f'num_vllm_engines: {num_vllm_engines}')
            vllm_engines = create_vllm_engines(
                num_engines=num_vllm_engines,
                tensor_parallel_size=vllm_tensor_parallel_size,
                enforce_eager=True,
                pretrain=pretrain_model_name,
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

            new_port = ray.get(master_actor.get_free_port.remote())
            print(f'new_port: {new_port}')
            refs = [
                engine.init_process_group.remote(
                    master_addr,
                    new_port,
                    i * vllm_tensor_parallel_size + 1,
                    dist.get_world_size() // 2 + 1,
                    'weight-update',
                    backend='nccl',
                ) for i, engine in enumerate(vllm_engines)
            ]
            refs.append(master_actor.add_process_group.remote(
                backend='nccl',
                master_addr=master_addr,
                master_port=new_port,
                world_size=dist.get_world_size() // 2 + 1,
                rank=0,
                group_name='weight-update',
            ))
            # should only get refs of both master and vllm_engines together, otherwise it will hang
            print(ray.get(refs))

            refs = [actor.sync_weight_and_gen.remote(vllm_engines) for actor in train_actors]
            ray.get(refs)
            print('sync weight and gen done')

            refs = [actor.train_1_iter.remote() for actor in train_actors]
            ray.get(refs)
            print('train 1 iter done')

            ref = vllm_engines[0].generate.remote(prompts)
            gen_results = ray.get(ref)
            for output in gen_results:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == '__main__':
    run()
