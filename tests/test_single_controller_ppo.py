# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0


# Copy the test file in the root of the repo
# NOTE: This actually runs GRPO instead of PPO
# cd compose-rl && cp tests/test_single_controller_ppo.py .
# run cmd: composer test_single_controller_ppo.py
# If I do ctrl+c to kill job
# Check with `ray status` to see if the actors are still running
# If they are, then run `ray stop`

import logging
import os
import pathlib
import time
import datetime
from functools import partial
from typing import Any, Optional

import pytest
import ray
import torch
import torch.distributed as dist
from composer import Trainer
from composer.core import get_precision_context
from composer.optim import DecoupledAdamW
from composer.utils import dist as composer_dist
from llmfoundry.data import build_dataloader
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from compose_rl.algorithms.online import (
    ComposerHFPolicyLM,
    ComposerHFCriticFreePolicyLM,
    SingleControllerOnPolicyCallback,
)
from compose_rl.algorithms.online.generation_utils import (
    broadcast_to_vllm,
    create_vllm_engines,
    vllm_generate,
)
from compose_rl.utils.ray_utils import start_ray_server
from tests.common import (
    BaseDistributedGPUActor,
    world_size,
)


@ray.remote(num_gpus=1)
class DistributedGPUActor(BaseDistributedGPUActor):
    """Distributed GPU actor for testing."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
    ):
        super().__init__(rank, world_size, master_addr, master_port)
        
        # Configure Ray actor logging - this will go to Ray logs
        self.logger = logging.getLogger(f"Actor-{rank}")
        self.logger.setLevel(logging.INFO)
        
        # Create console handler that will be captured by Ray
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'[ACTOR-{rank}] %(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.model = None
        self.model_update_group = None
        self.ref_path = None
        self._dataloader = None
        self._tokenizer = None
        self.ppo_callback = None
        self.ppo_trainer: Trainer = None  # type: ignore

        self.pretrain_model_name = None
        self.device_train_batch_size = None
        self.num_batches_per_update = None
        self.max_seq_len = None
        self.precision = None  # type: ignore
        self.train_config: dict = None  # type: ignore

    def build_train_config(self, pretrain_model_name: str):
        self.logger.info(f"Starting build_train_config with model: {pretrain_model_name}")
        self.pretrain_model_name = pretrain_model_name

        self.model_config = {
            'tokenizer': self.tokenizer,
            'pretrained_model_name_or_path': self.pretrain_model_name,
            'pretrained': True,
            'use_flash_attention_2': True,
            'allow_embedding_resizing': True,
            'name': 'hf_critic_free_lm',
            # 'init_device': 'mixed',
            # This throws: [rank0]: ValueError: Detected mixed initialization where some ranks have model on cpu or gpu and some ranks are on meta. Either keep all ranks on the same device or set parallelism_config['fsdp']['sync_module_states'] = True. Otherwise, some weights may be randomly initialized when loading a checkpoint.
            'loss_type': 'grpo',
            'target_kl': 0.1,
            'kl_estimator': 'k3',
            'kl_clip_range': 40,
            'use_auth_token': True,
            'compute_kl_loss': False,
            'policy_clip_ratio': 0.2,
            'normalize_advantage': True,
            'length_normalize_policy_loss': True,
            'attn_implementation': 'flash_attention_2'
        }
        self.global_train_batch_size = 64
        self.device_train_batch_size = self.global_train_batch_size // self.world_size
        self.num_batches_per_update = 8
        self.max_seq_len = 10240
        # self.max_gen_len = 8192
        self.max_gen_len = 1000
        self.precision = 'amp_bf16'

        ref_model_config = {
            'name': 'hf_causal_lm',
            'pretrained': self.model_config['pretrained'],
            'pretrained_model_name_or_path': self.pretrain_model_name,
            'use_auth_token': self.model_config['use_auth_token'],
            'use_flash_attention_2': self.model_config['use_flash_attention_2'], 
        }

        variables = {
            'gamma': 1,
            'lambda_gae': 1,
            'epoch_per_iteration': 1,
            'num_batches_per_update': self.num_batches_per_update,
            # 'num_train_nodes': 2,
            'generations_per_prompt': 8,
            'num_batches_per_update': 8,
            # 'vllm_tensor_parallel_size': 1,
            'device_generate_batch_size': 1,
            'vllm_enable_prefix_caching': True,
            'generation_kwargs': {
                'top_p': 1.0,
                'use_cache': True,
                'do_sample': False,
                'temperature': 1.0,
            },
            'eos_token_ids': [
                151643,
                151645
            ],
            'buffer': {
                'name': 'MinibatchRolloutBuffer',
                'max_buffer_size': self.num_batches_per_update,
            },
            'max_gen_len': self.max_gen_len,
            'kl_controller': {
                'init_kl_coef': 0.0, # no KL penalty
                'kl_ctl_type': 'fixed',
            },
            'reference_model': {
                'model_config': ref_model_config,
                'precision': self.precision,
                'load_path': self.ref_path,
            },
            'non_train_fsdp_config': self.fsdp_config,
            'rewards': {
                'math_verifier': {
                    'reward_type': 'math_verifier',
                    'reward': 4,
                },
                'bad_generation_end': {
                    'reward': -1,
                    'eos_penalty': True,
                    'reward_type': 'bad_generation_end'
                },
                'math_format_verifier': {
                    'reward': 1,
                    'reward_type': 'math_format_verifier'
                },
                'penalize_extra_short_responses': {
                    'reward': -1,
                    'reward_type': 'short_response_reward',
                    'len_threshold': 10
                },
            }
        }
        algorithm_config = {
            'gradient_clipping': {
                'clipping_type': 'norm',
                'clipping_threshold': 0.0001
            }
        }
        self.train_config = {
            'seed': 17,
            'model': self.model_config,
            'fsdp_config': self.fsdp_config,
            'precision': self.precision,
            'variables': variables,
            'algorithms': algorithm_config,
            'global_train_batch_size': self.device_train_batch_size * self.world_size,
            'device_train_batch_size': self.device_train_batch_size,
            'device_train_microbatch_size': self.device_train_batch_size,
            'save_folder': './checkpoints/grpo_single_controller',
            'log_config': True,
            'max_seq_len': self.max_seq_len,
            'python_log_level': 'debug',
            'console_log_interval': '1ba',
        }
        self.logger.info("Finished build_train_config")

    def build_dataloader(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dataset_dir = f"/tmp/dataset/prompt_{timestamp}/"
        train_loader_config = {
            'name': 'prompt',
            'dataset': {
                'local': temp_dataset_dir,
                'split': 'train',
                'remote': 'dbfs:/Volumes/datasets/ashutoshbaheti/orl_data/open_r1_filtered/q7b_open_r1_48k/',
                'shuffle': True,
                'max_gen_len': self.max_gen_len,
                'max_seq_len': self.max_seq_len,
                'shuffle_seed': 17,
                'download_timeout': 1800
            },
            'drop_last': True,
            'num_workers': 1,
        }
        foundry_dataspec = build_dataloader(
            cfg = train_loader_config,
            tokenizer = self.tokenizer,
            device_batch_size = self.train_config['device_train_batch_size'],
        )
        self.logger.info(f"Foundry dataloader built successfully from class {type(foundry_dataspec)}")
        foundry_dataloader = foundry_dataspec.dataloader
        return foundry_dataloader

    def build_tokenizer(self):
        # TODO (algo): decide if we should use tokens or messages given
        # we may need token level log prob
        # TODO (infra): use the tokenizer/texts for prompt dataloader but
        # token (ids) for the experience buffer/manager
        kwargs = {
            'padding': 'longest',
            'pad_token': '<|endoftext|>',
            'truncation': True,
            'padding_side': 'left',
            'model_max_length': self.max_seq_len,
            'trust_remote_code': True,
        }
        tokenizer = AutoTokenizer.from_pretrained(self.pretrain_model_name, **kwargs)
        return tokenizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.build_tokenizer()
        return self._tokenizer

    @property
    def fsdp_config(self):
        # TODO (infra): use actual fsdp1 config
        return {}

    def init_composer_dist(self):
        composer_dist.initialize_dist('gpu')

    def build_ppo_trainer(self):
        name = self.model_config.pop('name')
        
        self.logger.info(f"Model type: {name}")
        if name == 'hf_ppo_lm':
            self.logger.info("Creating ComposerHFPolicyLM")
            model = ComposerHFPolicyLM(**self.model_config)
        elif name == 'hf_critic_free_lm':
            self.logger.info("Creating ComposerHFCriticFreePolicyLM")
            model = ComposerHFCriticFreePolicyLM(**self.model_config)
        self.logger.info("Model created successfully")

        optimizer = DecoupledAdamW(model.parameters(), lr=1e-8)

        # TODO (infra): pull the rest of the training logic from the callback
        # to this class, e.g, how to interact with env, calculate rewards etc
        # NOTE: SingleControllerOnPolicyCallback is currently over-writing the iteration_start method
        self.ppo_callback = SingleControllerOnPolicyCallback(
            train_config=self.train_config,
        )

        self.ppo_trainer = Trainer(
            model=model,
            optimizers=optimizer,
            callbacks=self.ppo_callback,
            train_dataloader=self.build_dataloader(),
            precision=self.precision,
            parallelism_config={'fsdp': self.fsdp_config},
            max_duration='5iter',
            device_train_microbatch_size=1,
            load_path=self.ref_path,
        )

    def train_1_iter(self):
        # TODO (algo): implement the top level PPO algo here instead of the
        # callback. Algorithmic researchers are expected to implement this
        # function along with above policy/value/reward/ref trainers or models
        # TODO (infra): try multiple fit to see if the (mlflow) logger, etc
        # TODO (infra): fault tolerance at iteration level first
        # TODO (infra): enable batch level control

        # NOTE: Trainer has a train microbatches function that should be used here to get low level control.
        # fit() checks if there is existing checkpoint, make a full forward pass, it will run eval pass and save pass.
        # We potentially want to run this https://github.com/mosaicml/composer/blob/dev/composer/trainer/trainer.py#L2826
        # fit() can also potentially overwrite the mlflow
        self.ppo_trainer.fit(duration='1iter')
        self.logger.info(f"#### Finished training 1 iter with loss: {self.ppo_trainer.state.loss}")

    def update_inference_model(self, vllm_engines: list[Any]):
        start_time = time.time()
        print('Before broadcast to vLLM')
        # TODO (infra) instead of direcly broadcasting to vllm, we should
        # push the model parameters to a parameter buffer manager and have
        # the buffer manager initiate broadcast of parameters to vllm engines
        broadcast_to_vllm(
            self.ppo_callback.actor_critic,
            vllm_engines,
            self.model_update_group,
            device=torch.device('cuda'),
            loss_type=self.ppo_callback.actor_critic.loss_type,  # type: ignore
        )
        print('Finished broadcasting to vLLM')
        print(f'Took: {time.time() - start_time} to broadcast to vllm.')
        dist.barrier()

    def query_inference_engines(self, vllm_engines: list[Any]):
        """Round trip to inference engines.

        Args:
            vllm_engines (list[Any]): The vllm engines to round trip to.
        """
        # TODO (infra): we should use the rollout agent to generate sequences
        # instead of the trainer actor, e.g,. reimplment _get_next_iter_prompts
        # in the rollout agent
        batch = self.ppo_trainer.state.device.batch_to_device(
            self.ppo_callback._get_next_iter_prompts(),
        )
        max_gen_len = self.train_config['variables']['max_gen_len']
        generation_kwargs = self.train_config['variables']['generation_kwargs']
        with get_precision_context(self.precision), torch.no_grad():
            # TODO (infra): refactor this code to isolate gather of
            # prompts on the trainer actor and gather/scatter of sequences
            # on the trainer actor, the first half is uesless while
            # the second half should be managed throught a experience manager
            sequences = vllm_generate(
                vllm_engines=vllm_engines,
                batch=batch,
                max_gen_len=max_gen_len,
                generation_kwargs=generation_kwargs,
                tokenizer=self.tokenizer,  # type: ignore
                vllm_generate_function='generate',
            )
        # Add the prepared sequences to the batch again
        batch['sequences'] = sequences
        self.ppo_callback.batch_rollouts = batch  # type: ignore


def setup_process_groups(
    master_actor: Any,
    vllm_engines: list[Any],
    vllm_tensor_parallel_size: int,
):
    """Initialize process groups for vLLM engines and master actor."""
    # Get a new port for the weight-update process group
    master_addr, _ = ray.get(
        master_actor.get_master_address.remote(),
    )  # type: ignore
    new_port = ray.get(master_actor.get_free_port.remote())  # type: ignore
    print(f'new_port: {new_port}')

    world_size = dist.get_world_size()

    # Initialize process groups for vLLM engines
    refs = [
        engine.init_process_group.remote(
            master_addr,
            new_port,
            i * vllm_tensor_parallel_size + 1,
            world_size // 2 + 1,
            'weight-update',
            backend='nccl',
        ) for i, engine in enumerate(vllm_engines)
    ]

    # Add master actor to the process group
    refs.append(
        master_actor.add_process_group.remote(
            backend='nccl',
            master_addr=master_addr,
            master_port=new_port,
            world_size=world_size // 2 + 1,
            rank=0,
            group_name='weight-update',
        ),
    )

    # Wait for all process groups to be initialized
    print(ray.get(refs))


class SPMDActorGroup:
    # TODO (infra): refactor this to a proper base class

    def __init__(self, num_train_actors: int):
        self.num_train_actors = num_train_actors

        self._train_actors = []
        """Create and initialize all training actors."""
        print(f'\n=== STARTING DISTRIBUTED TRAINING WITH RAY ACTORS ===')

        # Create master actor first
        self._master_actor = DistributedGPUActor.remote(
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
            actor = DistributedGPUActor.remote(
                i,
                self.num_train_actors,
                master_addr,  # type: ignore
                master_port,
            )
            self._train_actors.append(actor)

    @property
    def train_actors(self):
        return self._train_actors

    @property
    def master_actor(self):
        return self._master_actor


class TrainActorGroup(SPMDActorGroup):
    # TODO: this class is mainly pass through gang scheduler,
    # we should refactor this class to be more generic and reusable

    def build_models(self, pretrain_model_name: str):
        """Build reference models and PPO trainers for all actors."""
        build_train_config_tasks = [
            actor.build_train_config.remote(pretrain_model_name)
            for actor in self._train_actors
        ]
        ray.get(build_train_config_tasks)

        init_task = [
            actor.init_composer_dist.remote() for actor in self._train_actors
        ]
        ray.get(init_task)

        # Build PPO trainers
        build_ppo_trainer_tasks = [
            actor.build_ppo_trainer.remote() for actor in self._train_actors
        ]
        ray.get(build_ppo_trainer_tasks)
        print('build ppo trainer done')

    def update_inference_model(self, vllm_engines: list[Any]):
        refs = [
            actor.update_inference_model.remote(vllm_engines)
            for actor in self._train_actors
        ]
        ray.get(refs)
        print('update inference model done')

    def query_inference_engines(self, vllm_engines: list[Any]):
        refs = [
            actor.query_inference_engines.remote(vllm_engines)
            for actor in self._train_actors
        ]
        ray.get(refs)
        print('query inference engines done')

    def train_iteration(self):
        """Run one training iteration on all actors."""
        refs = [actor.train_1_iter.remote() for actor in self._train_actors]
        ray.get(refs)
        print('train 1 iter done')


class RolloutAgent:

    def __init__(self, vllm_engines: list, vllm_tensor_parallel_size: int):
        self.vllm_engines = vllm_engines
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size

    @property
    def num_vllm_engines(self):
        return len(self.vllm_engines)

    def generate(self, prompts: list[str]):
        # TODO (infra): try integrate this with the multi-turn rollout
        # repo
        ref = self.vllm_engines[0].generate.remote(prompts)
        gen_results = ray.get(ref)
        for output in gen_results:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f'Prompt: {prompt!r}, Generated text: {generated_text!r}')


# TODO (infra): implement parameter buffer manager and experience manager
class PPOController:

    def __init__(
        self,
        train_actor: TrainActorGroup,
        inference_client: RolloutAgent,
        pretrain_model_name: str,
    ):
        self.train_actor = train_actor
        self.inference_client = inference_client

        self.train_actor.build_models(pretrain_model_name)
        setup_process_groups(
            self.train_actor.master_actor,
            self.inference_client.vllm_engines,
            self.inference_client.vllm_tensor_parallel_size,
        )

    def train(self):
        for _ in range(5):  # Example: train for 5 iterations
            # NOTE: this loop is represents the logic happening in the current `iteration_start` of the OnPolicyCallback
            self.train_actor.update_inference_model(
                self.inference_client.vllm_engines
            )
            self.train_actor.query_inference_engines(
                self.inference_client.vllm_engines
            )
            self.train_actor.train_iteration()


def _run_single_controller_ppo(
    pretrain_model_path: str,
    world_size: int = 0,
):
    """Shared function for running single controller PPO.

    Args:
        pretrain_model_path: Path to the pretrained model
        world_size: Number of distributed processes
        prompts: List of prompts to test generation with
    """
    # Set vLLM attention backend to FLASH_ATTN otherwise FlashInfer backend
    # takes too long to jit compile
    os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'

    prompts = [
        'what is RAY?',
        'what is vLLM?',
    ]

    with start_ray_server() as _address:
        if dist.get_rank() == 0:
            # only rank 0 is the master controller

            # create SPMD training actors of the system
            if world_size == 0:
                world_size = dist.get_world_size()
            num_train_actors = world_size // 2
            train_actor = TrainActorGroup(num_train_actors)

            # Create vLLM engines (or inference actors)
            vllm_tensor_parallel_size = world_size - num_train_actors
            num_vllm_engines = (
                world_size - num_train_actors
            ) // vllm_tensor_parallel_size
            # TODO: Encapsulate this into a inference server manager class
            vllm_engines = create_vllm_engines(
                num_engines=num_vllm_engines,
                tensor_parallel_size=vllm_tensor_parallel_size,
                enforce_eager=True,
                pretrain=pretrain_model_path,
                revision=None,
                seed=1,
                enable_prefix_caching=False,
                max_model_len=1000,
                device_bundle={
                    'GPU': 1,
                    'CPU': 1,
                    'worker_node': 0,
                },
            )
            inference_client = RolloutAgent(
                vllm_engines,
                vllm_tensor_parallel_size,
            )

            ppo_controller = PPOController(
                train_actor,
                inference_client,
                pretrain_model_path,
            )
            ppo_controller.train()

            inference_client.generate(prompts)


@pytest.mark.gpu
@world_size(4)  # TODO change this to 2 for CI testing (hit fatal python error)
def test_single_controller_ppo(
    world_size: int,
    tiny_llama_model: PreTrainedModel,
    tiny_gpt2_tokenizer: PreTrainedTokenizerBase,
    tmp_path: pathlib.Path,
):
    """Test single controller PPO with Ray actors and vLLM engines."""
    # Save the model and tokenizer to a temporary directory
    local_save_path = str(tmp_path / 'llama_model')
    tiny_llama_model.save_pretrained(local_save_path)
    tiny_gpt2_tokenizer.save_pretrained(local_save_path)

    _run_single_controller_ppo(
        pretrain_model_path=local_save_path,
        world_size=world_size,
    )


if __name__ == '__main__':
    # This is an example of how to move the controller logic from PPO Callback
    # to a separate trainer actor above and this main single controller
    # function.
    _run_single_controller_ppo(
        pretrain_model_path='Qwen/Qwen2.5-1.5B-Instruct',
    )