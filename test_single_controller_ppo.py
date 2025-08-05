# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0


# Copy the test file in the root of the repo
# NOTE: This actually runs GRPO instead of PPO
# cd compose-rl
# run cmd: composer test_single_controller_ppo.py
# If I do ctrl+c to kill job
# Check with `ray status` to see if the actors are still running
# If they are, then run `ray stop`

import argparse
import logging
import os
import time
import datetime
from itertools import chain
from functools import partial
from typing import Any, Optional, Union, MutableMapping
from multiprocessing import get_context
from multiprocessing.pool import AsyncResult, Pool

from composer.loggers import MLFlowLogger
import ray
import spacy
import torch
import torch.distributed as dist
from composer import Trainer
from composer.core import get_precision_context, Precision
from composer.optim import DecoupledAdamW
from composer.utils import dist as composer_dist
from llmfoundry.data import build_dataloader
from llmfoundry.utils import build_composer_model
from llmfoundry.utils.config_utils import process_init_device  # type: ignore
from omegaconf import OmegaConf as om
from transformers import AutoTokenizer

from compose_rl.algorithms.online import (
    ComposerHFPolicyLM,
    ComposerHFCriticFreePolicyLM,
    SingleControllerOnPolicyCallback,
)
from compose_rl.algorithms.online.generation_utils import (
    broadcast_to_vllm,
    create_vllm_engines,
    _vllm_generate,
)
from compose_rl.utils.ray_utils import start_ray_server, uninstall_megablocks_if_exists
from compose_rl.controllers import BaseDistributedGPUActor, SPMDActorGroup
from compose_rl.controllers.buffer import Buffer
from compose_rl.algorithms.online.callback_utils import preprocess_batches
from compose_rl.registry_builders import build_reward
from compose_rl.registry import rewards as rewards_registry
from compose_rl.interfaces.base_kl_controller import BaseKLController
from compose_rl.algorithms.reward_modeling import (
    BadGenerationEndReward,
    BaseReward,
    InferenceRewardModel,
    Reward,
    RewardModel,
)
from compose_rl.utils import (
    approx_kl,
    batch_process_fine_granularities,
    get_log_probs,
    scatter_gather_rewards,
    switch_left_to_right_padding,
    mask_eos,
    get_decoded_sequence,
)
from compose_rl.algorithms.online.reward_manager import (
    ReferenceOutput,
    RewardOutput,
)

_MAX_SEQ_LEN = 6000
_MAX_GEN_LEN = 4000

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
        self.model_config = None
        self.global_train_batch_size = None
        self.max_gen_len = None

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
        self.max_seq_len = _MAX_SEQ_LEN
        self.max_gen_len = _MAX_GEN_LEN
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
            'generations_per_prompt': 8,
            'num_batches_per_update': 8,
            'device_generate_batch_size': 1,
            'vllm_enable_prefix_caching': True,
            'generation_kwargs': {
                'top_p': 1.0,
                'use_cache': True,
                'do_sample': False,
                'temperature': 1.0,
            },
            'eos_token_ids': [
                128001,
                128008,
                128009,
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
            'rewards': {}, # Testing with no rewards
            'non_train_fsdp_config': self.fsdp_config,
        }
        algorithm_config = {
            'gradient_clipping': {
                'clipping_type': 'norm',
                'clipping_threshold': 1.0
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

    def build_tokenizer(self):
        # TODO (algo): decide if we should use tokens or messages given
        # we may need token level log prob
        # TODO (infra): use the tokenizer/texts for prompt dataloader but
        # token (ids) for the experience buffer/manager
        kwargs = {
            'padding': 'longest',
            'pad_token': '<|finetune_right_pad_id|>',
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

        optimizer = DecoupledAdamW(model.parameters(), lr=1e-6)

        # TODO (infra): pull the rest of the training logic from the callback
        # to this class, e.g, how to interact with env, calculate rewards etc
        # NOTE: SingleControllerOnPolicyCallback is currently over-writing the iteration_start method
        self.ppo_callback = SingleControllerOnPolicyCallback(
            train_config=self.train_config,
        )

        # Create a dummy dataloader to make sure trainer can call .fit() with
        # the dataloader that exists at ITERATION_START. This dataloader
        # will NOT be used for training.
        dummy_dataset = torch.utils.data.TensorDataset(torch.randn(16, 1))
        dummy_distributed_sampler = torch.utils.data.distributed.DistributedSampler(dummy_dataset)
        dummy_dataloader = torch.utils.data.DataLoader(dummy_dataset, sampler=dummy_distributed_sampler)

        mlflow_logger = MLFlowLogger(
            experiment_name='test_single_controller_ppo',
            run_name='test_single_controller_grpo_reward_out',
            tracking_uri='databricks',
        )

        self.ppo_trainer = Trainer(
            model=model,
            optimizers=optimizer,
            callbacks=self.ppo_callback,
            train_dataloader=dummy_dataloader,
            precision=self.precision,
            parallelism_config={'fsdp': self.fsdp_config},
            max_duration='5iter',
            loggers=[mlflow_logger],
            device_train_microbatch_size=1,
            load_path=self.ref_path,
        )

    def add_rollouts(self, current_rank_rollouts: dict[str, Any]):
        """Adds the current rank's rollouts to the callback."""
        for k, v in current_rank_rollouts.items():
            assert isinstance(v, torch.Tensor) or isinstance(v, list) or isinstance(v, dict), f"Expected a tensor or list or dict, got {type(v)}"
            if isinstance(v, torch.Tensor):
                current_rank_rollouts[k] = v.to(torch.device('cuda'))
            elif isinstance(v, dict):
                # This is the case with the rewards dict where it has (key, tensor) pairs
                rewards_dict_for_rank = {}
                for reward_key, reward_tensor in v.items():
                    rewards_dict_for_rank[reward_key] = reward_tensor.to(torch.device('cuda'))
                current_rank_rollouts[k] = rewards_dict_for_rank
            elif not (isinstance(v, list)):
                raise ValueError(f"Expected a tensor or list or dict of tensors, got {type(v)}")
        self.ppo_callback.batch_rollouts = current_rank_rollouts

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


class TrainActorGroup(SPMDActorGroup):
    """Group of training actors for PPO."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def build_models(self, pretrain_model_name: str):
        """Build reference models and PPO trainers for all actors."""
        self.collective_methods.build_train_config(pretrain_model_name)
        self.collective_methods.init_composer_dist()

        # Build PPO trainers
        self.collective_methods.build_ppo_trainer()
        print('build ppo trainer done')

    def _partition_rollouts_across_ranks(self, rollouts: dict[str, Any]):
        """Partition the rollouts across all actors."""
        partitioned_rollouts = []
        per_rank_data_size = rollouts['prompt'].shape[0] // self.num_train_actors
        for i in range(self.num_train_actors):
            current_rank_start = i * per_rank_data_size
            current_rank_end = (i + 1) * per_rank_data_size
            current_rank_rollouts = {}
            for k, v in rollouts.items():
                if isinstance(v, torch.Tensor) or isinstance(v, list):
                    current_rank_rollouts[k] = v[current_rank_start:current_rank_end]
                elif isinstance(v, dict):
                    # This is the case with the rewards dict where it has (key, tensor) pairs
                    rewards_dict_for_rank = {}
                    for reward_key, reward_tensor in v.items():
                        rewards_dict_for_rank[reward_key] = reward_tensor[current_rank_start:current_rank_end]
                    current_rank_rollouts[k] = rewards_dict_for_rank
                else:
                    raise ValueError(f"Expected a tensor or list or dict of tensors, got {type(v)}")
            partitioned_rollouts.append(current_rank_rollouts)
        return partitioned_rollouts

    def add_latest_rollouts_from_buffer(self, experience_buffer: "ExperienceBuffer"):
        assert experience_buffer is not None, "Experience buffer is not set"
        assert len(experience_buffer) > 0, "Experience buffer is empty"
        latest_rollouts = experience_buffer.popleft()
        # Extract rewards dict separately from latest_rollouts
        partitioned_rollouts = self._partition_rollouts_across_ranks(latest_rollouts)
        assert len(partitioned_rollouts) == self.num_train_actors, "Number of partitioned rollouts should be equal to the number of train actors"
        ray.get([train_actor.add_rollouts.remote(partition) for train_actor, partition in zip(self.train_actors, partitioned_rollouts)])


class InferenceServer:
    """Inference server with vLLM engines."""

    def __init__(self, num_vllm_engines: int, vllm_tensor_parallel_size: int, pretrain_model_name: str):
        self.num_vllm_engines = num_vllm_engines
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.vllm_engines = create_vllm_engines(
                num_engines=num_vllm_engines,
                tensor_parallel_size=vllm_tensor_parallel_size,
                enforce_eager=True,
                pretrain=pretrain_model_name,
                revision=None,
                seed=1,
                enable_prefix_caching=False,
                max_model_len=_MAX_GEN_LEN,
                device_bundle={
                    'GPU': 1,
                    'CPU': 1,
                    'worker_node': 0,
                },
            )

    @property
    def engines(self):
        return self.vllm_engines

class ParameterBuffer(Buffer):
    """Buffer for updating the inference model."""

    def update_inference_model(self, actor: DistributedGPUActor, inference_server: InferenceServer):
        start_time = time.time()
        print('Before broadcast to vLLM')
        # TODO (infra) instead of direcly broadcasting to vllm, we should
        # push the model parameters to a parameter buffer manager and have
        # the buffer manager initiate broadcast of parameters to vllm engines
        broadcast_to_vllm(
            actor.ppo_callback.actor_critic,
            inference_server.engines,
            actor.model_update_group,
            device=torch.device('cuda'),
            loss_type=actor.ppo_callback.actor_critic.loss_type,  # type: ignore
        )
        print('Finished broadcasting to vLLM')
        print(f'Took: {time.time() - start_time} to broadcast to vllm.')
        dist.barrier()

    def put(self, struct: dict[str, Any]):
        # prefers to implement the model update logic in the Buffer class as the buffer is a bridge between the trainer actor and the inference server
        # and knows the best way to transfer the model parameters. Trainer just needs to put necessary struct to this api
        struct['actor_group'].collective_methods.execute(partial(self.update_inference_model, inference_server=struct['inference_server']))


# TODO: Move this experience buffer earlier so that we can avoid
# using "ExperienceBuffer" (with quotes) as a type hint.
class ExperienceBuffer(Buffer):
    """Buffer for storing experiences."""

    def put(self, struct: dict[str, Any]):
        self.buffer.append(struct)

    def get(self, struct: Optional[dict[str, Any]] = None):
        return self.buffer[0]

    def popleft(self, struct: Optional[dict[str, Any]] = None):
        return self.buffer.pop(0)

    def __len__(self):
        return len(self.buffer)


class StreamingDatasetActor(BaseDistributedGPUActor):
    """Streaming actor for loading prompts onto the experience buffer."""

    def __init__(self):
        # Setting up the distributed environment (WORLD_SIZE = 1)
        super().__init__(
            rank=0,
            world_size=1,
            master_addr=None,
            master_port=None,
        )

        # Setting up all of the configs
        # TODO: We should move these to dataclasses
        # TODO: In a future PR, create all configs in the main function and populate
        # the correct configs across all entities (e.g. DistributedGPUActor, StreamingDatasetActor, etc)
        self.pretrain_model_name = 'meta-llama/Llama-3.1-8B-Instruct'
        self.prompt_handler_config = {
            "global_train_batch_size": 64,
            "generations_per_prompt": 8,
            "num_batches_per_update": 8,
            "max_seq_len": _MAX_SEQ_LEN,
            "max_gen_len": _MAX_GEN_LEN,
        }
        self.tokenizer_config = {
            'padding': 'longest',
            'pad_token': '<|finetune_right_pad_id|>',
            'truncation': True,
            'padding_side': 'left',
            'model_max_length': self.prompt_handler_config['max_seq_len'],
            'trust_remote_code': True,
        }
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dataset_dir = f"/tmp/dataset/prompt_{timestamp}/"
        self.dataloader_config = {
            'name': 'prompt',
            'dataset': {
                'local': temp_dataset_dir,
                'split': 'train',
                'remote': 'dbfs:/Volumes/datasets/ashutoshbaheti/orl_data/math_lighteval/llama3_8b_math_prompts/',
                'shuffle': True,
                'max_gen_len': self.prompt_handler_config['max_gen_len'],
                'max_seq_len': self.prompt_handler_config['max_seq_len'],
                'shuffle_seed': 17,
                'download_timeout': 1800
            },
            'drop_last': True,
            'num_workers': 1,
        }

        # Key variables
        global_train_batch_size = self.prompt_handler_config['global_train_batch_size']
        self.generations_per_prompt = self.prompt_handler_config['generations_per_prompt']
        num_batches_per_update = self.prompt_handler_config['num_batches_per_update']
        total_num_generations = global_train_batch_size * num_batches_per_update
        self.num_prompts_per_iteration = total_num_generations // self.generations_per_prompt

        # Validate that the total number of generations is divisible by the number of generations per prompt
        assert total_num_generations % self.generations_per_prompt == 0, "total_num_generations must be divisible by generations_per_prompt"

        # Creating main entities
        self.tokenizer = self._build_tokenizer()
        self.dataloader = self._build_dataloader()
        self.dataloader_iter = iter(self.dataloader)

    def _build_dataloader(self):
        foundry_dataspec = build_dataloader(
            cfg = self.dataloader_config,
            tokenizer = self.tokenizer,
            device_batch_size = self.num_prompts_per_iteration,
        )
        return foundry_dataspec.dataloader

    def _build_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.pretrain_model_name, **self.tokenizer_config)
        return tokenizer

    def get_prompt_handler_config(self):
        return self.prompt_handler_config

    def get_tokenizer_pad_token_id(self):
        return self.tokenizer.pad_token_id

    def _get_single_iter_prompts(self):
        """Gets a single iteration's prompts from the dataloader."""
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            return next(self.dataloader_iter)

    def get_next_iter_prompts(self):
        """Gets the next iteration's prompts across all ranks and prepares them for the rollout agent."""
        batches = [self._get_single_iter_prompts()]

        return preprocess_batches(batches, self.generations_per_prompt, self.tokenizer.pad_token_id)

class RewardActor(BaseDistributedGPUActor):
    """Streaming actor for adding rewards on top the experience buffer."""

    def __init__(self):
        # Setting up the distributed environment (WORLD_SIZE = 1)
        super().__init__(
            rank=0,
            world_size=1,
            master_addr=None,
            master_port=None,
        )
        # Configure Ray actor logging - this will go to Ray logs
        self.logger = logging.getLogger(f"REWARD-ACTOR")
        self.logger.setLevel(logging.INFO)
        
        # Create console handler that will be captured by Ray
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'[REWARD-ACTOR] %(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # For fine-grained rewards. Not necessarily used.
        self.max_seq_len = _MAX_SEQ_LEN
        self.max_seq_len = _MAX_SEQ_LEN
        self.precision = Precision.AMP_BF16
        self.parser = spacy.load('en_core_web_sm')
        self.pretrain_model_name = 'meta-llama/Llama-3.1-8B-Instruct'
        self.tokenizer_config = {
            'padding': 'longest',
            'pad_token': '<|finetune_right_pad_id|>',
            'truncation': True,
            'padding_side': 'left',
            'model_max_length': self.max_seq_len,
            'trust_remote_code': True,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrain_model_name, **self.tokenizer_config)

        self.all_rewards = {}
        self.reward_coefficients: dict[str, float] = {}
        self.granularities: dict[str, str] = {}

        self.inference_rewards: list[str] = []
        self.functional_rewards: list[str] = []
        self.local_reward_models: list[str] = []

        self.reward_config = {
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

        for reward_name, reward_config in self.reward_config.items():
            assert isinstance(reward_name, str)
            if reward_name in self.all_rewards:
                raise KeyError(
                    f'The reward manager already has a model with {reward_name=}',
                )

            self.logger.info(f'Initializing reward with name {reward_name}')

            # TODO: Validate reward_config
            reward_type = reward_config.pop('reward_type')
            reward_cls = rewards_registry.get(reward_type)

            if issubclass(reward_cls, Reward):
                # TODO: This assumes that all functional rewards are document level rewards.
                # This is not necessarily true, but is a reasonable assumption for now.
                self.granularities[reward_name] = 'document'
                model = build_reward(
                    name=reward_type,
                    tokenizer=self.tokenizer,
                    kwargs=reward_config,
                )
                self.functional_rewards.append(reward_name)

            elif issubclass(reward_cls, RewardModel):
                self.granularities[reward_name] = reward_config.get(
                    'granularity',
                )

                if reward_cls == InferenceRewardModel:
                    model = build_reward(
                        name=reward_type,
                        tokenizer=self.tokenizer,
                        kwargs=reward_config,
                    )
                    self.inference_rewards.append(reward_name)

                else:
                    # TODO: Local reward models will note be supported in the single controller cpu only RM manager.
                    reward_model_config = reward_config.get(
                        'model_config',
                        None,
                    )
                    assert reward_model_config is not None, 'model_config must be provided in reward_config'
                    model = self.initialize_composer_model(
                        model_config=reward_config.get('model_config'),
                        model_name=reward_name,
                        precision=reward_config.get('precision', self.precision),
                        load_path=reward_config.get('load_path', None),
                    )
                    self.local_reward_models.append(reward_name)
            else:
                raise TypeError(
                    f'Reward class {reward_cls} is not a subclass of either Reward or RewardModel.',
                )

            self.all_rewards[reward_name] = model
            self.reward_coefficients[reward_name] = reward_config.get(
                'reward_coefficient',
                1.0,
            )

        self.granularity_types = list(set(self.granularities.values()))

        self.pool = None
        if self.inference_rewards or self.functional_rewards:
            self.pool = Pool(
                processes=len(self.inference_rewards) +
                len(self.functional_rewards),
                context=get_context('spawn'),
            )
    
    @property
    def fsdp_config(self):
        # TODO (infra): This should be the non_train_fsdp_config from the callback
        return {}
    
    def initialize_composer_model(
        self,
        model_config: dict[str, Any],
        model_name: str,
        precision: Precision = Precision.FP32,
        load_path: Optional[str] = None,
    ) -> torch.nn.Module:
        """Create the reference model."""
        self.logger.info(f'Initializing {model_name} model')
        name = model_config.pop('name')

        init_context = process_init_device(
            model_config,
            self.fsdp_config,
        )
        model = build_composer_model(
            name=name,
            cfg=model_config,
            tokenizer=self.tokenizer,
            init_context=init_context,
            master_weights_dtype=model_config.get('master_weights_dtype', None),
        )

        parallelism_config = {'fsdp': self.fsdp_config}

        # Create a Trainer object to load from checkpoint and FSDP the model
        _ = Trainer(
            model=model,
            parallelism_config=parallelism_config,
            precision=precision,
            load_weights_only=True,
            load_strict_model_weights=False,
            load_path=load_path,
            python_log_level='debug',
        )

        self.logger.info(f'Initialized {model_name} model')
        return model

    @staticmethod
    def make_zero_reward(ref_tensor: torch.Tensor):
        """Helper to instantiate an empty reward tensor.

        The output will be a zero tensor with the same shape, device, and dtype
        as ref_tensor
        """
        return torch.zeros_like(ref_tensor).to(
            ref_tensor.device,
        ).type(ref_tensor.dtype)

    @staticmethod
    def _to_cpu(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.cpu()
        elif isinstance(x, (tuple, list)):
            return [RewardActor._to_cpu(x_) for x_ in x]
        elif isinstance(x, dict):
            return {k: RewardActor._to_cpu(v) for k, v in x.items()}
        else:
            return x

    @staticmethod
    def call_reward_model(
        reward_model: RewardModel,
        batch: MutableMapping,
    ):
        """Calls the reward model and extract rewards.

        This function will call the reward model (local or inference) and extract
        the rewards from the model output. The extracted rewards will be scattered
        into a reward tensor.

        Args:
            reward_model (RewardModel): the reward model to call.
            batch (MutableMapping): the batch of data to compute the reward.

        Returns:
            rewards (Tensor): a tensor of rewards. This is the result of scattering
                the extracted rewards into the zero_rewards tensor.
        """
        # We need to do this to handle getting rewards at multiple points in a
        # single input sequence with a deployed RM.
        if isinstance(reward_model, InferenceRewardModel):
            rm_seq_lens = [
                [idx + prompt_len
                 for idx in gather_indices]
                for gather_indices, prompt_len in
                zip(batch['end_idxs_gather'], batch['reward_prompt_lens'])
            ]
        else:
            rm_seq_lens = batch['reward_seq_lens']

        reward_batch = {
            'input_ids': batch['tok_formatted_reward_inputs'],
            'attention_mask': batch['tok_formatted_reward_attn_masks'],
            'seq_lens': rm_seq_lens,
            'is_inference': True,
            'seq_reward': True,
        }

        # Note this uses separate seq lengths to account for potential
        # changes made during reward string formatting
        curr_rewards = reward_model(
            reward_batch,
        ).to(
            dtype=batch['zero_rewards'].dtype,
        )

        assert isinstance(curr_rewards, torch.Tensor)
        # Passing in reward_seq_lens to make sure RL formatting in env_generate
        # and special reward formatting idxs match up before scattering rewards
        output: torch.Tensor = scatter_gather_rewards(
            temp_rews=batch['zero_rewards'],
            curr_rewards=curr_rewards,
            reward_prompt_lens=batch['reward_prompt_lens'],
            prompt_lens=batch['prompt_lens'],
            reward_generated_lens=batch['reward_generated_lens'],
            generated_lens=batch['generated_lens'],
            end_idxs_gather=batch['end_idxs_gather'],
            end_idxs_scatter=batch['end_idxs_scatter'],
            reward_seq_lens=batch['reward_seq_lens'],
            seq_lens=batch['seq_lens'],
        )
        return output

    def calculate_reward(
        self,
        raw_untokenized_texts: list[tuple[str, str]],
        right_padded_obses: torch.Tensor,
        attention_masks: torch.Tensor,
        seq_lens: torch.Tensor,
        generated_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        max_gen_length: int,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        verified_answers: Optional[list[str]] = None,
    ) -> RewardOutput:
        """Collect rewards for generations.

        Args:
            raw_untokenized_texts (list): A list of (prompt, generation) string
                pairs from decoding the tokens seen/produced by the policy model.
            right_padded_obses (tensor): The right padded prompt+generation
                token sequences from calling generate on the policy model.
            attention_masks (tensor): A mask tensor indicating which tokens
                in right_padded_obses are padding tokens.
            seq_lens (tensor): The combined prompt+generation token length of
                each sequence.
            generated_lens (tensor): The number of tokens generated by the policy
                for each sequence.
            prompt_lens (tensor): The number of tokens in the prompt given to
                the policy.
            max_gen_len (int): The maximum number of tokens the policy is able
                to generate.
            actions (tensor): The (right padded) tokens generated by the policy.
            action_log_probs (tensor): The log probability of generating each action.
            verified_answers (Optional[list[str]]): A list of answers for verifiable rewards.

        Returns:
            RewardOutput: A dictionary of float tensors, with an entry for each reward
                model managed by the reward manager. For reward models that are called
                async, the associated value is an AsyncResult object that will return
                the reward tensor from its `.get()` method.
        """
        device = right_padded_obses.device.type

        # Only process text for the existing granularity types of the rewards
        processed_inputs = batch_process_fine_granularities(
            raw_untokenized_texts=raw_untokenized_texts,
            granularity_types=self.granularity_types,
            generated_lens=generated_lens.cpu().tolist(),
            prompt_lens=prompt_lens.cpu().tolist(),
            original_obses=right_padded_obses.cpu().tolist(),
            parser=self.parser,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            device=device,
        )

        computed_rewards: RewardOutput = {}

        # Base batch that we will adjust per reward mdoel
        batch = {
            'input_ids': right_padded_obses,
            'attention_mask': attention_masks,
            'actions': actions,
            'prompt_len': prompt_lens,
            'max_gen_len': max_gen_length,
            'generated_lens': generated_lens,
            'seq_lens': seq_lens,
            'action_log_probs': action_log_probs,
        }
        if verified_answers is not None:
            batch['verified_answers'] = verified_answers

        for reward_name in chain(
            self.functional_rewards,
            self.inference_rewards,
            self.local_reward_models,
        ):
            curr_reward = self.all_rewards[reward_name]
            curr_batch = self._create_batch(
                self.all_rewards[reward_name],
                reward_name,
                processed_inputs,
                batch,
                raw_untokenized_texts,
            )
            curr_batch['zero_rewards'] = self.make_zero_reward(action_log_probs)
            func = curr_reward
            args = (self._to_cpu(curr_batch),)

            if isinstance(
                curr_reward,
                Reward,
            ) or isinstance(curr_reward, InferenceRewardModel):
                if isinstance(curr_reward, InferenceRewardModel):
                    func = self.call_reward_model
                    args = (
                        self.all_rewards[reward_name],
                        self._to_cpu(curr_batch),
                    )

                assert self.pool is not None
                computed_rewards[reward_name] = self.pool.apply_async(
                    func=func,
                    args=args,
                )
            elif isinstance(curr_reward, RewardModel):
                computed_rewards[reward_name] = self.call_reward_model(
                    self.all_rewards[reward_name],
                    curr_batch,
                )
            else:
                raise TypeError(
                    f'Unknown reward model type {type(curr_reward)}. Expected `Reward` or `RewardModel`.',
                )

        batch['zero_rewards'] = self.make_zero_reward(action_log_probs)

        # convert all AsyncResult objects to tensors because ray cannot return Pool objects
        for reward_name, subreward in computed_rewards.items():
            if isinstance(subreward, AsyncResult):
                computed_rewards[reward_name] = subreward.get()
            else:
                computed_rewards[reward_name] = subreward
        return computed_rewards

    def _create_batch(
        self,
        reward_model: BaseReward,
        reward_name: str,
        processed_inputs: dict[str, Any],
        base_batch: dict[str, Any],
        raw_untokenized_texts: list[tuple[str, str]],
    ) -> dict[str, Any]:
        """Helper to get the callable and the input kwargs for the reward.

        Args:
            reward_model (BaseReward): the reward model to create the batch for.
            reward_name (str): the name of the reward to create the batch for.
            processed_inputs (dict): the processed inputs for the reward, based on granularity.
            base_batch (dict): the base batch to add the reward inputs to.
            raw_untokenized_texts (list): the raw untokenized texts.
        """
        if isinstance(reward_model, Reward):
            return {
                **base_batch,
                'raw_untokenized_texts': raw_untokenized_texts,
            }
        elif isinstance(reward_model, RewardModel):
            granularity = self.granularities[reward_name]
            curr_inputs = processed_inputs['end_reward_inputs_dict'][granularity
                                                                    ]
            tok_formatted_reward_inputs = torch.tensor(
                curr_inputs.input_ids,
            ).type(base_batch['input_ids'].dtype)
            tok_formatted_reward_attn_masks = torch.tensor(
                curr_inputs.attention_mask,
            ).type(base_batch['attention_mask'].dtype)

            return {
                'tok_formatted_reward_inputs':
                    tok_formatted_reward_inputs,
                'tok_formatted_reward_attn_masks':
                    tok_formatted_reward_attn_masks,
                'reward_seq_lens':
                    processed_inputs['reward_seq_lens_dict'][granularity],
                'reward_prompt_lens':
                    processed_inputs['reward_prompt_lens_dict'][granularity],
                'reward_generated_lens':
                    processed_inputs['reward_generated_lens_dict'][granularity],
                'end_idxs_gather':
                    processed_inputs['end_idxs_gather_dict'][granularity],
                'end_idxs_scatter':
                    processed_inputs['end_idxs_scatter_dict'][granularity],
                'prompt_lens':
                    base_batch['prompt_len'],
                'generated_lens':
                    base_batch['generated_lens'],
                'seq_lens':
                    base_batch['seq_lens'],
            }
        else:
            raise TypeError(
                f'Unknown reward model type {type(reward_model)}. Expected `Reward` or `RewardModel`.',
            )

    def resolve_outputs(
        self,
        ref_output: ReferenceOutput,
        reward_output: RewardOutput,
        kl_ctl: BaseKLController,
        action_mask: torch.Tensor,
        center_reward_mean: Optional[float] = None,
    ) -> dict[str, torch.Tensor]:
        """Resolve async results and finalize reward dict.

        Note: This method will wait for any AsyncResults to finish, so the associated async
        calls become blocking once this method is called. This method is separated from
        the __call__ method to make it easier to perform this (potentially) blocking step
        as long after __call__ as possible (ie, to best leverage the async setup).

        Args:
            ref_output (ReferenceOutput): The first output of the __call__ method.
                The ReferenceOutput tuple has two elements: the reference KL and
                the reference log probs, in that order.
            reward_output (RewardOutput): The second output of the __call__ method.
            kl_ctl (BaseKLController): KL controller object that provides the
                coefficient of the KL penalty in the aggregate reward.
            action_mask (Tensor): A mask tensor indicating which action tokens
                are padding.
            center_reward_mean (float, optional): An offset to subtract from the
                aggregate environment rewards (subtracted before the KL penalty is
                added). Default: no offset is subtracted.

        Returns:
            outputs: a dictionary capturing all the reward outputs, including the
                aggregate reward for RL training, as well as the rewards from each
                reward model.
        """
        device = action_mask.device

        # Resolve any output elements that are being computed async,
        # waiting for them to finish where necessary.
        resolved_reward_outputs: dict[str, torch.Tensor] = {}
        bad_end_generation_mask = None
        bad_end_generation_name = None
        for name, subreward in reward_output.items():
            if isinstance(subreward, AsyncResult):
                resolved_reward: torch.Tensor = subreward.get()
            else:
                resolved_reward: torch.Tensor = subreward
            resolved_reward_outputs[name] = resolved_reward.to(device=device)
            if isinstance(self.all_rewards[name], BadGenerationEndReward):
                bad_end_generation_name = name
                bad_generation_row_mask = torch.any(resolved_reward != 0, dim=1)

                bad_end_generation_mask = (
                    ~bad_generation_row_mask
                ).unsqueeze(1).expand_as(resolved_reward)
                bad_end_generation_mask = bad_end_generation_mask.to(
                    device=device,
                )

        ref_kl = ref_output[0].to(device=device)
        ref_log_probs = ref_output[1].to(device=device)

        if self.kl_penalty_in_reward:
            rewards: torch.Tensor = -kl_ctl.value * ref_kl.detach()
        else:
            rewards: torch.Tensor = torch.zeros_like(ref_kl)

        env_rewards = self.make_zero_reward(rewards)

        rews_dict_out: dict[str, torch.Tensor] = {}
        for name, subreward in resolved_reward_outputs.items():
            if name not in self.reward_coefficients:
                raise KeyError(
                    f'Reward with {name=} is not recognized by the reward manager.',
                )
            env_rewards += subreward.detach() * self.reward_coefficients[name]

            # In the output, make sure each key has 'reward' in it to engage
            # proper logging (see .loss of policy class)
            out_name = name + '_reward' if 'reward' not in name else ''
            rews_dict_out[out_name] = subreward.detach() * action_mask

        # Masking out all rewards if the generation ends with a bad token
        # And strictly adding a penalty for bad generation ending.
        if bad_end_generation_mask is not None and bad_end_generation_name is not None:
            env_rewards *= bad_end_generation_mask
            env_rewards += (
                resolved_reward_outputs[bad_end_generation_name].detach() *
                self.reward_coefficients[bad_end_generation_name]
            )

        # Optionally apply an offset to the environment rewards
        if center_reward_mean is not None:
            env_rewards -= center_reward_mean

        # Final rewards is total env rewards + KL penalties
        rewards += env_rewards

        # Zero rewards at padded tokens
        rewards *= action_mask
        env_rewards *= action_mask

        outputs = {
            'rewards': rewards.detach(),
            'env_rewards': env_rewards.detach(),
            'ift_log_probs': ref_log_probs.detach(),
            'ift_kl': ref_kl.detach(),
        }
        outputs.update(rews_dict_out)

        return outputs


class RolloutAgent:
    """Rollout agent for generating sequences from the inference server."""

    def __init__(
        self,
        inference_server: InferenceServer,
        streaming_dataset_actor: StreamingDatasetActor,
        reward_actor: RewardActor,
    ):
        self.inference_server = inference_server
        self.streaming_dataset_actor = streaming_dataset_actor
        self.reward_actor = reward_actor

        self.max_gen_len = _MAX_GEN_LEN
        self.max_seq_len = _MAX_SEQ_LEN
        self.pretrain_model_name = 'meta-llama/Llama-3.1-8B-Instruct'
        self.tokenizer_config = {
            'padding': 'longest',
            'pad_token': '<|finetune_right_pad_id|>',
            'truncation': True,
            'padding_side': 'left',
            'model_max_length': self.max_seq_len,
            'trust_remote_code': True,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrain_model_name, **self.tokenizer_config)
        self.generation_kwargs = {
            'top_p': 1.0,
            'use_cache': True,
            'do_sample': False,
            'temperature': 1.0,
        }
        self.precision = 'amp_bf16'
        self.tokenizer_pad_token_id = ray.get(self.streaming_dataset_actor.get_tokenizer_pad_token_id.remote())
        self.prompt_handler_config = ray.get(self.streaming_dataset_actor.get_prompt_handler_config.remote())
        self.max_gen_len = self.prompt_handler_config['max_gen_len']

    def get_next_iter_rollouts(self):
        """
        Gets the next rollouts from the inference server.

        Since all ranks should see different data, we need to get the rollouts for each rank.
        """
        iter_data = ray.get(self.streaming_dataset_actor.get_next_iter_prompts.remote())
        all_prompts = iter_data['prompt']
        # TODO: Since this functionality is (somewhat) shared across the OnPolicyCallback and the RolloutAgent,
        # we should move this to the separate util file.
        with get_precision_context(self.precision), torch.no_grad():
            sequences = _vllm_generate(
                vllm_engines=self.inference_server.engines,
                max_gen_len=self.max_gen_len,
                generation_kwargs=self.generation_kwargs,
                pad_token_id=self.tokenizer_pad_token_id,
                all_prompts=all_prompts,
                batch_sizes=[len(all_prompts)],
            )

        sequences = sequences[0]
        max_vllm_generated_len = max([len(response) for response in sequences])
        padded_responses = []
        for sequence in sequences:
            sequence = list(sequence)
            if len(sequence) < max_vllm_generated_len:
                sequence = sequence + [self.tokenizer_pad_token_id] * (max_vllm_generated_len - len(sequence))
            padded_responses.append(sequence)

        padded_responses = torch.tensor(
            padded_responses,
            dtype=all_prompts.dtype,
            device=torch.device('cpu'),
        )

        processed_sequences = torch.cat([all_prompts, padded_responses], dim=-1)
        iter_data['sequences'] = processed_sequences

        # Calculate the rewards here
        # Initialize the required variables from the reward actor
        tokenizer = self.tokenizer
        max_gen_len = self.max_gen_len
        eos_token_ids = [
                128001,
                128008,
                128009,
            ]


        # NOTE: Borrowing the right snippets from env_rewards to create inputs for RewardActor
        batch = iter_data
        prompt_tokens = batch['prompt']
        batch_size, _ = prompt_tokens.shape
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError(
                'Tokenizer does not have a pad token id. Please use a different tokenizer or add a pad token id.',
            )
        prompt_len = batch['prompt_len']
        verified_answers = batch.get('verified_answer', None)
        prompt_id = batch['prompt_id']
        cur_device = prompt_tokens.device
        prompt_dtype = prompt_tokens.dtype

        assert 'sequences' in batch, f'sequences is not in batch {batch.keys()=}'

        sequences = batch['sequences']
        generated_len = torch.ones(
            batch_size,
            device=cur_device,
            dtype=prompt_dtype,
        ) * max_gen_len

        # If all the processes early exit generate, then we need to manually pad everything
        # we can pad this with pad tokens, since we switch the padding between left and right
        # padding based on the sequence length + max_sequence_length.
        if prompt_tokens.size(1) + max_gen_len > sequences.size(1):
            len_to_pad = max_gen_len - (
                sequences.size(1) - prompt_tokens.size(1)
            )

            extra_padding = torch.ones(
                (batch_size, len_to_pad),
                device=cur_device,
                dtype=prompt_dtype,
            ) * pad_token_id
            sequences = torch.cat(
                [sequences, extra_padding],  # type: ignore
                dim=-1,  # type: ignore
            )

        # Sanity checking we're adding max_gen_len to prompt_tokens
        if prompt_tokens.size(1) + max_gen_len != sequences.size(1):
            raise ValueError(
                f'Prompts {prompt_tokens.size(1)} + max_gen_len {max_gen_len} != sequences {sequences.size(1)}',
            )

        # Actions are what tokens the current policy would generate.
        actions = sequences[:, -max_gen_len:]

        right_padded_obs = switch_left_to_right_padding(
            sequences,
            prompt_len,
            max_gen_len,
            pad_token_id,  # type: ignore
        )
        right_padded_attn_mask = torch.logical_not(
            torch.eq(right_padded_obs, pad_token_id),  # type: ignore
        )

        (
            right_padded_obs,
            right_padded_attn_mask,
            generated_len,
            action_mask,
        ) = mask_eos(
            actions=actions,
            right_padded_obs=right_padded_obs,
            right_padded_attn_mask=right_padded_attn_mask,
            prompt_len=prompt_len,
            generated_len=generated_len,
            max_gen_len=max_gen_len,
            eos_token_ids=eos_token_ids,  # type: ignore
            pad_token=pad_token_id,  # type: ignore
        )

        untokenized_prompt_and_responses = []
        for i in range(batch_size):
            prompt = tokenizer.decode(  # type: ignore
                right_padded_obs[i, :prompt_len[i]])
            generated_text = tokenizer.decode(  # type:  ignore
                get_decoded_sequence(actions[i], generated_len[i],
                                            max_gen_len))
            untokenized_prompt_and_responses.append((prompt, generated_text),)
        # Future implementations may change the way reward_seq_len is defined
        # e.g., if special formatting is applied
        reward_seq_len = prompt_len + generated_len

        # TODO: Don't think reward_manager is actually using action log probs... its likely just a tensor for shape management
        dummy_log_probs = torch.zeros_like(
            actions,
            dtype=torch.float32,
        )
        all_rewards = ray.get(self.reward_actor.calculate_reward.remote(
            raw_untokenized_texts=untokenized_prompt_and_responses,
            right_padded_obses=right_padded_obs,
            attention_masks=right_padded_attn_mask,
            seq_lens=reward_seq_len,
            generated_lens=generated_len,
            prompt_lens=prompt_len,
            max_gen_length=max_gen_len,
            actions=actions,
            action_log_probs=dummy_log_probs,
            verified_answers=verified_answers,
        ))
        all_rewards_dict = all_rewards
        prompts_and_gens = untokenized_prompt_and_responses

        # Shove all the necessary info in the iter_data for custom handling later
        iter_data["all_rewards_dict"] = all_rewards_dict
        print(f'Rollout agent generated {len(prompts_and_gens)} rollouts')
        print(f'With {len(all_rewards_dict)} rewards containing {list(all_rewards_dict.keys())} keys')
        return iter_data


class PPOController:
    """PPO controller for training the policy and value networks."""

    def __init__(
        self,
        train_actor: TrainActorGroup,
        inference_server: InferenceServer,
        rollout_agent: RolloutAgent,
        parameter_buffer: ParameterBuffer,
        experience_buffer: ExperienceBuffer,
        pretrain_model_name: str,
    ):
        self.train_actor = train_actor
        self.inference_server = inference_server
        self.rollout_agent = rollout_agent
        self.parameter_buffer = parameter_buffer
        self.experience_buffer = experience_buffer
        self.train_actor.build_models(pretrain_model_name)
        setup_process_groups(
            self.train_actor.master_actor,
            inference_server.engines,
            inference_server.vllm_tensor_parallel_size,
        )

    def train(self):
        for _ in range(5):  # Example: train for 5 iterations
            # NOTE: this loop is represents the logic happening in the current `iteration_start` of the OnPolicyCallback
            self.parameter_buffer.put({'actor_group': self.train_actor, 'inference_server': self.inference_server})
            # Simple example of adding elements to the experience buffer
            self.experience_buffer.put(self.rollout_agent.get_next_iter_rollouts())
            # Populate the train actor group with the rollouts and then train
            self.train_actor.add_latest_rollouts_from_buffer(self.experience_buffer)
            self.train_actor.collective_methods.train_1_iter()


def _run_single_controller_ppo(
    config: Any,
):
    """Shared function for running single controller PPO.

    Args:
        config: OmegaConf configuration object containing all parameters
    """
    # Set vLLM attention backend to FLASH_ATTN otherwise FlashInfer backend
    # takes too long to jit compile
    os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'

    with start_ray_server() as _address:
        # only rank 0 is the master controller
        if dist.get_rank() == 0:
            world_size = getattr(config, "world_size", 0)
            if world_size == 0:
                world_size = dist.get_world_size()

            # Create buffers for the parameter and experience buffers
            # first since they don't have external dependencies
            parameter_buffer = ParameterBuffer()
            experience_buffer = ExperienceBuffer()

            # create SPMD training actors of the system
            num_train_actors = world_size // 2
            train_actor = TrainActorGroup(num_train_actors, DistributedGPUActor)

            # Create vLLM engines (or inference actors)
            vllm_tensor_parallel_size = world_size - num_train_actors
            num_vllm_engines = (
                world_size - num_train_actors
            ) // vllm_tensor_parallel_size
            # TODO: Encapsulate this into a inference server manager class
            pretrain_model_name = config.pretrain_model_name
            inference_server = InferenceServer(
                num_vllm_engines=num_vllm_engines,
                vllm_tensor_parallel_size=vllm_tensor_parallel_size,
                pretrain_model_name=pretrain_model_name,
            )

            # We are using a CPU worker for the StreamingActor
            # and this involves a super hacky workaround by
            # uninstalling megablocks if it exists. Better solutions
            # would include:
            # 1) decouple StreamingActor from llm-foundry altogether
            # 2) don't broadly import llm-foundry in compose-rl (only
            # import it into codepaths/files that will only be used by
            # GPUActors as opposed to CPUActors)
            # 3) Setting up ray actors with correct environments (which
            # would involve creating a BaseDistributedActor instead of a
            # BaseDistributedGPUActor so that we can use CPUs)
            # We uninstall megablocks after the Train Actors have been
            # created so that those actors still have megablocks functionality.
            uninstall_megablocks_if_exists()
            streaming_dataset_actor = ray.remote(num_gpus=0)(StreamingDatasetActor).remote()
            reward_actor = ray.remote(num_gpus=0)(RewardActor).remote()
            rollout_agent = RolloutAgent(inference_server, streaming_dataset_actor, reward_actor)

            ppo_controller = PPOController(
                train_actor,
                inference_server,
                rollout_agent,
                parameter_buffer,
                experience_buffer,
                pretrain_model_name,
            )
            ppo_controller.train()


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run single controller PPO with configuration file')
    parser.add_argument('--file_path', type=str, required=False, default=None,
                       help='Path to the OmegaConf YAML configuration file')
    args = parser.parse_args()
    
    # Load configuration using OmegaConf
    if args.file_path is not None:
        config = om.load(args.file_path)
    else:
        config = om.create({
            'pretrain_model_name': 'meta-llama/Llama-3.1-8B-Instruct',
        })
    
    # This is an example of how to move the controller logic from PPO Callback
    # to a separate trainer actor above and this main single controller
    # function.
    _run_single_controller_ppo(config)
