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
import asyncio
import contextlib
import copy
from contextlib import contextmanager
import logging
import os
import pickle
import socket
import sys
import time
import datetime
from itertools import chain
from functools import partial
from typing import Any, Optional, Union, MutableMapping
from multiprocessing import get_context
from multiprocessing.context import TimeoutError as MultiprocessingTimeoutError
from multiprocessing.pool import AsyncResult, Pool

from composer.loggers import MLFlowLogger
import requests
import torch
from composer import Trainer
from composer.core import get_precision_context, Precision
from composer.core.data_spec import _default_split_batch
from composer.trainer.trainer import _get_initial_device_train_microbatch_size
from compose_rl.data.buffer import MinibatchRolloutBuffer
from composer.optim import DecoupledAdamW
from composer.utils import dist
from llmfoundry.data import build_dataloader
from llmfoundry.utils import build_composer_model
from llmfoundry.utils.config_utils import process_init_device  # type: ignore
from omegaconf import DictConfig, OmegaConf as om
from transformers import AutoTokenizer
from composer.callbacks import MemoryMonitor, SpeedMonitor, LRMonitor

from compose_rl.registry_builders import build_kl_controller
from compose_rl.algorithms.online import (
    ComposerHFPolicyLM,
    ComposerHFCriticFreePolicyLM,
    SingleControllerOnPolicyCallback,
)
from orl_servers.generation_utils import (
    broadcast_to_vllm,
    vllm_generate_sync,
)
from orl_servers.vllm_remote import RemoteVLLMEngine
from orl_servers.structs import InferenceEngineConfig, WeightUpdateMeta
from orl_servers.async_utils import run_async_sync
from orl_servers.client import ArealOpenAI
from compose_rl.utils.ray_utils import start_ray_server, uninstall_megablocks_if_exists
from compose_rl.controllers import BaseDistributedGPUActor, SPMDActorGroup
import subprocess
import signal
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
    dist_compute_masked_mean_and_var,
    get_log_probs,
    get_entropies,
    scatter_gather_rewards,
    switch_left_to_right_padding,
    mask_eos,
    masked_sum,
    masked_mean,
    get_decoded_sequence,
)
from compose_rl.algorithms.online.reward_manager import (
    ReferenceOutput,
    RewardOutput,
)

from compose_rl.algorithms.online.model_methods import OnPolicyEnum

from orl_servers.vllm_worker_wrap import stateless_init_process_group
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from orl_servers.http_utils import arequest_with_retry

log = logging.getLogger(__name__)


@contextmanager
def time_it(name: str):
    start_time = time.time()
    pst_start_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-8)))
    log.info(f"[{name}] started at {pst_start_time.strftime('%Y-%m-%d %H:%M PST')}")
    yield
    end_time = time.time()
    pst_end_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-8)))
    log.info(f"[{name}] finished at {pst_end_time.strftime('%Y-%m-%d %H:%M PST')}")
    log.info(f"[{name}] took {end_time - start_time:.2f} seconds")


def get_and_validate_num_prompts_per_iteration(config: Any):
    generations_per_prompt = config.variables.generations_per_prompt
    num_batches_per_update = config.variables.num_batches_per_update
    total_num_generations = config.global_train_batch_size * num_batches_per_update
    num_prompts_per_iteration = total_num_generations // generations_per_prompt

    assert total_num_generations % generations_per_prompt == 0, "total_num_generations must be divisible by generations_per_prompt"

    return num_prompts_per_iteration


async def launch_vllm_servers(
    pretrain_model_name: str,
    tensor_parallel_size: int,
    data_parallel_size: int,
    num_vllm_servers: int,
    num_train_actors: int,
    max_model_len: int,
    enable_prefix_caching: bool,
    use_existing: bool = False,
) -> tuple[RemoteVLLMEngine, list[subprocess.Popen]]:
    """Launch multiple vLLM HTTP servers and return processes and a RemoteVLLMEngine.

    Servers are started on localhost with ports starting at 8000.
    Each server is assigned a disjoint set of GPUs based on training_world_size.
    """
    processes: list[subprocess.Popen] = []
    addresses: list[str] = []

    log.info(f'num_vllm_servers: {num_vllm_servers}, num_train_actors: {num_train_actors}, tensor_parallel_size: {tensor_parallel_size}, data_parallel_size: {data_parallel_size}')

    for server_idx in range(num_vllm_servers):
        env = os.environ.copy()
        gpu_start = num_train_actors + server_idx * tensor_parallel_size
        gpu_ids = list(range(gpu_start, gpu_start + tensor_parallel_size))
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

        port = 8000 + server_idx
        if not use_existing:
            cmd = [
                'orl-vllm-server',
                '--model', pretrain_model_name,
                '--worker-extension-cls', 'orl_servers.vllm_worker_wrap.WorkerWrap',
                '--max-model-len', str(max_model_len),
                '--tensor-parallel-size', str(tensor_parallel_size),
                '--data-parallel-size', str(data_parallel_size),
                '--seed', '1',
                '--enable-prefix-caching' if enable_prefix_caching else '--no-enable-prefix-caching',
                # '--enforce-eager',  # TODO: check if we need to enforce eager
                # '--disable-custom-all-reduce',  # A100 does not like it
                '--port', str(port),
                '--enable-log-requests',
            ]
            # Log to stdout and stderr
            p = subprocess.Popen(cmd, env=env)
            processes.append(p)
            print(' '.join(cmd))
        addresses.append(f'localhost:{port}')

    vllm_engine = RemoteVLLMEngine(
        config=InferenceEngineConfig(
            setup_timeout=180.0,
            request_timeout=300.0,
            request_retries=3,
        ),
        addresses=addresses,
    )
    log.info(f'Initializing vLLM engine for addresses: {addresses}')
    vllm_engine.initialize()
    log.info(f'Initialized vLLM engine')
    return vllm_engine, processes


def partition_rollouts_across_ranks(num_train_actors: int, rollouts: dict[str, Any]) -> list[dict[str, Any]]:
    """Partition the rollouts across all actors."""
    partitioned_rollouts = []
    per_rank_data_size = rollouts['prompt'].shape[0] // num_train_actors
    for i in range(num_train_actors):
        current_rank_start = i * per_rank_data_size
        current_rank_end = (i + 1) * per_rank_data_size
        current_rank_rollouts = {}
        for k, v in rollouts.items():
            if isinstance(v, torch.Tensor) or isinstance(v, list):
                sliced_v = v[current_rank_start:current_rank_end]
                if isinstance(sliced_v, torch.Tensor):
                    sliced_v = sliced_v.tolist()
                current_rank_rollouts[k] = sliced_v
            elif isinstance(v, dict):
                # This is the case with the rewards dict where it has (key, tensor) pairs
                rewards_dict_for_rank = {}
                for reward_key, reward_tensor in v.items():
                    sliced_reward_tensor = reward_tensor[current_rank_start:current_rank_end]
                    if isinstance(sliced_reward_tensor, torch.Tensor):
                        sliced_reward_tensor = sliced_reward_tensor.tolist()
                    rewards_dict_for_rank[reward_key] = sliced_reward_tensor
                current_rank_rollouts[k] = rewards_dict_for_rank
            else:
                raise ValueError(f"Expected a tensor or list or dict of tensors, got {type(v)}")
        partitioned_rollouts.append(current_rank_rollouts)
    return partitioned_rollouts


@contextlib.contextmanager
def _patch_env(**environs: str):
    """Returns a context manager that patches ``os.environ`` with ``environs``.

    The original ``os.environ`` values are restored at the end.
    """
    # Adapted loosely from https://stackoverflow.com/a/34333710
    # Capture the original environ values
    original_environs = {k: os.environ.get(k) for k in environs}

    # Patch the environment
    for k, v in environs.items():
        os.environ[k] = v
    try:
        # Run the context manager
        yield
    finally:
        # Restore the original environ values
        for k, v in original_environs.items():
            if v is None:
                del os.environ[k]
            else:
                os.environ[k] = v

# Note: This needs to be re-worked once the repos are migrated.
class EvalAgent:
    """An async agent for handling evals."""

    def __init__(
        self,
        vllm_engine: RemoteVLLMEngine,
        config: Any,
    ):
        self.vllm_engine = vllm_engine
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name, **config.tokenizer.kwargs)
        self.vllm_client = ArealOpenAI(vllm_engine, self.tokenizer)
        self.config = config

        # Variables from the config used in the eval_agent
        # TODO: Support eval_interval_num in a more generic way (e.g. handle more than just `iter`)
        self.eval_interval_num = int(config.eval_interval.strip("iter"))
        self.num_batches_per_update = config.variables.num_batches_per_update
        self.experiment_name = config.loggers.mlflow.experiment_name
        self.run_name = config.loggers.mlflow.tags.run

        self.callback = self.build_callback()

    def build_callback(self):
        from llmfoundry.utils.builders import build_callback
        # Creating the evals and eval_overrides from the config.
        kwargs = om.to_container(self.config.callbacks.orl_eval, resolve=True)
        # Using a minimal (fake) train_config to built the callback correctly.
        # The setup actually doesn't matter as we just want to expose the
        # run_evaluation function to this eval agent.
        fake_train_config = {
            'eval_interval': f'{self.eval_interval_num}iter',
            'python_log_level': 'debug',
        }
        callback = build_callback(
            name='orl_eval',
            kwargs=kwargs,
            train_config=fake_train_config,
        )
        # Need to create a fake state to pass to fit_start to help the callback register correctly.
        class _State:
            vllm_client = None
        fake_state = _State()
        fake_state.vllm_client = self.vllm_client
        # fit_start needs to be called to allow us to call _run_evaluation
        callback.fit_start(fake_state, logger=None)
        return callback

    def run_evaluation(self, step: int = 0):
        """Run evaluation after weights are broadcast to vLLM engines."""
        # _run_evaluation requires that mlflow_logger is not None (even though it is not used)
        # As a consequence, we set it to 1 (as a placeholder) to circumvent the issue.
        self.callback.mlflow_logger = 1
        with time_it("run_evaluation"):
            self.callback._run_evaluation(self.experiment_name, self.run_name, step)

    async def run(self, num_iterations: int, lock: asyncio.Lock, eval_semaphore: asyncio.Semaphore):
        """Async loop on driver to trigger evaluations.

        We don't need to treat this as a Ray actor since we don't need to set a world_size or
        use GPUs for this process.
        """
        # TODO: We could potentially use an async queue instead of a semaphore to trigger the eval
        # We could potentially circumvent this iteration loop in that scenario.
        for iteration in range(0, num_iterations, self.eval_interval_num):
            await eval_semaphore.acquire()
            async with lock:
                await asyncio.to_thread(self.run_evaluation, step=iteration*self.num_batches_per_update)


class StreamingDatasetActor:
    """Streaming actor for loading prompts onto the experience buffer."""

    def __init__(self, config: Any):

        # Setting up all of the configs
        # TODO: We should move these to dataclasses
        # TODO: In a future PR, create all configs in the main function and populate
        # the correct configs across all entities (e.g. DistributedGPUActor, StreamingDatasetActor, etc)
        self.pretrain_model_name = config.model.pretrained_model_name_or_path
        self.prompt_handler_config = {
            'global_train_batch_size': config.global_train_batch_size,
            'generations_per_prompt': config.variables.generations_per_prompt,
            'num_batches_per_update': config.variables.num_batches_per_update,
            'max_seq_len': config.max_seq_len,
            'max_gen_len': config.variables.max_gen_len,
        }
        self.tokenizer_config = config.tokenizer.kwargs
        self.dataloader_config = config.train_loader

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dataloader_config['dataset']['local'] = \
            self.dataloader_config['dataset']['local'].format(timestamp=timestamp)

        # Key variables
        self.generations_per_prompt = config.variables.generations_per_prompt
        self.num_prompts_per_iteration = get_and_validate_num_prompts_per_iteration(config)

        # Creating main entities
        log.info(f'Building tokenizer')
        self.tokenizer = self._build_tokenizer()
        log.info(f'Building dataloader')
        self.dataloader = self._build_dataloader()
        log.info(f'Building dataloader iterator')
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

    def get_tokenizer(self):
        return self.tokenizer

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

    def get_dataloader_state_dict(self):
        return self.dataloader.state_dict()
    
    def load_dataloader_state_dict(self, state_dict: dict):
        self.dataloader.load_state_dict(state_dict)


class RewardActor:
    """Streaming actor for adding rewards on top the experience buffer."""

    def __init__(self, config: Any):
        self.max_seq_len = config.max_seq_len

        self.reward_config = om.to_container(config.variables.rewards, resolve=True)
        self.all_rewards = {}

        tokenizer_config = om.to_container(config.tokenizer.kwargs, resolve=True)
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name, **tokenizer_config)
        for reward_name, reward_config in self.reward_config.items():
            assert isinstance(reward_name, str)
            if reward_name in self.all_rewards:
                raise KeyError(
                    f'The reward already has a model with {reward_name=}',
                )

            log.info(f'Initializing reward with name {reward_name}')

            reward_type = reward_config.pop('reward_type')
            reward_cls = rewards_registry.get(reward_type)
            assert issubclass(reward_cls, Reward)
            model = build_reward(
                name=reward_type,
                tokenizer=tokenizer,
                kwargs=reward_config,
            )

            self.all_rewards[reward_name] = model

        self.pool = None
        self._num_reward_procs = len(self.all_rewards)
        # TODO: evaluate whether multiprocessing or Raylet group is better for the reward actor
        self.pool = Pool(
            processes=self._num_reward_procs,
            context=get_context('spawn'),
        )

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

        for reward_name, curr_reward in self.all_rewards.items():
            curr_reward = self.all_rewards[reward_name]
            curr_batch = {
                **batch,
                'raw_untokenized_texts': raw_untokenized_texts,
            }
            curr_batch['zero_rewards'] = torch.zeros_like(action_log_probs)

            func = curr_reward
            args = (self._to_cpu(curr_batch),)

            computed_rewards[reward_name] = self.pool.apply_async(
                func=func,
                args=args,
            )

        # convert all AsyncResult objects to tensors because ray cannot return Pool objects
        encountered_timeout = False
        for reward_name, subreward in computed_rewards.items():
            if isinstance(subreward, AsyncResult):
                # computed_rewards[reward_name] = subreward.get()
                try:
                    computed_rewards[reward_name] = subreward.get(
                        timeout=self.all_rewards[reward_name].BLOCKING_TIMEOUT,
                    )
                except (TimeoutError, MultiprocessingTimeoutError):
                    encountered_timeout = True
                    log.error(
                        f'Timeout while waiting for {reward_name} reward to finish. ' +
                        'This may indicate a problem with the reward. Using a default reward of 0.',
                    )
                    computed_rewards[reward_name] = torch.zeros_like(action_log_probs).to(
                        torch.float32,
                    )
            else:
                computed_rewards[reward_name] = subreward

        # Rather than trying to signal to the stuck process, or do anything more careful,
        # since we know we are fully done with reward computation, we can just recreate the pool
        # to ensure that all processes are cleaned up and we can continue without resources leaking.
        if encountered_timeout and self.pool is not None:
            self.pool.terminate()
            self.pool.join()

            self.pool = Pool(
                processes=self._num_reward_procs,
                context=get_context('spawn'),
            )

        return computed_rewards


class RolloutAgent:
    """Rollout agent for generating sequences from the inference server."""

    def __init__(
        self,
        vllm_engine: RemoteVLLMEngine,
        streaming_dataset_actor: StreamingDatasetActor,
        reward_actor: RewardActor,
        config: Any,
    ):
        # Actors
        self.vllm_engine = vllm_engine
        self.streaming_dataset_actor = streaming_dataset_actor
        self.reward_actor = reward_actor

        self.generation_kwargs = config.variables.generation_kwargs
        self.precision = config.precision

        self.tokenizer = self.streaming_dataset_actor.get_tokenizer()
        self.tokenizer_pad_token_id = self.streaming_dataset_actor.get_tokenizer_pad_token_id()
        if self.tokenizer_pad_token_id is None:
            raise ValueError(
                'Tokenizer does not have a pad token id. Please use a different tokenizer or add a pad token id.',
            )

        self.prompt_handler_config = self.streaming_dataset_actor.get_prompt_handler_config()
        self.max_gen_len = self.prompt_handler_config['max_gen_len']

        # TODO: get from config
        self.eos_token_ids = [
                128001,
                128008,
                128009,
            ]

        # Load iter_num from the checkpoint
        self.save_folder = os.path.join(config.save_folder, 'RolloutAgent')

        self.iter_num = 0

        # Load the latest checkpoint
        self.latest_checkpoint = os.path.join(self.save_folder, 'latest.symlink')

        if config.autoresume and os.path.exists(self.latest_checkpoint):
            log.info(f'Autoresuming from checkpoint for RolloutAgent.')
            with open(self.latest_checkpoint, 'rb') as f:
                checkpoint = pickle.load(f)
            self.iter_num = checkpoint['iter_num']
            log.info(f'Loading streaming dataloader state dict for RolloutAgent.', checkpoint['streaming_dataloader'])
            self.streaming_dataset_actor.load_dataloader_state_dict(checkpoint['streaming_dataloader'])

    def get_next_iter_rollouts(self):
        """
        Gets the next rollouts from the inference server.

        Since all ranks should see different data, we need to get the rollouts for each rank.
        """
        iter_data = self.streaming_dataset_actor.get_next_iter_prompts()
        all_prompts = iter_data['prompt']
        # TODO: Since this functionality is (somewhat) shared across the OnPolicyCallback and the RolloutAgent,
        # we should move this to the separate util file.
        with get_precision_context(self.precision), torch.no_grad():
            with time_it("rollout"):
                sequences, vllm_logprobs = vllm_generate_sync(
                    remote_engine=self.vllm_engine,
                    max_gen_len=self.max_gen_len,
                    generation_kwargs=self.generation_kwargs,
                    pad_token_id=self.tokenizer_pad_token_id,
                    all_prompts=all_prompts,
                    batch_sizes=[len(all_prompts)],
                )

        sequences = sequences[0]
        vllm_logprobs = vllm_logprobs[0]

        max_vllm_generated_len = max([len(response) for response in sequences])
        
        # TODO: clean this up since this padded_response and padded_log_probs share similarity with the generation_utils.py
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

        padded_logprobs = []
        for logprobs in vllm_logprobs:
            logprobs = list(logprobs)
            if len(logprobs) < max_vllm_generated_len:
                logprobs = logprobs + [0] * (max_vllm_generated_len - len(logprobs))
            padded_logprobs.append(logprobs)

        padded_logprobs = torch.tensor(
            padded_logprobs,
            dtype=torch.float,
            device=torch.device('cpu'),
        )


        temp_zeros = torch.zeros_like(all_prompts, dtype=torch.float, device=torch.device('cpu'))
        processed_logprobs = torch.cat([temp_zeros, padded_logprobs], dim=-1)
        iter_data['vllm_logprobs'] = processed_logprobs
        assert processed_logprobs.shape == processed_sequences.shape, f'vllm_logprobs and sequences have different shapes {processed_logprobs.shape=}, {processed_sequences.shape=}'


        # Calculate the rewards here
        # Initialize the required variables from the reward actor
        tokenizer = self.tokenizer
        max_gen_len = self.max_gen_len
        eos_token_ids = self.eos_token_ids
        pad_token_id = self.tokenizer_pad_token_id

        # NOTE: Borrowing the right snippets from env_rewards to create inputs for RewardActor
        prompt_tokens = iter_data['prompt']
        batch_size, _ = prompt_tokens.shape
        prompt_len = iter_data['prompt_len']
        verified_answers = iter_data.get('verified_answer', None)
        cur_device = prompt_tokens.device
        prompt_dtype = prompt_tokens.dtype

        assert 'sequences' in iter_data, f'sequences is not in iter_data {iter_data.keys()=}'

        sequences = iter_data['sequences']
        vllm_logprobs = iter_data['vllm_logprobs']
        generated_len = torch.ones(
            batch_size,
            device=cur_device,
            dtype=prompt_dtype,
        ) * max_gen_len

        # If all the processes early exit generate, then we need to manually pad everything
        # we can pad this with pad tokens, since we switch the padding between left and right
        # padding based on the sequence length + max_sequence_length.
        #TODO: check if this padding is needed? i assume so. 
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

            extra_zero_padding = torch.zeros(
                (batch_size, len_to_pad),
                device=cur_device,
                dtype=torch.float,
            )
            vllm_logprobs = torch.cat(
                [vllm_logprobs, extra_zero_padding],  # type: ignore
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
            _,
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
        # TODO: cleanup reward actor methods
        dummy_log_probs = torch.zeros_like(
            actions,
            dtype=torch.float32,
        )

        # TODO: we should parallelize reward_actor and vllm_generate
        with time_it("Calculating Rewards from Reward Actor"):
            all_rewards = self.reward_actor.calculate_reward(
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
            )
        all_rewards_dict = all_rewards
        prompts_and_gens = untokenized_prompt_and_responses

        # Shove all the necessary info in the iter_data for custom handling later
        iter_data["all_rewards_dict"] = all_rewards_dict
        log.info(f'Rollout agent generated {len(prompts_and_gens)} rollouts')
        log.info(f'With {len(all_rewards_dict)} rewards containing {list(all_rewards_dict.keys())} keys')

        # Checkpointing
        save_folder_iter = os.path.join(self.save_folder, f'iter_{self.iter_num}')
        checkpoint_path = os.path.join(save_folder_iter, 'checkpoint.pt')
        self.iter_num += 1

        streaming_dataloader_state_dict = self.streaming_dataset_actor.get_dataloader_state_dict()
        log.info(f'Streaming dataloader state dict for RolloutAgent.', streaming_dataloader_state_dict)

        # make sure that the folder path can exist
        os.makedirs(save_folder_iter, exist_ok=True)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'iter_data': iter_data,
                'iter_num': self.iter_num,
                'streaming_dataloader': streaming_dataloader_state_dict,
            }, f)

        if os.path.exists(self.latest_checkpoint):
            os.remove(self.latest_checkpoint)
        os.symlink(checkpoint_path, self.latest_checkpoint)
        return iter_data


async def _produce_rollouts(rollout_agent: RolloutAgent, num_train_actors: int, experience_buffer: asyncio.Queue, num_iterations: int):
    for _ in range(num_iterations):
        rollouts = await asyncio.to_thread(rollout_agent.get_next_iter_rollouts)
        partitioned_rollouts = partition_rollouts_across_ranks(num_train_actors, rollouts)

        await experience_buffer.put(partitioned_rollouts)


class TrainEngine:
    def __init__(self, num_train_actors: int):
        self.addresses = [
            f'localhost:{8500 + i}' for i in range(num_train_actors)
        ]
        self.setup_timeout = 60
    
    def _wait_for_server(self, address: str):
        base_url = f"http://{address}"
        tik = time.time()
        while time.time() - tik < self.setup_timeout:
            if self.check_health(base_url):
                return
            time.sleep(1)
        raise RuntimeError("server launch failed")

    def check_health(self, base_url: str):
        # Check server endpoint
        try:
            response = requests.get(f"{base_url}/health", timeout=30)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    async def initialize(self):
        log.info("Waiting for server ready...")
        for addr_ in self.addresses:
            log.info(f'Waiting for server {addr_} to be ready')
            self._wait_for_server(addr_)
            log.info(f'Server {addr_} is ready')
        log.info("Servers are all ready!")


    async def build_trainer(self, config: Any):
        if isinstance(config, DictConfig):
            # convert to dict
            config = om.to_container(config, resolve=True)
        await asyncio.gather(*[
            arequest_with_retry(
            addr,
            endpoint='/initialize',
                method='POST',
                payload={'config': config},
            )
            for addr in self.addresses
        ])
        return

    async def create_online_minibatches(self, all_rollouts: list[dict[str, Any]]):
        await asyncio.gather(*[
            arequest_with_retry(
            addr,
            endpoint='/create_online_minibatches',
            method='POST',
            payload={'current_rank_rollouts': all_rollouts[i]},
            )
        for i, addr in enumerate(self.addresses)])
        return

    async def initialize_model_update_group(self, new_port: int, num_vllm_servers: int, gen_tp_size: int):
        await asyncio.gather(*[
            arequest_with_retry(
            addr,
            endpoint='/initialize_model_update_group',
            method='POST',
            payload={'new_port': new_port, 'num_vllm_servers': num_vllm_servers, 'gen_tp_size': gen_tp_size},
        )
        for addr in self.addresses])
        return

    async def broadcast_to_vllm(self, addresses: list[str]):
        await asyncio.gather(*[
            arequest_with_retry(
            addr,
            endpoint='/broadcast_to_vllm',
            method='POST',
            payload={'addresses': addresses},
        )
        for addr in self.addresses])
        return

    async def train_1_iter(self):
        await asyncio.gather(*[
            arequest_with_retry(
            addr,
            endpoint='/train_1_iter',
            method='POST',
            )
        for addr in self.addresses])
        return


async def _train(
            train_engine: TrainEngine, 
            experience_buffer: asyncio.Queue, 
            num_iterations: int, 
            vllm_engine: RemoteVLLMEngine,
):
    for i in range(num_iterations):
        log.info(f'Training iteration {i}')
        all_rollouts = await experience_buffer.get()
        await train_engine.create_online_minibatches(all_rollouts)
        await train_engine.train_1_iter()

        start_time = time.time()
        log.info('Before broadcast to vLLM')
        # TODO (infra) instead of direcly broadcasting to vllm, we should
        # push the model parameters to a parameter buffer manager and have
        # the buffer manager initiate broadcast of parameters to vllm engines
        # TODO fix this monkey patching
        await train_engine.broadcast_to_vllm(vllm_engine.addresses)
        log.info('Finished broadcasting to vLLM')
        log.info(f'Took: {time.time() - start_time} to broadcast to vllm.')
        log.info(f'Training iteration {i} completed')

async def _setup_process_groups(
    vllm_engine: RemoteVLLMEngine,
    train_engine: TrainEngine,
    gen_tp_size: int,
    num_vllm_servers: int,
):
    """Initialize trainer and vLLM servers' weight-update process group.

    This mirrors the logic used in test_single_controller_vllm.py by:
      - Getting a free TCP port from the master actor
      - Initializing the vLLM servers' NCCL communicators via HTTP
      - Adding a matching process group on the trainer side (rank 0)
    """

    new_port = 9000


    meta = WeightUpdateMeta(
        nccl_master_address="127.0.0.1",
        nccl_master_port=new_port,
        gen_tp_size=gen_tp_size,
        gen_world_size=num_vllm_servers * gen_tp_size,
    )
    # Initialize both sides concurrently
    await asyncio.gather(
        vllm_engine.ainit_weight_update_group(meta),
        train_engine.initialize_model_update_group(new_port, num_vllm_servers, gen_tp_size),
    )




async def launch_train_servers(num_train_actors: int) -> tuple[TrainEngine, list[subprocess.Popen]]:
    train_procs = []
    cmd = [
        'composer',
        '-n', str(num_train_actors),
        '--world_size', str(num_train_actors),
        'train_server.py',
    ]
    with open("train_server.out", "w") as outfile:
        p = subprocess.Popen(cmd, stdout=outfile, stderr=outfile)
    log.info(' '.join(cmd))
    train_procs.append(p)
    log.info(f'Started train servers {[p.pid for p in train_procs]}')

    train_engine = TrainEngine(num_train_actors)

    await train_engine.initialize()
    return train_engine, train_procs

async def _run_single_controller_ppo(
    config: Any,
    num_train_actors: int,
    num_vllm_servers: int,
):
    """Shared function for running single controller PPO.

    Args:
        config: OmegaConf configuration object containing all parameters
    """
    # only rank 0 is the master controller
    vllm_procs = []
    try:
        vllm_tensor_parallel_size = config.vllm_tensor_parallel_size
        enable_prefix_caching=config.vllm_enable_prefix_caching
        rollout_agent = None
        model_update_group = None
        log.info(f'Launching train servers')
        train_engine, train_procs = await launch_train_servers(num_train_actors)

        log.info(f'Setting up streaming dataset actor')
        streaming_dataset_actor = StreamingDatasetActor(config)
        # # create SPMD training actors of the system
        # log.info(f'Initilizing default training process group')
        # train_actor.init_composer_dist()
        # log.info(f'Initialized default training process group')


        # Launch vLLM server
        log.info(f'Building trainer')
        await train_engine.build_trainer(config)

        log.info(f'Launching vLLM servers')


        vllm_procs = []
        
        vllm_engine, vllm_procs = await launch_vllm_servers(
            pretrain_model_name=config.model.pretrained_model_name_or_path,
            tensor_parallel_size=vllm_tensor_parallel_size,
            data_parallel_size=1,
            num_vllm_servers=num_vllm_servers,
            num_train_actors=num_train_actors,
            max_model_len=config.max_seq_len,
            enable_prefix_caching=enable_prefix_caching,
            use_existing=False
        )
        log.info(f'Started vLLM servers {[vllm_proc.pid for vllm_proc in vllm_procs]}')

        log.info(f'Setting up reward actor')
        reward_actor = RewardActor(config)
        log.info(f'Setting up rollout agent')
        rollout_agent = RolloutAgent(vllm_engine, streaming_dataset_actor, reward_actor, config)

        log.info(f'Setting up process groups for weight_update')
        model_update_group = await _setup_process_groups(
            vllm_engine, train_engine, vllm_tensor_parallel_size, num_vllm_servers)


        
        num_prompts_per_iteration = get_and_validate_num_prompts_per_iteration(config)
        assert num_prompts_per_iteration % num_train_actors == 0, "Number of prompts per iteration must be divisible by number of train actors to ensure accurate advantage calculations."

        # num_iterations = int(config.max_duration.strip("iter"))
        num_iterations = 2
        experience_buffer = asyncio.Queue()
        await asyncio.gather(
            _produce_rollouts(rollout_agent, num_train_actors, experience_buffer, num_iterations),
            _train(train_engine, experience_buffer, num_iterations, vllm_engine),
        )
    except Exception as e:
        log.error(f'Error in _run_single_controller_ppo: {e}')
        raise e
    finally:
        log.info(f'Shutting down vLLM servers {[vllm_proc.pid for vllm_proc in vllm_procs]}')
        for vllm_proc in vllm_procs:
            vllm_proc.send_signal(signal.SIGTERM)
            vllm_proc.wait(timeout=10)  # Wait up to 10 seconds for graceful shutdown

        for train_proc in train_procs:
            train_proc.send_signal(signal.SIGTERM)
            train_proc.wait(timeout=10)  # Wait up to 10 seconds for graceful shutdown


if __name__ == '__main__':
    # Parse command line arguments

    logging.basicConfig(
        # Example of format string
        # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: Message here
        format=
        f'%(asctime)s: rank{dist.get_global_rank()}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s',
        force=True,
    )
    logging.getLogger(__name__).setLevel(
        'INFO',
    )  # Train script

    parser = argparse.ArgumentParser(description='Run single controller PPO with configuration file')
    parser.add_argument(
        '--file_path',
        type=str,
        required=False,
        default=None,
        help='Path to the OmegaConf YAML configuration file',
    )
    parser.add_argument(
        'overrides',
        nargs='*',
        help='Override config parameters (e.g., n_nodes=1 actor.type._class=qwen3)',
    )
    parser.add_argument(
        '--start_vllm_servers',
        type=bool,
        required=False,
        default=False,
        
    )
    args = parser.parse_args()

        # Load configuration using OmegaConf
    if args.file_path is None:
        config = om.load('yamls/single-controller-grpo-workflow.yaml').parameters
    else:
        config = om.load(args.file_path)

    if args.start_vllm_servers:
        vllm_procs = []
        vllm_engine, vllm_procs = launch_vllm_servers(
            pretrain_model_name=config.model.pretrained_model_name_or_path,
            tensor_parallel_size=1,
            data_parallel_size=1,
            num_vllm_servers=4,
            num_train_actors=4,
            max_model_len=config.max_seq_len,
            enable_prefix_caching=config.vllm_enable_prefix_caching,
            use_existing=False,
        )
        print(f'Started vLLM servers {[vllm_proc.pid for vllm_proc in vllm_procs]}')
        [vllm_proc.wait() for vllm_proc in vllm_procs]
        sys.exit()



    # Apply command line overrides
    if args.overrides:
        # Convert list of key=value strings to OmegaConf overrides
        override_config = om.from_dotlist(args.overrides)
        config = om.merge(config, override_config)
        log.info(f'args.overrides: {args.overrides}')
        log.info(f'Config after overrides: {config}')

    log.info(f'config.model.pretrained_model_name_or_path: {config.model.pretrained_model_name_or_path}')
    num_train_actors = dist.get_world_size() // 2
    num_vllm_servers = dist.get_world_size() // 2
    with _patch_env(WORLD_SIZE='1', LOCAL_WORLD_SIZE='1'):
        asyncio.run(_run_single_controller_ppo(config, num_train_actors, num_vllm_servers))

    # train_engine, _ = launch_train_servers(4)
    # train_engine = TrainEngine(4)
    # train_engine.initialize()

