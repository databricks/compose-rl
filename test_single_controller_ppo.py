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
import copy
from contextlib import contextmanager
import logging
import os
import pickle
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
from composer.core.data_spec import _default_split_batch
from composer.trainer.trainer import _get_initial_device_train_microbatch_size
from compose_rl.data.buffer import MinibatchRolloutBuffer
from composer.optim import DecoupledAdamW
from composer.utils import dist as composer_dist
from llmfoundry.data import build_dataloader
from llmfoundry.utils import build_composer_model
from llmfoundry.utils.config_utils import process_init_device  # type: ignore
from omegaconf import OmegaConf as om
from transformers import AutoTokenizer
from composer.callbacks import MemoryMonitor, SpeedMonitor, LRMonitor

from compose_rl.registry_builders import build_kl_controller
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


@contextmanager
def time_it(name: str):
    start_time = time.time()
    pst_start_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-8)))
    print(f"[{name}] started at {pst_start_time.strftime('%Y-%m-%d %H:%M PST')}")
    yield
    end_time = time.time()
    pst_end_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-8)))
    print(f"[{name}] finished at {pst_end_time.strftime('%Y-%m-%d %H:%M PST')}")
    print(f"[{name}] took {end_time - start_time:.2f} seconds")


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

        self.config = None
        self.model = None
        self.reference_model = None
        self.model_update_group = None
        self.ref_path = None
        self._dataloader = None
        self._tokenizer = None
        self.ppo_callback = None
        self.ppo_trainer: Trainer = None  # type: ignore
        self.buffer: MinibatchRolloutBuffer = None  # type: ignore

        self.pretrain_model_name = None
        self.device_train_batch_size = None
        self.num_batches_per_update = None
        self.max_seq_len = None
        self.precision = None  # type: ignore
        self.train_config: dict = None  # type: ignore
        self.variables_config: dict = None  # type: ignore
        self.model_config = None
        self.ref_model_config = None
        self.global_train_batch_size = None
        self.max_gen_len = None

        # KL Penalty and Controller
        self.kl_ift = []
        self.kl_controller = None
        self.kl_controller_config = None
        self.kl_penalty_in_reward = None

        # Reward info
        self.reward_coefficients: dict = None  # type: ignore

    def build_train_config(self, config: Any):
        self.config = config
        self.logger.info(f"Starting build_train_config with model: {self.config.model.pretrained_model_name_or_path}")
        self.pretrain_model_name = self.config.model.pretrained_model_name_or_path

        self.model_config = om.to_container(self.config.model, resolve=True)
        self.model_config['tokenizer'] = self.tokenizer

        # Reference Model Initializing
        self.ref_model_config = om.to_container(self.config.variables.reference_model.model_config, resolve=True)

        self.global_train_batch_size = self.config.global_train_batch_size
        self.device_train_batch_size = self.global_train_batch_size // self.world_size
        self.num_batches_per_update = self.config.variables.num_batches_per_update
        self.max_seq_len = self.config.max_seq_len
        self.max_gen_len = self.config.variables.max_gen_len
        self.precision = self.config.precision

        # NOTE: if compute kl loss then no reward penalty
        # TODO: we should be more explicit about this toggle / make each kl regularization mechanism explicit
        self.kl_controller_config = om.to_container(self.config.variables.kl_controller, resolve=True)
        self.kl_penalty_in_reward = not self.model_config.get('compute_kl_loss', False)

        # Reward Coefficients
        all_rewards_config = om.to_container(self.config.variables.rewards, resolve=True)
        self.reward_coefficients = {}
        for reward_name, reward_config in all_rewards_config.items():
            self.reward_coefficients[reward_name] = reward_config.get(
                'reward_coefficient',
                1.0,
            )

        variables = om.to_container(self.config.variables, resolve=True)
        self.variables_config = variables
        algorithm_config = self.config.algorithms

        self.train_config = {
            'seed': self.config.seed,
            'model': self.model_config,
            'ref_model': self.ref_model_config,
            'fsdp_config': self.config.fsdp_config,
            'kl_controller': self.kl_controller_config,
            'non_train_fsdp_config': self.variables_config['non_train_fsdp_config'],
            'precision': self.precision,
            'variables': variables,
            'algorithms': algorithm_config,
            'global_train_batch_size': self.device_train_batch_size * self.world_size,
            'device_train_batch_size': self.device_train_batch_size,
            'device_train_microbatch_size': self.device_train_batch_size,
            'save_folder': self.config.save_folder,
            'log_config': self.config.log_config,
            'max_seq_len': self.max_seq_len,
            'python_log_level': self.config.python_log_level,
            'console_log_interval': self.config.console_log_interval,
        }
        self.logger.info("Finished build_train_config")

    def build_buffer(self):
        self.buffer = MinibatchRolloutBuffer(self.variables_config['buffer'])
        self.logger.info(f'Initialized minibatch buffer.')

    def build_tokenizer(self):
        # TODO (algo): decide if we should use tokens or messages given
        # we may need token level log prob
        # TODO (infra): use the tokenizer/texts for prompt dataloader but
        # token (ids) for the experience buffer/manager
        kwargs = self.config.tokenizer.kwargs
        tokenizer = AutoTokenizer.from_pretrained(self.pretrain_model_name, **kwargs)
        return tokenizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.build_tokenizer()
        return self._tokenizer

    def init_composer_dist(self):
        print('Initializing composer dist', composer_dist.get_local_rank(), composer_dist.get_global_rank(), composer_dist.get_world_size())
        composer_dist.initialize_dist('gpu')

    def build_kl_controller(self):
        kl_controller_name = self.kl_controller_config.pop('kl_ctl_type')
        self.kl_controller = build_kl_controller(
            name=kl_controller_name,
            kwargs=self.kl_controller_config,
        )
        self.logger.info(f'Built KL Controller')

    def build_reference_model(self):
        name = self.ref_model_config.pop('name')

        init_context = process_init_device(
            self.ref_model_config,
            self.variables_config['non_train_fsdp_config'],
        )

        self.reference_model = build_composer_model(
            name=name,
            cfg=self.ref_model_config,
            tokenizer=self.tokenizer,
            init_context=init_context,
            master_weights_dtype=self.ref_model_config.get('master_weights_dtype', None),
        )

        parallelism_config = {'fsdp': self.variables_config['non_train_fsdp_config']}

        load_path = self.variables_config['reference_model'].get('load_path', None)

        # Create a Trainer object to load from checkpoint and FSDP the model
        _ = Trainer(
            model=self.reference_model,
            parallelism_config=parallelism_config,
            precision=self.precision,
            load_weights_only=True,
            load_strict_model_weights=False,
            load_path=load_path,
            python_log_level='debug',
        )
        self.logger.info(f'Initialized {name} reference model')

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

        # TODO: Add weight decay
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
            experiment_name=self.config.loggers.mlflow.experiment_name,
            run_name=self.config.loggers.mlflow.tags.run,
            tracking_uri=self.config.loggers.mlflow.tracking_uri,
        )

        callbacks = [
            self.ppo_callback,
            # callbacks for scheduled garbage collection
            # this helps improve throughput by garbage collecting
            # at regular intervals on all training processes
            # ScheduledGarbageCollector(
            #     batch_interval='1000',
            # ), # TODO: Add it back after we resolve some error because we are using a dummy dataloader
            # callbacks for monitoring other metrics
            LRMonitor(),
            MemoryMonitor(),
            SpeedMonitor(window_size=10),
        ]

        self.ppo_trainer = Trainer(
            model=model,
            optimizers=optimizer,
            callbacks=callbacks,
            train_dataloader=dummy_dataloader,
            precision=self.precision,
            parallelism_config={'fsdp': self.config.fsdp_config},
            max_duration=self.config.max_duration,
            loggers=[mlflow_logger],
            device_train_microbatch_size=self.config.device_train_microbatch_size,
            load_path=self.ref_path,
            save_folder=self.config.save_folder,
            save_interval=self.config.save_interval,
            autoresume=self.config.autoresume,
        )

    def close_trainer(self):
        self.ppo_trainer.close()

    # TODO: maybe make the name more informative?
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

        device = torch.device('cuda')
        with get_precision_context(self.precision), torch.no_grad():
            # 1) Compute Log Probs and Entropy
            partial_batch = self.get_log_probs_and_entropy(current_rank_rollouts, device)
            # 2) Compute Reference Log Probs and KL
            reference_output = self.get_reference_log_probs_and_kl(partial_batch)
 
            # Log to callback for KL Controller Update
            mean_ift = masked_mean(
                reference_output['kl'],
                partial_batch['action_mask'],
            )
            self.kl_ift.append(mean_ift.cpu())

            # 3) Scale rewards and apply KL Penalty
            reward_output = self.update_rewards(current_rank_rollouts['all_rewards_dict'], reference_output, partial_batch['action_mask'], device)
            # 4) Compute Advantages
            advantage_output = self.compute_advantages(partial_batch, reward_output)

            # Construct batch
            bs = partial_batch['prompt_id'].shape[0]
            batch = {
                'max_gen_len': torch.ones(bs).to(torch.int32) * self.max_gen_len,
                'ift_kl_scalar': torch.ones(bs) * self.kl_controller.value,
                **partial_batch,
                **reference_output,
                **reward_output,
                **advantage_output,
            }

            # Moving minibatches to CPU to not take additional GPU memory
            for k, v in batch.items():
                if hasattr(v, 'cpu'):
                    batch[k] = v.cpu()

        # NOTE: Probably should break things up but putting it here for now for clarity
        # Delete Non-tensor keys for training batch
        for key in ['verified_answer', 'messages']:
            if key in batch.batch_rollouts.keys():
                del batch[key]

        # We need to split the resolved outputs into minibatches
        for idx in range(
            batch['prompt_id'].shape[0] // self.device_train_batch_size,
        ):
            minibatch = self._extract_minibatch(
                self.batch_rollouts,
                idx,
                self.device_train_batch_size,
            )
            self.buffer.add(minibatch)

        # Making sure we correctly parsed the minibatches
        assert len(
            self.buffer,
        ) == self.num_batches_per_update, f'{len(self.buffer)} != {self.num_batches_per_update}'

        self.ppo_trainer.state.model.train()

        # Reset and initialize state train dataloader
        self.logger.warning(
            'trainer._train_data_spec should be updated whenever the dataloader is updated',
        )
        # Train Dataloader
        self.ppo_trainer.state.set_dataloader(self.buffer, 'ep')
        self.ppo_trainer.state.train_dataloader = self.ppo_trainer.state.dataloader
        self.ppo_trainer.state.device_train_microbatch_size = _get_initial_device_train_microbatch_size(
            self.ppo_trainer.state.device_train_microbatch_size,
            self.ppo_trainer.state.auto_microbatching,
            self.ppo_trainer.state.train_dataloader,
        )

        self._update_ift_kl()

    def _update_ift_kl(self):
        local_kl = torch.stack(self.kl_ift)

        global_ift_kl = torch.cat(dist.all_gather_object(local_kl))
        ift_kl_update = torch.mean(global_ift_kl)

        self.kl_controller.update(
            ift_kl_update,
            self.num_batches_per_update * self.device_train_batch_size *  # type: ignore
            dist.get_world_size(),
        )

        self.kl_ift = []

    def _extract_minibatch(
        self,
        batch: dict[str, torch.Tensor],
        idx: int,
        minibatch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Extracts a minibatch from a composite batch.

        This helper is used to extract a particular minibatch of size
        minibatch_size from `batch`, where `batch` may
        have a batch size that exceeds the minibatch size.

        Args:
            batch (dict[str, torch.Tensor]): an arbitrary batch, where
                each entry has batch size >= minibatch_size,
                representing the concatenation of >= 1 minibatches.
            idx (int): The index of the batch (see above description) to extract.

        Returns:
            curr_gen_batch (dict[str, torch.Tensor]): The gen_batch_idx'th
                gen_batch extracted from the batch input.
        """
        start_idx = idx * minibatch_size
        end_idx = (idx + 1) * minibatch_size
        curr_gen_batch = {
            batch_key: tensor[start_idx:end_idx]
            for batch_key, tensor in batch.items()
        }
        return curr_gen_batch

    def get_log_probs_and_entropy(self, current_rank_rollouts: dict[str, Any], device: torch.device):
        prompt_tokens = current_rank_rollouts['prompt']
        batch_size, _ = prompt_tokens.shape
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_ids = self.variables_config['eos_token_ids']
        prompt_len = current_rank_rollouts['prompt_len']
        prompt_id = current_rank_rollouts['prompt_id']

        #cur_device = prompt_tokens.device

        prompt_dtype = prompt_tokens.dtype

        assert 'sequences' in current_rank_rollouts, f'sequences is not in batch {current_rank_rollouts.keys()=}'

        sequences = current_rank_rollouts['sequences']
        generated_len = torch.ones(
            batch_size,
            device=device,
            dtype=prompt_dtype,
        ) * self.max_gen_len

        # If all the processes early exit generate, then we need to manually pad everything
        # we can pad this with pad tokens, since we switch the padding between left and right
        # padding based on the sequence length + max_sequence_length.
        if prompt_tokens.size(1) + self.max_gen_len > sequences.size(1):
            len_to_pad = self.max_gen_len - (
                sequences.size(1) - prompt_tokens.size(1)
            )

            extra_padding = torch.ones(
                (batch_size, len_to_pad),
                device=device,
                dtype=prompt_dtype,
            ) * pad_token_id
            sequences = torch.cat(
                [sequences, extra_padding],  # type: ignore
                dim=-1,  # type: ignore
            )

        # Sanity checking we're adding max_gen_len to prompt_tokens
        if prompt_tokens.size(1) + self.max_gen_len != sequences.size(1):
            raise ValueError(
                f'Prompts {prompt_tokens.size(1)} + max_gen_len {self.max_gen_len} != sequences {sequences.size(1)}',
            )

        # Actions are what tokens the current policy would generate.
        actions = sequences[:, -self.max_gen_len:]

        right_padded_obs = switch_left_to_right_padding(
            sequences,
            prompt_len,
            self.max_gen_len,
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
            max_gen_len=self.max_gen_len,
            eos_token_ids=eos_token_ids,  # type: ignore
            pad_token=pad_token_id,  # type: ignore
        )
        log_probs = []
        entropies = []
        values = []

        input_model_kwargs = {
            'obs': right_padded_obs,
            'right_padded_attn_mask': right_padded_attn_mask,
            'prompt_len': prompt_len,
            'max_gen_len': self.max_gen_len,
            'action_mask': action_mask,
            'actions': actions,
        }

        microbatch_splits = _default_split_batch(
            batch=input_model_kwargs,
            microbatch_size=self.config.device_train_microbatch_size,
        )
        # Compute the device_train_microbatch_log_probs inside the for loop to reduce the softmax overhead
        for split in microbatch_splits:
            curr_kwargs = split

            cur_output = self.ppo_trainer.state.model(curr_kwargs)
            cur_logits = cur_output['logits']
            # need to pull out current actions and prompt len
            cur_actions = curr_kwargs['actions']
            cur_action_mask = curr_kwargs['action_mask']
            cur_prompt_len = curr_kwargs['prompt_len']

            cur_log_probs = get_log_probs(
                logits=cur_logits,
                actions=cur_actions,
                prompt_len=cur_prompt_len,
                max_gen_len=self.max_gen_len,
            )
            cur_entropies = get_entropies(
                logits=cur_logits,
                action_mask=cur_action_mask,
                prompt_len=cur_prompt_len,
                max_gen_len=self.max_gen_len,
            )
            log_probs.append(cur_log_probs)
            entropies.append(cur_entropies)
            # Ignore values when the model doesn't have a value head
            if 'values' in cur_output:
                cur_values = cur_output['values']
                values.append(cur_values)

        device_train_microbatch_log_probs = torch.cat(log_probs)
        device_train_microbatch_entropies = torch.cat(entropies)

        partial_env_output = {
            'prompt_id': prompt_id,
            'old_log_probs': device_train_microbatch_log_probs,
            'old_entropies': device_train_microbatch_entropies,
            'obs': right_padded_obs,
            'right_padded_attn_mask': right_padded_attn_mask,
            'actions': actions,
            'action_mask': action_mask,
            'generated_len': generated_len,
            'prompt_len': prompt_len,
        }
        if len(values) > 0:
            device_train_microbatch_values = torch.cat(values)

            # Need to add in the padding for the value function
            value_action_mask = torch.cat([
                action_mask,
                torch.zeros((batch_size, 1), device=device),
            ],
                                          dim=-1)
            device_train_microbatch_values *= value_action_mask
            partial_env_output['values'] = device_train_microbatch_values

        # TODO: old_log_probs, old_entropies, metadata as a clearer output
        return partial_env_output

    def get_reference_log_probs_and_kl(self, batch):
        """
        This function computes the reference log probs and computes KL estimates between pi and pi_ref.
        """
        kl = []
        ref_model_log_probs = []

        microbatch_splits = _default_split_batch(
            batch=batch,
            microbatch_size=self.config.device_train_microbatch_size,
        )
        for split in microbatch_splits:
            curr_batch = split
            curr_ref_output = self.reference_model({
                "input_ids": curr_batch['obs'],
                "attention_mask": curr_batch['right_padded_attn_mask'],
            })
            curr_ref_log_probs = get_log_probs(
                logits=curr_ref_output.logits,
                actions=curr_batch['actions'],
                prompt_len=curr_batch['prompt_len'],
                max_gen_len=self.max_gen_len,
                temperature=self.variables_config['generation_kwargs']['temperature'],
            )

            kl_dict = approx_kl(
                log_p=curr_ref_log_probs,
                log_q=curr_batch['old_log_probs'],
                kl_clip_range=self.model_config['kl_clip_range'],  # pyright: ignore
            )
            curr_kl = kl_dict[self.model_config['kl_estimator']]  # pyright: ignore

            kl.append(curr_kl)
            ref_model_log_probs.append(curr_ref_log_probs)

        kl = torch.cat(kl)
        ref_model_log_probs = torch.cat(ref_model_log_probs)
        ref_output = {
            "kl": kl,
            # TODO: rename to reference_log_probs
            #"reference_log_probs": ref_model_log_probs,
            "ift_log_probs": ref_model_log_probs,
        }
        return ref_output

    def update_rewards(self, raw_rewards_dict, ref_output, action_mask, device):
        resolved_reward_outputs: dict[str, torch.Tensor] = {}
        bad_end_generation_name, bad_end_generation_mask = None, None
        for name, subreward in raw_rewards_dict.items():
            # Functional Rewards
            resolved_reward_outputs[name] = subreward.to(device=device)

            # NOTE: all rewards is not accesible here
            #if isinstance(self.all_rewards[name], BadGenerationEndReward):
            if name == "bad_generation_end":
                bad_end_generation_name = name
                bad_generation_row_mask = torch.any(subreward != 0, dim=1)

                bad_end_generation_mask = (
                    ~bad_generation_row_mask
                ).unsqueeze(1).expand_as(subreward)
                bad_end_generation_mask = bad_end_generation_mask.to(
                    device=device,
                )

        # Reward Penalty
        ref_kl = ref_output['kl'].to(device=device)

        if self.kl_penalty_in_reward:
            rewards: torch.Tensor = -self.kl_controller.value * ref_kl.detach()
        else:
            rewards: torch.Tensor = torch.zeros_like(ref_kl)

        env_rewards = torch.zeros_like(rewards)
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
        # TODO: General scaling of reward values through whitening should be revisited
        # if center_reward_mean is not None:
        #    env_rewards -= center_reward_mean
        #

        # Final rewards is total env rewards + KL penalties
        rewards += env_rewards

        # Zero rewards at padded tokens
        rewards *= action_mask
        env_rewards *= action_mask

        outputs = {
            'rewards': rewards.detach(),
            'env_rewards': env_rewards.detach(),
        }
        outputs.update(rews_dict_out)

        return outputs

    def compute_advantages(self, batch: Any, reward_output: Any):
        # compute GRPO advantages
        bs = batch['prompt_id'].shape[0]
        prompt_id = batch['prompt_id']
        rewards = reward_output['rewards']

        # Flatten the rewards by summing on sequence length/action_mask
        flat_rewards = masked_sum(
            rewards,
            batch['action_mask'],
            dim=-1,
        )

        # Get unique prompt IDs and their indices
        unique_prompt_ids, inverse_indices = torch.unique(
            prompt_id,
            return_inverse=True,
        )

        # Use scatter to compute means and standard deviations
        # First, we'll create a tensor to track counts, sums, and sum of squares
        n_unique = len(unique_prompt_ids)
        counts = torch.zeros(n_unique, device=prompt_id.device)
        sums = torch.zeros(n_unique, device=prompt_id.device)
        sum_squares = torch.zeros(n_unique, device=prompt_id.device)

        # Use scatter_add to accumulate values
        counts.scatter_add_(
            0,
            inverse_indices,
            torch.ones_like(flat_rewards),
        )
        sums.scatter_add_(0, inverse_indices, flat_rewards)
        sum_squares.scatter_add_(0, inverse_indices, flat_rewards**2)

        # Compute means and standard deviations
        means = sums / counts
        variances = (sum_squares / counts) - (means**2)
        stds = torch.sqrt(variances)

        # Map back to original tensor shape
        mean_rewards = means[inverse_indices]
        std_rewards = stds[inverse_indices]

        # Calculate GRPO advantage
        grpo_advantage = (flat_rewards - mean_rewards)
        # Only normalize the advantage if flag is set
        if self.model_config['normalize_advantage']:  # type: ignore
            grpo_advantage /= (std_rewards + 1e-4)

        # Create advantages of the same shape as original rewards
        advantages = torch.zeros_like(rewards)
        # Copy the flat grpo_advantage according to action_mask
        expanded_advantages = grpo_advantage.unsqueeze(1).expand_as(
            batch['action_mask'],
        )
        advantages = torch.where(
            batch['action_mask'].bool(),
            expanded_advantages,
            advantages,
        )

        batch_adv_mean, batch_adv_var = dist_compute_masked_mean_and_var(
            advantages,
            batch['action_mask'],
        )

        advantage_output = {
            'advantages': advantages,
            'adv_masked_mean': torch.ones(bs) * batch_adv_mean.cpu(),
            'adv_masked_var': torch.ones(bs) * batch_adv_var.cpu(),
            'reward_std': torch.ones(bs) * rewards.std().to('cpu'),
        }
        return advantage_output


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

    # TODO: Maybe rename to build components?
    def build_models(self, config: Any):
        """Build reference models and PPO trainers for all actors."""
        self.collective_methods.build_train_config(config)
        self.collective_methods.init_composer_dist()

        # Build PPO trainers
        self.collective_methods.build_ppo_trainer()
        print('build ppo trainer done')

        # Build Minibatch Buffer
        self.collective_methods.build_buffer()

        # Build Reference Model
        self.collective_methods.build_reference_model()

        # Build KL Controller
        self.collective_methods.build_kl_controller()

    def _partition_rollouts_across_ranks(self, rollouts: dict[str, Any]) -> list[dict[str, Any]]:
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

    def _add_latest_rollouts(self, rollouts: dict[str, Any]):
        partitioned_rollouts = self._partition_rollouts_across_ranks(rollouts)
        assert len(partitioned_rollouts) == self.num_train_actors, "Number of partitioned rollouts should be equal to the number of train actors"
        ray.get([train_actor.add_rollouts.remote(partition) for train_actor, partition in zip(self.train_actors, partitioned_rollouts)])

    def train_1_iter(self):
        # added this method to time the collectivetraining time otherwise we can time each rank but the print/logging becomes messy to read
        with time_it("training"):
            self.collective_methods.train_1_iter()

    async def run(self, num_iterations: int, experience_buffer: 'ExperienceBuffer', parameter_buffer: 'ParameterBuffer', inference_server: 'InferenceServer', lock: asyncio.Lock, rollout_semaphore: asyncio.Semaphore, eval_semaphore: asyncio.Semaphore):
        # the overall design rn is we have a async def run function for each of the subcontroller that is responsible for async primitives but leave the rest of the logic to be sync function and use
        # asyncio.to_thread to bridge the async and sync world
        for _ in range(num_iterations):
            # Simple example of adding elements to the experience buffer
            # Populate the train actor group with the rollouts and then train
            latest_rollouts = await experience_buffer.get()
            self._add_latest_rollouts(latest_rollouts)
            await asyncio.to_thread(self.train_1_iter)
            # TODO decide where should we use the lock and the semaphore
            # it is more explicit to use them at this level but more abstracted away from trainer if we put them as input to the parameter buffer
            await parameter_buffer.put({'actor_group': self, 'inference_server': inference_server, 'lock': lock, 'rollout_semaphore': rollout_semaphore, 'eval_semaphore': eval_semaphore})

class InferenceServer:
    """Inference server with vLLM engines."""

    def __init__(self, num_vllm_engines: int, pretrain_model_name: str, config: Any):
        self.num_vllm_engines = num_vllm_engines
        self.vllm_tensor_parallel_size = config.vllm_tensor_parallel_size
        self.vllm_engines = create_vllm_engines(
                num_engines=num_vllm_engines,
                tensor_parallel_size=self.vllm_tensor_parallel_size,
                enforce_eager=True,
                pretrain=pretrain_model_name,
                revision=None,
                seed=1,
                enable_prefix_caching=config.vllm_enable_prefix_caching,
                max_model_len=config.max_seq_len,
                device_bundle={
                    'GPU': 1,
                    'CPU': 1,
                    'worker_gpu': 0,
                },
            )

    @property
    def engines(self):
        return self.vllm_engines

# Note: This needs to be re-worked once the repos are migrated.
class EvalAgent:
    """An async agent for handling evals."""

    def __init__(
        self,
        vllm_engines: list[Any],
        config: Any,
    ):
        self.vllm_engines = vllm_engines
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
            vllm_engines = []
        fake_state = _State()
        fake_state.vllm_engines = self.vllm_engines
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


class ParameterBuffer(Buffer):
    """Buffer for updating the inference model."""

    def __init__(self, config: Any):
        super().__init__()
        self.num_times_param_updated = 0
        # TODO: Support eval_interval_num in a more generic way (e.g. handle more than just `iter`)
        self.eval_interval_num = int(config.eval_interval.strip("iter"))

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

    async def put(self, struct: dict[str, Any]):
        # prefers to implement the model update logic in the Buffer class as the buffer is a bridge between the trainer actor and the inference server
        # and knows the best way to transfer the model parameters. Trainer just needs to put necessary struct to this api
        async with struct['lock']:
            struct['actor_group'].collective_methods.execute(partial(self.update_inference_model, inference_server=struct['inference_server']))
        # allow next rollout/generation step
        struct['rollout_semaphore'].release()
        # schedule eval if interval reached
        self.num_times_param_updated += 1
        # Since we updated the params, we need to check if we need to schedule an eval
        # based on the previous value of num_times_param_updated as we want to run an eval
        # at timestep 0.
        if (self.num_times_param_updated - 1) % self.eval_interval_num == 0:
            struct['eval_semaphore'].release()


class ExperienceBuffer(Buffer):
    """Buffer for storing experiences."""

    async def put(self, struct: dict[str, Any]):
        await self.buffer.put(struct)

    async def get(self, struct: Optional[dict[str, Any]] = None):
        return await self.buffer.get()

    def __len__(self):
        return len(self.buffer)


class StreamingDatasetActor(BaseDistributedGPUActor):
    """Streaming actor for loading prompts onto the experience buffer."""

    def __init__(self, config: Any):
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
        global_train_batch_size = config.global_train_batch_size
        self.generations_per_prompt = config.variables.generations_per_prompt
        num_batches_per_update = config.variables.num_batches_per_update
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


class RewardActor(BaseDistributedGPUActor):
    """Streaming actor for adding rewards on top the experience buffer."""

    def __init__(self, config: Any):
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

            self.logger.info(f'Initializing reward with name {reward_name}')

            # TODO: Validate reward_config
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
        self.pool = Pool(
            processes=len(self.all_rewards),
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
            batch['raw_untokenized_texts'] = raw_untokenized_texts
            batch['zero_rewards'] = torch.zeros_like(action_log_probs)

            func = curr_reward
            args = (self._to_cpu(batch),)

            computed_rewards[reward_name] = self.pool.apply_async(
                func=func,
                args=args,
            )

        # convert all AsyncResult objects to tensors because ray cannot return Pool objects
        for reward_name, subreward in computed_rewards.items():
            if isinstance(subreward, AsyncResult):
                computed_rewards[reward_name] = subreward.get()
            else:
                computed_rewards[reward_name] = subreward

        return computed_rewards


class RolloutAgent:
    """Rollout agent for generating sequences from the inference server."""

    def __init__(
        self,
        inference_server: InferenceServer,
        streaming_dataset_actor: StreamingDatasetActor,
        reward_actor: RewardActor,
        config: Any,
    ):
        # Actors
        self.inference_server = inference_server
        self.streaming_dataset_actor = streaming_dataset_actor
        self.reward_actor = reward_actor

        self.generation_kwargs = config.variables.generation_kwargs
        self.precision = config.precision

        self.tokenizer = ray.get(self.streaming_dataset_actor.get_tokenizer.remote())
        self.tokenizer_pad_token_id = ray.get(self.streaming_dataset_actor.get_tokenizer_pad_token_id.remote())
        if self.tokenizer_pad_token_id is None:
            raise ValueError(
                'Tokenizer does not have a pad token id. Please use a different tokenizer or add a pad token id.',
            )

        self.prompt_handler_config = ray.get(self.streaming_dataset_actor.get_prompt_handler_config.remote())
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
            print(f'Autoresuming from checkpoint for RolloutAgent.')
            with open(self.latest_checkpoint, 'rb') as f:
                checkpoint = pickle.load(f)
            self.iter_num = checkpoint['iter_num']
            print(f'Loading streaming dataloader state dict for RolloutAgent.', checkpoint['streaming_dataloader'])
            self.streaming_dataset_actor.load_dataloader_state_dict.remote(checkpoint['streaming_dataloader'])

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

        # Checkpointing
        save_folder_iter = os.path.join(self.save_folder, f'iter_{self.iter_num}')
        checkpoint_path = os.path.join(save_folder_iter, 'checkpoint.pt')
        self.iter_num += 1

        streaming_dataloader_state_dict = ray.get(self.streaming_dataset_actor.get_dataloader_state_dict.remote())
        print(f'Streaming dataloader state dict for RolloutAgent.', streaming_dataloader_state_dict)

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

    async def run(self, num_iterations: int, experience_buffer: 'ExperienceBuffer', lock: asyncio.Lock, rollout_semaphore: asyncio.Semaphore):
        for _ in range(num_iterations):
            # semaphore has be to acquired before the lock is acquired
            # otherwise it could hang the parameter_buffer due to lock is already acquired
            await rollout_semaphore.acquire()
            async with lock:
                rollouts = await asyncio.to_thread(self.get_next_iter_rollouts)
            await experience_buffer.put(rollouts)

class PPOController:
    """PPO controller for training the policy and value networks."""

    def __init__(
        self,
        train_actor: TrainActorGroup,
        inference_server: InferenceServer,
        rollout_agent: RolloutAgent,
        parameter_buffer: ParameterBuffer,
        experience_buffer: ExperienceBuffer,
        eval_agent: EvalAgent,
        config: Any,
    ):
        self.train_actor = train_actor
        self.inference_server = inference_server
        self.rollout_agent = rollout_agent
        self.parameter_buffer = parameter_buffer
        self.experience_buffer = experience_buffer
        self.train_actor.build_models(config)
        self.eval_agent = eval_agent
        setup_process_groups(
            self.train_actor.master_actor,
            inference_server.engines,
            inference_server.vllm_tensor_parallel_size,
        )
        self.lock = asyncio.Lock()
        self.rollout_semaphore = asyncio.Semaphore(config.max_async_step)
        self.eval_semaphore = asyncio.Semaphore(0)
        self.config = config
    
    async def train_async(self, max_duration: int | str):
        if isinstance(max_duration, str):
            num_iterations = int(max_duration.replace('iter', ''))
        else:
            num_iterations = max_duration

        # we need to sync the train actor and the rollout agent once otherwise in async the rollout agent could start with params not synced with the train actor
        await self.parameter_buffer.put({'actor_group': self.train_actor, 'inference_server': self.inference_server, 'lock': self.lock, 'rollout_semaphore': self.rollout_semaphore, 'eval_semaphore': self.eval_semaphore})
        rollout_task = asyncio.create_task(self.rollout_agent.run(num_iterations, self.experience_buffer, self.lock, self.rollout_semaphore))
        eval_task = asyncio.create_task(self.eval_agent.run(num_iterations, self.lock, self.eval_semaphore))
        train_task = asyncio.create_task(self.train_actor.run(num_iterations, self.experience_buffer, self.parameter_buffer, self.inference_server, self.lock, self.rollout_semaphore, self.eval_semaphore))
        await asyncio.gather(rollout_task, eval_task, train_task)
        self.train_actor.collective_methods.close_trainer()

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

    # Disable setting CUDA_VISIBLE_DEVICES by ray, we will set it manually
    os.environ['RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES'] = '1'

    with start_ray_server() as _address:
        # only rank 0 is the master controller
        if dist.get_rank() == 0:
            world_size = getattr(config, "world_size", 0)
            if world_size == 0:
                world_size = dist.get_world_size()

            # Create buffers for the parameter and experience buffers
            # first since they don't have external dependencies
            parameter_buffer = ParameterBuffer(config)
            experience_buffer = ExperienceBuffer()

            # create SPMD training actors of the system
            num_train_actors = world_size // 2
            train_actor = TrainActorGroup(num_train_actors, DistributedGPUActor)

            # Create vLLM engines (or inference actors)
            vllm_tensor_parallel_size = config.vllm_tensor_parallel_size
            num_vllm_engines = (
                world_size - num_train_actors
            ) // vllm_tensor_parallel_size
            # TODO: Encapsulate this into a inference server manager class
            inference_server = InferenceServer(
                num_vllm_engines=num_vllm_engines,
                pretrain_model_name=config.model.pretrained_model_name_or_path,
                config=config,
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
            streaming_dataset_actor = ray.remote(num_gpus=0)(StreamingDatasetActor).remote(config)
            reward_actor = ray.remote(num_gpus=0)(RewardActor).remote(config)
            rollout_agent = RolloutAgent(inference_server, streaming_dataset_actor, reward_actor, config)

            # EvalAgent doesn't need to be a Ray actor since we don't need to
            # set a world_size or use GPUs for this process.
            eval_agent = EvalAgent(inference_server.engines, config)

            ppo_controller = PPOController(
                train_actor,
                inference_server,
                rollout_agent,
                parameter_buffer,
                experience_buffer,
                eval_agent,
                config,
            )
            asyncio.run(ppo_controller.train_async(config.max_duration))


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run single controller PPO with configuration file')
    parser.add_argument('--file_path', type=str, required=False, default=None,
                       help='Path to the OmegaConf YAML configuration file')
    args = parser.parse_args()
    
    # Load configuration using OmegaConf
    if args.file_path is None:
        config = om.load("yamls/single-controller-grpo-workflow.yaml").parameters
    else:
        config = om.load(args.file_path)
    
    # This is an example of how to move the controller logic from PPO Callback
    # to a separate trainer actor above and this main single controller
    # function.
    _run_single_controller_ppo(config)

