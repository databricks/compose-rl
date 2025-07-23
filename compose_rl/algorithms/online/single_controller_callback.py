# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Online On-Policy RL callback."""

from __future__ import annotations

import logging
import time
from typing import Any, Union

import torch
from composer.core import (
    State,
    TimeUnit,
    ensure_time,
    get_precision_context,
)
from composer.loggers import Logger, MLFlowLogger, WandBLogger
from composer.trainer.trainer import _get_initial_device_train_microbatch_size
from composer.utils import dist, ensure_tuple
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from compose_rl.algorithms.online.generation_utils import (
    broadcast_to_vllm,
    vllm_generate,
)
from compose_rl.algorithms.online.model import (
    ComposerHFPolicyLM,
    ComposerMPTPolicyLM,
)
from compose_rl.algorithms.online.model_methods import (
    OnPolicyEnum,
)
from compose_rl.algorithms.online.reward_manager import (
    ReferenceOutput,
    RewardManager,
    RewardOutput,
)
from compose_rl.data.buffer import MinibatchRolloutBuffer
from compose_rl.registry_builders import build_kl_controller
from compose_rl.utils import (
    compute_advantages,
    dist_compute_masked_mean_and_var,
    flatten,
    masked_mean,
    masked_sum,
)

# Import the base class
from compose_rl.algorithms.online.callback import OnPolicyCallback, env_reward

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
Policy = Union[ComposerHFPolicyLM, ComposerMPTPolicyLM]

__all__ = ['SingleControllerOnPolicyCallback', 'env_reward']

log = logging.getLogger(__name__)


class SingleControllerOnPolicyCallback(OnPolicyCallback):
    """Callback for managing on-policy training in an RLHF loop.

    Args:
        train_config (dict): Training config passed to callback via foundry train.py as
            callback is registered under callbacks_with_config registry.
    """

    def __init__(
        self,
        train_config: dict,
    ):
        var_config = train_config['variables']

        # The maximum generation length.
        self.max_gen_len: int = var_config.get('max_gen_len', 32)
        # Gamma discounting for computing returns.
        self.gamma = var_config.get('gamma', 1.0)
        # Value used in the generalized advantage estimate calculation.
        self.lambda_gae = var_config.get('lambda_gae', 1.0)

        # Other algo specific hparams

        # Which kl estimator to use
        if 'kl_estimator' not in train_config['model']:
            # TODO: Modify PPO to nuke config_overrides in the future
            # Check in model's config_overrides
            kl_estimator = train_config['model']['config_overrides'].get(
                'kl_estimator',
                'k1',
            )
        else:
            kl_estimator = train_config['model'].get('kl_estimator', 'k1')
        if kl_estimator not in ['k1', 'k2', 'k3', 'k3_offpolicy']:
            raise ValueError(
                f'Invalid kl estimator: {kl_estimator}. ' +
                'Valid options are: k1, k2, k3, k3_offpolicy.',
            )
        self.kl_estimator = kl_estimator

        if 'kl_clip_range' not in train_config['model']:
            # TODO: Modify PPO to nuke config_overrides in the future
            # Check in model's config_overrides
            kl_clip_range = train_config['model']['config_overrides'].get(
                'kl_clip_range',
                40.0,
            )
        else:
            kl_clip_range = train_config['model'].get('kl_clip_range', 40.0)
        if kl_clip_range <= 0:
            raise ValueError(
                f'Invalid kl clip range: {kl_clip_range}. ' +
                'Must be greater than 0.',
            )
        # check for precision and clip range
        precision = train_config['precision']
        if precision != 'fp32':
            if kl_clip_range > 50.0:
                log.warning(
                    f'Clip value of {kl_clip_range=} will not be effective with {precision=} as range for tensors is too small',
                )
        self.kl_clip_range = kl_clip_range

        # Generation keyword arguments.
        self.generation_kwargs = var_config.get('generation_kwargs')
        # The value to center the reward mean around.
        self.center_reward_mean = var_config.get('center_reward_mean', None)

        # The reward config which we will use to make the RewardManager.
        self.reward_cfg = var_config['rewards']
        self.max_seq_len = train_config['max_seq_len']
        self.non_train_fsdp_config = var_config.get(
            'non_train_fsdp_config',
            train_config['fsdp_config'],
        )
        self.ref_config = var_config['reference_model']

        # Per-device generate size.
        self.device_generate_batch_size: int = var_config.get(
            'device_generate_batch_size',
            1,
        )
        self.device_train_batch_size: int = train_config.get(
            'device_train_batch_size',
            None,
        )
        assert self.device_train_batch_size is not None

        # Number of batches to use for a single PPO epoch.
        self.num_batches_per_update = var_config.get(
            'num_batches_per_update',
            1,
        )
        # Number of generations per prompt for a single PPO epoch.
        self.generations_per_prompt: int = var_config.get(
            'generations_per_prompt',
            1,
        )

        if self.num_batches_per_update % self.generations_per_prompt != 0:
            raise ValueError(
                f'{self.num_batches_per_update=} must be divisible by {self.generations_per_prompt=}',
            )

        self.epochs_per_iteration = ensure_time(
            var_config.get('epoch_per_iteration', 1),
            TimeUnit.EPOCH,
        )
        assert self.epochs_per_iteration.unit == TimeUnit.EPOCH

        # Programmatically setting the max buffer size instead of the yaml
        var_config['buffer']['max_buffer_size'] = self.num_batches_per_update
        self.buffer = MinibatchRolloutBuffer(var_config['buffer'])

        # Build the KL controller through registries
        kl_ctl_name = var_config['kl_controller'].pop('kl_ctl_type')
        self.kl_ctl = build_kl_controller(
            name=kl_ctl_name,
            kwargs=var_config['kl_controller'],
        )

        self.kl_ift = []

        self.wandb_logger = None
        self.mlflow_logger = None
        self.prompts_and_gens = []
        self.prompt_ids_rewards_and_answers = []
        self.iter_num = 0
        self.train_prompt_loader_state_dict = None
        self.train_prompt_loader = None

        self.input_eos_token_ids = var_config.get('eos_token_ids', None)

        if train_config.get('python_log_level', None) is not None:
            logging.getLogger('compose_rl').setLevel(
                train_config['python_log_level'].upper(),
            )
            logging.getLogger(__name__).setLevel(
                train_config['python_log_level'].upper(),
            )

        self.batch_rollouts = None


    def init(self, state: State, logger: Logger):
        self.pad_token_idx = state.model.tokenizer.pad_token_id  # type: ignore
        self.actor_critic = state.model

        if self.actor_critic.loss_type == OnPolicyEnum.GRPO:
            assert self.generations_per_prompt > 1, \
                'GRPO requires multiple generations per prompt. ' + \
                f'Current generations_per_prompt is: {self.generations_per_prompt}.'

        # TODO (#158): do this through composer.
        for destination in ensure_tuple(logger.destinations):
            if isinstance(destination, WandBLogger):
                self.wandb_logger = destination
            if isinstance(destination, MLFlowLogger):
                self.mlflow_logger = destination

        # Set iteration_length
        state._iteration_length = self.epochs_per_iteration

        self.precision = state.precision
        self.device_train_microbatch_size: int = state.device_train_microbatch_size  # type: ignore
        if self.device_train_microbatch_size == 'auto':  # type: ignore
            raise ValueError('auto microbatching is not supported for PPO')

        self.iter_batch_size = self.num_batches_per_update * self.device_train_batch_size

        # The KL penalty in the reward should only exist if we aren't minimizing
        # the KL directly in the loss.
        kl_penalty_in_reward = True

        if hasattr(self.actor_critic, 'compute_kl_loss'):
            kl_penalty_in_reward = not self.actor_critic.compute_kl_loss

        self.reward_manager = RewardManager(
            config=self.reward_cfg,
            ref_config=self.ref_config,
            tokenizer=self.actor_critic.tokenizer, # type: ignore
            max_seq_len=self.max_seq_len,
            fsdp_config=self.non_train_fsdp_config,
            precision=state.precision,
            kl_penalty_in_reward=kl_penalty_in_reward,
        )

        # This is needed to ensure PyTorch 2.4 checkpointing doesn't break
        self.actor_critic.tokenizer.batch_encode_plus( # type: ignore
            batch_text_or_text_pairs=['Dummy input'],
            padding='longest',
            truncation=True,
            return_attention_mask=True,
        )


    def round_trip_to_inference_engines(self, device: Any, vllm_engines: list[Any], model_update_group: dist.ProcessGroup):
        """Round trip to inference engines.
        
        Args:
            vllm_engines (list[Any]): The vllm engines to round trip to.
        """
        batch = device.batch_to_device(self._get_next_iter_prompts())
        # self._update_inference_model(batch, vllm_engines, model_update_group)
        self.batch_rollouts = self._interact_with_env(batch, vllm_engines)

    def iteration_start(self, state: State, logger: Logger):
        del logger  # unused

        self._get_reward(self.batch_rollouts)

        # Reset and initialize state train dataloader
        log.warning(
            'trainer._train_data_spec should be updated whenever the dataloader is updated',
        )
        # Train Dataloader
        state.set_dataloader(self.buffer, 'ep')
        state.train_dataloader = state.dataloader
        state.device_train_microbatch_size = _get_initial_device_train_microbatch_size(
            state.device_train_microbatch_size,
            state.auto_microbatching,
            state.train_dataloader,
        )

        # Update IFT KL
        self._update_ift_kl()

    # epoch_end and iteration_end methods are inherited from OnPolicyCallback

    def _get_next_iter_prompts(self):
        """Gets the next iteration's batch of prompts."""
        # Sample fewer batches for the Online RL interation depending on the number of generations per prompt
        n_unique_batches = self.num_batches_per_update // self.generations_per_prompt
        batches = [
            self._get_single_batch_prompts() for _ in range(n_unique_batches)
        ]

        ret_batch = {}
        assert 'prompt_id' in batches[0], 'prompt_id must be in the batch'
        for key in batches[0].keys():
            curr_values = []

            max_len = 0
            if isinstance(batches[0][key], torch.Tensor):
                max_len = max([batch[key].shape[-1] for batch in batches])

            padding_key = None
            for batch in batches:
                # Explode the batch into multiple batches for each generation
                for _ in range(self.generations_per_prompt):
                    # For keys that do not require additional processing
                    if key in ['prompt_len', 'verified_answer', 'prompt_id']:
                        curr_values.append(batch[key])
                        continue

                    bs, seq_len = batch[key].shape

                    if key == 'prompt':
                        padding_key = self.pad_token_idx
                        if (batch[key][:, -1] == padding_key).any():
                            raise ValueError(
                                'The last token in the prompt should not be the pad token. Please double '
                                +
                                'check the dataloader and prompt and dataloader.',
                            )
                    elif key == 'prompt_attention_mask':
                        padding_key = False

                    # Compute the required padding and concatenate with the batch tensor
                    pad = torch.ones(
                        (bs, max_len - seq_len),
                        dtype=batch[key].dtype,
                    ) * padding_key  # type: ignore
                    curr_values.append(torch.cat([pad, batch[key]], dim=-1))

            # For tensor fields, use torch.cat to combine the values; for string fields, just use the list
            if isinstance(curr_values[0], torch.Tensor):
                ret_batch[key] = torch.cat(curr_values)
            else:
                if key == 'verified_answer':
                    ret_batch[key] = list(flatten(curr_values))
                else:
                    # this is an edge case that we will not hit currently, but just handling it as needed
                    ret_batch[key] = curr_values

        return ret_batch

    # _get_single_batch_prompts method is inherited from OnPolicyCallback

    def _interact_with_env(self, batch: dict[str, torch.Tensor], vllm_engines: list[Any]):
        """Have the policy interact with the environment.

        Here, we redo microbatching, and run generate appropriately. We add the environment
        interactions to the buffer.

        Args:
            batch (dict): the iteration level batch we want to interact with the environment.
        """
        max_gen_len = self.max_gen_len
        generation_kwargs = self.generation_kwargs
        with get_precision_context(self.precision), torch.no_grad():
            # If vllm engines are available, we use them to generate sequences in one go
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
        return batch

    def _get_reward(self, batch: dict[str, torch.Tensor]):
        log.debug('Beginning reward computation for the rollout.')
        start_reward_time = time.time()
        env_outputs, prompts_and_gens, ref_outputs, all_rewards_dict = env_reward(
            actor_critic=self.actor_critic,  # pyright: ignore
            reward_manager=self.reward_manager,
            batch=batch,
            max_gen_len=self.max_gen_len,
            precision=self.precision,
            device_train_microbatch_size=self.device_train_microbatch_size,
            tokenizer=self.tokenizer,  # type: ignore
            eos_token_ids=self.eos_token_ids,  # type: ignore
            kl_estimator=self.kl_estimator,
            kl_clip_range=self.kl_clip_range,
        )

        end_reward_time = time.time()
        total_reward_time = end_reward_time - start_reward_time
        log.debug(
            f'Finished reward computation for the rollout in {total_reward_time:.4f} seconds.',
        )

        self.prompts_and_gens.extend(prompts_and_gens)

        gen_batch_partial_outputs = (env_outputs, ref_outputs, all_rewards_dict)
        # For every partial output we want to resolve them together
        # And compute the global per iteration batch advantage's mean and variance
        resolved_outputs = self._resolve_outputs(
            batch,
            gen_batch_partial_outputs,
        )

        # We need to split the resolved outputs into minibatches
        for idx in range(self.iter_batch_size // self.device_train_batch_size):
            minibatch = self._extract_minibatch(
                resolved_outputs,
                idx,
                self.device_train_batch_size,
            )
            self.buffer.add(minibatch)

        # Making sure we correctly parsed the minibatches
        assert len(self.buffer) == self.num_batches_per_update

        self.actor_critic.train()


    def _update_inference_model(self, batch: dict[str, torch.Tensor], vllm_engines: list[Any], model_update_group: dist.ProcessGroup):
        start_time = time.time()
        log.info('Before broadcast to vLLM')
        broadcast_to_vllm(
            self.actor_critic,
            vllm_engines,
            model_update_group,
            batch,
            #loss_type=self.actor_critic.loss_type.value,  # type: ignore
            loss_type=self.actor_critic.loss_type,  # type: ignore
        )
        log.info('Finished broadcasting to vLLM')
        log.info(f'Took: {time.time() - start_time} to broadcast to vllm.')
        dist.barrier()
