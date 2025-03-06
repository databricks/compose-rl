# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""PPO callback."""

from __future__ import annotations

import gc
import logging
import time
from itertools import chain
from typing import Any, Union

import torch
import wandb
from composer.core import (
    Precision,
    State,
    TimeUnit,
    ensure_time,
    get_precision_context,
)
from composer.loggers import Logger, MLFlowLogger, WandBLogger
from composer.trainer.trainer import _get_initial_device_train_microbatch_size
from composer.utils import dist, ensure_tuple
from llmfoundry.interfaces import CallbackWithConfig
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from compose_rl.ppo.buffer import MinibatchRolloutBuffer
from compose_rl.ppo.model import ComposerHFPolicyModel, ComposerMosaicPolicy
from compose_rl.ppo.reward_manager import (
    ReferenceOutput,
    RewardManager,
    RewardOutput,
)
from compose_rl.registry_builders import build_kl_controller
from compose_rl.utils import (
    add_right_padding,
    compute_advantages,
    dist_compute_masked_mean_and_var,
    flip_pad_token_usage_for_generate,
    flip_pad_token_usage_in_ffn,
    get_decoded_sequence,
    get_log_probs,
    mask_eos,
    masked_mean,
    switch_left_to_right_padding,
)

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
Policy = Union[ComposerHFPolicyModel, ComposerMosaicPolicy]

__all__ = ['PPOCallback', 'env_generate']

log = logging.getLogger(__name__)


def env_generate(
    actor_critic: Policy,
    reward_manager: RewardManager,
    batch: dict,
    max_gen_len: int,
    precision: Precision,
    device_train_microbatch_size: int,
    generation_kwargs: dict,
    tokenizer: Tokenizer,
) -> tuple[dict[str, torch.Tensor],
           list[tuple[str, str]],
           ReferenceOutput,
           RewardOutput,
          ]:
    """Run generate from the model.

    Runs generate over a set of prompts in the batch. It also does extra computation
    that is required for later loss computation.

    Args:
        actor_critic (ComposerMosaicPolicy): Actor critic model to run generate over.
        reward_manager (RewardManager): Composes the reference IFT model and all reward models.
        batch (dict): The batch of data to run generate over.
        max_gen_len (int): Maximum generation length.
        precision (Precision): Precision to run computation.
        device_train_microbatch_size (int): Device train microbatch size for the training job.
            We need to do all log_prob computation with this in order to maintain numerics.
        generation_kwargs (dict): Generation keyword arguments.
        tokenizer (Tokenizer): The actor critic's tokenizer.

    Returns:
        partial_env_output (dict[str, tensor]): Partially complete dictionary of return elements suitable
            for PPO training
        untokenized_prompt_and_responses (list): List of [str, str] tuples, each containing the decoded
            prompt and responses tokens sequences, respectively
        ref_output (ReferenceOutput): Pair of tensors corresponding to the KL penalty and
            log prob sequences obtained from the reference model. If the reference model is non-blocking,
            this will be an AsyncResult object that will resolve to the described output.
        all_rewards (RewardOutput): Dictionary of tensors containing the reward output
            from each reward model managed by the reward manager. If reward model "X" is non-blocking,
            then all_rewards["X"] will be an AsyncResult object that will resolve to associated reward tensor.

    Note:
        Use the .get() method on an AsyncResult object (see Returns, above) to resolve it. This method
        is blocking.
    """
    prompt_tokens = batch['prompt']

    batch_size, _ = prompt_tokens.shape

    policy = actor_critic.model
    policy.eval()

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    if eos_token_id is None:
        raise ValueError(
            'Tokenizer does not have an eos token id. Please use a different tokenizer or add an eos token id.',
        )
    if pad_token_id is None:
        raise ValueError(
            'Tokenizer does not have a pad token id. Please use a different tokenizer or add a pad token id.',
        )

    with get_precision_context(precision):
        prompt_len = batch['prompt_len']

        with torch.no_grad():
            cur_device = prompt_tokens.device
            prompt_dtype = prompt_tokens.dtype

            # Adding a dummy forwards call.
            # We need this otherwise FSDP throws an error during a standard forward pass.
            policy(
                torch.tensor([[0]], dtype=torch.long, device=cur_device),
                attention_mask=torch.tensor([[1]],
                                            dtype=torch.bool,
                                            device=cur_device),
            )

            # Generate doesn't work if we unpad the FFN. So we need to check if we
            # need to flip the flag in the model.
            flipped_usage = flip_pad_token_usage_for_generate(policy)

            start_gen_time = time.time()
            # We don't need to include EOS tokens since we mask out EOS tokens below
            generated_dict = policy.generate(
                prompt_tokens,
                max_new_tokens=max_gen_len,
                return_dict_in_generate=True,
                synced_gpus=True,
                attention_mask=batch['prompt_attention_mask'],
                pad_token_id=pad_token_id,
                **generation_kwargs,
            )

            # We should flip the flag back after generate as needed.
            if flipped_usage:
                flip_pad_token_usage_in_ffn(policy)

            # Sequences are [batch, seq_len + generated_len], covering the initial prompt and generated values
            sequences = generated_dict['sequences']  # type: ignore

            num_tokens_generated = sequences.size(1) - prompt_tokens.size(1)

            log.info(
                f'It took {time.time() - start_gen_time} to generate {num_tokens_generated} tokens',
            )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            generated_len = torch.ones(
                batch_size,
                device=cur_device,
                dtype=prompt_dtype,
            ) * max_gen_len

            # If all the processes early exit generate, then we need to manually pad everything
            if prompt_tokens.size(1) + max_gen_len > sequences.size(1):
                len_to_pad = max_gen_len - (
                    sequences.size(1) - prompt_tokens.size(1)
                )

                extra_padding = torch.ones(
                    (batch_size, len_to_pad),
                    device=cur_device,
                    dtype=prompt_dtype,
                ) * eos_token_id
                sequences = torch.cat([sequences, extra_padding], dim=-1)

            # Sanity checking we're adding max_gen_len to prompt_tokens
            assert prompt_tokens.size(1) + max_gen_len == sequences.size(1)

            # Actions are what tokens the current policy would generate.
            actions = sequences[:, -max_gen_len:]

            right_padded_obs = switch_left_to_right_padding(
                sequences,
                prompt_len,
                max_gen_len,
                pad_token_id,
            )
            right_padded_attn_mask = torch.logical_not(
                torch.eq(right_padded_obs, pad_token_id),
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
                eos_token=eos_token_id,
                pad_token=pad_token_id,
            )

            untokenized_prompt_and_responses = []
            for i in range(batch_size):
                prompt = tokenizer.decode(  # type: ignore
                    right_padded_obs[i, :prompt_len[i]])
                generated_text = tokenizer.decode(  # type:  ignore
                    get_decoded_sequence(actions[i], generated_len[i],
                                               max_gen_len))
                untokenized_prompt_and_responses.append(
                    (prompt, generated_text),
                )

            # Making logits [batch_size, generated_len, vocab_size]
            # We need to recompute the logits here. Otherwise there are numerical differences
            # We also need to do it on the size of `device_train_microbatch_size` otherwise
            # there are numerical differences at training time.
            logits = []
            values = []

            input_model_kwargs = {
                'obs': right_padded_obs,
                'right_padded_attn_mask': right_padded_attn_mask,
                'prompt_len': prompt_len,
                'max_gen_len': max_gen_len,
                'action_mask': action_mask,
                'actions': actions,
            }

            for i in range(batch_size // device_train_microbatch_size):
                curr_kwargs = {
                    key: value[i * device_train_microbatch_size:(i + 1) *
                               device_train_microbatch_size]
                    if isinstance(value, torch.Tensor) else value
                    for key, value in input_model_kwargs.items()
                }
                cur_output = actor_critic(curr_kwargs)
                logits.append(cur_output['logits'])
                values.append(cur_output['values'])

            device_train_microbatch_logits = torch.cat(logits)
            device_train_microbatch_values = torch.cat(values)

            device_train_microbatch_log_probs = get_log_probs(
                logits=device_train_microbatch_logits,
                actions=actions,
                prompt_len=prompt_len,
                max_gen_len=max_gen_len,
            )

            # Need to add in the padding for the value function
            value_action_mask = torch.cat([
                action_mask,
                torch.zeros((batch_size, 1), device=cur_device),
            ],
                                          dim=-1)
            device_train_microbatch_values *= value_action_mask

            partial_env_output = {
                'actions': actions.detach(),
                'old_log_probs': device_train_microbatch_log_probs.detach(),
                'obs': right_padded_obs.detach(),
                'generated_len': generated_len,
                'action_mask': action_mask,
                'values': device_train_microbatch_values.detach(),
            }

            # Future implementations may change the way reward_seq_len is defined
            # e.g., if special formatting is applied
            reward_seq_len = prompt_len + generated_len

            ref_output, all_rewards = reward_manager(
                raw_untokenized_texts=untokenized_prompt_and_responses,
                right_padded_obses=right_padded_obs,
                attention_masks=right_padded_attn_mask,
                seq_lens=reward_seq_len,
                generated_lens=generated_len,
                prompt_lens=prompt_len,
                max_gen_length=max_gen_len,
                actions=actions,
                action_log_probs=device_train_microbatch_log_probs,
                device_train_microbatch_size=device_train_microbatch_size,
            )

    return (
        partial_env_output,
        untokenized_prompt_and_responses,
        ref_output,
        all_rewards,
    )


class PPOCallback(CallbackWithConfig):
    """Callback for managing PPO training in an RLHF loop.

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

        self.epochs_per_iteration = ensure_time(
            var_config.get('epoch_per_iteration', 1),
            TimeUnit.EPOCH,
        )
        assert self.epochs_per_iteration.unit == TimeUnit.EPOCH

        self.buffer = MinibatchRolloutBuffer(var_config['buffer'])
        self.kl_ctl = build_kl_controller(var_config['kl_controller'])

        self.kl_ift = []

        self.wandb_logger = None
        self.mlflow_logger = None
        self.prompts_and_gens = []
        self.iter_num = 0
        self.train_prompt_loader_state_dict = None
        self.train_prompt_loader = None

        if train_config.get('python_log_level', None) is not None:
            logging.getLogger('compose_rl').setLevel(
                train_config['python_log_level'].upper(),
            )
            logging.getLogger(__name__).setLevel(
                train_config['python_log_level'].upper(),
            )

    def init(self, state: State, logger: Logger):
        self.pad_token_idx = state.model.tokenizer.pad_token_id
        self.actor_critic = state.model

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

        self.iter_batch_size = self.num_batches_per_update * self.device_train_batch_size

        # The KL penalty in the reward should only exist if we aren't minimizing
        # the KL directly in the loss.
        kl_penalty_in_reward = True
        if hasattr(self.actor_critic.model.config, 'compute_kl_loss'):
            kl_penalty_in_reward = not self.actor_critic.model.config.compute_kl_loss

        self.reward_manager = RewardManager(
            config=self.reward_cfg,
            ref_config=self.ref_config,
            tokenizer=self.actor_critic.tokenizer,
            max_seq_len=self.max_seq_len,
            fsdp_config=self.non_train_fsdp_config,
            precision=state.precision,
            kl_penalty_in_reward=kl_penalty_in_reward,
        )

        # This is needed to ensure PyTorch 2.4 checkpointing doesn't break
        self.actor_critic.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=['Dummy input'],
            padding='longest',
            truncation=True,
            return_attention_mask=True,
        )

    def before_load(self, state: State, logger: Logger):
        del logger
        self.train_prompt_loader = state.train_dataloader

    def after_load(self, state: State, logger: Logger):
        del logger  # unused
        # This needs to be done here becuase callbacks are init'd before we attach
        # the dataloader as a property to state
        self.tokenizer = state.model.tokenizer
        self.train_prompt_loader_iter = iter(
            self.train_prompt_loader,  # pyright: ignore
        )

        if self.train_prompt_loader_state_dict is not None:
            self.train_prompt_loader.load_state_dict( # pyright: ignore
                self.train_prompt_loader_state_dict,
            )

    def iteration_start(self, state: State, logger: Logger):
        del logger  # unused

        batch = self._get_next_iter_prompts()
        # Update dataloader
        batch = state.device.batch_to_device(batch)
        self._interact_with_env(batch)
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

    def epoch_end(self, state: State, logger: Logger):
        del logger  # unused
        assert self.epochs_per_iteration == state._iteration_length
        if self.actor_critic.determine_early_stop():
            state.timestamp.epoch_in_iteration = self.epochs_per_iteration

    def iteration_end(self, state: State, logger: Logger):
        del logger  # unused
        self._log_generations_to_logger(state)
        self._increment_rl_iter()
        self.buffer.reset()
        self.buffer.set_state_dict(
            self.train_prompt_loader.state_dict(), # pyright: ignore
            0,
        )

    def _get_next_iter_prompts(self):
        """Gets the next iteration's batch of prompts."""
        batches = [
            self._get_single_batch_prompts()
            for _ in range(self.num_batches_per_update)
        ]

        ret_batch = {}
        for key in batches[0].keys():
            curr_values = []
            max_len = max([batch[key].shape[-1] for batch in batches])
            padding_key = None
            for batch in batches:
                # Take care of the prompt length here, no need for extra processing
                if key == 'prompt_len':
                    curr_values.append(batch[key])
                    continue

                bs, seq_len = batch[key].shape

                if key == 'prompt':
                    padding_key = self.pad_token_idx
                    if (batch[key][:, -1] == padding_key).any():
                        raise ValueError(
                            'The last token in the prompt should not be the pad token. Please double '
                            + 'check the dataloader and prompt and dataloader.',
                        )
                elif key == 'prompt_attention_mask':
                    padding_key = False

                pad = torch.ones(
                    (bs, max_len - seq_len),
                    dtype=batch[key].dtype,
                ) * padding_key
                curr_values.append(torch.cat([pad, batch[key]], dim=-1))

            ret_batch[key] = torch.cat(curr_values)

        return ret_batch

    def _get_single_batch_prompts(self):
        """Gets a single batch of prompts from the dataloader."""
        try:
            return next(self.train_prompt_loader_iter)
        except StopIteration:
            # Reset the iterator to the beginning of the dataloader
            self.train_prompt_loader_iter = iter(
                self.train_prompt_loader,  # pyright: ignore
            )
            # Get the first sample from the dataloader
            return next(self.train_prompt_loader_iter)

    def _interact_with_env(self, batch: dict[str, torch.Tensor]):
        """Have the policy interact with the environment.

        Here, we redo microbatching, and run generate appropriately. We add the environment
        interactions to the buffer.

        Args:
            batch (dict): the iteration level batch we want to interact with the environment.
        """
        # Determine the number of generating calls we want to make
        # We can have the generate size be greater than the device train microbatch size
        num_gen_calls = self.num_batches_per_update * self.device_train_batch_size // self.device_generate_batch_size

        gen_batch_partial_outputs = []
        for i in range(num_gen_calls):
            gen_batch = self._extract_minibatch(
                batch=batch,
                idx=i,
                minibatch_size=self.device_generate_batch_size,
            )

            env_outputs, prompts_and_gens, ref_outputs, all_rewards_dict = env_generate(
                actor_critic=self.actor_critic, # pyright: ignore
                reward_manager=self.reward_manager,
                batch=gen_batch,
                max_gen_len=self.max_gen_len,
                precision=self.precision,
                device_train_microbatch_size=self.device_train_microbatch_size,
                generation_kwargs=self.generation_kwargs,
                tokenizer=self.tokenizer,
            )

            self.prompts_and_gens.extend(prompts_and_gens)

            gen_batch_partial_outputs.append(
                (env_outputs, ref_outputs, all_rewards_dict),
            )

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

    def _resolve_outputs(
        self,
        iter_batch: dict[str, torch.Tensor],
        partial_outputs: list[tuple[dict, ReferenceOutput, RewardOutput]],
    ) -> dict[str, torch.Tensor]:
        """Resolve env/reference/reward outputs into a PPO minibatch.

        Args:
            iter_batch (dict): The batch for the current iteration.
            partial_outputs (list): A list of (env_output, reference_output, reward_output) tuples,
                one tuple for each generate batch size. This tuple is created from `env_generate`.

        Returns:
            output_minibatch (dict): The final minibatch from the environment, with all AsyncResult
                objects resolved and outputs processed for PPO training.
        """
        outputs = []
        for env_outs, ref_outs, rew_dict in partial_outputs:
            rew_outs = self.reward_manager.resolve_outputs(
                ref_output=ref_outs,
                reward_output=rew_dict,
                kl_ctl=self.kl_ctl,
                action_mask=env_outs['action_mask'],
                center_reward_mean=self.center_reward_mean,
            )
            env_outs.update(rew_outs)
            outputs.append(env_outs)

        env_outputs = {}
        # Repadding all observations, right padded attention masks to the same length for composer.
        max_len = max([output['obs'].shape[-1] for output in outputs])
        for output in outputs:
            output['obs'] = add_right_padding(
                output['obs'],
                max_len,
                self.pad_token_idx,
            )
            output['right_padded_attn_mask'] = torch.logical_not(
                torch.eq(output['obs'], self.pad_token_idx),
            )

        for key in outputs[0].keys():
            env_outputs[key] = torch.cat([output[key] for output in outputs])

        # Now that rewards are resolved, we can compute advantages
        env_outputs['advantages'] = compute_advantages(
            rewards=env_outputs['rewards'],
            values=env_outputs['values'],
            gamma=self.gamma,
            lambda_gae=self.lambda_gae,
        )

        batch_adv_mean, batch_adv_var = dist_compute_masked_mean_and_var(
            env_outputs['advantages'],
            env_outputs['action_mask'],
        )

        mean_ift = masked_mean(
            env_outputs['ift_kl'],
            env_outputs['action_mask'],
        )

        self.kl_ift.append(mean_ift.cpu())

        iter_batch.update(env_outputs)

        iter_batch.update({
            'max_gen_len':
                torch.ones(self.iter_batch_size).to(torch.int32) *
                self.max_gen_len,
            'adv_masked_mean':
                torch.ones(self.iter_batch_size) * batch_adv_mean.cpu(),
            'adv_masked_var':
                torch.ones(self.iter_batch_size) * batch_adv_var.cpu(),
            'ift_kl_scalar':
                torch.ones(self.iter_batch_size) * self.kl_ctl.value,
            'reward_std':
                torch.ones(self.iter_batch_size) *
                env_outputs['rewards'].std().to('cpu'),
        })

        # Moving minibatches to CPU to not take additional GPU memory
        for k, v in iter_batch.items():
            if hasattr(v, 'cpu'):
                iter_batch[k] = v.cpu()

        return iter_batch

    def _log_generations_to_logger(self, state: State):
        prompts_and_gens = list(
            chain(*dist.all_gather_object(self.prompts_and_gens)),
        )

        if dist.get_global_rank() == 0:
            if self.wandb_logger is not None:
                assert wandb.run is not None, 'wandb should have started the run'

                artifact = wandb.Artifact(
                    'generate_samples_' + str(wandb.run.id),
                    type='predictions',
                )

                text_table = wandb.Table(
                    data=prompts_and_gens,
                    columns=['prompt', 'generation'],
                )

                artifact.add(text_table, 'predictions')
                wandb.log_artifact(artifact)
                wandb.log({'generations': text_table},
                          step=state.timestamp.batch.value)

            if self.mlflow_logger is not None:
                self.mlflow_logger.log_table(
                    columns=['prompt', 'generations'],
                    rows=prompts_and_gens,
                    name=f'Prompt_generations_{self.iter_num}',
                )

        self.prompts_and_gens = []

    def _update_ift_kl(self):
        local_kl = torch.stack(self.kl_ift)

        global_ift_kl = torch.cat(dist.all_gather_object(local_kl))
        ift_kl_update = torch.mean(global_ift_kl)

        self.kl_ctl.update(
            ift_kl_update,
            self.num_batches_per_update * self.device_train_batch_size *
            dist.get_world_size(),
        )

        self.kl_ift = []

    def _increment_rl_iter(self):
        self.iter_num += 1

    def state_dict(self):
        return {
            'KL_ctl_state_dict': self.kl_ctl.state_dict(),
            'iter_num': self.iter_num,
            'train_prompt_loader':
                self.train_prompt_loader.state_dict(),  # pyright: ignore
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.kl_ctl.load_state_dict(state_dict['KL_ctl_state_dict'])
        self.iter_num = state_dict['iter_num']
        self.train_prompt_loader_state_dict = state_dict['train_prompt_loader']
