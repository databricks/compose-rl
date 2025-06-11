# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum
from typing import MutableMapping, Optional

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

import compose_rl.utils as utils


class OnPolicyEnum(Enum):
    PPO = 'ppo'
    GRPO = 'grpo'


class ALGORITHM_TYPE(set, Enum):
    CRITIC_FREE = {OnPolicyEnum.GRPO}
    ACTOR_CRITIC = {OnPolicyEnum.PPO}
    CLIPPED_PG = {OnPolicyEnum.PPO, OnPolicyEnum.GRPO}


@dataclass
class CausalLMOutputWithPastAndValues(CausalLMOutputWithPast):
    """Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        values (`torch.FloatTensor) of shape `(batch_size, sequence_length)`:
            Value function output for each token in the sequence.
    """

    loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    values: Optional[torch.Tensor] = None


def prepare_critic_values_for_training(
    all_values: torch.FloatTensor,
    prompt_len: torch.Tensor,
    max_gen_len: int,
    action_mask: torch.Tensor,
    zero_pad: bool,
):
    """Prepare the values for learning the critic.

    Args:
        all_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`): Value function output for each token
            in the sequence.
        prompt_len (`torch.Tensor` of shape `(batch_size,)`): Length of the prompt.
        max_gen_len (int): Maximum generation length of the model.
        action_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`): Mask for the actions.

    Returns:
        `torch.FloatTensor` of shape `(batch_size, max_gen_len)`: Values for learning the critic.
    """
    bs, _ = all_values.shape

    # Getting the appropriate generated values
    values = utils.get_batched_generated_values(
        all_values,
        prompt_len,
        max_gen_len,
    )
    values *= action_mask

    if zero_pad:
        zero_pad_tensor = torch.zeros((bs, 1),
                                      device=values.device,
                                      dtype=values.dtype)
        values = torch.cat([values, zero_pad_tensor], dim=-1)

    return values


def composer_online_rl_forward(
    batch: MutableMapping,
    model: torch.nn.Module,
    loss_type: OnPolicyEnum,
) -> MutableMapping:
    """Forward pass for the Composer PPO model.

    Args:
        batch (MutableMapping): The batch to run forward over.
        model (torch.nn.Module): The PPO Actor Critic model to run forwards over.
        loss_type (str): The loss type which decides whether to use critic-free or not. Defaults to ``ppo``.
    """
    model_forward_kwargs = {
        'attention_mask': batch['right_padded_attn_mask'],
        'output_hidden_states': True,
    }

    model_forward_kwargs['prompt_len'] = batch['prompt_len']
    model_forward_kwargs['action_mask'] = batch['action_mask']
    model_forward_kwargs['max_gen_len'] = batch['max_gen_len']

    actor_output = model(batch['obs'], **model_forward_kwargs)

    logits = actor_output.logits

    log_prob_outputs = utils.get_log_probs(
        logits,
        batch['actions'],
        batch['prompt_len'],
        batch['max_gen_len'],
    )

    return_dict = {
        'online_log_probs': log_prob_outputs,
        'logits': logits,
    }

    if loss_type == OnPolicyEnum.PPO:
        if 'values' not in actor_output:
            raise ValueError(
                'The actor output does not contain values. Please check the model.',
            )
        values = actor_output.values
        return_dict['values'] = values

    return return_dict


def critic_loss(
    outputs: MutableMapping,
    batch: MutableMapping,
    loss_type: OnPolicyEnum,
    value_clip_range: float = 0.2,
) -> MutableMapping:
    if loss_type == OnPolicyEnum.PPO:
        advantages = batch['advantages']
        v_preds = outputs['values'][:, :-1] * batch['action_mask']
        v_preds = v_preds.to(advantages.dtype)

        values = batch['values'][:, :-1] * batch['action_mask']

        returns = advantages + values
        returns_mean = utils.sample_wise_masked_mean(
            returns,
            batch['action_mask'],
        )
        returns_var = utils.masked_var(returns, batch['action_mask'])

        v_pred_clipped = torch.clamp(
            v_preds,
            values - value_clip_range,
            values + value_clip_range,
        )

        value_loss_1 = (v_preds - returns)**2
        value_loss_2 = (v_pred_clipped - returns)**2

        value_loss = 0.5 * utils.sample_wise_masked_mean(
            torch.max(value_loss_1, value_loss_2),
            batch['action_mask'],
        )
        value_clip_frac = utils.sample_wise_masked_mean(
            torch.gt(value_loss_1, value_loss_2).double(),
            batch['action_mask'],
        )

        val_error = utils.sample_wise_masked_mean((v_preds - returns)**2,
                                                  batch['action_mask'])

        critic_dict = {
            'loss/value_loss':
                value_loss,
            'value_loss/values':
                utils.sample_wise_masked_mean(
                    values,  # pyright: ignore
                    batch['action_mask'],
                ),
            'value_loss/vpred':
                utils.sample_wise_masked_mean(
                    v_preds,  # pyright: ignore
                    batch['action_mask'],
                ),
            'value_loss/clip_frac':
                value_clip_frac,
            'value_loss/returns_mean':
                returns_mean,
            'value_loss/returns_var':
                returns_var,
            'value_loss/value_error':
                val_error,
        }
        return critic_dict
    else:
        raise ValueError(f'Critic Loss not implemented for {loss_type}')


def policy_loss(
    advantages: torch.Tensor,
    outputs: MutableMapping,
    batch: MutableMapping,
    loss_type: OnPolicyEnum,
    policy_clip_ratio: float = 0.15,
    policy_clip_high_ratio: float | None = None,
    length_normalize_policy_loss: bool = True,
    kl_estimator: Optional[str] = 'k1',
    kl_clip_range: Optional[float] = 40.0,
) -> MutableMapping:

    if loss_type in ALGORITHM_TYPE.CLIPPED_PG:
        online_log_probs, old_log_probs = outputs['online_log_probs'], batch[
            'old_log_probs']
        old_entropies = batch['old_entropies']

        policy_kl_dict = utils.approx_kl(
            log_p=online_log_probs,
            log_q=old_log_probs,
            kl_clip_range=kl_clip_range,
        )
        online_ift_kl_dict = utils.approx_kl(
            log_p=batch['ift_log_probs'],
            log_q=outputs['online_log_probs'],
            kl_clip_range=kl_clip_range,
        )

        # Normalize the KL divergence by the length depending on the flag
        if length_normalize_policy_loss:
            policy_kl = utils.sample_wise_masked_mean(
                policy_kl_dict[kl_estimator], # pyright: ignore
                batch['action_mask'],
            )
            online_ift_kl = utils.sample_wise_masked_mean(
                online_ift_kl_dict[kl_estimator], # pyright: ignore
                batch['action_mask'],
            )
        else:
            policy_kl = utils.masked_sum(
                policy_kl_dict[kl_estimator], # pyright: ignore
                batch['action_mask'],
            )
            online_ift_kl = utils.masked_sum(
                online_ift_kl_dict[kl_estimator], # pyright: ignore
                batch['action_mask'],
            )

        ratio = torch.exp(online_log_probs - old_log_probs)
        policy_loss_1 = -advantages * ratio

        # Use the same clip ratio for both sides if clip high ratio is not provided
        if policy_clip_high_ratio is None:
            policy_clip_high_ratio = policy_clip_ratio

        policy_loss_2 = -advantages * torch.clamp(
            ratio,
            1 - policy_clip_ratio,
            1 + policy_clip_high_ratio,
        )

        policy_loss = torch.max(policy_loss_1, policy_loss_2)
        policy_clip_frac = utils.masked_mean(
            torch.gt(policy_loss_1, policy_loss_2).double(),
            batch['action_mask'],
        )

        if length_normalize_policy_loss:
            policy_loss = utils.sample_wise_masked_mean(
                policy_loss,
                batch['action_mask'],
            )
        else:
            policy_loss = utils.masked_sum(policy_loss, batch['action_mask'])

        policy_token_kl_logging_dict = {
            f'token_kl/policy_token_kl_{k}_estimate':
                utils.sample_wise_masked_mean(
                    v,
                    batch['action_mask'],
                ) for k, v in policy_kl_dict.items()
        }
        policy_seq_kl_logging_dict = {
            f'seq_kl/policy_seq_kl_{k}_estimate':
                utils.masked_sum(
                    v,
                    batch['action_mask'],
                ) for k, v in policy_kl_dict.items()
        }
        online_ift_token_kl_logging_dict = {
            f'token_kl/online_ift_token_kl_{k}_estimate':
                utils.sample_wise_masked_mean(
                    v,
                    batch['action_mask'],
                ) for k, v in online_ift_kl_dict.items()
        }
        online_ift_seq_kl_logging_dict = {
            f'seq_kl/online_ift_seq_kl_{k}_estimate':
                utils.masked_sum(
                    v,
                    batch['action_mask'],
                ) for k, v in online_ift_kl_dict.items()
        }

        policy_dict = {
            'loss/policy_loss':
                policy_loss,
            'kl/policy_kl':
                policy_kl,
            'kl/online_ift_kl':
                online_ift_kl,
            'kl/ift_kl_scalar':
                batch['ift_kl_scalar'],
            **policy_token_kl_logging_dict,
            **policy_seq_kl_logging_dict,
            **online_ift_token_kl_logging_dict,
            **online_ift_seq_kl_logging_dict,
            'policy_loss/clip_frac':
                policy_clip_frac,
            'policy_loss/ratio':
                utils.sample_wise_masked_mean(ratio, batch['action_mask']),
            'gen/gen_length':
                batch['action_mask'].sum(dim=1).to(torch.float32),
            'gen/entropy':
                old_entropies,
            'advantages/mean':
                utils.sample_wise_masked_mean(advantages, batch['action_mask']),
        }
        return policy_dict
    else:
        raise ValueError(f'Policy loss not implemented for {loss_type}')


def online_rl_loss(
    outputs: MutableMapping,
    batch: MutableMapping,
    loss_type: OnPolicyEnum,
    value_clip_range: float = 0.2,
    value_loss_weight: float = 0.2,
    policy_clip_ratio: float = 0.15,
    policy_clip_high_ratio: float | None = None,
    length_normalize_policy_loss: bool = True,
    add_direct_kl_loss: bool = False,
    kl_estimator: Optional[str] = 'k1',
    kl_clip_range: Optional[float] = 40.0,
) -> MutableMapping:
    """Compute the online RL loss.

    Args:
        outputs (MutableMapping): The outputs from the forward pass.
        batch (MutableMapping): The batch to compute the loss over.
        loss_type (str): The loss type which decides whether to use critic-free or not. Defaults to ``ppo``.
        value_clip_range (float): The value clip range.
        value_loss_weight (float): The value loss weight.
        policy_clip_ratio (float): The policy clip ratio.
        policy_clip_high_ratio (float | None): The high policy clip ratio. Default: ``None``.
        length_normalize_policy_loss (bool): Whether to normalize the policy loss by the length of the sequence. Default: ``True``.
        add_direct_kl_loss (bool): Whether to add the KL loss directly to the loss. Default: ``False``.
        kl_estimator (str): The KL estimator to use. Default: ``'k1'``.
        kl_clip_range (float): The clip range for the KL divergence. Default: ``40.0``.
    """
    # log_probs: [bs, gen_len] log probability of each action
    # action_mask: [bs, gen_len] action mask
    # advantages: [bs, gen_len] advantage computation from PPO or GRPO
    # v_preds: [bs, gen_len + 1] maps each sequence to a scalar. With zero padding
    # values: [bs, gen_len + 1] maps each sequence to a scalar. With zero padding
    # Note: `values` are the outputs of the critic model at the start of the PPO epoch and are fixed throughout the epoch,
    # and `v_preds` are the outputs of the critic model using its current weights.
    # Tensors in `batch` are fixed throughout the PPO epoch, and
    # tensors in `outputs` are recomputed at the start of each step in the epoch.

    return_dict = {}
    advantages = batch['advantages']

    # 1. Critic Loss
    if loss_type in ALGORITHM_TYPE.ACTOR_CRITIC:
        value_dict = critic_loss(
            outputs=outputs,
            batch=batch,
            value_clip_range=value_clip_range,
            loss_type=loss_type,
        )

        adv_masked_mean = batch['adv_masked_mean']
        adv_masked_var = batch['adv_masked_var']

        # If adv masked mean isn't just a scalar, then should be duplicated across all dimensions
        # TODO: add check for the tensor is duplicated, make this into a utils?
        if adv_masked_mean.dim() > 0:
            adv_masked_mean = adv_masked_mean[0]
        if adv_masked_var.dim() > 0:
            adv_masked_var = adv_masked_var[0]

        # Normalizing advantages over each minibatch
        advantages = utils.masked_normalize(
            batch['advantages'],
            adv_masked_mean,
            adv_masked_var,
        )

        return_dict.update(**value_dict)

    advantages = advantages.detach()

    # 2. Policy Loss
    policy_dict = policy_loss(
        advantages=advantages,
        outputs=outputs,
        batch=batch,
        loss_type=loss_type,
        policy_clip_ratio=policy_clip_ratio,
        policy_clip_high_ratio=policy_clip_high_ratio,
        length_normalize_policy_loss=length_normalize_policy_loss,
        kl_estimator=kl_estimator,
        kl_clip_range=kl_clip_range,
    )

    return_dict.update(**policy_dict)

    for key, value in batch.items():
        # This logic handles reward logging a little differently than other quantities.
        # For rewards shaped as [batch, actions] we log (1) the per-sequence masked average
        # and (2) the per-sequence masked sum over actions, both size [batch].
        # We then average over [batch], so the interpretation is (1) the average per-token
        # reward, and (2) the average total reward.
        if 'reward' in key:
            if value.shape == batch['action_mask'].shape:
                # Average reward per timestep
                return_dict['env/' + str(key) + '_mean'] = utils.masked_mean(
                    value,
                    batch['action_mask'],
                    dim=1,
                ).mean(dim=0)
                # Total reward over timesteps
                return_dict['env/' + str(key) + '_total'] = utils.masked_sum(
                    value,
                    batch['action_mask'],
                    dim=1,
                ).mean(dim=0)
            else:
                # If this value is not [batch, actions] shaped, just do a
                # vanilla mean.
                return_dict['env/' + str(key)] = value.mean(dim=0)
        if 'ift_kl' == key:
            return_dict['kl/' + str(key)] = utils.masked_mean(
                value,
                batch['action_mask'],
            )

    # 3. Compute the total loss
    return_dict['total'] = return_dict['loss/policy_loss']
    if loss_type in ALGORITHM_TYPE.ACTOR_CRITIC:
        # Add value loss to total loss
        return_dict['total'] += value_loss_weight * return_dict[
            'loss/value_loss']  # pyright: ignore
    # If we want to directly minimize the KL Divergence, we can do so here
    # and it will not include the KL in the reward.
    if add_direct_kl_loss:
        return_dict['total'] += batch['ift_kl_scalar'][0] * return_dict[
            'kl/online_ift_kl']
        return_dict['loss/online_ift_kl'] = (
            batch['ift_kl_scalar'][0] * return_dict['kl/online_ift_kl']
        )
    if 'lbl' in outputs and outputs['lbl'] is not None:
        return_dict['loss/lbl'] = outputs['lbl']
        return_dict['total'] += outputs['lbl']

    # Detaching all return_dict values
    for key, value in return_dict.items():
        if key not in 'total':
            return_dict[key] = value.detach().cpu()

    return return_dict