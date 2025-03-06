# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import MutableMapping, Optional

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

import compose_rl.utils as utils


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


def composer_ppo_forward(
    batch: MutableMapping,
    model: torch.nn.Module,
) -> MutableMapping:
    """Forward pass for the Composer PPO model.

    Args:
        batch (MutableMapping): The batch to run forward over.
        model (torch.nn.Module): The PPO Actor Critic model to run forwards over.
    """
    model_forward_kwargs = {
        'attention_mask': batch['right_padded_attn_mask'],
        'output_hidden_states': True,
    }

    model_forward_kwargs['prompt_len'] = batch['prompt_len']
    model_forward_kwargs['action_mask'] = batch['action_mask']
    model_forward_kwargs['max_gen_len'] = batch['max_gen_len']

    actor_output = model(batch['obs'], **model_forward_kwargs)

    values = actor_output.values
    logits = actor_output.logits

    log_prob_outputs = utils.get_log_probs(
        logits,
        batch['actions'],
        batch['prompt_len'],
        batch['max_gen_len'],
    )

    return {
        'online_log_probs': log_prob_outputs,
        'logits': logits,
        'values': values,
    }


def ppo_loss(
    outputs: MutableMapping,
    batch: MutableMapping,
    value_clip_range: float,
    policy_clip_ratio: float,
    value_loss_weight: float,
    add_direct_kl_loss: bool = False,
) -> tuple[MutableMapping, torch.Tensor]:
    """Compute the PPO loss.

    Args:
        outputs (MutableMapping): The outputs from the forward pass.
        batch (MutableMapping): The batch to compute the loss over.
        value_clip_range (float): The value clip range.
        policy_clip_ratio (float): The policy clip ratio.
        value_loss_weight (float): The value loss weight.
        add_direct_kl_loss (bool): Whether to add the KL loss directly to the loss. Default: ``False``.
    """
    # v_preds: [bs, gen_len + 1] maps each sequence to a scalar. With zero padding
    # values: [bs, gen_len + 1] maps each sequence to a scalar. With zero padding
    # advantages: [bs, gen_len] advantage computation from ppo
    # log_probs: [bs, gen_len] log probability of each action
    # action_mask: [bs, gen_len] action mask

    advantages = batch['advantages']
    # Note: `values` are the outputs of the critic model at the start of the PPO epoch and are fixed throughout the epoch,
    # and `v_preds` are the outputs of the critic model using its current weights.
    # Tensors in `batch` are fixed throughout the PPO epoch, and
    # tensors in `outputs` are recomputed at the start of each step in the epoch.
    v_preds = outputs['values'][:, :-1] * batch['action_mask']
    v_preds = v_preds.to(advantages.dtype)

    values = batch['values'][:, :-1] * batch['action_mask']

    returns = advantages + values
    returns_mean = utils.masked_mean(returns, batch['action_mask'])
    returns_var = utils.masked_var(returns, batch['action_mask'])

    v_pred_clipped = torch.clamp(
        v_preds,
        values - value_clip_range,
        values + value_clip_range,
    )

    value_loss_1 = (v_preds - returns)**2
    value_loss_2 = (v_pred_clipped - returns)**2

    value_loss = 0.5 * utils.masked_mean(
        torch.max(value_loss_1, value_loss_2),
        batch['action_mask'],
    )
    value_clip_frac = utils.masked_mean(
        torch.gt(value_loss_1, value_loss_2).double(),
        batch['action_mask'],
    )

    online_log_probs, old_log_probs = outputs['online_log_probs'], batch[
        'old_log_probs']

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
    advantages = advantages.detach()

    policy_kl = utils.masked_mean(
        utils.approx_kl(online_log_probs, old_log_probs),
        batch['action_mask'],
    )
    online_ift_kl = utils.masked_mean(
        outputs['online_log_probs'] - batch['ift_log_probs'],
        batch['action_mask'],
    )

    ratio = torch.exp(online_log_probs - old_log_probs)

    policy_loss_1 = -advantages * ratio
    policy_loss_2 = -advantages * torch.clamp(
        ratio,
        1 - policy_clip_ratio,
        1 + policy_clip_ratio,
    )

    policy_loss = torch.max(policy_loss_1, policy_loss_2)
    policy_clip_frac = utils.masked_mean(
        torch.gt(policy_loss_1, policy_loss_2).double(),
        batch['action_mask'],
    )

    policy_loss = utils.masked_mean(policy_loss, batch['action_mask'])

    val_error = utils.masked_mean((v_preds - returns)**2, batch['action_mask'])

    return_dict = {
        'loss/value_loss': value_loss,
        'loss/policy_loss': policy_loss,
        'kl/policy_kl': policy_kl,
        'kl/ift_kl_scalar': batch['ift_kl_scalar'],
        'value_loss/clip_frac': value_clip_frac,
        'policy_loss/clip_frac': policy_clip_frac,
        'kl/online_ift_kl': online_ift_kl,
        'value_loss/returns_mean': returns_mean,
        'value_loss/returns_var': returns_var,
        'value_loss/value_error': val_error,
        'advantages/mean': utils.masked_mean(advantages, batch['action_mask']),
        'policy_loss/ratio': utils.masked_mean(ratio, batch['action_mask']),
        'value_loss/values': utils.masked_mean(values, batch['action_mask']),
        'value_loss/vpred': utils.masked_mean(v_preds, batch['action_mask']),
        'length/gen_length': batch['action_mask'].sum(dim=1).to(torch.float32),
    }

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

    # Detaching all values
    for key, value in return_dict.items():
        return_dict[key] = value.detach().cpu()

    return_dict['total'] = policy_loss + value_loss_weight * value_loss

    # If we want to directly minimize the KL Divergence, we can do so here
    # and it will not include the KL in the reward.
    if add_direct_kl_loss:
        return_dict['total'] += batch['ift_kl_scalar'][0] * online_ift_kl
        return_dict['loss/online_ift_kl'] = (
            batch['ift_kl_scalar'][0] * online_ift_kl
        )

    if 'lbl' in outputs and outputs['lbl'] is not None:
        return_dict['loss/lbl'] = outputs['lbl']
        return_dict['total'] += outputs['lbl']

    return return_dict, policy_kl.detach().cpu()
