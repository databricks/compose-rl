# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""DPO Utils."""

from enum import Enum
from typing import Mapping, MutableMapping, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

from compose_rl.utils import (
    clear_mb_load_balancing_loss,
    extract_packed_chosen_rejected,
    get_batch_logp,
    get_mb_load_balancing_loss,
)


class DPOEnum(Enum):
    DPO = 'dpo'
    RPO = 'rpo'
    RCDPO = 'rcdpo'
    REBEL = 'rebel'
    IPO = 'ipo'
    KTO = 'kto'


def dpo_forward(
    model: nn.Module,
    tokenizer: Tokenizer,
    batch: MutableMapping,
    average_log_prob: bool = False,
    policy_model_config: Optional[PretrainedConfig] = None,
    use_attention_sequence_id: bool = False,
) -> dict[str, torch.Tensor]:
    """Forwards the model for dpo and get the chosen and rejected log probs.

    Args:
        model (nn.Module): Model we are forwarding.
        tokenizer (Tokenizer): Tokenizer for the model.
        batch (Dict[str, torch.LongTensor]): Batch over which we should forward the model.
            Note: this batch has chosen and rejected concated along the sequence dimension.
        average_log_prob (bool): Whether should we average the log probabilities.
        policy_model_config: Policy model config.
        use_attention_sequence_id (bool): Whether we should use the attention sequence id.
    """
    if policy_model_config is not None and hasattr(model, 'transformer'):
        clear_mb_load_balancing_loss(policy_model_config, model.transformer)

    batch_size, concat_seq_len = batch['input_ids'].shape
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError('Tokenizer must have a PAD token.')

    # If we can use attention sequence ID, we use this logic branch.
    # This is determined by a value set in `train_dpo.py`
    if use_attention_sequence_id:
        output_logits = model(
            batch['input_ids'],
            attention_mask=batch['text_attention_mask'],
            sequence_id=batch['sequence_ids'],
        ).logits

        chosen_logits, rejected_logits = extract_packed_chosen_rejected(
            output_logits,
            batch['chosen_len'],
            batch['rejected_len'],
            concat_seq_len // 2,
            pad_token_id=pad_token_id,
        )

    else:
        # If we can't use attn_seq_id then we need to unpack each batch and
        # Pack along the batch dimension instead.

        chosen_inputs, rejected_inputs = extract_packed_chosen_rejected(
            batch['input_ids'],
            batch['chosen_len'],
            batch['rejected_len'],
            concat_seq_len // 2,
            pad_token_id=pad_token_id,
        )

        chosen_attention_mask, rejected_attention_mask = extract_packed_chosen_rejected(
            batch['text_attention_mask'],
            batch['chosen_len'],
            batch['rejected_len'],
            concat_seq_len // 2,
            pad_token_id=0,
        )

        batch_cat_inputs = torch.cat([chosen_inputs, rejected_inputs], dim=0)
        batch_attn_mask = torch.cat(
            [
                chosen_attention_mask,
                rejected_attention_mask,
            ],
            dim=0,
        )

        output_logits = model(
            batch_cat_inputs,
            attention_mask=batch_attn_mask,
        ).logits

        # Extract out the chosen and rejected logits along the batch dimension
        chosen_logits = output_logits[:batch_size]
        rejected_logits = output_logits[batch_size:]

    chosen_labels, rejected_labels = extract_packed_chosen_rejected(
        batch['input_ids'],
        batch['chosen_len'],
        batch['rejected_len'],
        concat_seq_len // 2,
        pad_token_id=0,
    )

    chosen_logps = get_batch_logp(
        chosen_labels,
        chosen_logits,
        batch['prompt_len'],
        batch['chosen_len'],
        average_log_prob,
    )

    rejected_logps = get_batch_logp(
        rejected_labels,
        rejected_logits,
        batch['prompt_len'],
        batch['rejected_len'],
        average_log_prob,
    )

    outputs: dict[str, torch.Tensor] = {
        'policy_chosen_logp': chosen_logps,
        'policy_rejected_logp': rejected_logps,
    }

    if policy_model_config is not None and hasattr(model, 'transformer'):
        lbl = get_mb_load_balancing_loss(policy_model_config, model.transformer)
        if lbl is not None:
            outputs['lbl'] = lbl

    return outputs


def dpo_loss(
    outputs: CausalLMOutputWithPast,
    batch: Mapping,
    loss_type: DPOEnum,
    beta: float,
    label_smoothing: float,
    sft_alpha: float,
) -> dict[str, torch.Tensor]:
    """Computes DPO loss.

    Given precomputed values, the batch, and dpo-oriented specific values, this will compute the dpo_loss.

    Args:
        outputs (CausalLMOutputWithPast): Outputs from forwarding the model over the batch.
        batch (Mapping): Input batch of data.
        loss_type (str): Loss type that we should compute (e.g. dpo, ipo, or kto),
        beta (float): How much to regularize the policy model. We regularizethe policy less with
            the reference model as beta -> 0.
        label_smoothing: Represents conservativeness for the DPO loss. This assumes that
            preferences as noisy (preferences are flipped with probability label_smoothing).
        sft_alpha (float): Regularization weight for supervised finetuning loss (SFT) to
            be added to DPO type loss.
    """
    policy_chosen_logp = outputs['policy_chosen_logp']
    policy_rejected_logp = outputs['policy_rejected_logp']
    ref_chosen_logp = batch.get(
        'ref_chosen',
        torch.zeros_like(policy_chosen_logp),
    )
    ref_rejected_logp = batch.get(
        'ref_rejected',
        torch.zeros_like(policy_rejected_logp),
    )

    pi_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp

    logits = pi_logratios - ref_logratios  # Also known as h_{\pi_\theta}^{y_w,y_l}

    losses = torch.zeros_like(logits)
    if loss_type == DPOEnum.DPO:
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing) -
            F.logsigmoid(-beta * logits) * label_smoothing
        )
    elif loss_type == DPOEnum.RCDPO:
        # Adding reward-difference based label_smoothing = 1 - reward_bt_prob
        chosen_reward = batch['chosen_reward']
        rejected_reward = batch['rejected_reward']
        reward_diff = chosen_reward - rejected_reward
        reward_bt_prob = torch.sigmoid(reward_diff)
        rcdpo_losses = -F.logsigmoid(
            beta * logits,
        ) * reward_bt_prob - F.logsigmoid(
            -beta * logits,
        ) * (1 - reward_bt_prob)
        losses = rcdpo_losses
    elif loss_type == DPOEnum.RPO:
        # Reproducing the RPO loss from NVIDIA's paper: https://arxiv.org/pdf/2406.11704v1 page 13
        # Code: https://github.com/NVIDIA/NeMo-Aligner/blob/c92a3bf9c2d6312581982a8d1db30591855394c5/nemo_aligner/models/nlp/gpt/megatron_gpt_dpo_model.py#L261-L273
        eta = 1  # NOTE: Hardcoding this to be 1 as per the paper's recommendation
        chosen_reward = batch['chosen_reward']
        rejected_reward = batch['rejected_reward']
        reward_diff = chosen_reward - rejected_reward

        logsigmoid_a = F.logsigmoid(beta * logits)
        logsigmoid_b = F.logsigmoid(eta * reward_diff)
        logsigmoid_not_a = F.logsigmoid(-beta * logits)
        logsigmoid_not_b = F.logsigmoid(-eta * reward_diff)

        losses = torch.exp(logsigmoid_b) * (
            logsigmoid_b - logsigmoid_a
        ) + torch.exp(logsigmoid_not_b) * (logsigmoid_not_b - logsigmoid_not_a)
    elif loss_type == DPOEnum.REBEL:
        # Reproducing the REBEL loss from paper: https://arxiv.org/pdf/2404.16767 page 4
        # Code: https://github.com/ZhaolinGao/REBEL/blob/e0a6a190108a45c70b4920b58a4ccac8a09ab22b/src/tldr/rebel.py#L761-L777
        pi_logratios = policy_chosen_logp - policy_rejected_logp
        ref_logratios = ref_chosen_logp - ref_rejected_logp

        logits = pi_logratios - ref_logratios  # Also known as h_{\pi_\theta}^{y_w,y_l}

        chosen_reward = batch['chosen_reward']
        rejected_reward = batch['rejected_reward']
        reward_diff = chosen_reward - rejected_reward
        losses = (beta * logits - reward_diff)**2
        # beta represents 1/eta hparam from the paper
    elif loss_type == DPOEnum.IPO:
        losses = (logits - 1 / (2 * beta))**2
    elif loss_type == DPOEnum.KTO:
        chosen_KL = (policy_chosen_logp - ref_chosen_logp).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logp -
                       ref_rejected_logp).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logp - ref_chosen_logp
        rejected_logratios = policy_rejected_logp - ref_rejected_logp
        losses = torch.cat(
            (
                1 - F.sigmoid(beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        )
    else:
        raise ValueError(f'Loss type: {loss_type} is not supported.')
    if sft_alpha > 0:
        sft_losses = -1 * sft_alpha * policy_chosen_logp
        sft_losses_normalized = sft_losses / batch['chosen_len']
        losses_before_sft = losses.clone().detach()
        losses += sft_losses_normalized

    losses = losses.mean()

    chosen_rewards = beta * (policy_chosen_logp - ref_chosen_logp).detach()
    rejected_rewards = beta * (policy_rejected_logp -
                               ref_rejected_logp).detach()

    # Logging KL margins for comparing different methods
    chosen_KL = (policy_chosen_logp - ref_chosen_logp).detach()
    rejected_KL = (policy_rejected_logp - ref_rejected_logp).detach()
    margin_KL = (chosen_KL - rejected_KL).detach()
    loss_dict = {
        'chosen_rewards': chosen_rewards,
        'rejected_rewards': rejected_rewards,
        'margin': chosen_rewards - rejected_rewards,
        'chosen_KL': chosen_KL,
        'rejected_KL': rejected_KL,
        'margin_KL': margin_KL,
        'accuracy': (chosen_rewards > rejected_rewards).to(torch.float32),
    }
    if loss_type in [DPOEnum.RPO, DPOEnum.RCDPO, DPOEnum.REBEL]:
        # reward_diff is always defined if loss_type is RPO, RCDPO, or REBEL
        loss_dict['reward_diff'] = reward_diff.detach()  # type: ignore
    if sft_alpha > 0:
        # sft_losses_normalized is always defined if sft_alpha>0
        snl = sft_losses_normalized.detach()  # type: ignore
        loss_dict['sft_regularization_loss'] = snl
        # losses_before_sft is always defined if sft_alpha>0
        loss_dict[f'{loss_type.value}_loss'] = losses_before_sft  # type: ignore

    if 'lbl' in outputs:
        losses += outputs['lbl']
        loss_dict['lbl'] = outputs['lbl']

    loss_dict['total'] = losses

    return loss_dict
