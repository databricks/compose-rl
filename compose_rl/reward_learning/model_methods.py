# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Reward Model Utilies."""

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

from compose_rl.reward_learning.hf_utils import SequenceClassifierOutput
from compose_rl.utils import (
    clear_mb_load_balancing_loss,
    extract_packed_chosen_rejected,
    get_mb_load_balancing_loss,
)

Tokenizer = Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]

LM_LOGIT_LOSSES = []


class PairwiseRewardEnum(Enum):
    BT = 'bt'
    BT_EURUS = 'bt_eurus'
    BELLMAN = 'bellman'
    BELLMAN_EURUS = 'bellman_eurus'


class ClassifierRewardEnum(Enum):
    BCE = 'bce'


def pairwise_forward(
    model: nn.Module,
    tokenizer: Tokenizer,
    batch: MutableMapping,
    policy_model_config: Optional[PretrainedConfig] = None,
    use_attention_sequence_id: bool = False,
    return_last: bool = True,
    return_lm_logits: bool = False,
) -> dict[str, torch.Tensor]:
    """Forwards the model for Reward Models.

    Args:
        model (nn.Module): Model we are forwarding.
        tokenizer (Tokenizer): Tokenizer for the model.
        batch (Dict[str, torch.LongTensor]): Batch over which we should forward the model.
            Note: this batch has chosen and rejected concated along the sequence dimension.
        policy_model_config: Policy model config.
        use_attention_sequence_id (bool): Whether we should use the attention sequence id.
        return_last (bool): Whether to only return the final score, or a value for every token
        return_lm_logits (bool): Whether to only return the logits from the lm_head
    """
    if policy_model_config is not None and hasattr(model, 'transformer'):
        clear_mb_load_balancing_loss(policy_model_config, model.transformer)

    batch_size, concat_seq_len = batch['input_ids'].shape
    pad_token_id = tokenizer.pad_token_id  # type: ignore
    if pad_token_id is None:
        raise ValueError('Tokenizer must have a PAD token.')

    # If we can use attention sequence ID, we use this logic branch.
    if use_attention_sequence_id:
        model_output = model(
            batch['input_ids'],
            attention_mask=batch['text_attention_mask'],
            sequence_id=batch['sequence_ids'],
            return_lm_logits=return_lm_logits,
        )

        chosen_scores, rejected_scores = extract_packed_chosen_rejected(
            input_tensor=model_output.scores,
            chosen_len=batch['chosen_len'],
            rejected_len=batch['rejected_len'],
            max_seq_len=concat_seq_len // 2,
            pad_token_id=pad_token_id,
        )

    else:
        # If we can't use attn_seq_id then we need to unpack each batch and
        # Pack along the batch dimension instead.

        chosen_inputs, rejected_inputs = extract_packed_chosen_rejected(
            input_tensor=batch['input_ids'],
            chosen_len=batch['chosen_len'],
            rejected_len=batch['rejected_len'],
            max_seq_len=concat_seq_len // 2,
            pad_token_id=pad_token_id,
        )

        chosen_attention_mask, rejected_attention_mask = extract_packed_chosen_rejected(
            input_tensor=batch['text_attention_mask'],
            chosen_len=batch['chosen_len'],
            rejected_len=batch['rejected_len'],
            max_seq_len=concat_seq_len // 2,
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

        # Dynamic Padding
        max_length = max(max(batch['chosen_len']), max(batch['rejected_len']))
        batch_cat_inputs = batch_cat_inputs[:, :max_length]
        batch_attn_mask = batch_attn_mask[:, :max_length]

        model_output = model(
            batch_cat_inputs,
            attention_mask=batch_attn_mask,
            return_lm_logits=return_lm_logits,
        )

        output_scores = model_output.scores

        # Extract out the chosen and rejected logits along the batch dimension
        # Expected Shape: (Batch Size, Max Seq. Length)
        chosen_scores = output_scores[:batch_size]
        rejected_scores = output_scores[batch_size:]

    if return_last:
        # Expected Shape: (Batch Size, 1)
        chosen_scores = torch.gather(
            chosen_scores,
            dim=1,
            index=batch['chosen_len'].view(-1, 1) - 1,
        )
        rejected_scores = torch.gather(
            rejected_scores,
            dim=1,
            index=batch['rejected_len'].view(-1, 1) - 1,
        )

    outputs: dict[str, torch.Tensor] = {
        'chosen_scores': chosen_scores,
        'rejected_scores': rejected_scores,
    }

    chosen_logits, rejected_logits = None, None
    if return_lm_logits:
        # Expected Shape: (Batch Size, Max Seq. Length, Vocab Size)
        chosen_logits = model_output.logits[:batch_size]
        rejected_logits = model_output.logits[batch_size:]

        outputs['chosen_logits'] = chosen_logits
        outputs['rejected_logits'] = rejected_logits

    if policy_model_config is not None and hasattr(model, 'transformer'):
        lbl = get_mb_load_balancing_loss(policy_model_config, model.transformer)
        if lbl is not None:
            outputs['lbl'] = lbl

    return outputs


def classifier_forward(
    model: nn.Module,
    tokenizer: Tokenizer,
    batch: MutableMapping,
    policy_model_config: Optional[PretrainedConfig] = None,
    use_attention_sequence_id: bool = False,
    return_last: bool = True,
    return_lm_logits: bool = False,
) -> dict[str, torch.Tensor]:

    model_output = model(
        batch['text'],
        attention_mask=batch['text_attention_mask'],
        return_lm_logits=return_lm_logits,
    )

    output_scores = model_output.scores
    if return_last:
        # Expected Shape: (Batch Size, 1)
        output_scores = torch.gather(
            output_scores,
            dim=1,
            index=batch['text_len'].view(-1, 1) - 1,
        )

    # We need to add the labels here to compute metrics
    outputs: dict[str, torch.Tensor] = {
        'output_scores': output_scores,
        'labels': batch['labels'],
    }

    return outputs


def pairwise_loss(
    outputs: SequenceClassifierOutput,
    batch: Mapping,
    loss_type: PairwiseRewardEnum,
) -> dict[str, torch.Tensor]:
    """Computes Reward Model loss.

    Given precomputed values this will compute the specified reward model loss.

    Args:
        outputs (SequenceClassifierOutput): Outputs from forwarding the model over the batch.
        batch (Mapping): Input batch of data.
        loss_type (str): Loss type that we should compute (e.g. dpo, ipo, or kto),
    """
    del batch
    chosen_scores = outputs['chosen_scores']
    rejected_scores = outputs['rejected_scores']

    partial_loss_dict = {}
    losses = torch.zeros_like(chosen_scores)
    if loss_type == PairwiseRewardEnum.BT:
        losses = -F.logsigmoid(chosen_scores - rejected_scores)
    elif loss_type == PairwiseRewardEnum.BT_EURUS:
        bt_loss = -F.logsigmoid(chosen_scores - rejected_scores)
        score_loss = -F.logsigmoid(
            chosen_scores,
        ) - F.logsigmoid(-rejected_scores)
        losses = bt_loss + score_loss

        partial_loss_dict = {
            'bt_loss': bt_loss,
            'score_loss': score_loss,
        }
    else:
        raise ValueError(f'Loss type: {loss_type} is not supported.')

    losses = losses.mean()

    loss_dict = {
        'chosen_rewards':
            chosen_scores.detach(),
        'rejected_rewards':
            rejected_scores.detach(),
        'margin': (chosen_scores - rejected_scores).detach(),
        'accuracy':
            (chosen_scores > rejected_scores).detach().to(torch.float32),
    }

    loss_dict.update(partial_loss_dict)

    if 'lbl' in outputs:
        losses += outputs['lbl']
        loss_dict['lbl'] = outputs['lbl']

    loss_dict['total'] = losses

    return loss_dict


def classifier_loss(
    outputs: SequenceClassifierOutput,
    batch: Mapping,
    loss_type: ClassifierRewardEnum,
) -> dict[str, torch.Tensor]:
    """Computes Classifier loss.

    Given precomputed values this will compute the specified classifier loss.

    Args:
        outputs (SequenceClassifierOutput): Outputs from forwarding the model over the batch.
        batch (Mapping): Input batch of data.
        loss_type (str): Loss type that we should compute (e.g. bce),
    """
    output_scores = outputs['output_scores']

    if loss_type == ClassifierRewardEnum.BCE:
        loss = F.binary_cross_entropy_with_logits(
            output_scores,
            batch['labels'],
        )
    else:
        raise NotImplementedError(f'Loss type: {loss_type} is not supported.')

    loss_dict = {
        'total': loss,
    }

    return loss_dict
