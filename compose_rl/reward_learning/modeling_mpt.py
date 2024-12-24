# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
from llmfoundry.models.mpt.configuration_mpt import MPTConfig
from llmfoundry.models.mpt.modeling_mpt import MPTForCausalLM

from compose_rl.reward_learning.hf_utils import (
    SequenceClassifierOutput,
    ValueHead,
)


class MPTForSequenceClassification(MPTForCausalLM):

    def __init__(self, config: MPTConfig):
        super().__init__(config)
        self.value_head = ValueHead(
            n_labels=config.n_labels,
            hidden_size=config.d_model,
            p_dropout=config.attn_config['attn_pdrop'],
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_lm_logits: Optional[torch.FloatTensor] = None,
    ) -> SequenceClassifierOutput:

        outputs = super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            labels=labels,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=True,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
        )

        logits = None
        if return_lm_logits:
            logits = outputs.logits

        # Classification Head
        scores = self.value_head(
            outputs.hidden_states[-1],  # type: ignore
        ).squeeze(-1)  # (Batch Size, Sequence Length)

        return SequenceClassifierOutput(
            loss=outputs.loss,
            scores=scores,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
