# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Prompt-guided reward model."""

from abc import ABC, abstractmethod
import backoff
import logging
from typing import Any, MutableMapping
import requests

import torch


from compose_rl.reward_learning.base_reward import RewardModel, Tokenizer
from compose_rl.reward_learning.inference_model import InferenceRewardModel
from compose_rl.registry import PGRM_FORMATTER_REGISTRY

log = logging.getLogger(__name__)

# TODO: add assertion that PromptGuidedRewardModel is always in "document" mode

class PromptGuidedRewardModel(InferenceRewardModel):

    def __init__(self, cfg: dict[Any, Any], tokenizer: Tokenizer):
        super().__init__(cfg, tokenizer)
        # select the formatter class based on the cfg
        self.input_batch_to_prompt_strs_formatter = PGRM_FORMATTER_REGISTRY[cfg['pgrm_formatter']](cfg, tokenizer)
    def __call__(self, batch: MutableMapping) -> torch.Tensor:
        if 'raw_untokenized_texts' not in batch:
            raise ValueError("PromptGuidedRewardModel requires raw_untokenized_texts in batch")
        batch_size = batch['input_ids'].shape[0]
        
        @backoff.on_exception(  # pyright: ignore[reportUntypedFunctionDecorator]
            backoff.expo,
            exception=Exception,
            max_tries=self.max_retries + 1,
            max_value=30,
        )
        def call_predict_with_backoff(batch: MutableMapping) -> list[float]:
            input_strs: list[str] = self.input_batch_to_prompt_strs_formatter(batch)
            unprocessed_rewards = self._call_rm(input_strs)
            return unprocessed_rewards
        
        try:
            unprocessed_rewards = call_predict_with_backoff(batch)
        except Exception as e:
            # Retry limit has been reached. Raise the error :(
            error_msg = (
                'PROMPT GUIDED REWARD MODEL BACKOFF LIMIT EXCEEDED. ' +
                'Printing batch prompt+generations then raising last error...' +
                f'\nraw_untokenized_texts:\n{batch["raw_untokenized_texts"]}'
            )
            raise RuntimeError(error_msg) from e
        
        # Zero-pad and batch the rewards from the outputs (this is the sequence reward)
        # TODO: check that this is equivalent to the more verbose version in inference_model (e.g., lines 193-onwards)
        padded_reward_seqs = torch.zeros(batch['input_ids'].shape)
        padded_reward_seqs[batch['batch_indices'].squeeze(), batch['reward_indices'].squeeze()] = torch.tensor(unprocessed_rewards)
        return self.postprocess_reward(padded_reward_seqs)
    
            
    def _call_rm(self, input_str: list[str]) -> list[float]:
        """
        Calls the prompt-guided reward model.

        Args:
            input_str (list[str]): The input strings to call the reward model with. These should have the chat template
            applied already, but still be untokenized.

        Returns:
            list[float]: The rewards from the reward model.
        """
        requests.post(
            self._deployment_details['post_url'],
            headers=self._headers,
            json={
                'model': self._deployment_details['model'],
                'inputs': input_str,
            },
        )
        response = response.json()
        # These are document-level rewards, so only 1 reward score per "document" (i.e. per input_str)
        return [r['score'][0] for r in response['data']]
