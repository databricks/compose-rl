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

log = logging.getLogger(__name__)

# TODO: add assertion that PromptGuidedRewardModel is always in "document" mode

class PromptGuidedRewardModel(InferenceRewardModel):

    def __init__(self, cfg: dict[Any, Any], tokenizer: Tokenizer):
        # TODO: fix the circular import issue and move this import to the top of the file
        from compose_rl.registry import PGRM_FORMATTER_REGISTRY
        # select the formatter class based on the cfg
        self.input_batch_to_prompt_strs_formatter = PGRM_FORMATTER_REGISTRY[cfg['pgrm_formatter']](cfg, tokenizer)  
        super().__init__(cfg, tokenizer)

    def perform_health_check_on_model(self):
        """
        Performs a health check on the model by passing a dummy batch through it.
        """
        right_padded_obses = torch.tensor([[0] * 56])
        seq_lens = [[55]]
        dummy_batch = {
            'input_ids': right_padded_obses,
            'seq_lens': seq_lens,
            'seq_reward': True,
            'is_inference': True,
            'raw_untokenized_texts': [
"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The capital of France is Paris.<|eot_id|>
""",
            ],
        }
        output = self(dummy_batch)
        log.debug(f'PGRM perform_health_check_on_model output: {output}')

    def __call__(self, batch: MutableMapping) -> torch.Tensor:
        log.debug(f'PGRM __call__ received batch: {batch}')
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
        padded_reward_seqs[torch.arange(batch_size), torch.tensor(batch['seq_lens']).squeeze()] = torch.tensor(unprocessed_rewards)
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
        # post_dict = {
        #     'post_url': self._deployment_details['post_url'],
        #     'headers': self._headers,
        #     'json_input': {
        #         'model': self._deployment_details['model'],
        #         'input': input_str,
        #     },
        # }
        # print('!!!!ABOUT TO POST TO RM!!!!\n'*10)
        # print(f'{post_dict=}')
        response = requests.post(
            self._deployment_details['post_url'],
            headers=self._headers,
            json={
                'model': self._deployment_details['model'],
                'input': input_str,
            },
        )
        # print('!!!!POST TO RM COMPLETE!!!!\n'*10)
        response = response.json()
        # print(f'{response=}')
        # These are document-level rewards, so only 1 reward score per "document" (i.e. per input_str)
        return [r['score'][0] for r in response['data']]
