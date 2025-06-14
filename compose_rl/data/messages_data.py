# Copyright 2025 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Build a prompt dataset and dataloader for training."""

import logging
from typing import Any

import torch
from streaming import StreamingDataset
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
)

import compose_rl.utils as utils

log = logging.getLogger(__name__)


def messages_dataset_collate_fn(
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    batch: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Collator for messages data.

    Args:
        batch (List[Dict[str, Any]]): A list of data samples to collate.
        tokenizer (PreTrainedTokenizer): The model's tokenizer.
        max_seq_len (int): The maximum sequence length of the model.
    """
    if tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer must have a PAD token.')

    ref_collate_fn = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        mlm_probability=0.0,
    )

    keys = batch[0].keys()
    collated_batch: dict[str, torch.Tensor] = {}
    for key in keys:
        cur_values = [item[key] for item in batch]
        if key in ['prompt_len']:
            collated_batch[key] = torch.stack(cur_values).squeeze(dim=1)
            continue
        if key == 'prompt_id':
            collated_batch[key] = torch.tensor(cur_values)
            continue
        if key in ['verified_answer']:
            collated_batch[key] = list(  # pyright: ignore[reportGeneralTypeIssues]
                utils.flatten(cur_values),
            )
            continue

        collated_batch[key] = ref_collate_fn(cur_values)['input_ids']

    collated_batch['prompt_attention_mask'] = torch.logical_not(
        torch.eq(collated_batch['prompt'],
                 tokenizer.pad_token_id),  # type: ignore
    )
    return collated_batch


class MessagesStreamingDataset(StreamingDataset):
    """Dataloader for streaming in messages and converting to prompts."""

    def __init__(
        self,
        max_gen_len: int,
        max_seq_len: int,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs: dict[str, Any],
    ):
        self.max_gen_len = max_gen_len
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        super().__init__(**kwargs)

    def _validate_messages(self, messages: list[dict[str, str]]) -> bool:
        """Validate the messages. A valid message is a list of dictionaries with the following keys:
        - role: str
        - content: str
        
        Args:
            messages (list[dict[str, str]]): The messages to validate.
        Returns:
            bool: True if the messages are valid, False otherwise.
        """
        if not isinstance(messages, list):
            return False
        for message in messages:
            if not isinstance(message, dict):
                return False
            if 'role' not in message:
                return False
            if 'content' not in message:
                return False
            if not isinstance(message['content'], str):
                return False
        return True

    def _tokenize_messages(self, messages: list[str]) -> torch.Tensor:
        if not self._validate_messages(messages):
            raise ValueError(f'Invalid messages received. Got: {messages=}')
        return torch.Tensor(self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        ))

    # How to process a sample
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get an item from StreamingDataset at a given index.

        Args:
            idx (int): the index where we fetch the data in the StreamingDataset.
        """
        sample = super().__getitem__(idx)
        messages = sample['messages']
        prompt: torch.Tensor = self._tokenize_messages(messages)

        # TODO (bcui): Maybe add in an option to truncate a prompt by a given length?
        if len(prompt) + self.max_gen_len > self.max_seq_len:
            truncate_len = len(prompt) + self.max_gen_len - self.max_seq_len
            log.info(f'Truncating prompt by: {truncate_len}')
            prompt = prompt[:-truncate_len]

        prompt_len = torch.Tensor([len(prompt)]).to(dtype=torch.int64)
        # Send the prompt id along with prompt data
        item_dict = {
            'prompt_id': idx,
            'prompt': prompt,
            'prompt_len': prompt_len,
            'messages': messages,
        }

        verified_answer = sample.get('verified_answer', None)
        if verified_answer:
            if isinstance(verified_answer, str):
                _answer = verified_answer
            else:
                try:
                    _answer = verified_answer.decode('utf-8', errors='strict')
                except UnicodeDecodeError as e:
                    log.error(
                        f'Failed to decode verifed_answer with error: {e}',
                    )
                    _answer = ''

            item_dict['verified_answer'] = _answer  # type: ignore

        return item_dict
