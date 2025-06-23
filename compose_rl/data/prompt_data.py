# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Build a prompt dataset and dataloader for training."""

import logging
from typing import Any

import numpy as np
import torch
from streaming import StreamingDataset
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
)

import compose_rl.utils as utils

log = logging.getLogger(__name__)


def prompt_dataset_collate_fn(
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    batch: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Collator for prompt data.

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
        if key == 'vstar':
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


class PromptStreamingDataset(StreamingDataset):
    """Dataloader for streaming in prompts."""

    def __init__(
        self,
        max_gen_len: int,
        max_seq_len: int,
        **kwargs: dict[str, Any],
    ):
        self.max_gen_len = max_gen_len
        self.max_seq_len = max_seq_len
        super().__init__(**kwargs)

    def _read_binary_tokenized_sample(self, sample: dict[str, Any], key: str):
        decoded_arr = torch.from_numpy(
            np.frombuffer(sample[key],
                          dtype=np.int64)[:self.max_seq_len].copy(),
        )
        return decoded_arr

    # How to process a sample
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get an item from StreamingDataset at a given index.

        Args:
            idx (int): the index where we fetch the data in the StreamingDataset.
        """
        sample = super().__getitem__(idx)
        prompt = self._read_binary_tokenized_sample(sample, 'prompt')

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

        #vstar
        vstar = sample.get('vstar', None)
        if vstar is not None:
            item_dict['vstar'] = torch.Tensor([vstar])

        return item_dict
