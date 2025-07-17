# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Build a reward dataset and dataloader for training."""

import logging
from typing import Any

import numpy as np
import torch
from streaming import StreamingDataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer

log = logging.getLogger(__name__)


def pairwise_preference_dataset_collate_fn(
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
    data: list[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Collator for preference data.

    This will concatenate chosen and rejected and create the appropriate attention mask
    along with adding a sequence ID to the batch.

    Args:
        tokenizer (Tokenizer): The model's tokenizer.
        max_seq_len (int): The maximum sequence length of the model.
        data (list[dict[str, torch.Tensor]]): The preference data to collate.
    """
    if max_seq_len % 2 != 0:
        raise (
            ValueError(
                f'max_seq_len must be even for splitting evenly between chosen and rejected sequences. Found {max_seq_len=}',
            )
        )

    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer must have an EOS token.')
    if tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer must have a PAD token.')

    ref_collate_fn = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        mlm_probability=0.0,
    )

    input_ids = []
    attention_masks = []
    chosen_lens = []
    rejected_lens = []
    prompt_lens = []
    sequence_id = []
    chosen_rewards = []
    rejected_rewards = []

    for sample in data:
        chosen = sample['chosen']
        rejected = sample['rejected']
        prompt_len = sample['prompt_len']
        chosen_len = sample['chosen_len']
        rejected_len = sample['rejected_len']

        # Note: if we do any truncation, we force the last token to be EOS
        # https://github.com/mosaicml/RLHF/issues/101

        # Add the eos token if it's not in the chosen sample
        if chosen[-1] != tokenizer.eos_token_id:
            chosen[-1] = tokenizer.eos_token_id  # type: ignore
        if rejected[-1] != tokenizer.eos_token_id:
            rejected[-1] = tokenizer.eos_token_id  # type: ignore

        pad_len = max_seq_len - chosen_len - rejected_len
        cat_batch = torch.cat([chosen, rejected], dim=-1)

        if pad_len < 0:
            # We should truncate chosen and rejected by the same amount
            truncate_len = abs(pad_len // 2) + 1

            log.warning((
                f'Chosen length: {len(chosen)} Rejected length: {len(rejected)}'
                f' are too long for max_seq_len: {max_seq_len}'
                f' truncating each sequence by {truncate_len[0]} tokens.'
            ))

            # Truncate each value by truncate length, and make the last token EOS
            chosen = chosen[:-truncate_len]
            chosen[-1] = tokenizer.eos_token_id  # type: ignore

            rejected = rejected[:-truncate_len]
            rejected[-1] = tokenizer.eos_token_id  # type: ignore

            cat_batch = torch.cat([chosen, rejected], dim=-1)

            chosen_len = torch.tensor([len(chosen)])
            rejected_len = torch.tensor([len(rejected)])

            pad_len = max_seq_len - chosen_len - rejected_len

        if pad_len > 0:
            cat_batch = torch.cat(
                [
                    cat_batch,
                    torch.ones(int(pad_len.item()), dtype=cat_batch.dtype) *
                    tokenizer.pad_token_id,  # type: ignore
                ],
                dim=-1,  # type: ignore
            )

        attention_mask = torch.logical_not(
            torch.eq(cat_batch, tokenizer.pad_token_id),  # type: ignore
        )

        cur_sequence_id = torch.tensor(([0] * chosen_len) +
                                       ([1] * rejected_len) +
                                       ([-1] * max(0, int(pad_len.item()))),)
        sequence_id.append(cur_sequence_id)

        input_ids.append(cat_batch)
        attention_masks.append(attention_mask)
        chosen_lens.append(chosen_len)
        rejected_lens.append(rejected_len)
        prompt_lens.append(prompt_len)
        if 'chosen_reward' in sample:
            chosen_rewards.append(sample['chosen_reward'])
            rejected_rewards.append(sample['rejected_reward'])

    input_ids = ref_collate_fn(input_ids)['input_ids']
    attention_masks = torch.stack(attention_masks)
    sequence_id = torch.stack(sequence_id)

    chosen_lens = torch.cat(chosen_lens)
    rejected_lens = torch.cat(rejected_lens)
    prompt_lens = torch.cat(prompt_lens)
    return_dict = {
        'chosen_len': chosen_lens,
        'rejected_len': rejected_lens,
        'prompt_len': prompt_lens,
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'sequence_id': sequence_id,
    }
    if len(chosen_rewards) > 0:
        chosen_rewards = torch.stack(chosen_rewards)
        rejected_rewards = torch.stack(rejected_rewards)
        return_dict['chosen_reward'] = chosen_rewards
        return_dict['rejected_reward'] = rejected_rewards
    return return_dict


def finegrained_preference_dataset_collate_fn(
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
    data: dict,
) -> dict[str, Any]:
    """Collator for fine-grained preference data.

    Args:
        tokenizer (Tokenizer): The model's tokenizer.
        max_seq_len (int): The maximum sequence length of the model.
        data (dict): The preference data to collate.
    """
    del max_seq_len
    if tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer must have a PAD token.')
    if tokenizer.padding_side != "right":
        raise ValueError('Tokenizer must use right padding.')
    ref_collate_fn = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        mlm_probability=0.0,
    )

    batch = {}
    all_text = [item['text'] for item in data]
    batch['text'] = ref_collate_fn(all_text)['input_ids']
    batch['attention_mask'] = torch.logical_not(
        torch.eq(batch['text'], tokenizer.pad_token_id),  # type: ignore
    )
    bs, max_len = batch['text'].size() # max length in this padded batch

    all_padded_labels = []
    all_masks = []
    for sample in data:
        text = sample["text"]  # get raw text (not padded)
        labels = sample["labels"]  # get raw label
        text_len = int(sample["text_len"].item())
        assert text.shape == labels.shape and text.size(0) == text_len
        # mask: maskes out positions where we don't need to predict values
        mask = sample.get('mask', None)    

        pad_len = max_len - text_len
        assert pad_len >= 0  
        # pad the labels with right padding and label = -100, right padding
        cat_labels = torch.cat(
            [
                labels, 
                torch.ones(pad_len, dtype = labels.dtype)*
                (-100)
            ],
            dim = -1, 
        )
        if mask is None:
            cat_mask = torch.ones_like(cat_labels)
            if pad_len > 1:
                cat_mask[-pad_len:] = 0      
        else:
            assert mask.shape == text.shape
            # padding zeros to the mask
            cat_mask = torch.cat(
                mask, 
                torch.zeros(pad_len, dtype = mask.dtype)*0
            )

        all_padded_labels.append(cat_labels)
        all_masks.append(cat_mask)

    batch["labels"] = torch.stack(all_padded_labels, dim = 0)
    batch['mask'] = torch.stack(all_masks, dim = 0)
    # text, label, mask, attention_mask should have the same shape: bs x max_len
    assert batch['text'].shape == batch['labels'].shape
    assert batch["text"].shape == batch["mask"].shape
    assert batch["text"].shape == batch["attention_mask"].shape
    
    return batch


class PairwisePreferenceStreamingDataset(StreamingDataset):
    """Dataloader for streaming in preference data."""

    def __init__(self, max_seq_len: int, **kwargs: dict[str, Any]):
        self.max_seq_len = max_seq_len
        super().__init__(**kwargs)
        self.num_truncated = 0
        self.num_read = 0

    def _read_binary_tokenized_sample(self, sample: dict[str, Any], key: str):
        self.num_read += 1
        temp_sample = torch.from_numpy(np.frombuffer(sample[key]))
        if len(temp_sample) > self.max_seq_len:
            log.info(f'Truncating sample: {self.num_truncated} {self.num_read}')
            self.num_truncated += 1
            truncated = torch.from_numpy(
                np.frombuffer(sample[key][self.max_seq_len:], dtype=np.int64),
            )
            log.info(f'Truncating: {truncated}')
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
        # Handle prompt if available
        if 'prompt' in sample:
            # Prepend the prompt to the chosen and rejected responses
            sample['chosen'] = sample['prompt'] + sample['chosen']
            sample['rejected'] = sample['prompt'] + sample['rejected']
        chosen = self._read_binary_tokenized_sample(sample, 'chosen')
        rejected = self._read_binary_tokenized_sample(sample, 'rejected')

        if 'prompt' in sample:
            prompt = self._read_binary_tokenized_sample(sample, 'prompt')
            prompt_len = len(prompt)
        else:
            # Only use prefix matching version of prompt_len when
            # 'prompt' is not directly given in the sample
            prompt_len = self.find_prompt_length(chosen, rejected)
        chosen_len, rejected_len = len(chosen), len(rejected)
        return_dict = {
            'chosen': chosen,
            'rejected': rejected,
            'chosen_len': torch.Tensor([chosen_len]).to(torch.int64),
            'rejected_len': torch.Tensor([rejected_len]).to(torch.int64),
            'prompt_len': torch.Tensor([prompt_len]).to(torch.int64),
        }
        # If rewards are given, add them to the return dict
        if 'chosen_reward' in sample:
            chosen_reward = torch.Tensor([sample['chosen_reward']])
            rejected_reward = torch.Tensor([sample['rejected_reward']])
            return_dict['chosen_reward'] = chosen_reward
            return_dict['rejected_reward'] = rejected_reward
        return return_dict

    def find_prompt_length(self, seq_1: torch.Tensor, seq_2: torch.Tensor):
        """Finds the length of the common prompt given two sequences.

        Args:
            seq_1 (torch.Tensor): A sequence of tokens
            seq_2 (torch.Tensor): A sequence of tokens
        """
        overlap_length = 0
        for a, b in zip(seq_1, seq_2):
            if a == b:
                overlap_length += 1
            else:
                break
        return overlap_length


class FinegrainedPreferenceStreamingDataset(StreamingDataset):
    """Dataloader for streaming with fine-grained preference data."""

    def __init__(self, max_seq_len: int, **kwargs: Any):
        self.max_seq_len = max_seq_len
        super().__init__(**kwargs)
        self.num_truncated = 0
        self.num_read = 0

    def _read_binary_tokenized_sample(self, sample: dict[str, Any], key: str):
        self.num_read += 1
        temp_sample = torch.from_numpy(np.frombuffer(sample[key], dtype=np.int64))
        if len(temp_sample) > self.max_seq_len:
            log.info(
                f'Truncating sample {self.num_read}. Number truncated: {self.num_truncated}.',
            )
            self.num_truncated += 1
            truncated = torch.from_numpy(
                np.frombuffer(sample[key][self.max_seq_len:], dtype=np.int64),
            )
            log.info(f'Truncated sample: {truncated}')
            decoded_arr = torch.from_numpy(
                np.frombuffer(sample[key],
                              dtype=np.int64)[:self.max_seq_len].copy(),
            )
        else:
            decoded_arr = torch.from_numpy(
                np.frombuffer(sample[key], dtype=np.int64).copy(),
            )
        return decoded_arr

    # How to process a sample
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get an item from StreamingDataset at a given index.

        Args:
            idx (int): the index where we fetch the data in the StreamingDataset.
        """
        sample = super().__getitem__(idx)
        # truncated by max_seq_len and convert bytes to int64
        if isinstance(sample['input'], bytes) and isinstance(sample['label'], bytes):
            text = self._read_binary_tokenized_sample(sample, 'input')
            label =self._read_binary_tokenized_sample(sample, 'label')
        elif isinstance(sample['input'], np.ndarray) and isinstance(sample['label'], np.ndarray):
            text = torch.from_numpy(sample['input'][:self.max_seq_len]).to(torch.int64)
            label = torch.from_numpy(sample['label'][:self.max_seq_len]).to(torch.int64)

        text_len = len(text)
        if 'mask' in sample: # mask is for masking out positions where we do need to predict value
            if isinstance(sample['mask'], bytes):
                mask = self._read_binary_tokenized_sample(sample, 'mask') #truncate and 
            elif isinstance(sample['mask'], np.ndarray):
                mask = sample['mask'][:self.max_seq_len]
            return {
                'text': text,
                'labels': label,
                'text_len': torch.Tensor([text_len]).to(torch.int64),
                'mask': torch.tensor(mask).to(torch.int64),
            }
        else:
            return {
                'text': text,
                'labels': label,
                'text_len': torch.Tensor([text_len]).to(torch.int64),
            }
