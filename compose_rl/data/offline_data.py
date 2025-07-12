# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Build a reward dataset and dataloader for training."""

import logging
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from streaming import StreamingDataset
from torchvision import transforms
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer, AutoProcessor

log = logging.getLogger(__name__)


def offline_dataset_collate_fn(
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
    data: list[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Collator for offline data.

    Args:
        tokenizer (Tokenizer): The model's tokenizer.
        max_seq_len (int): The maximum sequence length of the model.
        data (list[dict[str, torch.Tensor]]): The preference data to collate.
    """
    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer must have an EOS token.')
    if tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer must have a PAD token.')

    ref_collate_fn = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        mlm_probability=0.0,
    )

    batch_input_ids = []
    attention_masks = []
    sequence_lens = []
    prompt_lens = []
    rewards = []
    vstars = []

    # For VLMs
    batch_token_type_ids = []
    pixel_values = []

    for sample in data:
        input_ids = sample['input_ids']
        prompt_len = sample['prompt_len']
        sequence_len = sample['sequence_len']

        is_multimodal = 'pixel_values' in sample.keys()
        if is_multimodal:
            pixel_vals = sample['pixel_values']
            token_type_ids = sample['token_type_ids']
        else:
            pixel_vals = None
            token_type_ids = None

        # Note: if we do any truncation, we force the last token to be EOS
        # https://github.com/mosaicml/RLHF/issues/101

        # Add the eos token if it's not in the chosen sample
        if input_ids[-1] != tokenizer.eos_token_id:
            input_ids[-1] = tokenizer.eos_token_id  # type: ignore

        pad_len = max_seq_len - sequence_len

        if pad_len < 0:
            # We should truncate with an additional token left for eos
            truncate_len = abs(pad_len) + 1

            log.warning((
                f'Sequence length: {sequence_len}'
                f' are too long for max_seq_len: {max_seq_len}'
                f' truncating by {truncate_len[0]} tokens.'
            ))

            # Truncate each value by truncate length, and make the last token EOS
            input_ids = input_ids[:-truncate_len]
            input_ids[-1] = tokenizer.eos_token_id  # type: ignore

            if is_multimodal:
                token_type_ids = token_type_ids[:-truncate_len]
                # NOTE: GEMMA specific: 0 == text token
                token_type_ids[-1] = 0

            sequence_len = torch.tensor([len(sequence_len)])

            pad_len = max_seq_len - sequence_len

        if pad_len > 0:
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.ones(int(pad_len.item()), dtype=input_ids.dtype) *
                    tokenizer.pad_token_id,  # type: ignore
                ],
                dim=-1,  # type: ignore
            )
            if is_multimodal:
                token_type_ids = torch.cat(
                    [
                        token_type_ids,  # type: ignore
                        torch.zeros(
                            int(pad_len.item()),
                            dtype=token_type_ids.dtype,  # type: ignore
                        ),
                    ],
                    dim=-1,
                )

        attention_mask = torch.logical_not(
            torch.eq(input_ids, tokenizer.pad_token_id),  # type: ignore
        )

        batch_input_ids.append(input_ids)
        attention_masks.append(attention_mask)
        sequence_lens.append(sequence_len)
        prompt_lens.append(prompt_len)
        if 'reward' in sample:
            rewards.append(sample['reward'])
        if 'vstar' in sample:
            vstars.append(sample['vstar'])

        if is_multimodal:
            batch_token_type_ids.append(token_type_ids)  # type: ignore
            pixel_values.append(pixel_vals)

    batch_input_ids = ref_collate_fn(batch_input_ids)['input_ids']
    attention_masks = torch.stack(attention_masks)

    sequence_lens = torch.cat(sequence_lens)
    prompt_lens = torch.cat(prompt_lens)
    return_dict = {
        'sequence_len': sequence_lens,
        'prompt_len': prompt_lens,
        'input_ids': batch_input_ids,
        'attention_mask': attention_masks,
    }
    if len(rewards) > 0:
        rewards = torch.cat(rewards)
        return_dict['reward'] = rewards
    if len(vstars) > 0:
        vstars = torch.cat(vstars)
        return_dict['vstar'] = vstars

    if is_multimodal:  # type: ignore
        token_type_ids = torch.stack(batch_token_type_ids)
        pixel_values = torch.stack(pixel_values)
        return_dict['token_type_ids'] = token_type_ids
        return_dict['pixel_values'] = pixel_values

    return return_dict


class OfflineStreamingDataset(StreamingDataset):
    """Dataloader for streaming in preference data."""

    def __init__(self, max_seq_len: int, processor_name: Optional[str] = None, **kwargs: dict[str, Any]):
        self.max_seq_len = max_seq_len
        super().__init__(**kwargs)
        self.num_truncated = 0
        self.num_read = 0

        # For proper multimodal HF checkpointing
        self.processor = None
        if processor_name is not None:
            self.processor = AutoProcessor.from_pretrained(processor_name)

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

        # Read Samples
        input_ids, prompt = [], []
        if isinstance(sample['prompt'], bytes):
            sample['input_ids'] = sample['prompt'] + sample['response']
            input_ids = self._read_binary_tokenized_sample(sample, 'input_ids')
            prompt = self._read_binary_tokenized_sample(sample, 'prompt')
        elif isinstance(sample['prompt'], np.ndarray):
            input_ids = np.concatenate([sample['prompt'], sample['response']])
            input_ids = torch.from_numpy(input_ids[:self.max_seq_len])
            prompt = torch.from_numpy(sample['prompt'])
        else:
            token_type = type(sample['input_ids'])
            raise ValueError(
                f'Expect prompt and response to be bytes or numpy.ndarray type, but got {token_type}',
            )

        # Get Lenghts
        prompt_len = len(prompt)
        sequence_len = len(input_ids)

        return_dict = {
            'input_ids': input_ids,
            'sequence_len': torch.Tensor([sequence_len]).to(torch.int64),
            'prompt_len': torch.Tensor([prompt_len]).to(torch.int64),
        }
        # If rewards are given, add them to the return dict
        if 'reward' in sample:
            return_dict['reward'] = torch.Tensor([sample['reward']])

        if 'vstar' in sample:
            return_dict['vstar'] = torch.Tensor([sample['vstar']])

        if 'v-star' in sample:
            return_dict['vstar'] = torch.Tensor([sample['v-star']])

        if 'pixel_values' in sample:
            if isinstance(sample['pixel_values'], np.ndarray):
                pixel_values = torch.from_numpy(sample['pixel_values'])
            elif isinstance(sample['pixel_values'], Image.Image):
                pil_to_tensor_transform = transforms.PILToTensor()
                pixel_values = pil_to_tensor_transform(sample['pixel_values'])
            else:
                pixel_values_type = type(sample['pixel_values'])
                raise ValueError(
                    f'Expect pixel values to be numpy.ndarray or PIL.Image type, but got {pixel_values_type}',
                )

            if isinstance(sample['token_type_ids'], bytes):
                token_type_ids = self._read_binary_tokenized_sample(
                    sample,
                    'token_type_ids',
                )
            elif isinstance(sample['token_type_ids'], np.ndarray):
                token_type_ids = torch.from_numpy(
                    sample['token_type_ids'][:self.max_seq_len],
                )
            else:
                token_type = type(sample['token_type_ids'])
                raise ValueError(
                    f'Expect token_type_ids to be numpy.ndarray or bytes, but got {token_type}',
                )

            return_dict['pixel_values'] = pixel_values
            return_dict['token_type_ids'] = token_type_ids

        return return_dict
