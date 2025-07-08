# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Build a reward dataset and dataloader for training."""

import logging
from typing import Any

import numpy as np
import torch
from PIL import Image
from streaming import StreamingDataset
from torchvision import transforms
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

    # For VLMs
    token_type_ids = []
    pixel_values = []

    for sample in data:
        chosen = sample['chosen']
        rejected = sample['rejected']
        prompt_len = sample['prompt_len']
        chosen_len = sample['chosen_len']
        rejected_len = sample['rejected_len']

        is_multimodal = 'pixel_values' in sample.keys()
        if is_multimodal:
            pixel_vals = sample['pixel_values']
            chosen_token_type_ids = sample['chosen_token_type_ids']
            rejected_token_type_ids = sample['rejected_token_type_ids']
        else:
            pixel_vals = None
            chosen_token_type_ids = None
            rejected_token_type_ids = None
            cat_token_type_ids = None

        # Note: if we do any truncation, we force the last token to be EOS
        # https://github.com/mosaicml/RLHF/issues/101

        # Add the eos token if it's not in the chosen sample
        if chosen[-1] != tokenizer.eos_token_id:
            chosen[-1] = tokenizer.eos_token_id  # type: ignore
        if rejected[-1] != tokenizer.eos_token_id:
            rejected[-1] = tokenizer.eos_token_id  # type: ignore

        pad_len = max_seq_len - chosen_len - rejected_len
        cat_batch = torch.cat([chosen, rejected], dim=-1)

        if is_multimodal:
            cat_token_type_ids = torch.cat([
                chosen_token_type_ids,
                rejected_token_type_ids,
            ],
                                           dim=-1)

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

            if is_multimodal:
                chosen_token_type_ids = chosen_token_type_ids[:
                                                              -truncate_len  # type: ignore
                                                             ]
                rejected_token_type_ids = rejected_token_type_ids[:  # type: ignore
                                                                  -truncate_len]

                # NOTE: GEMMA specific: 0 == text token
                chosen_token_type_ids[-1] = 0
                rejected_token_type_ids[-1] = 0
                cat_token_type_ids = torch.cat([
                    chosen_token_type_ids,
                    rejected_token_type_ids,
                ],
                                               dim=-1)

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
            if is_multimodal:
                cat_token_type_ids = torch.cat(
                    [
                        cat_token_type_ids,  # type: ignore
                        torch.zeros(
                            int(pad_len.item()),
                            dtype=cat_token_type_ids.dtype,  # type: ignore
                        ),
                    ],
                    dim=-1,
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

        if is_multimodal:
            token_type_ids.append(cat_token_type_ids)  # type: ignore
            pixel_values.append(pixel_vals)

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

    if is_multimodal:  # type: ignore
        token_type_ids = torch.stack(token_type_ids)
        pixel_values = torch.stack(pixel_values)
        return_dict['token_type_ids'] = token_type_ids
        return_dict['pixel_values'] = pixel_values

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
    ref_collate_fn = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        mlm_probability=0.0,
    )

    keys = data[0].keys()
    batch = {}
    for key in keys:
        cur_values = [item[key] for item in data]
        if key == 'prompt_mask':
            max_len = max([len(val) for val in cur_values])
            mask = torch.stack([
                torch.cat([torch.Tensor(val),
                           torch.ones(max_len - len(val))])
                for val in cur_values
            ])
            mask = ~mask.to(torch.bool)
            batch[key] = mask.to(torch.int8)
            continue
        elif key in ['prompt_len', 'text_len']:
            batch[key] = torch.stack(cur_values).squeeze(dim=1)
            continue
        elif key in ['label']:
            cur_values = [a.unsqueeze(0) for a in cur_values]
            batch[key] = torch.cat(cur_values, dim=0)
            continue

        batch[key] = ref_collate_fn(cur_values)['input_ids']
    batch['attention_mask'] = torch.logical_not(
        torch.eq(batch['text'], tokenizer.pad_token_id),  # type: ignore
    )

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
        if isinstance(sample['chosen'], bytes):
            # Prepend the prompt to the chosen and rejected responses
            if 'prompt' in sample:
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

        elif isinstance(sample['chosen'], np.ndarray):
            if 'prompt' in sample:
                sample['chosen'] = np.concatenate([
                    sample['prompt'],
                    sample['chosen'],
                ])
                sample['rejected'] = np.concatenate([
                    sample['prompt'],
                    sample['rejected'],
                ])

            chosen = torch.from_numpy(sample['chosen'][:self.max_seq_len])
            rejected = torch.from_numpy(sample['rejected'][:self.max_seq_len])

            if 'prompt' in sample:
                prompt_len = len(sample['prompt'])
            else:
                # Only use prefix matching version of prompt_len when
                # 'prompt' is not directly given in the sample
                prompt_len = self.find_prompt_length(chosen, rejected)
        else:
            token_type = type(sample['chosen'])
            raise ValueError(
                f'Expect prompt and response to be bytes or numpy.ndarray type, but got {token_type}',
            )

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

            if isinstance(sample['chosen_token_type_ids'], bytes):
                chosen_token_type_ids = self._read_binary_tokenized_sample(
                    sample,
                    'chosen_token_type_ids',
                )
                rejected_token_type_ids = self._read_binary_tokenized_sample(
                    sample,
                    'rejected_token_type_ids',
                )
            elif isinstance(sample['chosen_token_type_ids'], np.ndarray):
                chosen_token_type_ids = torch.from_numpy(
                    sample['chosen_token_type_ids'][:self.max_seq_len],
                )
                rejected_token_type_ids = torch.from_numpy(
                    sample['rejected_token_type_ids'][:self.max_seq_len],
                )
            else:
                token_type = type(sample['chosen_token_type_ids'])
                raise ValueError(
                    f'Expect token_type_ids to be numpy.ndarray or bytes, but got {token_type}',
                )

            return_dict['pixel_values'] = pixel_values
            return_dict['chosen_token_type_ids'] = chosen_token_type_ids
            return_dict['rejected_token_type_ids'] = rejected_token_type_ids

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
        temp_sample = torch.from_numpy(np.frombuffer(sample[key]))
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
        text = self._read_binary_tokenized_sample(sample, 'input')
        label = torch.from_numpy(np.frombuffer(sample['label'], dtype=np.uint8))
        # This needs to be a float tensor for BCE
        label = label.to(torch.float32)

        text_len = len(text)

        return {
            'text': text,
            'labels': label,
            'text_len': torch.Tensor([text_len]).to(torch.int64),
        }
