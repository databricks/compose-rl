# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import torch
from compose_rl.utils import flatten

def preprocess_batches(batches: list, generations_per_prompt: int, pad_token_idx: int):
    ret_batch = {}
    for key in batches[0].keys():
        curr_values = []

        max_len = 0
        if isinstance(batches[0][key], torch.Tensor):
            max_len = max([batch[key].shape[-1] for batch in batches])

        padding_key = None
        for batch in batches:
            # Explode the batch into multiple batches for each generation
            for _ in range(generations_per_prompt):
                # For keys that do not require additional processing
                if key in [
                    'prompt_len',
                    'verified_answer',
                    'prompt_id',
                    'vstar',
                    'messages',
                ]:
                    curr_values.append(batch[key])
                    continue

                bs, seq_len = batch[key].shape

                if key == 'prompt':
                    padding_key = pad_token_idx
                    if (batch[key][:, -1] == padding_key).any():
                        raise ValueError(
                            'The last token in the prompt should not be the pad token. Please double '
                            +
                            'check the dataloader and prompt and dataloader.',
                        )
                elif key == 'prompt_attention_mask':
                    padding_key = False

                # Compute the required padding and concatenate with the batch tensor
                pad = torch.ones(
                    (bs, max_len - seq_len),
                    dtype=batch[key].dtype,
                ) * padding_key  # type: ignore
                curr_values.append(torch.cat([pad, batch[key]], dim=-1))

        # For tensor fields, use torch.cat to combine the values; for string fields, just use the list
        if isinstance(curr_values[0], torch.Tensor):
            ret_batch[key] = torch.cat(curr_values)
        else:
            if key in ['verified_answer', 'vstar']:
                ret_batch[key] = list(flatten(curr_values))
            else:
                ret_batch[key] = curr_values

    return ret_batch