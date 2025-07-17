# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Dataloader builders."""

import logging
from functools import partial
from typing import Any, Callable, Optional, Union

from composer.core.data_spec import DataSpec
from streaming import Stream, StreamingDataLoader, StreamingDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from compose_rl.data.messages_data import (
    MessagesStreamingDataset,
    messages_dataset_collate_fn,
)
from compose_rl.data.preference_data import (
    FinegrainedPreferenceStreamingDataset,
    PairwisePreferenceStreamingDataset,
    finegrained_preference_dataset_collate_fn,
    pairwise_preference_dataset_collate_fn,
)
from compose_rl.data.prompt_data import (
    PromptStreamingDataset,
    prompt_dataset_collate_fn,
)

__all__ = [
    'build_finegrained_preference_dataloader',
    'build_pairwise_preference_dataloader',
    'build_prompt_dataloader',
    'build_messages_dataloader',
]

log = logging.getLogger(__name__)


def generate_dataloader_builder(
    dataset_cls: type[StreamingDataset],
    collate_fn: Callable,
    token_counter_fn: Optional[Callable] = None,
) -> Callable:
    """Generates dataloader builder for a given dataset_cls and collate_fn."""

    def build_preference_dataloader(
        tokenizer: PreTrainedTokenizer,
        device_batch_size: int,
        dataset: dict[str, Any],
        drop_last: bool,
        num_workers: int,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        timeout: int = 0,
    ) -> Union[DataLoader, DataSpec]:
        """Builds a dataloader for prompt data.

        Args:
            tokenizer: the model's tokenizer.
            device_batch_size: batch size per device.
            dataset: the dataset configuration.
            drop_last: indicating if we should drop the last batch.
            num_workers: number of workers to use.
            pin_memory: indicating if we should pin memory.
            prefetch_factor: the prefetch factor.
            persistent_workers: indicating if we should use persistent workers.
            timeout: the timeout value.
        """
        dataset_cfg = dataset

        streams_dict = dataset_cfg.pop('streams', None)
        max_seq_len = dataset_cfg.get('max_seq_len', None)
        if max_seq_len is None:
            raise ValueError(
                'max_seq_len must be provided in the dataset configuration',
            )

        # Build streams
        streams = None
        if streams_dict is not None:
            streams = [Stream(**stream) for stream in streams_dict.values()]
        if issubclass(
            dataset_cls,
            MessagesStreamingDataset,
        ) and 'tokenizer' not in dataset_cfg:
            dataset_cfg['tokenizer'] = tokenizer

        streaming_dataset = dataset_cls(
            streams=streams,  # type: ignore
            batch_size=device_batch_size,  # type: ignore
            **dataset_cfg,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        dataloader = StreamingDataLoader(
            streaming_dataset,
            collate_fn=partial(collate_fn, tokenizer, max_seq_len),
            batch_size=device_batch_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            timeout=timeout,
        )
        data_spec_kwargs = {}
        if token_counter_fn is not None:
            data_spec_kwargs['get_num_tokens_in_batch'] = token_counter_fn
        return DataSpec(dataloader=dataloader, **data_spec_kwargs)

    return build_preference_dataloader


build_pairwise_preference_dataloader = generate_dataloader_builder(
    PairwisePreferenceStreamingDataset,
    pairwise_preference_dataset_collate_fn,
)

build_finegrained_preference_dataloader = generate_dataloader_builder(
    FinegrainedPreferenceStreamingDataset,
    finegrained_preference_dataset_collate_fn,
)


def get_num_tokens_in_batch_online(
    batch: dict[str, Any],
) -> int:
    """Get the number of tokens in batch including prompt + generated tokens.

    Uses action_mask and prompt_len for precise counting when available.
    """
    if 'action_mask' in batch and 'prompt_len' in batch:
        prompt_len_tokens = batch['prompt_len'].sum().item()
        generated_items = batch['action_mask'].sum().item()
        padding_tokens = batch['action_mask'].numel() - generated_items
        log.info(f'prompt tokens in batch: {prompt_len_tokens}')
        log.info(f'generated tokens in batch: {generated_items}')
        log.info(
            f'unused generation (padding) tokens in batch: {padding_tokens}',
        )
        return int(prompt_len_tokens + generated_items)
    else:
        log.warning(
            'No action_mask/prompt_len in batch. ' +
            'Using default value of 0 for num_tokens_in_batch.',
        )
        return 0


build_prompt_dataloader = generate_dataloader_builder(
    PromptStreamingDataset,
    prompt_dataset_collate_fn,
    get_num_tokens_in_batch_online,
)

build_messages_dataloader = generate_dataloader_builder(
    MessagesStreamingDataset,
    messages_dataset_collate_fn,
    get_num_tokens_in_batch_online,
)
