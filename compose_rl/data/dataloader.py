# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Dataloader builders."""

from functools import partial
from typing import Any, Callable

from streaming import Stream, StreamingDataLoader, StreamingDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

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
]


def generate_dataloader_builder(
    dataset_cls: type[StreamingDataset],
    collate_fn: Callable,
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
    ) -> DataLoader:
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

        streaming_dataset = dataset_cls(
            streams=streams,
            batch_size=device_batch_size,
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
        return dataloader

    return build_preference_dataloader


build_pairwise_preference_dataloader = generate_dataloader_builder(
    PairwisePreferenceStreamingDataset,
    pairwise_preference_dataset_collate_fn,
)

build_finegrained_preference_dataloader = generate_dataloader_builder(
    FinegrainedPreferenceStreamingDataset,
    finegrained_preference_dataset_collate_fn,
)

build_prompt_dataloader = generate_dataloader_builder(
    PromptStreamingDataset,
    prompt_dataset_collate_fn,
)
