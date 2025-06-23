# Copyright 2025 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""A unified script to create messages-based datasets with Mosaic Streaming."""

import argparse
import json
import logging
import os
from typing import Any, Iterator

import datasets as hf_datasets
import fsspec
from messages_preprocessing_utils import (
    prepare_gsm8k_messages,
    prepare_math_messages,
)
from streaming import MDSWriter
from torch.utils.data import IterableDataset

log = logging.getLogger(__name__)


class UnifiedMessagesDataset(IterableDataset):
    """An IterableDataset that returns messages + (optionally) metadata.

    Args:
        dataset_path (str): the path to the hf dataset or jsonl file to process
        split (str): the split of the hf dataset to process (only used if dataset_path is an hf dataset)
        subset (str | None): the subset of the dataset to process (only used if dataset_path is an hf dataset)
    """

    def __init__(
        self,
        dataset_path: str,
        split: str | None = None,
        subset: str | None = None,
        token: str | None = None,
    ):
        self.dataset_path = dataset_path
        self.split = split
        self.subset = subset
        self.dataset_preprocess_fn = self.get_preprocess_fn(dataset_path)
        self.dataset = self.load_dataset(
            dataset_path,
            split=split,
            subset=subset,
            token=token,
        )

    def load_dataset(
        self,
        dataset_path: str,
        split: str | None = None,
        subset: str | None = None,
        token: str | None = None,
    ):
        if dataset_path.endswith('.jsonl'):
            log.info(
                f'Assuming dataset path is a jsonl file. Loading from {dataset_path}',
            )
            dataset = []
            # Using fsspec to handle both local and remote files
            with fsspec.open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = [json.loads(line) for line in f]
            return dataset
        else:
            log.info(
                f'Assuming dataset path is an hf dataset. Loading from {dataset_path} with split: {split} and subset: {subset}',
            )
            return hf_datasets.load_dataset(
                path=dataset_path,
                split=split,
                name=subset,
                streaming=True,
                token=token,
            )

    def get_preprocess_fn(self, dataset_path: str):
        """Returns the preprocessing function for the dataset.

        Each preprocessing function should return a tuple of (messages, metadata).
        Messages should be a list of dictionaries with a 'role' key and a 'content' key.
        Metadata should be a dictionary with any additional metadata. If there is no metadata, then the metadata can just be None.
        Both the messages and metadata (if not None)must be json serializable.

        Args:
            dataset_path (str): the path to the dataset

        Returns:
            A function that takes in a sample and returns a tuple of (messages, metadata).
        """
        if 'gsm8k' in dataset_path.lower():
            log.info('Using GSM8k preprocessing function')
            return prepare_gsm8k_messages
        elif 'math' in dataset_path.lower():
            log.info('Using MATH preprocessing function')
            return prepare_math_messages
        else:
            log.warning(
                f'No preprocessing function found for dataset path: {dataset_path}. Defaulting to writing the dataset as is.',
            )
            return lambda x: (x, None)

    def __iter__(
        self,
    ) -> Iterator[dict[str, Any]]:
        """Iteratively yields messages + (optionally) metadata."""
        for sample in self.dataset:
            messages, metadata = self.dataset_preprocess_fn(sample)
            if metadata is None:
                metadata = {}

            # time for some good ol fashioned validation
            self._ensure_jsonable(messages, 'messages')
            self._ensure_jsonable(metadata, 'metadata')

            yield {'messages': messages, 'metadata': metadata}

    @staticmethod
    def _ensure_jsonable(obj: Any, label: str) -> None:
        """Raise ValueError if obj cannot be round-tripped through JSON."""
        try:
            json.loads(json.dumps(obj))
        except Exception as e:
            log.error(f'Error converting {label} to JSON: {e}')
            log.error(f'{label}: {obj}')
            raise ValueError(f'{label} is not JSON-serializable') from e


def main(
    dataset_path: str,
    compression: str,
    local_dir: str,
    hashes: list[str],
    splits: list[str],
    subset: str | None = None,
):
    num_written = 0
    for split in splits:
        with MDSWriter(
            columns={
                'messages': 'json',
                'metadata': 'json',
            },
            out=os.path.join(local_dir, split),
            compression=compression,
            hashes=hashes,
        ) as out:
            dataset = UnifiedMessagesDataset(
                dataset_path=dataset_path,
                split=split,
                subset=subset,
            )
            log.info('Converting to MDS format')
            for sample in dataset:
                num_written += 1
                out.write(sample)
        log.info(f'Finished writing {num_written} samples')
    log.info(f'Dataset has {num_written} samples')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to the dataset to process',
    )
    parser.add_argument('--compression', type=str, default='zstd')
    parser.add_argument('--local_dir', type=str, required=True)
    parser.add_argument(
        '--hashes',
        type=str,
        nargs='+',
        default=['sha1', 'xxh64'],
    )
    parser.add_argument('--subset', type=str, default=None)
    parser.add_argument('--splits', type=str, nargs='+', default=['train'])

    args = parser.parse_args()
    hf_token = os.environ.get('HF_TOKEN')
    main(
        dataset_path=args.dataset_path,
        compression=args.compression,
        local_dir=args.local_dir,
        hashes=args.hashes,
        splits=args.splits,
        subset=args.subset,
    )
