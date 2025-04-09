# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""A unified script to create prompt datasets for different data types."""

import argparse
import os
import re
from typing import Any, Iterator, Literal

import datasets as hf_datasets
import numpy as np
from streaming import MDSWriter
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class UnifiedTokenizedDataset(IterableDataset):
    """An IterableDataset that returns token samples.

    Args:
        dataset_name (str): the name of the hf dataset to process
        split (str): the split of the hf dataset to process
        tokenizer (PreTrainedTokenizerBase): the tokenizer used to process the dataset
        max_length (int): the maximum length of each sample
        dataset_type (str): type of dataset ('preference' or 'single_prompt')
        subset (str | None): the subset of the dataset to process
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        dataset_type: Literal['preference', 'single_prompt',
                              'verifible_answers'],
        subset: str | None = None,
    ):
        self.tokenizer = tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.max_length = max_length
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name

        print(f'Dataset name: {dataset_name}')
        if subset:
            print(f'Processing subset: {subset}')
        print(f'Processing split: {split}')
        print(f'Processing dataset type: {dataset_type}')

        self.hf_dataset = hf_datasets.load_dataset(
            path=dataset_name,
            name=subset,
            split=split,
            streaming=True,
        )

    def __iter__(self) -> Iterator[dict[str, bytes]]:
        for sample in self.hf_dataset:
            if self.dataset_type == 'preference':
                yield self._process_preference_sample(sample)
            elif self.dataset_type == 'single_prompt':
                result = self._process_single_prompt_sample(sample)
                if result is not None:
                    yield result
            elif self.dataset_type == 'verifible_answers':
                result = self._process_verifiable_answer_sample(sample)
                if result is not None:
                    yield result
            elif self.dataset_type == 'classifier':
                yield self._process_classifier_sample(sample)

    def _process_preference_sample(self, sample: Any):
        """Process a preference sample.

        Args:
            sample (Any): a sample from the dataset
        """
        chosen_messages = sample['chosen']
        rejected_messages = sample['rejected']

        curr_chosen = self.tokenizer.apply_chat_template(
            chosen_messages,
            tokenize=True,
        )
        curr_rejected = self.tokenizer.apply_chat_template(
            rejected_messages,
            tokenize=True,
        )

        return {
            'chosen': np.asarray(curr_chosen).tobytes(),
            'rejected': np.asarray(curr_rejected).tobytes(),
        }

    def _process_single_prompt_sample(self, sample: Any):
        """Process a prompt sample.

        Args:
            sample (Any): a sample from the dataset
        """
        prompt = sample['prompt']
        messages = [{
            'role':
                'user',
            'content':
                f'Can you summarize the following content in 50 words or less: {prompt}',
        }]
        encoded_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        if len(encoded_prompt) > self.max_length:
            return None

        return {'prompt': np.asarray(encoded_prompt).tobytes()}

    def _process_classifier_sample(self, sample: Any):
        """A dummy process a classifier sample.

        Args:
            sample (Any): a sample from the dataset
        """
        messages = [{
            'role': 'user',
            'content': f'This is a test',
        }]
        encoded_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
        )

        label = np.random.randint(0, 2, size=(1,))

        return {
            'input': np.asarray(encoded_prompt).tobytes(),
            'label': np.asarray(label).tobytes(),
        }

    def _get_processing_fn_from_dataset(self):
        """Get the processing function based on the dataset name.

        This function is currently hard-coded for the GSM8K dataset.
        """
        prompt_fn = prepare_gsm8k_prompt
        answer_fn = extract_gsm8k_answer

        return prompt_fn, answer_fn

    def _process_verifiable_answer_sample(self, sample: Any):
        """Process a prompt sample and extract the answer.

        This function is currently hard-coded for the GSM8K dataset.

        Args:
            sample (Any): a sample from the dataset
        """
        prompt_fn, answer_fn = self._get_processing_fn_from_dataset()

        prompt = prompt_fn(sample)
        messages = [
            {
                'role': 'user',
                'content': prompt,
            },
        ]
        verified_answer = answer_fn(sample)

        encoded_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        if len(encoded_prompt) > self.max_length:
            return None

        return {
            'prompt': np.asarray(encoded_prompt).tobytes(),
            'verified_answer': verified_answer,
        }


def extract_gsm8k_answer(sample: Any) -> str:
    """Extract the ground truth from the answer column using regex."""
    answer = sample['answer']
    numbers = re.findall(r'-?[\d,]*\.?\d+', answer)
    assert len(numbers) > 0, f'No numbers found in answer: {answer}'
    final_answer = numbers[-1].strip().lower().replace(',', '').replace('$', '')
    return final_answer


def prepare_gsm8k_prompt(sample: Any) -> str:
    """Prepare the prompt for GSM8k."""
    prompt = sample['question'].strip()
    _instruction = "Let's think step by step and output the final answer after \"####\"."
    final_prompt = f'Question: {prompt} ' + _instruction
    return final_prompt


def main(
    dataset_name: str,
    compression: str,
    local_dir: str,
    hashes: list[str],
    splits: list[str],
    tokenizer_name: str,
    dataset_type: Literal['preference', 'single_prompt', 'verifible_answers'],
    max_length: int = 2048,
    subset: str | None = None,
):
    columns = {
        'preference': {
            'chosen': 'bytes',
            'rejected': 'bytes',
        },
        'single_prompt': {
            'prompt': 'bytes',
        },
        'verifible_answers': {
            'prompt': 'bytes',
            'verified_answer': 'str',
        },
        'classifier': {
            'input': 'bytes',
            'label': 'bytes',
        },
    }[dataset_type]

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
    )
    tokenizer.model_max_length = int(1e30)

    print(f'Using tokenizer: {tokenizer}')

    num_written = 0
    for split in splits:
        with MDSWriter(
            columns=columns,
            out=os.path.join(local_dir, split),
            compression=compression,
            hashes=hashes,
        ) as out:
            dataset = UnifiedTokenizedDataset(
                dataset_name=dataset_name,
                split=split,
                max_length=max_length,
                tokenizer=tokenizer,
                dataset_type=dataset_type,
                subset=subset,
            )

            print('Converting to MDS format')

            for sample in dataset:
                num_written += 1
                out.write(sample)

        print(f'Finished writing {num_written} samples')
    print('Finished converting')
    print('Dataset has:', num_written, 'samples')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help='Name of the dataset to process',
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
    parser.add_argument(
        '--tokenizer_name',
        type=str,
        default='meta-llama/Llama-3.1-8B-Instruct',
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        choices=[
            'preference',
            'single_prompt',
            'classifier',
            'verifible_answers',
        ],
        required=True,
        help='Type of dataset to process',
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=2048,
        help='Maximum length of tokenized samples',
    )

    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        compression=args.compression,
        local_dir=args.local_dir,
        hashes=args.hashes,
        splits=args.splits,
        tokenizer_name=args.tokenizer_name,
        dataset_type=args.dataset_type,
        max_length=args.max_length,
        subset=args.subset,
    )
