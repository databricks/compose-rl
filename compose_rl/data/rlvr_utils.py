# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any

# GSM8k specific functions


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
