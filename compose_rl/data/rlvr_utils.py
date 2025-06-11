# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from typing import Any

from compose_rl.data.math_utils import (
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)

log = logging.getLogger(__name__)


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


def extract_math_answer(sample: Any) -> str | None:
    """Extract the ground truth from the solution column."""
    last_boxed_string = last_boxed_only_string(sample['solution'])
    if not last_boxed_string:  # No boxed string found
        return None

    unnormalized_answer = remove_boxed(last_boxed_string)
    return normalize_final_answer(unnormalized_answer)


def prepare_math_prompt(sample: Any) -> str:
    """Prepare the prompt for Math dataset."""
    _template = """Solve the math problem below step by step, showing your reasoning clearly.\n\nEnd with your final answer with the format: \\boxed{{}}\n\n{problem}""".strip(
    )
    return _template.format(problem=sample['problem'].strip())
