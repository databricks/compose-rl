# Copyright 2025 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Preprocessing functions for the messages dataset.

Each preprocessing function should return a tuple of (messages, metadata).
Messages should be a list of dictionaries with a 'role' key and a 'content' key.
Metadata should be a dictionary with any additional metadata. If there is no
metadata, then return an empty dictionary. Both the messages and metadata must
be json serializable.
"""

from typing import Any

from compose_rl.utils.rlvr_utils import (
    extract_gsm8k_answer,
    extract_math_answer,
    extract_verified_answer,
    prepare_gsm8k_prompt,
    prepare_math_prompt,
    prepare_prompt,
)


def prepare_gsm8k_messages(
    sample: Any,
) -> tuple[list[dict[str, str]], dict[str, str | None]]:
    user_prompt = prepare_gsm8k_prompt(sample)
    verified_answer = extract_gsm8k_answer(sample)
    messages = [
        {
            'role': 'user',
            'content': user_prompt,
        },
    ]
    return messages, {'verified_answer': verified_answer}


def prepare_math_messages(
    sample: Any,
) -> tuple[list[dict[str, str]], dict[str, str | None]]:
    user_prompt = prepare_math_prompt(sample)
    verified_answer = extract_math_answer(sample)
    messages = [
        {
            'role': 'user',
            'content': user_prompt,
        },
    ]
    return messages, {'verified_answer': verified_answer}


def prepare_ultrafeedback_summarization_messages(
    sample: Any,
) -> tuple[list[dict[str, str]], dict]:
    prompt = sample['prompt']
    messages = [{
        'role': 'user',
        'content': f'Can you summarize the following content in 50 words or less: {prompt}',
    }]
    return messages, {}


def prepare_messages(
    sample: Any,
) -> tuple[list[dict[str, str]], dict[str, str | None]]:
    user_prompt = prepare_prompt(sample)
    verified_answer = extract_verified_answer(sample)
    messages = [
        {
            'role': 'user',
            'content': user_prompt,
        },
    ]
    return messages, {'verified_answer': verified_answer}
