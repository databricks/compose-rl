# Copyright 2025 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from compose_rl.utils.rlvr_utils import (
    extract_gsm8k_answer,
    extract_math_answer,
    prepare_gsm8k_prompt,
    prepare_math_prompt,
)

def prepare_gsm8k_messages(sample: Any) -> dict[str, list[dict[str, str]]]:
    user_prompt = prepare_gsm8k_prompt(sample)
    verified_answer = extract_gsm8k_answer(sample)
    messages = [
        {
            'role': 'user',
            'content': user_prompt,
        }
    ]
    return {'messages': messages, 'verified_answer': verified_answer}

def prepare_math_messages(sample: Any) -> dict[str, Any]:
    user_prompt = prepare_math_prompt(sample)
    verified_answer = extract_math_answer(sample)
    messages = [
        {
            'role': 'user',
            'content': user_prompt,
        }
    ]
    return {'messages': messages, 'verified_answer': verified_answer}
