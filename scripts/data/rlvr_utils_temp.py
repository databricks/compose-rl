import re
from typing import Any


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


def last_boxed_only_string(string: str) -> str | None:
    """Extracts the last LaTeX boxed expression from a string."""
    idx = string.rfind('\\boxed')
    if '\\boxed ' in string:
        return '\\boxed ' + string.split('\\boxed ')[-1].split('$')[0]
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return None if right_brace_idx is None else string[idx:right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    """Removes LaTeX box delimiters from a string."""
    if '\\boxed ' in s:
        left = '\\boxed '
        assert s[:len(left)] == left
        return s[len(left):]
    if '\\boxed{' in s and s[-1] == '}':
        left = '\\boxed{'
        assert s[:len(left)] == left
        return s[len(left):-1]

    # Just remove any \boxed or \fbox prefix and any trailing brace
    s = s.replace('\\boxed', '').replace('\\fbox', '')
    return s.strip('{}')


def prepare_math_prompt(sample: Any) -> str:
    """Prepare the prompt for MATH."""
    prompt = sample['problem'].strip()
    _instruction = "Let's think step by step and output the final answer within \\boxed{}."
    final_prompt = f'Question: {prompt} ' + _instruction
    return final_prompt
