# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from typing import Any
import signal

import sympy
from sympy.parsing.latex import parse_latex

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


class timeout:

    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)



def is_equiv(x1: str, x2: str) -> bool:
    """Checks mathematical equivalence between two normalized LaTeX strings."""
    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.  # pyright: ignore[reportGeneralTypeIssues]
                errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                log.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2  # pyright: ignore[reportOptionalOperand]
            except TypeError:
                log.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                return sympy.simplify(diff) == 0
            except ValueError:
                log.debug(
                    f'Had some trouble simplifying when comparing {x1} and {x2}',
                )
                return False
    except TimeoutError:
        log.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        log.error(e)
        raise
    except Exception as e:
        log.debug(f'Failed comparing {x1} and {x2} with {e}')
        return False


SUBSTITUTIONS = [
    ('an ', ''),
    ('a ', ''),
    ('.$', '$'),
    ('\\$', ''),
    (r'\ ', ''),
    (' ', ''),
    ('mbox', 'text'),
    (',\\text{and}', ','),
    ('\\text{and}', ','),
    ('\\text{m}', '\\text{}'),
]

REMOVED_EXPRESSIONS = [
    'square',
    'ways',
    'integers',
    'dollars',
    'mph',
    'inches',
    'ft',
    'hours',
    'km',
    'units',
    '\\ldots',
    'sue',
    'points',
    'feet',
    'minutes',
    'digits',
    'cents',
    'degrees',
    'cm',
    'gm',
    'pounds',
    'meters',
    'meals',
    'edges',
    'students',
    'childrentickets',
    'multiples',
    '\\text{s}',
    '\\text{.}',
    '\\text{\ns}',
    '\\text{}^2',
    '\\text{}^3',
    '\\text{\n}',
    '\\text{}',
    r'\mathrm{th}',
    r'^\circ',
    r'^{\circ}',
    r'\;',
    r',\!',
    '{,}',
    '"',
    '\\dots',
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalizes a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split('=')[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')

    return final_answer


def extract_math_answer(sample: Any) -> str | None:
    """Extract the ground truth from the solution column."""
    last_boxed_string = last_boxed_only_string(sample['solution'])
    if not last_boxed_string:  # No boxed string found
        return None

    unnormalized_answer = remove_boxed(last_boxed_string)
    return normalize_final_answer(unnormalized_answer)


def prepare_math_prompt(sample: Any) -> str:
    """Prepare the prompt for Math dataset."""
    prompt = sample['problem'].strip()
    _instruction = " Let's think step by step and output the final answer within \\boxed{}."
    final_prompt = f'Question: {prompt} ' + _instruction
    return final_prompt
