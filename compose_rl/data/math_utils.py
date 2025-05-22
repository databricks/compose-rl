"""Utilities for math evaluation.

Note, this uses sympy for equivalence checking, in contrast to simple-evals, which uses LLM-as-a-judge
(with llms judging themselves, which is problematic).

Note: This implementation requires antlr4-python3-runtime==4.11 for latex parsing. Only pre-releases
of Omegaconf support such a recent version of antlr4-python3-runtime, so we use a pre-release of Omegaconf,
omegaconf==2.4.0.dev3. As of October 2024, Omegaconf has not made a release since February 2022.
"""

import logging
import re

import sympy
from sympy.parsing.latex import parse_latex


log = logging.getLogger(__name__)


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


def is_sympy_equivalent(x1: str, x2: str) -> bool:
    """Checks mathematical equivalence between two normalized LaTeX strings."""
    try:
        try:
            parsed_x1 = parse_latex(x1)
            parsed_x2 = parse_latex(x2)
        except (
            sympy.parsing.latex.errors.LaTeXParsingError,  # pyright: ignore[reportGeneralTypeIssues]
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
            log.debug(f'Had some trouble simplifying when comparing {x1} and {x2}')
            return False
    except ImportError as e:
        log.error(e)
        raise
    except Exception as e:
        log.debug(f'Failed comparing {x1} and {x2} with {e}')
        return False


SUBSTITUTIONS: list[tuple[str, str]] = [
    ('\\left', ''),
    ('\\right', ''),
    ('\\{', '{'),
    ('\\}', '}'),
    ('\\\\', '\\'),
    ('tfrac', 'frac'),
    ('dfrac', 'frac'),
    ('\\neq', '\\ne'),
    ('\\leq', '\\le'),
    ('\\geq', '\\ge'),
    ('^{\\circ}', ''),
    ('^\\circ', ''),
    ('\\$', ''),
    ('$', ''),
    ('\\%', ''),
    (r'\%', ''),
    ('\\(', ''),
    ('\\)', ''),
    ('infinity', '\\infty'),
    ('inf', '\\infty'),
    ('+\\infinity', '\\infty'),
    (r'\ ', ''),
    ('\\ldots', ''),
    ('\\dots', ''),
    (r'{,}', ''),
    (r'\mathrm{th}', ''),
    ('mbox', 'text'),
    (',\\text{and}', ','),
    ('\\text{and}', ','),
    (',\\text{or}', ','),
    ('\\text{or}', ','),
    (r'\;', ''),
    (r',\!', ''),
]

REMOVED_EXPRESSIONS: list[str] = [
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
    '"',
]


def get_unnormalized_answer(text: str) -> str | None:
    """Extracts raw answer from model output text."""
    invalid_answer = '[invalidanswer]'

    candidate_patterns = [
        r'Final Answer:\s*((?:[^<]|<[^<])*?)\n',
        r'Final Answer is:\s*((?:[^<]|<[^<])*?)\n',
        r'The answer is:\s*((?:[^<]|<[^<])*?)\n',
        r'Answer:\s*((?:[^<]|<[^<])*?)\n',
        r'Solution:\s*((?:[^<]|<[^<])*?)\n',
        r'The solution is:\s*((?:[^<]|<[^<])*?)\n',
    ]

    last_match = None
    last_position = -1
    for pattern in candidate_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if match.start() > last_position:
                last_position = match.start()
                last_match = match.group(1).strip()

    stop_words = [
        '</s>', '<|im_end|>', '<|endoftext|>', '<|eot_id|>', '<|eom_id|>', '<|end_of_text|>', '<|end▁of▁sentence|>'
    ]
    for stop_word in stop_words:
        if last_match and last_match.endswith(stop_word):
            last_match = last_match[:-len(stop_word)].strip()

    return last_match or invalid_answer


def _read_token(s: str, i: int) -> tuple[str, int]:
    """
    Return (token, next_index) where `token` is either
        '{...}'      balanced brace group,
        '\\command'  control sequence,
        'c'          single char (non-space),
    starting at position i.  If no token possible, return ('', i).
    """
    n = len(s)
    if i >= n:
        return '', i

    # Skip leading spaces
    while i < n and s[i].isspace():
        i += 1
    if i >= n:
        return '', i

    # 1) Brace group
    if s[i] == '{':
        depth = 1
        j = i + 1
        while j < n and depth:
            if s[j] == '{':
                depth += 1
            elif s[j] == '}':
                depth -= 1
            j += 1
        if depth == 0:  # balanced
            return s[i:j], j
        # un-balanced, treat as single char
        return s[i], i + 1

    # 2) Control sequence
    if s[i] == '\\':
        j = i + 1
        while j < n and s[j].isalpha():
            j += 1
        return s[i:j], j

    # 3) Single printable char
    return s[i], i + 1


def _fix_fracs(text: str) -> str:
    """
    Make every \\frac follow the canonical  \\frac{num}{den}  form.
    Tokens are parsed according to real LaTeX rules.
    Nothing else in the string is modified.
    """
    out, i, n = [], 0, len(text)

    while i < n:
        if text.startswith(r'\frac', i):
            out.append(r'\frac')
            i += 5  # len('\frac')
            # -------- numerator --------
            num, i = _read_token(text, i)
            if not num:  # malformed -> bail out
                out.append(text[i:])
                break
            if not num.startswith('{'):
                num = '{' + num + '}'
            # -------- denominator --------
            den, i = _read_token(text, i)
            if not den:  # malformed -> bail out
                out.append(num + text[i:])
                break
            if not den.startswith('{'):
                den = '{' + den + '}'
            out.append(num + den)
        else:
            out.append(text[i])
            i += 1

    return ''.join(out)


_SLASH_FRAC_RE = re.compile(
    r"""
    (?<![\\\w{])         # NOT preceded by '\', letter, or '{'  ⇒ avoid \frac, 3x/4, \frac{3/4}
    \s*                  # optional spaces
    (-?[0-9]+)           # group(1)  numerator  (signed ASCII int)
    \s*/\s*              # the slash, possibly surrounded by spaces
    (-?[0-9]+)           # group(2)  denominator (signed ASCII int)
    (?![\w])             # NOT immediately followed by letter or digit
    """,
    re.VERBOSE,
)


def _fix_a_slash_b(s: str) -> str:

    def replacement(m: re.Match[str]) -> str:
        numerator = m.group(1)
        denominator = m.group(2)

        # Avoid division by zero
        if denominator == '0':
            return m.group(0)  # Return original match unchanged

        return rf'\frac{{{numerator}}}{{{denominator}}}'

    return _SLASH_FRAC_RE.sub(replacement, s)


def _fix_sqrt(string: str) -> str:
    """
    Matches `\\sqrt`, optionally followed by a bracketed index (e.g., `[n]`), skips
    if already followed by a brace, then captures the next LaTeX command, parenthesis
    group, word, or single character to wrap in braces.
    """
    _sqrt_pat_final = re.compile(r'\\sqrt(\[[^\]]+\])?\s*(?!\{)(\\[A-Za-z]+|\([^()]*\)|[A-Za-z0-9_.]+|.)')

    def _repl(m: re.Match[str]) -> str:
        index = m.group(1) or ''
        arg = m.group(2)
        if arg.startswith('{'):
            return fr'\sqrt{index}{arg}'
        return fr'\sqrt{index}{{{arg}}}'

    return _sqrt_pat_final.sub(_repl, string)


def _remove_right_units(string: str) -> str:
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if '\\text{ ' in string:
        splits = string.split('\\text{ ')
        assert len(splits) == 2
        return splits[0]
    return string


def _strip_assignment(string: str) -> str:
    if '=' not in string:
        return string.strip()

    parts = [p.strip() for p in string.split('=')]

    # case 1: single '='  →  lhs = rhs
    if len(parts[0]) <= 2:
        return parts[1]

    # case 2: first side is a short symbol such as `k=...`
    lhs = parts[0]
    if len(lhs) <= 2:
        return '='.join(parts[1:]).strip()

    # case 3: chain of equalities, pick right-most chunk if it is "simple"
    # "simple"  =  contains no further '=', and contains at most digits,
    # letters, dots,  slashes, backslashes, braces, or parentheses
    last = parts[-1]
    if re.fullmatch(r'[0-9a-zA-Z.\-+*/\\{}()\s]+', last):
        return last.strip()

    return string.strip()


def normalize_string(string: str) -> str:
    """Normalizes a string to a quantitative reasoning question."""
    # remove linebreaks and `!`/inverse space
    string = string.replace('\n', '')
    string = string.replace('\\!', '')

    for before, after in SUBSTITUTIONS:
        string = string.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        string = string.replace(expr, '')

    # Remove right-hand units if present
    string = _remove_right_units(string)

    # Fix " ."/"{." patterns and leading dots ("." becomes "0.")
    string = string.replace(' .', ' 0.').replace('{.', '{0.}')
    if len(string) > 0 and string[0] == '.':
        string = '0' + string

    # Remove spaces (prefer after all basic fixes, to keep regex working right)
    string = string.replace(' ', '')

    # Handle equals sign if present as assignment
    string = _strip_assignment(string)

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    string = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', string)
    string = re.sub(r'(\\text\{)(.*?)(\})', '\\2', string)
    string = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', string)
    string = re.sub(r'(\\texttt\{)(.*?)(\})', '\\2', string)
    string = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', string)
    string = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', string)

    # Fraction and root fixes
    string = _fix_sqrt(string)
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)

    # Remove any remaining dollar signs (if any from matching regex below)
    string = string.replace('$', '')

    # Remove scientific formatting commas for numbers
    if string.replace(',', '').isdigit():
        string = string.replace(',', '')

    # Special case: 0.5 --> \frac{1}{2}
    if string == '0.5':
        string = '\\frac{1}{2}'

    return string.strip()


def is_hendrycks_equivalent(str1: str, str2: str) -> bool:
    try:
        ss1 = normalize_string(str1)
        ss2 = normalize_string(str2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def extract_answers(prediction: str) -> list[str]:
    """Extracts all potential answers from a string (given different potential formats)"""
    answers = []

    # Attempt extraction from \boxed{} string
    boxed_answer = last_boxed_only_string(prediction)
    if boxed_answer:
        answers.append(normalize_string(remove_boxed(boxed_answer)))

    # Attempt extraction via Minerva math format
    answer = get_unnormalized_answer(prediction)
    if answer and answer != '[invalidanswer]':
        answers.append(normalize_string(answer))

    if not answers:
        # Attempt extraction from the last LaTeX-formatted answer
        dollars = [m.start() for m in re.finditer(r'\$', prediction)]
        if len(dollars) > 1:
            answers.append(normalize_string(prediction[dollars[-2] + 1:dollars[-1]]))

    if not answers:
        # fallback to the full output if no answers were found
        answers.append(normalize_string(prediction))
        # add the full output as a fallback
        answers.append(prediction)

    return answers


def process_results(prediction: str, ground_truth: str) -> int:
    # Compare all extracted_answers with gold truth answer
    extracted_answers = extract_answers(prediction)
    for answer in extracted_answers:
        if (
            answer.strip() == ground_truth.strip() or is_sympy_equivalent(answer, ground_truth) or
            is_hendrycks_equivalent(answer, ground_truth)
        ):
            return 1

    return 0
