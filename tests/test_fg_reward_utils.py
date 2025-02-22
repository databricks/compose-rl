# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from textwrap import dedent

import pytest
import spacy
from transformers import AutoTokenizer

from compose_rl.utils import process_fine_granularities


@dataclass
class FineGrainedOutputs:
    reward_input: str
    reward_prompt_len: int
    reward_generated_len: int
    reward_seq_len: int
    end_indices_aligned_gather: list[int]
    end_indices_aligned_scatter: list[int]

    def __init__(
        self,
        reward_input: str,
        reward_prompt_len: int,
        reward_generated_len: int,
        reward_seq_len: int,
        end_indices_aligned_gather: list[int],
        end_indices_aligned_scatter: list[int],
    ) -> None:
        self.reward_input = reward_input
        self.reward_prompt_len = reward_prompt_len
        self.reward_generated_len = reward_generated_len
        self.reward_seq_len = reward_seq_len
        self.end_indices_aligned_gather = end_indices_aligned_gather
        self.end_indices_aligned_scatter = end_indices_aligned_scatter


@pytest.fixture
def parser():
    return spacy.load('en_core_web_sm')


@pytest.mark.parametrize(
    'granularity,test_string_prompt,test_string_generated,answer_config,tokenizer',
    [
        (
            'document',
            dedent(
                """\
                <|im_start|>user
                Give me a short 3 step process to make pasta.<|im_end|>
                """,
            ),
            dedent(
                """\
                <|im_start|>assistant
                - Boil water on a pot; and then add pasta to it.
                - Cook until al dente (as per package instructions).
                - Drain pasta in a colander; and then toss with sauce.<|im_end|><|endoftext|>""",
            ),
            FineGrainedOutputs(
                reward_input=dedent(
                    """\
                    <|im_start|>user
                    Give me a short 3 step process to make pasta.<|im_end|>
                    <|im_start|>assistant
                    - Boil water on a pot; and then add pasta to it.
                    - Cook until al dente (as per package instructions).
                    - Drain pasta in a colander; and then toss with sauce.<|im_end|><|endoftext|>""",
                ),
                reward_prompt_len=17,
                reward_generated_len=46,
                reward_seq_len=63,
                end_indices_aligned_gather=[45],
                end_indices_aligned_scatter=[45],
            ),
            'rajammanabrolu/gpt-4-chat',
        ),
        (
            'sentence',
            dedent(
                """\
                <|im_start|>user
                Give me a short 3 step process to make pasta.<|im_end|>
                """,
            ),
            dedent(
                """\
                <|im_start|>assistant
                - Boil water on a pot; and then add pasta to it.
                - Cook until al dente (as per package instructions).
                - Drain pasta in a colander; and then toss with sauce.<|im_end|><|endoftext|>""",
            ),
            FineGrainedOutputs(
                reward_input=dedent(
                    """\
                    <|im_start|>user
                    Give me a short 3 step process to make pasta.<|im_end|>
                    <|im_start|>assistant
                    - Boil water on a pot; and then add pasta to it.
                    - Cook until al dente (as per package instructions).
                    - Drain pasta in a colander; and then toss with sauce.<|im_end|><|endoftext|>""",
                ),
                reward_prompt_len=17,
                reward_generated_len=46,
                reward_seq_len=63,
                end_indices_aligned_gather=[17, 29, 45],
                end_indices_aligned_scatter=[17, 29, 45],
            ),
            'rajammanabrolu/gpt-4-chat',
        ),
        (
            'subsentence',
            dedent(
                """\
                <|im_start|>user
                Give me a short 3 step process to make pasta.<|im_end|>
                """,
            ),
            dedent(
                """\
                <|im_start|>assistant
                - Boil the water on a pot; and then add the pasta to it.
                - Cook until al dente (as per package instructions).
                - Drain the pasta in a colander; and then toss it with the sauce.<|im_end|><|endoftext|>""",
            ),
            FineGrainedOutputs(
                reward_input=dedent(
                    """\
                    <|im_start|>user
                    Give me a short 3 step process to make pasta.<|im_end|>
                    <|im_start|>assistant
                    - Boil the water on a pot; and then add the pasta to it.
                    - Cook until al dente (as per package instructions).
                    - Drain the pasta in a colander; and then toss it with the sauce.<|im_end|><|endoftext|>""",
                ),
                reward_prompt_len=17,
                reward_generated_len=51,
                reward_seq_len=68,
                end_indices_aligned_gather=[12, 19, 31, 41, 50],
                end_indices_aligned_scatter=[12, 19, 31, 41, 50],
            ),
            'rajammanabrolu/gpt-4-chat',
        ),
        (
            'document',
            dedent(
                """\
                <|im_start|>user
                Give me a short 3 step process to make pasta.<|im_end|>
                """,
            ),
            dedent(
                """\
                <|im_start|>assistant
                - Boil water on a pot; and then add pasta to it.
                - Cook until al dente (as per package instructions).
                - Drain pasta in a colander; and then toss with sauce.<|im_end|><|endoftext|>""",
            ),
            FineGrainedOutputs(
                reward_input=dedent(
                    """\
                    <|im_start|>user
                    Give me a short 3 step process to make pasta.<|im_end|>
                    <|im_start|>assistant
                    - Boil water on a pot; and then add pasta to it.
                    - Cook until al dente (as per package instructions).
                    - Drain pasta in a colander; and then toss with sauce.<|im_end|><|endoftext|>""",
                ),
                reward_prompt_len=17,
                reward_generated_len=46,
                reward_seq_len=63,
                end_indices_aligned_gather=[45],
                end_indices_aligned_scatter=[45],
            ),
            'rajammanabrolu/gpt-4-chat-2',
        ),
        (
            'sentence',
            dedent(
                """\
                <|im_start|>user
                Give me a short 3 step process to make pasta.<|im_end|>
                """,
            ),
            dedent(
                """\
                <|im_start|>assistant
                - Boil water on a pot; and then add pasta to it.
                - Cook until al dente (as per package instructions).
                - Drain pasta in a colander; and then toss with sauce.<|im_end|><|endoftext|>""",
            ),
            FineGrainedOutputs(
                reward_input=dedent(
                    """\
                    <|im_start|>user
                    Give me a short 3 step process to make pasta.<|im_end|>
                    <|im_start|>assistant
                    - Boil water on a pot; and then add pasta to it.
                    <|reward_token|>- Cook until al dente (as per package instructions).
                    <|reward_token|>- Drain pasta in a colander; and then toss with sauce.<|im_end|><|endoftext|>""",
                ),
                reward_prompt_len=17,
                reward_generated_len=46,
                reward_seq_len=63,
                end_indices_aligned_gather=[17, 30, 45],
                end_indices_aligned_scatter=[17, 29, 45],
            ),
            'rajammanabrolu/gpt-4-chat-2',
        ),
        (
            'subsentence',
            dedent(
                """\
                <|im_start|>user
                Give me a short 3 step process to make pasta.<|im_end|>
                """,
            ),
            dedent(
                """\
                <|im_start|>assistant
                - Boil the water on a pot; and then add the pasta to it.
                - Cook until al dente (as per package instructions).
                - Drain the pasta in a colander; and then toss it with the sauce.<|im_end|><|endoftext|>""",
            ),
            FineGrainedOutputs(
                reward_input=dedent(
                    """\
                    <|im_start|>user
                    Give me a short 3 step process to make pasta.<|im_end|>
                    <|im_start|>assistant
                    - Boil the water on a pot; <|reward_token|>and then add the pasta to it.
                    <|reward_token|>- Cook until al dente (as per package instructions).
                    <|reward_token|>- Drain the pasta in a colander; <|reward_token|>and then toss it with the sauce.<|im_end|><|endoftext|>""",
                ),
                reward_prompt_len=17,
                reward_generated_len=51,
                reward_seq_len=68,
                end_indices_aligned_gather=[12, 21, 34, 45, 50],
                end_indices_aligned_scatter=[12, 19, 31, 41, 50],
            ),
            'rajammanabrolu/gpt-4-chat-2',
        ),
    ],
)
def test_fine_granularities(
    granularity: str,
    test_string_prompt: str,
    test_string_generated: str,
    answer_config: FineGrainedOutputs,
    tokenizer: str,
    parser: spacy.Language,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    toks_prompt = tokenizer(test_string_prompt)['input_ids']
    prompt_len = len(toks_prompt)
    toks_generate = tokenizer(test_string_generated)['input_ids']
    generated_len = len(toks_generate)
    toks_total = tokenizer(test_string_prompt +
                           test_string_generated)['input_ids']
    total_len = len(toks_total)

    reward_input, reward_prompt_len, reward_generated_len, reward_seq_len, \
        end_indices_aligned_gather, end_indices_aligned_scatter = process_fine_granularities(
            prompt=test_string_prompt,
            prompt_len=prompt_len,
            generated=test_string_generated,
            generated_len=generated_len,
            original_obs=toks_total,
            granularity=granularity,
            parser=parser,
            tokenizer=tokenizer,
            max_seq_len=total_len,
        )
    assert answer_config.reward_input == reward_input
    assert answer_config.reward_prompt_len == reward_prompt_len
    assert answer_config.reward_generated_len == reward_generated_len
    assert answer_config.reward_seq_len == reward_seq_len
    assert answer_config.end_indices_aligned_gather == end_indices_aligned_gather
    assert answer_config.end_indices_aligned_scatter == end_indices_aligned_scatter