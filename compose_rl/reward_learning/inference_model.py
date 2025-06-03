# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Inference based reward model."""

import logging
import re
from typing import Any, MutableMapping

import backoff
import requests
import torch
from composer.utils import dist
from mcli import config as mcli_config, get_run as mcli_get_run

from compose_rl.reward_learning.base_reward import RewardModel, Tokenizer
from compose_rl.utils import get_remote_name

log = logging.getLogger(__name__)


def fetch_bpt_details(config_or_bpt_run: dict[str, Any] | str):
    """Fetch the BPT details for a given run.

    Args:
        config_or_bpt_run: Either a config dict with a 'bpt_run' key, or a BPT run name.

    Returns:
        dict[str, str]
    """
    if isinstance(config_or_bpt_run, str):
        bpt_run_name = config_or_bpt_run
    else:
        assert 'bpt_run' in config_or_bpt_run, 'InferenceRewardModel requires a BPT run name.'
        bpt_run_name = config_or_bpt_run['bpt_run']

    log.info(f'Fetching BPT details for run {bpt_run_name}')
    run = mcli_get_run(bpt_run_name, timeout=60)
    if (status := str(run.status)) != "RUNNING":
        raise RuntimeError(f"BPT run {bpt_run_name} is not running, got status {status}")
    metadata = run.metadata
    return {
        "base_url": metadata["base_url"],
        "api_key": metadata["api_key"],
        "model": metadata["model_name"],
        "bpt_run_name": bpt_run_name,
    }

def undo_llama3_chat_template(text: str) -> list[dict[str, str]]:
    messages = []
    # Regular expression to match the role and content
    pattern = re.compile(r"<\|start_header_id\|>(.*?)<\|end_header_id\|>\n(.*?)<\|eot_id\|>", re.DOTALL)
    for match in pattern.finditer(text):
        role = match.group(1).strip()
        content = match.group(2).strip()
        messages.append({"role": role, "content": content})
    return messages

def undo_qwen_chat_template(text: str) -> list[dict[str, str]]:
    """
    Parse a ChatML-formatted string produced by Qwen-2.5 *instruct* tokenizers
    back into a list of {role, content} dictionaries.

    The parser is *best-effort*: any malformed fragments are skipped rather
    than raising
    """
    assert isinstance(text, str), f"Expected a string, got {type(text)}"
    START, END = "<|im_start|>", "<|im_end|>"
    messages = []
    # 1. Extract every <|im_start|> ... (<|im_end|> or EOF) chunk.
    #    The END tag is optional so we capture until the *next* START or EOF.
    pattern = re.compile(
        rf"{re.escape(START)}"              # literal <|im_start|>
        r"(.*?)"                            # non-greedy payload
        rf"(?:(?:{re.escape(END)})|$)",     # until <|im_end|> *or* end-of-file
        flags=re.DOTALL,
    )
    potential_messages = list(pattern.finditer(text))
    if len(potential_messages) == 0:
        # we are likely seeing a single message, not a chatml string
        log.debug(f"No chatml string found in text. Returning as single user message. Raw text is:\n{text}")
        return [{"role": "user", "content": text}]
    for m in potential_messages:
        payload = m.group(1)

        # Empty payload → skip (can happen if the very first split is empty)
        if not payload.strip():
            continue

        # 2. Split payload into role  +  content (first newline separates them)
        if "\n" in payload:
            role, content = payload.split("\n", 1)
            role = role.strip()
            content = content.lstrip("\n")  # keep intentional leading \n in msg
        else:
            # No newline → could be the bare generation prompt 'assistant'
            role, content = payload.strip(), ""
        # 3. Basic sanity check on role; if weird, label as 'user'
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"
        messages.append({"role": role, "content": content})
    return messages


class InferenceRewardModel(RewardModel):

    # This can be run async
    BLOCKING = False

    def __init__(self, config: dict[Any, Any], tokenizer: Tokenizer):
        super().__init__(config, tokenizer)
        self.config = config
        self.max_retries = self.config.get('max_retries', 5)
        self.timeout = self.config.get('timeout', None)
        self.apply_sigmoid = self.config.get('apply_sigmoid', False)
        if 'qwen2.5-coder' in tokenizer.name_or_path.lower():
            self.input_str_to_messages_formatter = undo_qwen_chat_template
        elif 'llama3' in tokenizer.name_or_path.lower():
            self.input_str_to_messages_formatter = undo_llama3_chat_template
        else:
            raise ValueError(f"Tokenizer {tokenizer.name_or_path} not supported for inference reward model")
        

        if self.max_retries < 0:
            raise ValueError(
                f'max_retries must be at least 0 but was {self.max_retries}',
            )
        
        self._deployment_details = self.get_deployment_details()
        self._headers = {
            "Authorization": f"Bearer {self._deployment_details['api_key']}",
            "Content-Type": "application/json",
        }

        # If there is an issue reaching/using the deployment, we want to surface that error now.
        # To do so, we just call this RM with a dummy input.
        if dist.get_local_rank() == 0:
            self.perform_health_check_on_model()
        dist.barrier()

    def get_deployment_details(self) -> tuple[dict[str, str], dict[str, str]]:
        """Gets the details of the inference deployment.

        Returns:
            deployment_details: dict[str, str]. Contains the post_url and api_key for the inference deployment.
        """
        deployment_details = fetch_bpt_details(self.config)
        deployment_details['post_url'] = f"{deployment_details['base_url']}/chat/completions"
        return deployment_details
    
    def perform_health_check_on_model(self):
        """
        Performs a health check on the model by passing a dummy batch through it.
        """
        right_padded_obses = torch.tensor([[0] * 16])
        seq_lens = [[12, 15]]
        dummy_batch = {
            'input_ids': right_padded_obses,
            'raw_untokenized_texts': ['Hello World'],
            'seq_lens': seq_lens,
            'seq_reward': True,
            'is_inference': True,
        }
        self(dummy_batch)  # Indices here are arbitrary.

    def validate_config(self):
        # Incorrect config settings will raise errors already in __init__ above
        # Maybe refactor in the future to use this method?
        return

    def postprocess_reward(self, reward: torch.Tensor) -> torch.Tensor:
        reward_shape = reward.shape
        if len(reward_shape) == 3:
            reward = reward.mean(dim=2)
        elif len(reward_shape) != 2:
            raise ValueError(
                f'reward must be [batch_size, seq_len] or [batch_size, seq_len, n_labels], but has {reward_shape=}',
            )
        if self.apply_sigmoid:
            reward = torch.nn.functional.sigmoid(reward)
        return reward

    def __call__(self, batch: MutableMapping) -> torch.Tensor:
        log.debug(f'InferenceRewardModel __call__ received batch: {batch}')
        if 'raw_untokenized_texts' not in batch:
            raise ValueError(f"InferenceRewardModel requires raw_untokenized_texts in batch, got {batch=}")
        batch_size = batch['input_ids'].shape[0]
        
        @backoff.on_exception(  # pyright: ignore[reportUntypedFunctionDecorator]
            backoff.expo,
            exception=Exception,
            max_tries=self.max_retries + 1,
            max_value=60,
        )
        def call_predict_with_backoff(batch: MutableMapping) -> list[float]:
            batch_messages: list[list[dict[str, str]]] = [self.input_str_to_messages_formatter(text) for text in batch['raw_untokenized_texts']]
            unprocessed_rewards = [self._call_rm(messages) for messages in batch_messages]
            return unprocessed_rewards
        
        try:
            unprocessed_rewards = call_predict_with_backoff(batch)
        except Exception as e:
            # Retry limit has been reached. Raise the error :(
            error_msg = (
                'PROMPT GUIDED REWARD MODEL BACKOFF LIMIT EXCEEDED. ' +
                'Printing batch then raising last error...' +
                f'\n\n{batch=}'
            )
            raise RuntimeError(error_msg) from e
        
        # Zero-pad and batch the rewards from the outputs (this is the sequence reward)
        # TODO: check that this is equivalent to the more verbose version in the original inference_model (e.g., lines 193-onwards)
        padded_reward_seqs = torch.zeros(batch['input_ids'].shape)
        padded_reward_seqs[torch.arange(batch_size), torch.tensor(batch['seq_lens']).squeeze()] = torch.tensor(unprocessed_rewards)
        return self.postprocess_reward(padded_reward_seqs)
    
    def _call_rm(self, messages: list[dict[str, str]]) -> list[float]:
        """
        Calls the prompt-guided reward model.

        Args:
            messages (list[dict[str, str]]): The messages to call the reward model with.

        Returns:
            list[float]: The rewards from the reward model.
        """
        response = requests.post(
            self._deployment_details['post_url'],
            headers=self._headers,
            json={
                'model': self._deployment_details['model'],
                'messages': messages,
                'max_tokens': 1,
                'logprobs': True,
                'temperature': 0.0,
                'add_generation_prompt': False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()  # checking that the response is successful
        response = response.json()
        # These are document-level rewards, so only 1 reward score per "document" (i.e. per messages chain)
        return response['choices'][0]['logprobs']['content'][0]['logprob']