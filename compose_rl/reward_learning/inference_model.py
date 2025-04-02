# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Inference based reward model."""

import logging
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

class InferenceRewardModel(RewardModel):

    # This can be run async
    BLOCKING = False

    def __init__(self, cfg: dict[Any, Any], tokenizer: Tokenizer):
        super().__init__(cfg, tokenizer)
        self.max_retries = self.cfg.get('max_retries', 5)
        self.timeout = self.cfg.get('timeout', None)
        self.threshold = self.cfg.get('threshold')

        if self.max_retries < 0:
            raise ValueError(
                f'max_retries must be at least 0 but was {self.max_retries}',
            )

        if self.timeout is not None:
            raise NotImplementedError(
                'timeout not currently supported for the latest inference setup',
            )
        
        self._deployment_details = self.get_deployment_details()
        self._headers = {
            "Authorization": f"Bearer {self._deployment_details['api_key']}",
            "Content-Type": "application/json",
        }

        # If there is an issue reaching/using the deployment, we want to surface that error now.
        # To do so, we just call this RM with a dummy input.
        if dist.get_local_rank() == 0:
            right_padded_obses = torch.tensor([[0] * 16])
            seq_lens = [[12, 15]]
            dummy_batch = {
                'input_ids': right_padded_obses,
                'seq_lens': seq_lens,
                'seq_reward': True,
                'is_inference': True,
            }
            self(dummy_batch)  # Indices here are arbitrary.
        dist.barrier()

    def get_deployment_details(self) -> tuple[dict[str, str], dict[str, str]]:
        """Gets the details of the inference deployment.

        Returns:
            deployment_details: dict[str, str]. Contains the post_url and api_key for the inference deployment.
        """
        deployment_details = fetch_bpt_details(self.cfg)
        deployment_details['post_url'] = f"{deployment_details['base_url']}/rewards"
        return deployment_details

    def validate_config(self):
        # Incorrect config settings will raise errors already in __init__ above
        # Maybe refactor in the future to use this method?
        return

    def postprocess_reward(self, reward: torch.Tensor) -> torch.Tensor:
        # if self.threshold is not None:
        # reward = self.model_cls.apply_threshold(reward, self.threshold)
        reward_shape = reward.shape
        if len(reward_shape) == 3:
            reward = reward.mean(dim=2)
        elif len(reward_shape) != 2:
            raise ValueError(
                f'reward must be [batch_size, seq_len] or [batch_size, seq_len, n_labels], but has {reward_shape=}',
            )
        return reward

    def __call__(
        self,
        batch: MutableMapping,
    ) -> torch.Tensor:

        if not batch['seq_reward']:
            raise NotImplementedError(
                'InferenceRewardModel currently requires seq_reward=True',
            )
        if not batch['is_inference']:
            raise NotImplementedError(
                'InferenceRewardModel currently requires `is_inference`=True',
            )

        input_ids = batch['input_ids']
        seq_lens = batch['seq_lens']

        # We'll work in lists because we have to send some version of this through the inference request
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        assert isinstance(input_ids, list)

        if len(input_ids) != len(seq_lens):
            raise ValueError(
                'The batch dimension of input_ids and the length of seq_lens do not match.',
            )

        # We need to send each input sequence through one per reward index that it needs.
        # Build that out here.
        deployment_inputs = []
        batch_indices = []
        reward_indices = []
        for bidx, (seq_input_ids,
                   seq_reward_indices) in enumerate(zip(input_ids, seq_lens)):
            for seq_reward_index in seq_reward_indices:
                deployment_inputs.append({
                    'input_ids': seq_input_ids[:seq_reward_index + 1],
                })
                batch_indices.append(bidx)
                reward_indices.append(seq_reward_index)

        @backoff.on_exception(  # pyright: ignore[reportUntypedFunctionDecorator]
            backoff.expo,
            exception=Exception,
            max_tries=self.max_retries + 1,
            max_value=30,
        )
        def call_predict_with_backoff(
            inputs: list[torch.Tensor],
        ):
            data = {
                'custom_input': inputs,
                'prompt': 'UNUSED',
            }
            response = requests.post(
                self._deployment_details['post_url'],
                headers=self._headers,
                json=data,
            )
            response = response.json()
            # Currently, all outputs will contain a single reward, coming from the last token.
            rewards = [r['score'][0] for r in response['data']]
            return rewards

        try:
            inf_outputs = call_predict_with_backoff(
                deployment_inputs,
            )
        except Exception as e:
            # Retry limit has been reached. Raise the error :(
            error_msg = (
                'REWARD MODEL DEPLOYMENT BACKOFF LIMIT EXCEEDED. ' +
                'Printing deployment inputs then raising last error...' +
                f'\nDeployment inputs:\n{deployment_inputs}'
            )
            raise RuntimeError(error_msg) from e

        # Zero-pad and batch the rewards from the outputs (this is the sequence reward)
        max_len = max([len(seq) for seq in input_ids])
        padded_reward_seqs = torch.zeros((len(input_ids), max_len))
        for reward, bidx, ridx in zip(
            inf_outputs,
            batch_indices,
            reward_indices,
        ):
            padded_reward_seqs[bidx, ridx] = reward
        reward = torch.tensor(padded_reward_seqs)

        return self.postprocess_reward(reward)