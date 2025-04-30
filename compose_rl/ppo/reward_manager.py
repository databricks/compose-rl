# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Contains the reward manager implementation."""

import logging
from itertools import chain
from multiprocessing import get_context
from multiprocessing.pool import AsyncResult, Pool
from typing import Any, MutableMapping, Optional, Union

import spacy
import torch
from composer import Trainer
from composer.core import Precision
from llmfoundry.utils import build_composer_model
# pyright does not recognize process_init_device though it is a declared export
from llmfoundry.utils.config_utils import process_init_device  # type: ignore
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from compose_rl.ppo.kl_controller import BaseKLController
from compose_rl.registry import RL_REWARD_REGISTRY
from compose_rl.reward_learning import (
    BadGenerationEndReward,
    BaseReward,
    InferenceRewardModel,
    Reward,
    RewardModel,
)
from compose_rl.utils import (
    batch_process_fine_granularities,
    get_log_probs,
    scatter_gather_rewards,
)

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
ReferenceOutput = tuple[torch.Tensor, torch.Tensor]
RewardOutput = dict[str, Union[AsyncResult, torch.Tensor]]

log = logging.getLogger(__name__)


class RewardManager:

    def __init__(
        self,
        config: DictConfig,
        ref_config: dict[str, Any],
        tokenizer: Tokenizer,
        max_seq_len: int,
        fsdp_config: Optional[dict[str, Any]],
        precision: Precision,
        kl_penalty_in_reward: bool = True,
    ):
        """Manages all the rewards used during PPO training.

        Args:
            config (DictConfig): the overarching reward manager config used to create each of the rewards.
            ref_config(DictConfig): the config used to create the reference model.
            tokenizer (Tokenizer): the tokenizer that is used by policy, reference, reward models.
            max_seq_len (int): the max sequence length supported by the model.
            fsdp_config: the FSDP config to use to create reward models if they are local reward models.
            precision: the default precision that will be passed to `Trainer` to initialize models.
            kl_penalty_in_reward (bool): indicates if we should add the KL penalty to the reward or directly
                compute it in the training loss.
        """
        self.config = config
        self.ref_config = ref_config
        self.tokenizer = tokenizer
        self.fsdp_config = fsdp_config
        self.max_seq_len = max_seq_len
        self.kl_penalty_in_reward = kl_penalty_in_reward

        # For fine-grained rewards. Not necessarily used.
        self.parser = spacy.load('en_core_web_sm')

        self.all_rewards = {}
        self.reward_coefficients: dict[str, float] = {}
        self.granularities: dict[str, str] = {}

        self.inference_rewards: list[str] = []
        self.functional_rewards: list[str] = []
        self.local_reward_models: list[str] = []

        ref_model_config: dict[str,
                               Any] = self.ref_config.get('model_config', None)

        self.reference_model = self.initialize_composer_model(
            model_config=ref_model_config,
            model_name='reference',
            precision=self.ref_config.get('precision', precision),
            load_path=self.ref_config.get('load_path', None),
        )

        for reward_name, reward_config in self.config.items():
            assert isinstance(reward_name, str)

            if reward_name in self.all_rewards:
                raise KeyError(
                    f'The reward manager already has a model with {reward_name=}',
                )

            log.info(f'Initializing reward with name {reward_name}')

            # TODO: Validate reward_config
            reward_cls = RL_REWARD_REGISTRY[reward_config.get('reward_type')]

            if issubclass(reward_cls, Reward):
                # TODO: This assumes that all functional rewards are document level rewards.
                # This is not necessarily true, but is a reasonable assumption for now.
                self.granularities[reward_name] = 'document'
                model = reward_cls(reward_config, self.tokenizer)
                self.functional_rewards.append(reward_name)

            elif issubclass(reward_cls, RewardModel):
                self.granularities[reward_name] = reward_config.get(
                    'granularity',
                )

                if reward_cls == InferenceRewardModel:
                    model = InferenceRewardModel(
                        reward_config.get('config'),
                        self.tokenizer,
                    )
                    self.inference_rewards.append(reward_name)

                else:
                    reward_model_config = reward_config.get(
                        'model_config',
                        None,
                    )
                    assert reward_model_config is not None, 'model_config must be provided in reward_config'
                    model = self.initialize_composer_model(
                        model_config=reward_config.get('model_config'),
                        model_name=reward_name,
                        precision=reward_config.get('precision', precision),
                        load_path=reward_config.get('load_path', None),
                    )
                    self.local_reward_models.append(reward_name)
            else:
                raise TypeError(
                    f'Reward class {reward_cls} is not a subclass of either Reward or RewardModel.',
                )

            self.all_rewards[reward_name] = model
            self.reward_coefficients[reward_name] = reward_config.get(
                'reward_coefficient',
                1.0,
            )

        self.granularity_types = list(set(self.granularities.values()))

        self.pool = None
        if self.inference_rewards or self.functional_rewards:
            self.pool = Pool(
                processes=len(self.inference_rewards) +
                len(self.functional_rewards),
                context=get_context('spawn'),
            )

        if not self.kl_penalty_in_reward:
            log.info(
                'The IFT KL will be minimized directly in loss, rather than as a reward.',
            )

    def initialize_composer_model(
        self,
        model_config: dict[str, Any],
        model_name: str,
        precision: Precision = Precision.FP32,
        load_path: Optional[str] = None,
    ) -> torch.nn.Module:
        """Create the reference model."""
        log.info(f'Initializing {model_name} model')
        name = model_config.pop('name')

        init_context = process_init_device(
            model_config,
            self.fsdp_config,
        )
        model = build_composer_model(
            name=name,
            cfg=model_config,
            tokenizer=self.tokenizer,
            init_context=init_context,
            master_weights_dtype=model_config.get('master_weights_dtype', None),
        )

        parallelism_config = {'fsdp': self.fsdp_config}

        # Create a Trainer object to load from checkpoint and FSDP the model
        _ = Trainer(
            model=model,
            parallelism_config=parallelism_config,
            precision=precision,
            load_weights_only=True,
            load_strict_model_weights=False,
            load_path=load_path,
            python_log_level='debug',
        )

        log.info(f'Initialized {model_name} model')
        return model

    @staticmethod
    def make_zero_reward(ref_tensor: torch.Tensor):
        """Helper to instantiate an empty reward tensor.

        The output will be a zero tensor with the same shape, device, and dtype
        as ref_tensor
        """
        return torch.zeros_like(ref_tensor).to(
            ref_tensor.device,
        ).type(ref_tensor.dtype)

    @staticmethod
    def _to_cpu(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.cpu()
        elif isinstance(x, (tuple, list)):
            return [RewardManager._to_cpu(x_) for x_ in x]
        elif isinstance(x, dict):
            return {k: RewardManager._to_cpu(v) for k, v in x.items()}
        else:
            return x

    @staticmethod
    def call_reward_model(
        reward_model: RewardModel,
        batch: MutableMapping,
    ):
        """Calls the reward model and extract rewards.

        This function will call the reward model (local or inference) and extract
        the rewards from the model output. The extracted rewards will be scattered
        into a reward tensor.

        Args:
            reward_model (RewardModel): the reward model to call.
            batch (MutableMapping): the batch of data to compute the reward.

        Returns:
            rewards (Tensor): a tensor of rewards. This is the result of scattering
                the extracted rewards into the zero_rewards tensor.
        """
        # We need to do this to handle getting rewards at multiple points in a
        # single input sequence with a deployed RM.
        if isinstance(reward_model, InferenceRewardModel):
            rm_seq_lens = [
                [idx + prompt_len
                 for idx in gather_indices]
                for gather_indices, prompt_len in
                zip(batch['end_idxs_gather'], batch['reward_prompt_lens'])
            ]
        else:
            rm_seq_lens = batch['reward_seq_lens']

        reward_batch = {
            'input_ids': batch['tok_formatted_reward_inputs'],
            'attention_mask': batch['tok_formatted_reward_attn_masks'],
            'seq_lens': rm_seq_lens,
            'is_inference': True,
            'seq_reward': True,
        }

        # Note this uses separate seq lengths to account for potential
        # changes made during reward string formatting
        curr_rewards = reward_model(
            reward_batch,
        ).to(dtype=batch['zero_rewards'].dtype,)

        assert isinstance(curr_rewards, torch.Tensor)
        # Passing in reward_seq_lens to make sure RL formatting in env_generate
        # and special reward formatting idxs match up before scattering rewards
        output: torch.Tensor = scatter_gather_rewards(
            temp_rews=batch['zero_rewards'],
            curr_rewards=curr_rewards,
            reward_prompt_lens=batch['reward_prompt_lens'],
            prompt_lens=batch['prompt_lens'],
            reward_generated_lens=batch['reward_generated_lens'],
            generated_lens=batch['generated_lens'],
            end_idxs_gather=batch['end_idxs_gather'],
            end_idxs_scatter=batch['end_idxs_scatter'],
            reward_seq_lens=batch['reward_seq_lens'],
            seq_lens=batch['seq_lens'],
        )
        return output

    def __call__(
        self,
        raw_untokenized_texts: list[tuple[str, str]],
        right_padded_obses: torch.Tensor,
        attention_masks: torch.Tensor,
        seq_lens: torch.Tensor,
        generated_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        max_gen_length: int,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        device_train_microbatch_size: int,
        kl_estimator: str,
        verified_answers: Optional[list[str]] = None,
    ) -> tuple[ReferenceOutput, RewardOutput]:
        """Collect rewards for generations.

        Args:
            raw_untokenized_texts (list): A list of (prompt, generation) string
                pairs from decoding the tokens seen/produced by the policy model.
            right_padded_obses (tensor): The right padded prompt+generation
                token sequences from calling generate on the policy model.
            attention_masks (tensor): A mask tensor indicating which tokens
                in right_padded_obses are padding tokens.
            seq_lens (tensor): The combined prompt+generation token length of
                each sequence.
            generated_lens (tensor): The number of tokens generated by the policy
                for each sequence.
            prompt_lens (tensor): The number of tokens in the prompt given to
                the policy.
            max_gen_len (int): The maximum number of tokens the policy is able
                to generate.
            actions (tensor): The (right padded) tokens generated by the policy.
            action_log_probs (tensor): The log probability of generating each action.
            device_train_microbatch_size (int): The device train microbatch size, which we need to compute log_probs otherwise we see numerical differences.
            kl_estimator (str): Which kl estimator to use. Options are 'k1', 'k2', 'k3' and 'k3_offpolicy'.
            verified_answers (Optional[list[str]]): A list of answers for verifiable rewards.

        Returns:
            ReferenceOutput: A tuple of float tensors. The first tensor is the
                (estimated) per-token KL between the policy and the reference model.
                The second tensor is the log probability of generating each action
                according to the reference model.
            RewardOutput: A dictionary of float tensors, with an entry for each reward
                model managed by the reward manager. For reward models that are called
                async, the associated value is an AsyncResult object that will return
                the reward tensor from its `.get()` method.
        """
        device = right_padded_obses.device.type

        # Only process text for the existing granularity types of the rewards
        processed_inputs = batch_process_fine_granularities(
            raw_untokenized_texts=raw_untokenized_texts,
            granularity_types=self.granularity_types,
            generated_lens=generated_lens.cpu().tolist(),
            prompt_lens=prompt_lens.cpu().tolist(),
            original_obses=right_padded_obses.cpu().tolist(),
            parser=self.parser,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            device=device,
        )

        computed_rewards: RewardOutput = {}

        # Base batch that we will adjust per reward mdoel
        batch = {
            'input_ids': right_padded_obses,
            'attention_mask': attention_masks,
            'actions': actions,
            'prompt_len': prompt_lens,
            'max_gen_len': max_gen_length,
            'generated_lens': generated_lens,
            'seq_lens': seq_lens,
            'action_log_probs': action_log_probs,
        }
        if verified_answers is not None:
            batch['verified_answers'] = verified_answers

        for reward_name in chain(
            self.functional_rewards,
            self.inference_rewards,
            self.local_reward_models,
        ):
            curr_reward = self.all_rewards[reward_name]
            curr_batch = self._create_batch(
                self.all_rewards[reward_name],
                reward_name,
                processed_inputs,
                batch,
                raw_untokenized_texts,
            )
            curr_batch['zero_rewards'] = self.make_zero_reward(action_log_probs)
            func = curr_reward
            args = (self._to_cpu(curr_batch),)

            if isinstance(
                curr_reward,
                Reward,
            ) or isinstance(curr_reward, InferenceRewardModel):
                if isinstance(curr_reward, InferenceRewardModel):
                    func = self.call_reward_model
                    args = (
                        self.all_rewards[reward_name],
                        self._to_cpu(curr_batch),
                    )

                assert self.pool is not None
                computed_rewards[reward_name] = self.pool.apply_async(
                    func=func,
                    args=args,
                )
            elif isinstance(curr_reward, RewardModel):
                computed_rewards[reward_name] = self.call_reward_model(
                    self.all_rewards[reward_name],
                    curr_batch,
                )
            else:
                raise TypeError(
                    f'Unknown reward model type {type(curr_reward)}. Expected `Reward` or `RewardModel`.',
                )

        batch['zero_rewards'] = self.make_zero_reward(action_log_probs)
        # Lastly, call the reference model
        ref_output = self.compute_reference_model_kl(
            batch,
            device_train_microbatch_size,
            kl_estimator,
        )

        return ref_output, computed_rewards

    def _create_batch(
        self,
        reward_model: BaseReward,
        reward_name: str,
        processed_inputs: dict[str, Any],
        base_batch: dict[str, Any],
        raw_untokenized_texts: list[tuple[str, str]],
    ) -> dict[str, Any]:
        """Helper to get the callable and the input kwargs for the reward.

        Args:
            reward_model (BaseReward): the reward model to create the batch for.
            reward_name (str): the name of the reward to create the batch for.
            processed_inputs (dict): the processed inputs for the reward, based on granularity.
            base_batch (dict): the base batch to add the reward inputs to.
            raw_untokenized_texts (list): the raw untokenized texts.
        """
        if isinstance(reward_model, Reward):
            return {
                **base_batch,
                'raw_untokenized_texts': raw_untokenized_texts,
            }
        elif isinstance(reward_model, RewardModel):
            granularity = self.granularities[reward_name]
            curr_inputs = processed_inputs['end_reward_inputs_dict'][granularity
                                                                    ]
            tok_formatted_reward_inputs = torch.tensor(
                curr_inputs.input_ids,
            ).type(base_batch['input_ids'].dtype)
            tok_formatted_reward_attn_masks = torch.tensor(
                curr_inputs.attention_mask,
            ).type(base_batch['attention_mask'].dtype)

            return {
                'tok_formatted_reward_inputs':
                    tok_formatted_reward_inputs,
                'tok_formatted_reward_attn_masks':
                    tok_formatted_reward_attn_masks,
                'reward_seq_lens':
                    processed_inputs['reward_seq_lens_dict'][granularity],
                'reward_prompt_lens':
                    processed_inputs['reward_prompt_lens_dict'][granularity],
                'reward_generated_lens':
                    processed_inputs['reward_generated_lens_dict'][granularity],
                'end_idxs_gather':
                    processed_inputs['end_idxs_gather_dict'][granularity],
                'end_idxs_scatter':
                    processed_inputs['end_idxs_scatter_dict'][granularity],
                'prompt_lens':
                    base_batch['prompt_len'],
                'generated_lens':
                    base_batch['generated_lens'],
                'seq_lens':
                    base_batch['seq_lens'],
            }
        else:
            raise TypeError(
                f'Unknown reward model type {type(reward_model)}. Expected `Reward` or `RewardModel`.',
            )

    def compute_reference_model_kl(
        self,
        batch: MutableMapping,
        device_train_microbatch_size: int,
        kl_estimator: str,
    ) -> ReferenceOutput:
        """Computes the reference KL for a batch of data.

        Args:
            batch (MutableMapping): the batch of data to compute the reference KL.
            device_train_microbatch_size (int): The device train microbatch size.
            kl_estimator (str): Which kl estimator to use. Options are 'k1', 'k2', 'k3', 'k3_offpolicy'.
        """
        batch_size = batch['input_ids'].size(0)
        kl = []
        ref_model_log_probs = []
        for i in range(batch_size // device_train_microbatch_size):
            curr_batch = {
                key: value[i * device_train_microbatch_size:(i + 1) *
                           device_train_microbatch_size]
                if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            curr_ref_output = self.reference_model(curr_batch)
            curr_ref_log_probs = get_log_probs(
                logits=curr_ref_output.logits,
                actions=curr_batch['actions'],
                prompt_len=curr_batch['prompt_len'],
                max_gen_len=curr_batch['max_gen_len'],
            )

            logprob_diff = (
                curr_batch['action_log_probs'] - curr_ref_log_probs
            ).clamp(min=-40.0, max=40.0)
            approxkl_k1 = logprob_diff
            approxkl_k2 = 0.5 * (logprob_diff**2)
            approxkl_k3 = torch.expm1(-logprob_diff) + logprob_diff
            approxkl_k3_offpolicy = 1.0 - torch.exp(-logprob_diff)

            curr_kl = 0.0
            if kl_estimator == 'k1':
                curr_kl = approxkl_k1
            elif kl_estimator == 'k2':
                # The k2_loss is approximately equivalent to the one-step KL divergence penalty with the k1 estimator
                # used in https://arxiv.org/pdf/2310.10505.
                curr_kl = approxkl_k2
            elif kl_estimator == 'k3':
                # The k3 estimator is the non negative kl approximation in http://joschu.net/blog/kl-approx.html
                curr_kl = approxkl_k3
            elif kl_estimator == 'k3_offpolicy':
                # This is taken from https://hongyuzang.notion.site/The-critical-implementation-detail-of-KL-loss-in-GRPO-1ae3fe2c1ff9809a9307c5402e190373
                # This is specifically for off-policy learning and can be useful for async training.
                curr_kl = approxkl_k3_offpolicy

            kl.append(curr_kl)
            ref_model_log_probs.append(curr_ref_log_probs)

        kl = torch.cat(kl)
        ref_model_log_probs = torch.cat(ref_model_log_probs)
        ref_output = (kl, ref_model_log_probs)

        return ref_output

    def resolve_outputs(
        self,
        ref_output: ReferenceOutput,
        reward_output: RewardOutput,
        kl_ctl: BaseKLController,
        action_mask: torch.Tensor,
        center_reward_mean: Optional[float] = None,
    ) -> dict[str, torch.Tensor]:
        """Resolve async results and finalize reward dict.

        Note: This method will wait for any AsyncResults to finish, so the associated async
        calls become blocking once this method is called. This method is separated from
        the __call__ method to make it easier to perform this (potentially) blocking step
        as long after __call__ as possible (ie, to best leverage the async setup).

        Args:
            ref_output (ReferenceOutput): The first output of the __call__ method.
                The ReferenceOutput tuple has two elements: the reference KL and
                the reference log probs, in that order.
            reward_output (RewardOutput): The second output of the __call__ method.
            kl_ctl (BaseKLController): KL controller object that provides the
                coefficient of the KL penalty in the aggregate reward.
            action_mask (Tensor): A mask tensor indicating which action tokens
                are padding.
            center_reward_mean (float, optional): An offset to subtract from the
                aggregate environment rewards (subtracted before the KL penalty is
                added). Default: no offset is subtracted.

        Returns:
            outputs: a dictionary capturing all the reward outputs, including the
                aggregate reward for RL training, as well as the rewards from each
                reward model.
        """
        device = action_mask.device

        # Resolve any output elements that are being computed async,
        # waiting for them to finish where necessary.
        resolved_reward_outputs: dict[str, torch.Tensor] = {}
        bad_end_generation_mask = None
        bad_end_generation_name = None
        for name, subreward in reward_output.items():
            if isinstance(subreward, AsyncResult):
                resolved_reward: torch.Tensor = subreward.get()
            else:
                resolved_reward: torch.Tensor = subreward
            resolved_reward_outputs[name] = resolved_reward.to(device=device)
            if isinstance(self.all_rewards[name], BadGenerationEndReward):
                bad_end_generation_name = name
                bad_generation_row_mask = torch.any(resolved_reward != 0, dim=1)

                bad_end_generation_mask = (
                    ~bad_generation_row_mask
                ).unsqueeze(1).expand_as(resolved_reward)
                bad_end_generation_mask = bad_end_generation_mask.to(
                    device=device,
                )

        ref_kl = ref_output[0].to(device=device)
        ref_log_probs = ref_output[1].to(device=device)

        if self.kl_penalty_in_reward:
            rewards: torch.Tensor = -kl_ctl.value * ref_kl.detach()
        else:
            rewards: torch.Tensor = torch.zeros_like(ref_kl)

        env_rewards = self.make_zero_reward(rewards)

        rews_dict_out: dict[str, torch.Tensor] = {}
        for name, subreward in resolved_reward_outputs.items():
            if name not in self.reward_coefficients:
                raise KeyError(
                    f'Reward with {name=} is not recognized by the reward manager.',
                )
            env_rewards += subreward.detach() * self.reward_coefficients[name]

            # In the output, make sure each key has 'reward' in it to engage
            # proper logging (see .loss of policy class)
            out_name = name + '_reward' if 'reward' not in name else ''
            rews_dict_out[out_name] = subreward.detach() * action_mask

        # Masking out all rewards if the generation ends with a bad token
        # And strictly adding a penalty for bad generation ending.
        if bad_end_generation_mask is not None and bad_end_generation_name is not None:
            env_rewards *= bad_end_generation_mask
            env_rewards += (
                resolved_reward_outputs[bad_end_generation_name].detach() *
                self.reward_coefficients[bad_end_generation_name]
            )

        # Optionally apply an offset to the environment rewards
        if center_reward_mean is not None:
            env_rewards -= center_reward_mean

        # Final rewards is total env rewards + KL penalties
        rewards += env_rewards

        # Zero rewards at padded tokens
        rewards *= action_mask
        env_rewards *= action_mask

        outputs = {
            'rewards': rewards.detach(),
            'env_rewards': env_rewards.detach(),
            'ift_log_probs': ref_log_probs.detach(),
            'ift_kl': ref_kl.detach(),
        }
        outputs.update(rews_dict_out)

        return outputs
