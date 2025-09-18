import traceback
from typing import Any, Dict
import asyncio

from composer.loggers import MLFlowLogger
import torch
from composer import Trainer
from composer.core import get_precision_context
from composer.core.data_spec import _default_split_batch
from composer.trainer.trainer import _get_initial_device_train_microbatch_size
from compose_rl.data.buffer import MinibatchRolloutBuffer
from composer.optim import DecoupledAdamW
from composer.utils import dist
from llmfoundry.utils import build_composer_model
from llmfoundry.utils.config_utils import process_init_device  # type: ignore
from omegaconf import DictConfig, OmegaConf as om
from transformers import AutoTokenizer
from composer.callbacks import MemoryMonitor, SpeedMonitor, LRMonitor

from compose_rl.registry_builders import build_kl_controller
from compose_rl.algorithms.online import (
    ComposerHFPolicyLM,
    ComposerHFCriticFreePolicyLM,
    SingleControllerOnPolicyCallback,
)
from compose_rl.utils import (
    approx_kl,
    dist_compute_masked_mean_and_var,
    get_log_probs,
    get_entropies,
    switch_left_to_right_padding,
    mask_eos,
    masked_sum,
    masked_mean,
)

from compose_rl.algorithms.online.model_methods import OnPolicyEnum

import logging

from orl_servers.vllm_worker_wrap import stateless_init_process_group

from orl_servers.vllm_remote import RemoteVLLMEngine

from orl_servers.structs import InferenceEngineConfig

from orl_servers.async_utils import run_async_sync

from orl_servers.generation_utils import broadcast_to_vllm as external_broadcast_to_vllm
log = logging.getLogger(__name__)

# FastAPI imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Pydantic models for API requests/responses
class InitializeRequest(BaseModel):
    config: Dict[str, Any]

class InitializeModelUpdateGroupRequest(BaseModel):
    new_port: int
    num_vllm_servers: int
    gen_tp_size: int

class CreateMinibatchesRequest(BaseModel):
    current_rank_rollouts: Dict[str, Any]

class BroadcastToVLLMRequest(BaseModel):
    addresses: list[str]

class HealthResponse(BaseModel):
    status: str
    message: str

# FastAPI app instance
app = FastAPI(title="Distributed GPU Actor Server", version="1.0.0")


class DistributedGPUActor:
    """Distributed GPU actor for testing."""

    def __init__(self):
        self.config = None
        self.model = None
        self.reference_model = None
        self.model_update_group = None
        self.ref_path = None
        self._dataloader = None
        self._tokenizer = None
        self.ppo_callback = None
        self.ppo_trainer: Trainer = None  # type: ignore
        self.buffer: MinibatchRolloutBuffer = None  # type: ignore

        self.pretrain_model_name = None
        self.device_train_batch_size = None
        self.num_batches_per_update = None
        self.max_seq_len = None
        self.precision = None  # type: ignore
        self.train_config: dict = None  # type: ignore
        self.variables_config: dict = None  # type: ignore
        self.model_config = None
        self.ref_model_config = None
        self.global_train_batch_size = None
        self.max_gen_len = None
        self.loss_type = None

        # KL Penalty and Controller
        self.kl_ift = []
        self.kl_controller = None
        self.kl_controller_config = None
        self.kl_penalty_in_reward = None

        # Reward info
        self.reward_coefficients: dict = None  # type: ignore

        self.model_update_group = None

        # RL iteration variables
        self.rl_iter = 0

    def initialize(self, config: Any):
        self.init_composer_dist()
        self._build_train_config(config)

        # Build PPO trainers
        self._build_ppo_trainer()

        # Build Minibatch Buffer
        self._build_buffer()

        # Build Reference Model
        self._build_reference_model()

        # Build KL Controller
        self._build_kl_controller()
        log.info(f'Initialized train actor')

    
    def initialize_model_update_group(self, new_port: int, num_vllm_servers: int, gen_tp_size: int):
        if dist.get_global_rank() == 0:
            self.model_update_group = stateless_init_process_group(
                    "127.0.0.1", new_port, 0, num_vllm_servers * gen_tp_size + 1, torch.cuda.current_device())

    def broadcast_to_vllm(self, addresses: list[str]):
        vllm_engine = RemoteVLLMEngine(InferenceEngineConfig(), addresses)

        self.ppo_callback.actor_critic.model_update_group = self.model_update_group
        run_async_sync(external_broadcast_to_vllm(
            model=self.ppo_callback.actor_critic,
            vllm_engine=vllm_engine,
            device=torch.device(f'cuda:{dist.get_local_rank()}'),
            loss_type=self.ppo_callback.actor_critic.loss_type,  # type: ignore
            enable_prefix_caching=self.config.vllm_enable_prefix_caching,
        ))
        

    def _build_train_config(self, config: Any):
        self.config = config
        log.info(f"Starting build_train_config with model: {self.config.model.pretrained_model_name_or_path}")
        self.pretrain_model_name = self.config.model.pretrained_model_name_or_path

        self.model_config = om.to_container(self.config.model, resolve=True)
        self.model_config['tokenizer'] = self.tokenizer
        self.loss_type = self.model_config.get('loss_type', OnPolicyEnum.GRPO)
        log.info("--------------------------------")
        log.info(f'loss_type: {self.loss_type}')
        log.info("--------------------------------")

        # Reference Model Initializing
        self.ref_model_config = om.to_container(self.config.variables.reference_model.model_config, resolve=True)

        self.global_train_batch_size = self.config.global_train_batch_size
        self.device_train_batch_size = self.global_train_batch_size // dist.get_world_size()
        self.num_batches_per_update = self.config.variables.num_batches_per_update
        self.max_seq_len = self.config.max_seq_len
        self.max_gen_len = self.config.variables.max_gen_len
        self.precision = self.config.precision

        # NOTE: if compute kl loss then no reward penalty
        # TODO: we should be more explicit about this toggle / make each kl regularization mechanism explicit
        self.kl_controller_config = om.to_container(self.config.variables.kl_controller, resolve=True)
        self.kl_penalty_in_reward = not self.model_config.get('compute_kl_loss', False)

        # Reward Coefficients
        all_rewards_config = om.to_container(self.config.variables.rewards, resolve=True)
        self.reward_coefficients = {}
        for reward_name, reward_config in all_rewards_config.items():
            self.reward_coefficients[reward_name] = reward_config.get(
                'reward_coefficient',
                1.0,
            )

        variables = om.to_container(self.config.variables, resolve=True)
        self.variables_config = variables
        algorithm_config = self.config.algorithms

        self.train_config = {
            'seed': self.config.seed,
            'model': self.model_config,
            'ref_model': self.ref_model_config,
            'fsdp_config': self.config.fsdp_config,
            'kl_controller': self.kl_controller_config,
            'non_train_fsdp_config': self.variables_config.get('non_train_fsdp_config', self.config.fsdp_config),
            'precision': self.precision,
            'variables': variables,
            'algorithms': algorithm_config,
            'global_train_batch_size': self.device_train_batch_size * dist.get_world_size(),
            'device_train_batch_size': self.device_train_batch_size,
            'device_train_microbatch_size': self.device_train_batch_size,
            'save_folder': self.config.save_folder,
            'log_config': self.config.log_config,
            'max_seq_len': self.max_seq_len,
            'python_log_level': self.config.python_log_level,
            'console_log_interval': self.config.console_log_interval,
        }
        log.info("Finished build_train_config")

    def _build_buffer(self):
        self.buffer = MinibatchRolloutBuffer(self.variables_config['buffer'])
        log.info(f'Initialized minibatch buffer.')

    def _build_tokenizer(self):
        # TODO (algo): decide if we should use tokens or messages given
        # we may need token level log prob
        # TODO (infra): use the tokenizer/texts for prompt dataloader but
        # token (ids) for the experience buffer/manager
        kwargs = self.config.tokenizer.kwargs
        tokenizer = AutoTokenizer.from_pretrained(self.pretrain_model_name, **kwargs)
        return tokenizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self._build_tokenizer()
        return self._tokenizer

    def init_composer_dist(self):
        log.info(f'Initializing composer dist {dist.get_local_rank()}, {dist.get_global_rank()}, {dist.get_world_size()}')
        dist.initialize_dist('gpu')

    def _build_kl_controller(self):
        kl_controller_name = self.kl_controller_config.pop('kl_ctl_type')
        self.kl_controller = build_kl_controller(
            name=kl_controller_name,
            kwargs=self.kl_controller_config,
        )
        log.info(f'Built KL Controller')

    def _build_reference_model(self):
        name = self.ref_model_config.pop('name')
        fsdp_config = self.variables_config.get('non_train_fsdp_config', self.config.fsdp_config)

        init_context = process_init_device(
            self.ref_model_config,
            fsdp_config,
        )

        self.reference_model = build_composer_model(
            name=name,
            cfg=self.ref_model_config,
            tokenizer=self.tokenizer,
            init_context=init_context,
            master_weights_dtype=self.ref_model_config.get('master_weights_dtype', None),
        )

        parallelism_config = {'fsdp': fsdp_config}

        load_path = self.variables_config['reference_model'].get('load_path', None)

        # Create a Trainer object to load from checkpoint and FSDP the model
        # TODO: use FSDP2 utils to FSDP module.
        _ = Trainer(
            model=self.reference_model,
            parallelism_config=parallelism_config,
            precision=self.precision,
            load_weights_only=True,
            load_strict_model_weights=False,
            load_path=load_path,
            python_log_level='debug',
        )
        log.info(f'Initialized {name} reference model')

    def _build_ppo_trainer(self):
        name = self.model_config.pop('name')

        log.info(f"Model type: {name}")
        if name == 'hf_ppo_lm':
            log.info("Creating ComposerHFPolicyLM")
            model = ComposerHFPolicyLM(**self.model_config)
        elif name == 'hf_critic_free_lm':
            log.info("Creating ComposerHFCriticFreePolicyLM")
            model = ComposerHFCriticFreePolicyLM(**self.model_config)
        log.info("Model created successfully")

        # TODO: Add weight decay
        optimizer = DecoupledAdamW(model.parameters(), lr=1e-6)

        # NOTE: there is no reliance on the callback anymore
        self.ppo_callback = SingleControllerOnPolicyCallback(
            train_config=self.train_config,
        )

        # Create a dummy dataloader to make sure trainer can call .fit() with
        # the dataloader that exists at ITERATION_START. This dataloader
        # will NOT be used for training.
        dummy_dataset = torch.utils.data.TensorDataset(torch.randn(16, 1))
        dummy_distributed_sampler = torch.utils.data.distributed.DistributedSampler(dummy_dataset)
        dummy_dataloader = torch.utils.data.DataLoader(dummy_dataset, sampler=dummy_distributed_sampler)

        mlflow_logger = MLFlowLogger(
            experiment_name=self.config.loggers.mlflow.experiment_name,
            run_name=self.config.loggers.mlflow.tags.run,
            tracking_uri=self.config.loggers.mlflow.tracking_uri,
        )

        callbacks = [
            self.ppo_callback,
            # callbacks for scheduled garbage collection
            # this helps improve throughput by garbage collecting
            # at regular intervals on all training processes
            # ScheduledGarbageCollector(
            #     batch_interval='1000',
            # ), # TODO: Add it back after we resolve some error because we are using a dummy dataloader
            # callbacks for monitoring other metrics
            LRMonitor(),
            MemoryMonitor(),
            SpeedMonitor(window_size=10),
        ]

        self.ppo_trainer = Trainer(
            model=model,
            optimizers=optimizer,
            callbacks=callbacks,
            train_dataloader=dummy_dataloader,
            precision=self.precision,
            parallelism_config={'fsdp': self.config.fsdp_config},
            max_duration=self.config.max_duration,
            loggers=[mlflow_logger],
            device_train_microbatch_size=self.config.device_train_microbatch_size,
            load_path=self.ref_path,
            save_folder=self.config.save_folder,
            save_interval=self.config.save_interval,
            autoresume=self.config.autoresume,
        )

    def close_trainer(self):
        self.ppo_trainer.close()

    # TODO: maybe make the name more informative?
    # TODO: think about how best to split this function up?
    def create_online_minibatches(self, current_rank_rollouts: dict[str, Any]):
        """Processes rollouts and creates minibatches for online learning.

        This function takes the rollouts, computes the log probs, kl, and advantages
        and splits them into minibatches for the PPO Trainer.
        """
        for k, v in current_rank_rollouts.items():
            assert isinstance(v, torch.Tensor) or isinstance(v, list) or isinstance(v, dict), f"Expected a tensor or list or dict, got {type(v)}"
            if isinstance(v, list) and not isinstance(v[0], str):
                v = torch.tensor(v)
            if isinstance(v, torch.Tensor):
                current_rank_rollouts[k] = v.to(torch.device('cuda'))
            elif isinstance(v, dict):
                # This is the case with the rewards dict where it has (key, tensor) pairs
                rewards_dict_for_rank = {}
                for reward_key, reward_tensor in v.items():
                    if isinstance(reward_tensor, list):
                        reward_tensor = torch.tensor(reward_tensor)
                    rewards_dict_for_rank[reward_key] = reward_tensor.to(torch.device('cuda'))
                current_rank_rollouts[k] = rewards_dict_for_rank
            elif not (isinstance(v, list)):
                raise ValueError(f"Expected a tensor or list or dict of tensors, got {type(v)}")

        device = torch.device('cuda')
        with get_precision_context(self.precision), torch.no_grad():
            # 1) Compute Log Probs and Entropy
            partial_batch = self.get_log_probs_and_entropy(current_rank_rollouts, device)
            # 2) Compute Reference Log Probs and KL
            reference_output = self.get_reference_log_probs_and_kl(partial_batch)
 
            # Log to callback for KL Controller Update
            mean_ift = masked_mean(
                reference_output['kl'],
                partial_batch['action_mask'],
            )
            self.kl_ift.append(mean_ift.cpu())

            # 3) Scale rewards and apply KL Penalty
            reward_output = self.update_rewards(current_rank_rollouts['all_rewards_dict'], reference_output, partial_batch['action_mask'], device)

            # 4) Compute Advantages
            # TODO: For full correctness do all gather
            advantage_output = self.compute_advantages(partial_batch, reward_output)

            # Construct batch
            bs = partial_batch['prompt_id'].shape[0]
            batch = {
                'max_gen_len': torch.ones(bs).to(torch.int32) * self.max_gen_len,
                'ift_kl_scalar': torch.ones(bs) * self.kl_controller.value,
                **partial_batch,
                **reference_output,
                **reward_output,
                **advantage_output,
            }

            # Moving minibatches to CPU to not take additional GPU memory
            for k, v in batch.items():
                if hasattr(v, 'cpu'):
                    batch[k] = v.cpu()

        # NOTE: Probably should break things up but putting it here for now for clarity
        # Delete Non-tensor keys for training batch
        for key in ['verified_answer', 'messages']:
            if key in batch.keys():
                del batch[key]

        # We need to split the resolved outputs into minibatches
        for idx in range(
            batch['prompt_id'].shape[0] // self.device_train_batch_size,
        ):
            minibatch = self._extract_minibatch(
                batch,
                idx,
                self.device_train_batch_size,
            )
            self.buffer.add(minibatch)

        # Making sure we correctly parsed the minibatches
        assert len(
            self.buffer,
        ) == self.num_batches_per_update, f'{len(self.buffer)} != {self.num_batches_per_update}'

        self.ppo_trainer.state.model.train()

        # Reset and initialize state train dataloader
        log.warning(
            'trainer._train_data_spec should be updated whenever the dataloader is updated',
        )
        # Train Dataloader
        self.ppo_trainer.state.set_dataloader(self.buffer, 'ep')
        self.ppo_trainer.state.train_dataloader = self.ppo_trainer.state.dataloader
        self.ppo_trainer.state.device_train_microbatch_size = _get_initial_device_train_microbatch_size(
            self.ppo_trainer.state.device_train_microbatch_size,
            self.ppo_trainer.state.auto_microbatching,
            self.ppo_trainer.state.train_dataloader,
        )

        self._update_ift_kl()

    def _update_ift_kl(self):
        local_kl = torch.stack(self.kl_ift)
        global_ift_kl = torch.cat(dist.all_gather_object(local_kl))
        ift_kl_update = torch.mean(global_ift_kl)

        self.kl_controller.update(
            ift_kl_update,
            self.num_batches_per_update * self.device_train_batch_size *  # type: ignore
            dist.get_world_size(),
        )

        self.kl_ift = []

    def _extract_minibatch(
        self,
        batch: dict[str, torch.Tensor],
        idx: int,
        minibatch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Extracts a minibatch from a composite batch.

        This helper is used to extract a particular minibatch of size
        minibatch_size from `batch`, where `batch` may
        have a batch size that exceeds the minibatch size.

        Args:
            batch (dict[str, torch.Tensor]): an arbitrary batch, where
                each entry has batch size >= minibatch_size,
                representing the concatenation of >= 1 minibatches.
            idx (int): The index of the batch (see above description) to extract.

        Returns:
            curr_gen_batch (dict[str, torch.Tensor]): The gen_batch_idx'th
                gen_batch extracted from the batch input.
        """
        start_idx = idx * minibatch_size
        end_idx = (idx + 1) * minibatch_size
        curr_gen_batch = {
            batch_key: tensor[start_idx:end_idx]
            for batch_key, tensor in batch.items()
        }
        return curr_gen_batch

    def get_log_probs_and_entropy(self, current_rank_rollouts: dict[str, Any], device: torch.device):
        prompt_tokens = current_rank_rollouts['prompt']
        batch_size, _ = prompt_tokens.shape
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_ids = self.variables_config['eos_token_ids']
        prompt_len = current_rank_rollouts['prompt_len']
        prompt_id = current_rank_rollouts['prompt_id']
        prompt_dtype = prompt_tokens.dtype

        assert 'sequences' in current_rank_rollouts, f'sequences is not in batch {current_rank_rollouts.keys()=}'
        assert 'vllm_logprobs' in current_rank_rollouts, f'vllm_logprobs is not in batch {current_rank_rollouts.keys()=}'
        sequences = current_rank_rollouts['sequences']
        vllm_logprobs = current_rank_rollouts['vllm_logprobs']
        generated_len = torch.ones(
            batch_size,
            device=device,
            dtype=prompt_dtype,
        ) * self.max_gen_len

        # If all the processes early exit generate, then we need to manually pad everything
        # we can pad this with pad tokens, since we switch the padding between left and right
        # padding based on the sequence length + max_sequence_length.
        if prompt_tokens.size(1) + self.max_gen_len > sequences.size(1):
            len_to_pad = self.max_gen_len - (
                sequences.size(1) - prompt_tokens.size(1)
            )

            extra_padding = torch.ones(
                (batch_size, len_to_pad),
                device=device,
                dtype=prompt_dtype,
            ) * pad_token_id
            sequences = torch.cat(
                [sequences, extra_padding],  # type: ignore
                dim=-1,  # type: ignore
            )

            extra_zero_padding = torch.zeros(
                (batch_size, len_to_pad),
                device=device,
                dtype=torch.float,
            )
            vllm_logprobs = torch.cat(
                [vllm_logprobs, extra_zero_padding],  # type: ignore
                dim=-1,  # type: ignore
            )

        # Sanity checking we're adding max_gen_len to prompt_tokens
        if prompt_tokens.size(1) + self.max_gen_len != sequences.size(1):
            raise ValueError(
                f'Prompts {prompt_tokens.size(1)} + max_gen_len {self.max_gen_len} != sequences {sequences.size(1)}',
            )

        # Actions are what tokens the current policy would generate.
        actions = sequences[:, -self.max_gen_len:]  # type: ignore
        vllm_logprobs_gen = vllm_logprobs[:, -self.max_gen_len:]  # type: ignore

        right_padded_obs = switch_left_to_right_padding(
            sequences,
            prompt_len,
            self.max_gen_len,
            pad_token_id,  # type: ignore
        )
        right_padded_attn_mask = torch.logical_not(
            torch.eq(right_padded_obs, pad_token_id),  # type: ignore
        )

        (
            right_padded_obs,
            right_padded_attn_mask,
            generated_len,
            action_mask,
        ) = mask_eos(
            actions=actions,
            right_padded_obs=right_padded_obs,
            right_padded_attn_mask=right_padded_attn_mask,
            prompt_len=prompt_len,
            generated_len=generated_len,
            max_gen_len=self.max_gen_len,
            eos_token_ids=eos_token_ids,  # type: ignore
            pad_token=pad_token_id,  # type: ignore
        )
        log_probs = []
        entropies = []
        values = []

        input_model_kwargs = {
            'obs': right_padded_obs,
            'right_padded_attn_mask': right_padded_attn_mask,
            'prompt_len': prompt_len,
            'max_gen_len': self.max_gen_len,
            'action_mask': action_mask,
            'actions': actions,
        }

        microbatch_splits = _default_split_batch(
            batch=input_model_kwargs,
            microbatch_size=self.config.device_train_microbatch_size,
        )
        # Compute the device_train_microbatch_log_probs inside the for loop to reduce the softmax overhead
        for split in microbatch_splits:
            curr_kwargs = split

            cur_output = self.ppo_trainer.state.model(curr_kwargs)
            cur_logits = cur_output['logits']
            # need to pull out current actions and prompt len
            cur_actions = curr_kwargs['actions']
            cur_action_mask = curr_kwargs['action_mask']
            cur_prompt_len = curr_kwargs['prompt_len']

            cur_log_probs = get_log_probs(
                logits=cur_logits,
                actions=cur_actions,
                prompt_len=cur_prompt_len,
                max_gen_len=self.max_gen_len,
            )
            cur_entropies = get_entropies(
                logits=cur_logits,
                action_mask=cur_action_mask,
                prompt_len=cur_prompt_len,
                max_gen_len=self.max_gen_len,
            )
            log_probs.append(cur_log_probs)
            entropies.append(cur_entropies)
            # Ignore values when the model doesn't have a value head
            if 'values' in cur_output:
                cur_values = cur_output['values']
                values.append(cur_values)

        device_train_microbatch_log_probs = torch.cat(log_probs)
        device_train_microbatch_entropies = torch.cat(entropies)

        assert vllm_logprobs_gen.shape == device_train_microbatch_log_probs.shape, f'vllm_logprobs_gen and device_train_microbatch_log_probs have different shapes {vllm_logprobs_gen.shape=}, {device_train_microbatch_log_probs.shape=}'


        partial_env_output = {
            'prompt_id': prompt_id,
            'old_log_probs': device_train_microbatch_log_probs,
            'old_entropies': device_train_microbatch_entropies,
            'obs': right_padded_obs,
            'right_padded_attn_mask': right_padded_attn_mask,
            'actions': actions,
            'action_mask': action_mask,
            'generated_len': generated_len,
            'prompt_len': prompt_len,
            'vllm_logprobs': vllm_logprobs_gen,
        }
        if len(values) > 0:
            device_train_microbatch_values = torch.cat(values)

            # Need to add in the padding for the value function
            value_action_mask = torch.cat([
                action_mask,
                torch.zeros((batch_size, 1), device=device),
            ],
                                          dim=-1)
            device_train_microbatch_values *= value_action_mask
            partial_env_output['values'] = device_train_microbatch_values

        # TODO: old_log_probs, old_entropies, metadata as a clearer output
        return partial_env_output

    def get_reference_log_probs_and_kl(self, batch: dict[str, Any]):
        """
        This function computes the reference log probs and computes KL estimates between pi and pi_ref.
        """
        kl = []
        ref_model_log_probs = []

        microbatch_splits = _default_split_batch(
            batch=batch,
            microbatch_size=self.config.device_train_microbatch_size,
        )
        for split in microbatch_splits:
            curr_batch = split
            curr_ref_output = self.reference_model({  # type: ignore
                "input_ids": curr_batch['obs'],
                "attention_mask": curr_batch['right_padded_attn_mask'],
            })
            curr_ref_log_probs = get_log_probs(
                logits=curr_ref_output.logits,
                actions=curr_batch['actions'],
                prompt_len=curr_batch['prompt_len'],
                max_gen_len=self.max_gen_len,
                temperature=self.variables_config['generation_kwargs']['temperature'],
            )

            kl_dict = approx_kl(
                log_p=curr_ref_log_probs,
                log_q=curr_batch['old_log_probs'],
                kl_clip_range=self.model_config['kl_clip_range'],  # pyright: ignore
            )
            curr_kl = kl_dict[self.model_config['kl_estimator']]  # pyright: ignore

            kl.append(curr_kl)
            ref_model_log_probs.append(curr_ref_log_probs)

        kl = torch.cat(kl)
        ref_model_log_probs = torch.cat(ref_model_log_probs)
        ref_output = {
            "kl": kl,
            # TODO: rename to reference_log_probs
            #"reference_log_probs": ref_model_log_probs,
            "ift_log_probs": ref_model_log_probs,
        }
        return ref_output

    def update_rewards(self, raw_rewards_dict: dict[str, Any], ref_output: dict[str, Any], action_mask: torch.Tensor, device: torch.device):
        resolved_reward_outputs: dict[str, torch.Tensor] = {}
        bad_end_generation_name, bad_end_generation_mask = None, None
        for name, subreward in raw_rewards_dict.items():
            # Functional Rewards
            resolved_reward_outputs[name] = subreward.to(device=device)

            # NOTE: all rewards is not accesible here
            #if isinstance(self.all_rewards[name], BadGenerationEndReward):
            if name == "bad_generation_end":
                bad_end_generation_name = name
                bad_generation_row_mask = torch.any(subreward != 0, dim=1)

                bad_end_generation_mask = (
                    ~bad_generation_row_mask
                ).unsqueeze(1).expand_as(subreward)
                bad_end_generation_mask = bad_end_generation_mask.to(
                    device=device,
                )

        # Reward Penalty
        ref_kl = ref_output['kl'].to(device=device)

        if self.kl_penalty_in_reward:
            rewards: torch.Tensor = -self.kl_controller.value * ref_kl.detach()
        else:
            rewards: torch.Tensor = torch.zeros_like(ref_kl)

        env_rewards = torch.zeros_like(rewards)
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
        # TODO: General scaling of reward values through whitening should be revisited
        # if center_reward_mean is not None:
        #    env_rewards -= center_reward_mean
        #

        # Final rewards is total env rewards + KL penalties
        rewards += env_rewards

        # Zero rewards at padded tokens
        rewards *= action_mask
        env_rewards *= action_mask

        outputs = {
            'rewards': rewards.detach(),
            'env_rewards': env_rewards.detach(),
        }
        outputs.update(rews_dict_out)

        return outputs

    # TODO: For different algorithms, have different Advantage functions. This one is specifically GRPO
    def compute_advantages(self, batch: dict[str, Any], reward_output: dict[str, Any]):
        # compute GRPO advantages
        bs = batch['prompt_id'].shape[0]
        prompt_id = batch['prompt_id']
        rewards = reward_output['rewards']

        # Flatten the rewards by summing on sequence length/action_mask
        flat_rewards = masked_sum(
            rewards,
            batch['action_mask'],
            dim=-1,
        )

        # Get unique prompt IDs and their indices
        unique_prompt_ids, inverse_indices = torch.unique(
            prompt_id,
            return_inverse=True,
        )

        # Use scatter to compute means and standard deviations
        # First, we'll create a tensor to track counts, sums, and sum of squares
        n_unique = len(unique_prompt_ids)
        counts = torch.zeros(n_unique, device=prompt_id.device)
        sums = torch.zeros(n_unique, device=prompt_id.device)
        sum_squares = torch.zeros(n_unique, device=prompt_id.device)

        # Use scatter_add to accumulate values
        counts.scatter_add_(
            0,
            inverse_indices,
            torch.ones_like(flat_rewards),
        )
        sums.scatter_add_(0, inverse_indices, flat_rewards)
        sum_squares.scatter_add_(0, inverse_indices, flat_rewards**2)

        # Compute means and standard deviations
        means = sums / counts
        variances = (sum_squares / counts) - (means**2)
        stds = torch.sqrt(variances)

        # Map back to original tensor shape
        mean_rewards = means[inverse_indices]
        std_rewards = stds[inverse_indices]

        # Calculate GRPO advantage
        grpo_advantage = (flat_rewards - mean_rewards)
        # Only normalize the advantage if flag is set
        if self.model_config['normalize_advantage']:  # type: ignore
            grpo_advantage /= (std_rewards + 1e-4)

        # Create advantages of the same shape as original rewards
        advantages = torch.zeros_like(rewards)
        # Copy the flat grpo_advantage according to action_mask
        expanded_advantages = grpo_advantage.unsqueeze(1).expand_as(
            batch['action_mask'],
        )
        advantages = torch.where(
            batch['action_mask'].bool(),
            expanded_advantages,
            advantages,
        )

        batch_adv_mean, batch_adv_var = dist_compute_masked_mean_and_var(
            advantages,
            batch['action_mask'],
        )

        advantage_output = {
            'advantages': advantages,
            'prompt_advantages': grpo_advantage,
            'adv_masked_mean': torch.ones(bs) * batch_adv_mean.cpu(),
            'adv_masked_var': torch.ones(bs) * batch_adv_var.cpu(),
            'reward_std': torch.ones(bs) * rewards.std().to('cpu'),
        }
        return advantage_output

    def train_1_iter(self):
        # TODO (algo): implement the top level PPO algo here instead of the
        # callback. Algorithmic researchers are expected to implement this
        # function along with above policy/value/reward/ref trainers or models
        # TODO (infra): try multiple fit to see if the (mlflow) logger, etc
        # TODO (infra): fault tolerance at iteration level first
        # TODO (infra): enable batch level control

        # NOTE: Trainer has a train microbatches function that should be used here to get low level control.
        # fit() checks if there is existing checkpoint, make a full forward pass, it will run eval pass and save pass.
        # We potentially want to run this https://github.com/mosaicml/composer/blob/dev/composer/trainer/trainer.py#L2826
        # fit() can also potentially overwrite the mlflow
        self.ppo_trainer.fit(duration='1iter')

        # After Iteration callback
        self.rl_iter += 1
        self.buffer.reset()
        log.info(f"#### Finished training 1 iter with loss: {self.ppo_trainer.state.loss}")


# Global actor instance
actor = DistributedGPUActor()


@app.post("/initialize")
async def initialize_actor(request: InitializeRequest):
    """Initialize the distributed GPU actor with the provided configuration."""
    try:
        # Run the synchronous initialize method in a thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, actor.initialize, DictConfig(request.config))
        return {"status": "success", "message": "Actor initialized successfully"}
    except Exception as e:
        log.error(f"Error initializing actor: {str(e)}, {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize actor: {str(e)}") from e


@app.post("/initialize_model_update_group")
async def initialize_model_update_group(request: InitializeModelUpdateGroupRequest):
    """Initialize the model update group with the provided configuration."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, actor.initialize_model_update_group, request.new_port, request.num_vllm_servers, request.gen_tp_size)
        return {"status": "success", "message": "Model update group initialized successfully"}
    except Exception as e:
        log.error(f"Error initializing model update group: {str(e)}, {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize model update group: {str(e)}") from e


@app.post("/create_online_minibatches")
async def create_online_minibatches(request: CreateMinibatchesRequest):
    """Create online minibatches from the provided rollouts."""
    try:
        # Run the synchronous create_online_minibatches method in a thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, actor.create_online_minibatches, request.current_rank_rollouts)
        return {"status": "success", "message": "Online minibatches created successfully"}
    except Exception as e:
        log.error(f"Error creating online minibatches: {str(e)}, {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to create online minibatches: {str(e)}") from e


@app.post("/broadcast_to_vllm")
async def broadcast_to_vllm(request: BroadcastToVLLMRequest):
    """Broadcast the model to the vLLM servers."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, actor.broadcast_to_vllm, request.addresses)
        return {"status": "success", "message": "Model broadcasted to vLLM servers successfully"}
    except Exception as e:
        log.error(f"Error broadcasting to vLLM servers: {str(e)}, {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to broadcast to vLLM servers: {str(e)}") from e


@app.post("/train_1_iter")
async def train_one_iteration():
    """Train the model for one iteration."""
    try:
        # Run the synchronous train_1_iter method in a thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, actor.train_1_iter)
        return {"status": "success", "message": "Training iteration completed successfully"}
    except Exception as e:
        log.error(f"Error training one iteration: {str(e)}, {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to train one iteration: {str(e)}") from e


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="Distributed GPU Actor server is running"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start the FastAPI Distributed GPU Actor server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8500 + dist.get_local_rank(), help="Port to run the server on (default: 8500)")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)