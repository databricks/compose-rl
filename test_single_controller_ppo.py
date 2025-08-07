# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0


# Copy the test file in the root of the repo
# NOTE: This actually runs GRPO instead of PPO
# cd compose-rl
# run cmd: composer test_single_controller_ppo.py
# If I do ctrl+c to kill job
# Check with `ray status` to see if the actors are still running
# If they are, then run `ray stop`

import argparse
from contextlib import contextmanager
import logging
import os
import pickle
import time
import datetime
from functools import partial
from typing import Any, Optional

from composer.loggers import MLFlowLogger
from mlflow.prompt.registry_utils import PromptVersion
import ray
import torch
import torch.distributed as dist
from composer import Trainer
from composer.core import get_precision_context
from composer.optim import DecoupledAdamW
from composer.utils import create_symlink_file, dist as composer_dist
from llmfoundry.data import build_dataloader
from omegaconf import OmegaConf as om
from transformers import AutoTokenizer
from composer.callbacks import MemoryMonitor, SpeedMonitor, LRMonitor

from compose_rl.algorithms.online import (
    ComposerHFPolicyLM,
    ComposerHFCriticFreePolicyLM,
    SingleControllerOnPolicyCallback,
)
from compose_rl.algorithms.online.generation_utils import (
    broadcast_to_vllm,
    create_vllm_engines,
    _vllm_generate,
)
from compose_rl.utils.ray_utils import start_ray_server, uninstall_megablocks_if_exists
from compose_rl.controllers import BaseDistributedGPUActor, SPMDActorGroup
from compose_rl.controllers.buffer import Buffer
from compose_rl.algorithms.online.callback_utils import preprocess_batches

GLOBAL_TRAIN_BATCH_SIZE = 64
GENERATIONS_PER_PROMPT = 8  
NUM_BATCHES_PER_UPDATE = 8
AUTORESUME = True
SAVE_FOLDER = '/checkpoints/grpo_single_controller'
NUM_TRAIN_ITERATIONS = 10
DO_SAMPLE = True

_MAX_SEQ_LEN = 10240
_MAX_GEN_LEN = 8192


@contextmanager
def time_it(name: str):
    start_time = time.time()
    print(f"[{name}] started at {time.strftime('%X')}")
    yield
    end_time = time.time()
    print(f"[{name}] finished at {time.strftime('%X')}")
    print(f"[{name}] took {end_time - start_time:.2f} seconds")



class DistributedGPUActor(BaseDistributedGPUActor):
    """Distributed GPU actor for testing."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
    ):
        super().__init__(rank, world_size, master_addr, master_port)
        
        # Configure Ray actor logging - this will go to Ray logs
        self.logger = logging.getLogger(f"Actor-{rank}")
        self.logger.setLevel(logging.INFO)
        
        # Create console handler that will be captured by Ray
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f'[ACTOR-{rank}] %(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.model = None
        self.model_update_group = None
        self.ref_path = None
        self._dataloader = None
        self._tokenizer = None
        self.ppo_callback = None
        self.ppo_trainer: Trainer = None  # type: ignore

        self.pretrain_model_name = None
        self.device_train_batch_size = None
        self.num_batches_per_update = None
        self.max_seq_len = None
        self.precision = None  # type: ignore
        self.train_config: dict = None  # type: ignore
        self.model_config = None
        self.global_train_batch_size = None
        self.max_gen_len = None

    def build_train_config(self, pretrain_model_name: str):
        self.logger.info(f"Starting build_train_config with model: {pretrain_model_name}")
        self.pretrain_model_name = pretrain_model_name

        self.model_config = {
            'tokenizer': self.tokenizer,
            'pretrained_model_name_or_path': self.pretrain_model_name,
            'pretrained': True,
            'use_flash_attention_2': True,
            'allow_embedding_resizing': True,
            'name': 'hf_critic_free_lm',
            # 'init_device': 'mixed',
            # This throws: [rank0]: ValueError: Detected mixed initialization where some ranks have model on cpu or gpu and some ranks are on meta. Either keep all ranks on the same device or set parallelism_config['fsdp']['sync_module_states'] = True. Otherwise, some weights may be randomly initialized when loading a checkpoint.
            'loss_type': 'grpo',
            'target_kl': 0.1,
            'kl_estimator': 'k3',
            'kl_clip_range': 40,
            'use_auth_token': True,
            'compute_kl_loss': False,
            'policy_clip_ratio': 0.2,
            'normalize_advantage': True,
            'length_normalize_policy_loss': True,
            'attn_implementation': 'flash_attention_2'
        }
        self.global_train_batch_size = GLOBAL_TRAIN_BATCH_SIZE
        self.device_train_batch_size = self.global_train_batch_size // self.world_size
        self.num_batches_per_update = NUM_BATCHES_PER_UPDATE
        self.max_seq_len = _MAX_SEQ_LEN
        self.max_gen_len = _MAX_GEN_LEN
        self.precision = 'amp_bf16'

        ref_model_config = {
            'name': 'hf_causal_lm',
            'pretrained': self.model_config['pretrained'],
            'pretrained_model_name_or_path': self.pretrain_model_name,
            'use_auth_token': self.model_config['use_auth_token'],
            'use_flash_attention_2': self.model_config['use_flash_attention_2'], 
        }

        variables = {
            'gamma': 1,
            'lambda_gae': 1,
            'epoch_per_iteration': 1,
            'num_batches_per_update': self.num_batches_per_update,
            'generations_per_prompt': GENERATIONS_PER_PROMPT,
            'device_generate_batch_size': 1,
            'vllm_enable_prefix_caching': True,
            'generation_kwargs': {
                'top_p': 1.0,
                'use_cache': True,
                'do_sample': DO_SAMPLE,
                'temperature': 1.0,
            },
            'eos_token_ids': [
                128001,
                128008,
                128009,
            ],
            'buffer': {
                'name': 'MinibatchRolloutBuffer',
                'max_buffer_size': self.num_batches_per_update,
            },
            'max_gen_len': self.max_gen_len,
            'kl_controller': {
                'init_kl_coef': 0.0, # no KL penalty
                'kl_ctl_type': 'fixed',
            },
            'reference_model': {
                'model_config': ref_model_config,
                'precision': self.precision,
                'load_path': self.ref_path,
            },
            'non_train_fsdp_config': self.fsdp_config,
            'rewards': {
                'math_verifier': {
                    'reward_type': 'math_verifier',
                    'reward': 4,
                },
                'bad_generation_end': {
                    'reward': -1,
                    'eos_penalty': True,
                    'reward_type': 'bad_generation_end'
                },
                'math_format_verifier': {
                    'reward': 1,
                    'reward_type': 'math_format_verifier'
                },
                'penalize_extra_short_responses': {
                    'reward': -1,
                    'reward_type': 'short_response_reward',
                    'len_threshold': 10
                },
            }
        }
        algorithm_config = {
            'gradient_clipping': {
                'clipping_type': 'norm',
                'clipping_threshold': 0.001
            }
        }
        self.train_config = {
            'seed': 17,
            'model': self.model_config,
            'fsdp_config': self.fsdp_config,
            'precision': self.precision,
            'variables': variables,
            'algorithms': algorithm_config,
            'global_train_batch_size': self.device_train_batch_size * self.world_size,
            'device_train_batch_size': self.device_train_batch_size,
            'device_train_microbatch_size': self.device_train_batch_size,
            'save_folder': SAVE_FOLDER,
            'log_config': True,
            'max_seq_len': self.max_seq_len,
            'python_log_level': 'debug',
            'console_log_interval': '1ba',
            'eval_interval': '1iter',
        }
        self.logger.info("Finished build_train_config")

    def build_tokenizer(self):
        # TODO (algo): decide if we should use tokens or messages given
        # we may need token level log prob
        # TODO (infra): use the tokenizer/texts for prompt dataloader but
        # token (ids) for the experience buffer/manager
        kwargs = {
            'padding': 'longest',
            'pad_token': '<|finetune_right_pad_id|>',
            'truncation': True,
            'padding_side': 'left',
            'model_max_length': self.max_seq_len,
            'trust_remote_code': True,
        }
        tokenizer = AutoTokenizer.from_pretrained(self.pretrain_model_name, **kwargs)
        return tokenizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.build_tokenizer()
        return self._tokenizer

    @property
    def fsdp_config(self):
        # TODO (infra): use actual fsdp1 config
        return {}

    def init_composer_dist(self):
        print('Initializing composer dist', composer_dist.get_local_rank(), composer_dist.get_global_rank(), composer_dist.get_world_size())
        composer_dist.initialize_dist('gpu')

    def build_orl_eval_callback(self):
        from llmfoundry.utils.builders import build_callback

        self.logger.info("Building ORL eval callback")
        kwargs = {
            'evals': [
                {
                    'name': 'gsm8k',
                },
                {
                    'name': 'math_500',
                },
            ],
            'eval_overrides': {
                'generation_params': {
                    'max_tokens': _MAX_GEN_LEN
                }
            },
        }
        return build_callback(
            name='orl_eval',
            kwargs=kwargs,
            train_config=self.train_config,
        )

    def build_ppo_trainer(self):
        name = self.model_config.pop('name')
        
        self.logger.info(f"Model type: {name}")
        if name == 'hf_ppo_lm':
            self.logger.info("Creating ComposerHFPolicyLM")
            model = ComposerHFPolicyLM(**self.model_config)
        elif name == 'hf_critic_free_lm':
            self.logger.info("Creating ComposerHFCriticFreePolicyLM")
            model = ComposerHFCriticFreePolicyLM(**self.model_config)
        self.logger.info("Model created successfully")

        optimizer = DecoupledAdamW(model.parameters(), lr=1e-6)

        # TODO (infra): pull the rest of the training logic from the callback
        # to this class, e.g, how to interact with env, calculate rewards etc
        # NOTE: SingleControllerOnPolicyCallback is currently over-writing the iteration_start method
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
            experiment_name='test_single_controller_ppo',
            run_name='test_single_controller_ppo',
            tracking_uri='databricks',
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

        # Try to add the ORL eval callback if the required dependencies are installed
        try:
            orl_eval_callback = self.build_orl_eval_callback()
            callbacks.append(orl_eval_callback)
        except Exception as e:
            self.logger.warning(f"Failed to build ORL eval callback: {e}")
            self.train_config.pop('eval_interval', None)

        self.ppo_trainer = Trainer(
            model=model,
            optimizers=optimizer,
            callbacks=callbacks,
            train_dataloader=dummy_dataloader,
            precision=self.precision,
            parallelism_config={'fsdp': self.fsdp_config},
            max_duration=f'{NUM_TRAIN_ITERATIONS}iter',
            loggers=[mlflow_logger],
            device_train_microbatch_size=1,
            load_path=self.ref_path,
            save_folder=SAVE_FOLDER,
            save_interval='1iter',
            autoresume=AUTORESUME,
        )

    def close_trainer(self):
        self.ppo_trainer.close()
    
    def attach_vllm_engines(self, vllm_engines: list[Any]):
        self.logger.info(f"Attaching {len(vllm_engines)} vLLM engines to the Training Actors")
        self.ppo_trainer.state.vllm_engines = vllm_engines

    def add_rollouts(self, current_rank_rollouts: dict[str, Any]):
        """Adds the current rank's rollouts to the callback."""
        for k, v in current_rank_rollouts.items():
            assert isinstance(v, torch.Tensor) or isinstance(v, list), f"Expected a tensor or list, got {type(v)}"
            if isinstance(v, torch.Tensor):
                current_rank_rollouts[k] = v.to(torch.device('cuda'))
        self.ppo_callback.batch_rollouts = current_rank_rollouts

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
        self.logger.info(f"#### Finished training 1 iter with loss: {self.ppo_trainer.state.loss}")


def setup_process_groups(
    master_actor: Any,
    vllm_engines: list[Any],
    vllm_tensor_parallel_size: int,
):
    """Initialize process groups for vLLM engines and master actor."""
    # Get a new port for the weight-update process group
    master_addr, _ = ray.get(
        master_actor.get_master_address.remote(),
    )  # type: ignore
    new_port = ray.get(master_actor.get_free_port.remote())  # type: ignore
    print(f'new_port: {new_port}')

    world_size = dist.get_world_size()

    # Initialize process groups for vLLM engines
    refs = [
        engine.init_process_group.remote(
            master_addr,
            new_port,
            i * vllm_tensor_parallel_size + 1,
            world_size // 2 + 1,
            'weight-update',
            backend='nccl',
        ) for i, engine in enumerate(vllm_engines)
    ]

    # Add master actor to the process group
    refs.append(
        master_actor.add_process_group.remote(
            backend='nccl',
            master_addr=master_addr,
            master_port=new_port,
            world_size=world_size // 2 + 1,
            rank=0,
            group_name='weight-update',
        ),
    )

    # Wait for all process groups to be initialized
    print(ray.get(refs))


class TrainActorGroup(SPMDActorGroup):
    """Group of training actors for PPO."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def build_models(self, pretrain_model_name: str):
        """Build reference models and PPO trainers for all actors."""
        self.collective_methods.build_train_config(pretrain_model_name)
        self.collective_methods.init_composer_dist()

        # Build PPO trainers
        self.collective_methods.build_ppo_trainer()
        print('build ppo trainer done')

    def _partition_rollouts_across_ranks(self, rollouts: dict[str, Any]) -> list[dict[str, Any]]:
        """Partition the rollouts across all actors."""
        partitioned_rollouts = []
        per_rank_data_size = rollouts['prompt'].shape[0] // self.num_train_actors
        for i in range(self.num_train_actors):
            current_rank_start = i * per_rank_data_size
            current_rank_end = (i + 1) * per_rank_data_size
            current_rank_rollouts = {}
            for k, v in rollouts.items():
                assert isinstance(v, torch.Tensor) or isinstance(v, list), f"Expected a tensor or list, got {type(v)}"
                current_rank_rollouts[k] = v[current_rank_start:current_rank_end]
            partitioned_rollouts.append(current_rank_rollouts)
        return partitioned_rollouts

    def add_latest_rollouts_from_buffer(self, experience_buffer: "ExperienceBuffer"):
        assert experience_buffer is not None, "Experience buffer is not set"
        assert len(experience_buffer) > 0, "Experience buffer is empty"
        latest_rollouts = experience_buffer.popleft()
        partitioned_rollouts = self._partition_rollouts_across_ranks(latest_rollouts)
        assert len(partitioned_rollouts) == self.num_train_actors, "Number of partitioned rollouts should be equal to the number of train actors"
        ray.get([train_actor.add_rollouts.remote(partition) for train_actor, partition in zip(self.train_actors, partitioned_rollouts)])
    
    def train_1_iter(self):
        # added this method to time the collectivetraining time otherwise we can time each rank but the print/logging becomes messy to read
        with time_it("training"):
            self.collective_methods.train_1_iter()


class InferenceServer:
    """Inference server with vLLM engines."""

    def __init__(self, num_vllm_engines: int, vllm_tensor_parallel_size: int, pretrain_model_name: str):
        self.num_vllm_engines = num_vllm_engines
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.vllm_engines = create_vllm_engines(
                num_engines=num_vllm_engines,
                tensor_parallel_size=vllm_tensor_parallel_size,
                enforce_eager=True,
                pretrain=pretrain_model_name,
                revision=None,
                seed=1,
                enable_prefix_caching=False,
                max_model_len=_MAX_GEN_LEN,
                device_bundle={
                    'GPU': 1,
                    'CPU': 1,
                    'worker_node': 0,
                },
            )

    @property
    def engines(self):
        return self.vllm_engines


class RolloutAgent:
    """Rollout agent for generating sequences from the inference server."""

    def __init__(
        self,
        inference_server: InferenceServer,
        streaming_dataset_actor: "StreamingDatasetActor",
    ):
        self.inference_server = inference_server
        self.streaming_dataset_actor = streaming_dataset_actor
        self.generation_kwargs = {
            'top_p': 1.0,
            'use_cache': True,
            'do_sample': DO_SAMPLE,
            'temperature': 1.0,
        }
        self.precision = 'amp_bf16'
        self.tokenizer_pad_token_id = ray.get(self.streaming_dataset_actor.get_tokenizer_pad_token_id.remote())
        self.prompt_handler_config = ray.get(self.streaming_dataset_actor.get_prompt_handler_config.remote())
        self.max_gen_len = self.prompt_handler_config['max_gen_len']

        # Load iter_num from the checkpoint
        self.save_folder = os.path.join(SAVE_FOLDER, 'RolloutAgent')

        self.iter_num = 0

        # Load the latest checkpoint
        self.latest_checkpoint = os.path.join(self.save_folder, 'latest.symlink')

        if AUTORESUME and os.path.exists(self.latest_checkpoint):
            print(f'Autoresuming from checkpoint for RolloutAgent.')
            with open(self.latest_checkpoint, 'rb') as f:
                checkpoint = pickle.load(f)
            self.iter_num = checkpoint['iter_num']
            print(f'Loading streaming dataloader state dict for RolloutAgent.', checkpoint['streaming_dataloader'])
            self.streaming_dataset_actor.load_dataloader_state_dict.remote(checkpoint['streaming_dataloader'])


    def get_next_iter_rollouts(self):
        """
        Gets the next rollouts from the inference server.

        Since all ranks should see different data, we need to get the rollouts for each rank.
        """
        iter_data = ray.get(self.streaming_dataset_actor.get_next_iter_prompts.remote())
        all_prompts = iter_data['prompt']
        # TODO: Since this functionality is (somewhat) shared across the OnPolicyCallback and the RolloutAgent,
        # we should move this to the separate util file.
        with get_precision_context(self.precision), torch.no_grad(), time_it("batch_inference"):
            sequences = _vllm_generate(
                vllm_engines=self.inference_server.engines,
                max_gen_len=self.max_gen_len,
                generation_kwargs=self.generation_kwargs,
                pad_token_id=self.tokenizer_pad_token_id,
                all_prompts=all_prompts,
                batch_sizes=[len(all_prompts)],
            )

        sequences = sequences[0]
        max_vllm_generated_len = max([len(response) for response in sequences])
        padded_responses = []
        for sequence in sequences:
            sequence = list(sequence)
            if len(sequence) < max_vllm_generated_len:
                sequence = sequence + [self.tokenizer_pad_token_id] * (max_vllm_generated_len - len(sequence))
            padded_responses.append(sequence)

        padded_responses = torch.tensor(
            padded_responses,
            dtype=all_prompts.dtype,
            device=torch.device('cpu'),
        )

        processed_sequences = torch.cat([all_prompts, padded_responses], dim=-1)
        iter_data['sequences'] = processed_sequences

        save_folder_iter = os.path.join(self.save_folder, f'iter_{self.iter_num}')
        checkpoint_path = os.path.join(save_folder_iter, 'checkpoint.pt')
        self.iter_num += 1

        streaming_dataloader_state_dict = ray.get(self.streaming_dataset_actor.get_dataloader_state_dict.remote())
        print(f'Streaming dataloader state dict for RolloutAgent.', streaming_dataloader_state_dict)

        # make sure that the folder path can exist
        os.makedirs(save_folder_iter, exist_ok=True)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'iter_data': iter_data,
                'iter_num': self.iter_num,
                'streaming_dataloader': streaming_dataloader_state_dict,
            }, f)

        if os.path.exists(self.latest_checkpoint):
            os.remove(self.latest_checkpoint)
        os.symlink(checkpoint_path, self.latest_checkpoint)
        return iter_data


class ParameterBuffer(Buffer):
    """Buffer for updating the inference model."""

    def update_inference_model(self, actor: DistributedGPUActor, inference_server: InferenceServer):
        start_time = time.time()
        print('Before broadcast to vLLM')
        # TODO (infra) instead of direcly broadcasting to vllm, we should
        # push the model parameters to a parameter buffer manager and have
        # the buffer manager initiate broadcast of parameters to vllm engines
        broadcast_to_vllm(
            actor.ppo_callback.actor_critic,
            inference_server.engines,
            actor.model_update_group,
            device=torch.device('cuda'),
            loss_type=actor.ppo_callback.actor_critic.loss_type,  # type: ignore
        )
        print('Finished broadcasting to vLLM')
        print(f'Took: {time.time() - start_time} to broadcast to vllm.')
        dist.barrier()

    def put(self, struct: dict[str, Any]):
        # prefers to implement the model update logic in the Buffer class as the buffer is a bridge between the trainer actor and the inference server
        # and knows the best way to transfer the model parameters. Trainer just needs to put necessary struct to this api
        struct['actor_group'].collective_methods.execute(partial(self.update_inference_model, inference_server=struct['inference_server']))


# TODO: Move this experience buffer earlier so that we can avoid
# using "ExperienceBuffer" (with quotes) as a type hint.
class ExperienceBuffer(Buffer):
    """Buffer for storing experiences."""

    def put(self, struct: dict[str, Any]):
        self.buffer.append(struct)

    def get(self, struct: Optional[dict[str, Any]] = None):
        return self.buffer[0]

    def popleft(self, struct: Optional[dict[str, Any]] = None):
        return self.buffer.pop(0)

    def __len__(self):
        return len(self.buffer)


class StreamingDatasetActor(BaseDistributedGPUActor):
    """Streaming actor for loading prompts onto the experience buffer."""

    def __init__(self):
        # Setting up the distributed environment (WORLD_SIZE = 1)
        super().__init__(
            rank=0,
            world_size=1,
            master_addr=None,
            master_port=None,
        )

        # Setting up all of the configs
        # TODO: We should move these to dataclasses
        # TODO: In a future PR, create all configs in the main function and populate
        # the correct configs across all entities (e.g. DistributedGPUActor, StreamingDatasetActor, etc)
        self.pretrain_model_name = 'meta-llama/Llama-3.1-8B-Instruct'
        self.prompt_handler_config = {
            "global_train_batch_size": GLOBAL_TRAIN_BATCH_SIZE,
            "generations_per_prompt": GENERATIONS_PER_PROMPT,
            "num_batches_per_update": NUM_BATCHES_PER_UPDATE,
            "max_seq_len": _MAX_SEQ_LEN,
            "max_gen_len": _MAX_GEN_LEN,
        }
        self.tokenizer_config = {
            'padding': 'longest',
            'pad_token': '<|finetune_right_pad_id|>',
            'truncation': True,
            'padding_side': 'left',
            'model_max_length': self.prompt_handler_config['max_seq_len'],
            'trust_remote_code': True,
        }
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dataset_dir = f"/tmp/dataset/prompt_{timestamp}/"
        self.dataloader_config = {
            'name': 'prompt',
            'dataset': {
                'local': temp_dataset_dir,
                'split': 'train',
                'remote': 'dbfs:/Volumes/datasets/ashutoshbaheti/orl_data/math_lighteval/llama3_8b_math_prompts/',
                'shuffle': True,
                'max_gen_len': self.prompt_handler_config['max_gen_len'],
                'max_seq_len': self.prompt_handler_config['max_seq_len'],
                'shuffle_seed': 17,
                'download_timeout': 1800
            },
            'drop_last': True,
            'num_workers': 1,
        }

        # Key variables
        global_train_batch_size = self.prompt_handler_config['global_train_batch_size']
        self.generations_per_prompt = self.prompt_handler_config['generations_per_prompt']
        num_batches_per_update = self.prompt_handler_config['num_batches_per_update']
        total_num_generations = global_train_batch_size * num_batches_per_update
        self.num_prompts_per_iteration = total_num_generations // self.generations_per_prompt

        # Validate that the total number of generations is divisible by the number of generations per prompt
        assert total_num_generations % self.generations_per_prompt == 0, "total_num_generations must be divisible by generations_per_prompt"

        # Creating main entities
        self.tokenizer = self._build_tokenizer()
        self.dataloader = self._build_dataloader()
        self.dataloader_iter = iter(self.dataloader)

    def _build_dataloader(self):
        foundry_dataspec = build_dataloader(
            cfg = self.dataloader_config,
            tokenizer = self.tokenizer,
            device_batch_size = self.num_prompts_per_iteration,
        )
        return foundry_dataspec.dataloader

    def _build_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.pretrain_model_name, **self.tokenizer_config)
        return tokenizer

    def get_prompt_handler_config(self):
        return self.prompt_handler_config

    def get_tokenizer_pad_token_id(self):
        return self.tokenizer.pad_token_id

    def _get_single_iter_prompts(self):
        """Gets a single iteration's prompts from the dataloader."""
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            return next(self.dataloader_iter)

    def get_next_iter_prompts(self):
        """Gets the next iteration's prompts across all ranks and prepares them for the rollout agent."""
        batches = [self._get_single_iter_prompts()]

        return preprocess_batches(batches, self.generations_per_prompt, self.tokenizer.pad_token_id)

    def get_dataloader_state_dict(self):
        return self.dataloader.state_dict()
    
    def load_dataloader_state_dict(self, state_dict: dict):
        self.dataloader.load_state_dict(state_dict)


class PPOController:
    """PPO controller for training the policy and value networks."""

    def __init__(
        self,
        train_actor: TrainActorGroup,
        inference_server: InferenceServer,
        rollout_agent: RolloutAgent,
        parameter_buffer: ParameterBuffer,
        experience_buffer: ExperienceBuffer,
        pretrain_model_name: str,
    ):
        self.train_actor = train_actor
        self.inference_server = inference_server
        self.rollout_agent = rollout_agent
        self.parameter_buffer = parameter_buffer
        self.experience_buffer = experience_buffer
        self.train_actor.build_models(pretrain_model_name)
        setup_process_groups(
            self.train_actor.master_actor,
            inference_server.engines,
            inference_server.vllm_tensor_parallel_size,
        )
        self.train_actor.collective_methods.attach_vllm_engines(self.inference_server.engines)

    def train(self):
        for _ in range(NUM_TRAIN_ITERATIONS):  # Example: train for 5 iterations
            # NOTE: this loop is represents the logic happening in the current `iteration_start` of the OnPolicyCallback
            self.parameter_buffer.put({'actor_group': self.train_actor, 'inference_server': self.inference_server})
            # Simple example of adding elements to the experience buffer
            self.experience_buffer.put(self.rollout_agent.get_next_iter_rollouts())
            # Populate the train actor group with the rollouts and then train
            self.train_actor.add_latest_rollouts_from_buffer(self.experience_buffer)
            self.train_actor.train_1_iter()
        
        self.train_actor.collective_methods.close_trainer()



def _run_single_controller_ppo(
    config: Any,
):
    """Shared function for running single controller PPO.

    Args:
        config: OmegaConf configuration object containing all parameters
    """
    # Set vLLM attention backend to FLASH_ATTN otherwise FlashInfer backend
    # takes too long to jit compile
    os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'

    # Disable setting CUDA_VISIBLE_DEVICES by ray, we will set it manually
    os.environ['RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES'] = '1'

    with start_ray_server() as _address:
        # only rank 0 is the master controller
        if dist.get_rank() == 0:
            world_size = getattr(config, "world_size", 0)
            if world_size == 0:
                world_size = dist.get_world_size()

            # Create buffers for the parameter and experience buffers
            # first since they don't have external dependencies
            parameter_buffer = ParameterBuffer()
            experience_buffer = ExperienceBuffer()

            # create SPMD training actors of the system
            num_train_actors = world_size // 2
            train_actor = TrainActorGroup(num_train_actors, DistributedGPUActor)

            # Create vLLM engines (or inference actors)
            vllm_tensor_parallel_size = world_size - num_train_actors
            num_vllm_engines = (
                world_size - num_train_actors
            ) // vllm_tensor_parallel_size
            # TODO: Encapsulate this into a inference server manager class
            pretrain_model_name = config.pretrain_model_name
            inference_server = InferenceServer(
                num_vllm_engines=num_vllm_engines,
                vllm_tensor_parallel_size=vllm_tensor_parallel_size,
                pretrain_model_name=pretrain_model_name,
            )

            # We are using a CPU worker for the StreamingActor
            # and this involves a super hacky workaround by
            # uninstalling megablocks if it exists. Better solutions
            # would include:
            # 1) decouple StreamingActor from llm-foundry altogether
            # 2) don't broadly import llm-foundry in compose-rl (only
            # import it into codepaths/files that will only be used by
            # GPUActors as opposed to CPUActors)
            # 3) Setting up ray actors with correct environments (which
            # would involve creating a BaseDistributedActor instead of a
            # BaseDistributedGPUActor so that we can use CPUs)
            # We uninstall megablocks after the Train Actors have been
            # created so that those actors still have megablocks functionality.
            uninstall_megablocks_if_exists()
            streaming_dataset_actor = ray.remote(num_gpus=0)(StreamingDatasetActor).remote()
            rollout_agent = RolloutAgent(inference_server, streaming_dataset_actor)

            ppo_controller = PPOController(
                train_actor,
                inference_server,
                rollout_agent,
                parameter_buffer,
                experience_buffer,
                pretrain_model_name,
            )
            ppo_controller.train()


if __name__ == '__main__':
    # # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run single controller PPO with configuration file')
    parser.add_argument('--file_path', type=str, required=False, default=None,
                       help='Path to the OmegaConf YAML configuration file')
    args = parser.parse_args()
    
    # Load configuration using OmegaConf
    if args.file_path is not None:
        config = om.load(args.file_path)
    else:
        config = om.create({
            'pretrain_model_name': 'meta-llama/Llama-3.2-1B-Instruct',
        })
    
    # This is an example of how to move the controller logic from PPO Callback
    # to a separate trainer actor above and this main single controller
    # function.
    _run_single_controller_ppo(config)

