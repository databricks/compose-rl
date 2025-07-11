import ray
import torch
import torch.distributed as dist
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from typing import Optional, Tuple
import argparse
from datetime import timedelta

from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer

from composer.utils import dist as composer_dist
from composer import Trainer
from composer.optim import DecoupledAdamW
from llmfoundry.models import ComposerHFCausalLM, ComposerMPTCausalLM
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.models.gpt2 import GPT2LMHeadModel

from compose_rl.algorithms.online import (
    ComposerHFPolicyLM,
    ComposerMPTPolicyLM,
    SingleControllerOnPolicyCallback,
)
from compose_rl.algorithms.online.model_methods import OnPolicyEnum
from compose_rl.algorithms.online.modeling_hf import ComposerHFPolicy
from compose_rl.data import prompt_dataset_collate_fn
from tests.common import PromptDataset, VerifiablePromptDataset, world_size
from tests.fixtures.fixtures import assets_tokenizer_helper

from compose_rl.algorithms.online.generation_utils import init_process_group, create_vllm_engines

from typing import Any


def ray_noset_visible_devices():
    return os.environ.get('RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES', '0') == '1'


def init_ray():
    # init ray on master node, rank 0
    if dist.get_rank() == 0:
        # Start head node
        subprocess.run(['ray', 'start', '--head'], check=True)
        ray.init('auto')
        # get existing ray ip and port 
        ctx = ray.get_runtime_context()
        address = ctx.gcs_address
        print(f'available gpus: {ray.available_resources()}')
    else:
        address = ''
    address_list = [address]
    # broadcast address to all other ranks
    dist.broadcast_object_list(address_list, src=0)
    if dist.get_rank() != 0 and os.environ.get('LOCAL_RANK', None) == '0':
        address = address_list[0]
        print(f'rank: {dist.get_rank()} connecting to address: {address}')
        subprocess.run(['ray', 'start', f'--address={address}'], check=True)
    dist.barrier()
    if dist.get_rank() == 0:
        # wait until num of gpus reach world_size
        num_gpus = int(ray.cluster_resources()['GPU'])
        counter = 0
        while num_gpus < dist.get_world_size():
            print(f'waiting for {dist.get_world_size() - num_gpus} gpus to be available')
            num_gpus = int(ray.cluster_resources()['GPU'])
            time.sleep(5)
            counter += 1
            if counter > 4:
                raise RuntimeError(f'Failed to start {dist.get_world_size()} gpus')
        print(f'Total available gpus: {ray.available_resources()}')
    return address


@ray.remote(num_gpus=1)
class DistributedGPUActor:
    def __init__(self, rank: int, world_size: int, master_addr: Optional[str] = None, master_port: Optional[int] = None):
        """Initialize the distributed GPU actor.
        
        Args:
            rank: The rank of this process in the distributed group
            world_size: Total number of processes in the distributed group
            master_addr: Master node address. If None, will allocate dynamically for rank 0
            master_port: Master node port. If None, will allocate dynamically for rank 0
        """
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        
        # Set up basic environment variables
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        
        # Set LOCAL_RANK based on Ray GPU allocation
        os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0]) if ray_noset_visible_devices() else "0"
        
        # If this is rank 0 and no master_addr/master_port provided, allocate them
        if rank == 0 and (master_addr is None or master_port is None):
            self._allocate_master_address()

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)

        self.model = None
        self.model_update_group = None

        self.pretrain_model_name = None
        self.ref_path = None
        self._dataloader = None
        self._tokenizer = None
        self.ppo_callback = None
        self.ppo_trainer: Trainer = None

    def build_dataloader(self):
        max_seq_len = 32
        prompt_len = 10

        dataset = VerifiablePromptDataset(prompt_len=prompt_len)
        dataloader = DataLoader(
            dataset,
            collate_fn=partial(
                prompt_dataset_collate_fn,
                self.tokenizer,
                max_seq_len,
            ),
            sampler=composer_dist.get_sampler(dataset),
            batch_size=4,
        )
        # We need to mock this method, since our dataset isn't a StreamingDataset
        dataloader.state_dict = lambda: {}
        dataloader.load_state_dict = lambda x: None
        return dataloader

    @property
    def dataloader(self):
        if self._dataloader is None:
            self._dataloader = self.build_dataloader()
        return self._dataloader

    def build_tokenizer(self):
        tokenizer = assets_tokenizer_helper(self.pretrain_model_name)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.build_tokenizer()
        return self._tokenizer

    @property
    def model_config(self):
        return {
            'tokenizer': self.tokenizer,
            'pretrained_model_name_or_path': self.pretrain_model_name,
            'pretrained': True,
            'use_flash_attention_2': True,
            'allow_embedding_resizing': True,
        }

    @property
    def fsdp_config(self):
        return dict()

    def build_ref_model(self):
        composer_dist.initialize_dist('gpu')

        tmp_model = ComposerHFCausalLM(**self.model_config)

        tmp_optimizer = DecoupledAdamW(tmp_model.parameters(), lr=1e-6)

        tmp_ref_path = str('./ref_checkpoints')

        temp_dataloader = [{
            'input_ids': torch.ones((2, 15)).to(dtype=torch.int64),
            'attention_mask': torch.ones((2, 15)),
            'labels': torch.ones((2, 15)).to(dtype=torch.int64),
        }]

        temp_trainer = Trainer(
            model=tmp_model,
            train_dataloader=temp_dataloader,
            optimizers=tmp_optimizer,
            max_duration='1ba',
            parallelism_config={'fsdp': self.fsdp_config},
            save_folder=tmp_ref_path,
            save_weights_only=True,
            device_train_microbatch_size=2,
        )

        temp_trainer.fit()

        # After making the reference model, we can proceed with the PPO training
        self.ref_path = os.path.join(tmp_ref_path, 'latest-rank0.pt')

    def build_ppo_trainer(self, pretrain_model_name: str):
        self.pretrain_model_name = pretrain_model_name
        composer_dist.initialize_dist('gpu')
        max_seq_len = 32
        precision = 'amp_bf16'

        model = ComposerHFPolicyLM(**self.model_config)

        optimizer = DecoupledAdamW(model.parameters(), lr=1e-8)

        num_batches_per_update = 2

        # ref_model_config = copy.deepcopy(self.model_config)
        ref_model_config = {**self.model_config, 'name': 'hf_causal_lm'}

        variables = {
            'buffer': {
                'name': 'MinibatchRolloutBuffer',
                'max_buffer_size': num_batches_per_update,
            },
            'max_gen_len': 8,
            'gamma': 0.99,
            'lambda_gae': 0.95,
            'generation_kwargs': {
                'use_cache': True,
                'do_sample': False,
            },
            'kl_controller': {
                'init_kl_coef': 0.2,
                'target': 0.01,
                'horizon': 12800,
                'kl_ctl_type': 'adaptive',
            },
            'reference_model': {
                'model_config': ref_model_config,
                'precision': precision,
                'load_path': self.ref_path,
                'non_train_fsdp_config': self.fsdp_config,
            },
            'device_generate_batch_size': 2,
            'epoch_per_iteration': 1,
            'num_batches_per_update': num_batches_per_update,
            'rewards': {
                'output_length': {
                    'reward_type': 'output_length',
                    'max_gen_len': 10,
                },
            },
        }
        train_config = {
            'model': {**self.model_config, 'kl_estimator': 'k1', 'kl_clip_range': 40.0},
            'fsdp_config': self.fsdp_config,
            'seed': 17,
            'precision': precision,
            'variables': variables,
            'max_seq_len': max_seq_len,
            'global_train_batch_size': 2,
            'device_train_batch_size': 2,
            'device_train_microbatch_size': 1,
        }

        tmp_save_path = str('./checkpoints')
        self.ppo_callback = SingleControllerOnPolicyCallback(train_config=train_config)
        self.ppo_trainer = Trainer(
            model=model,
            optimizers=optimizer,
            callbacks=self.ppo_callback,
            train_dataloader=self.dataloader,
            precision=precision,
            parallelism_config={'fsdp': self.fsdp_config},
            max_duration='3iter',
            device_train_microbatch_size=1,
            load_path=self.ref_path,
            # save_folder=tmp_save_path,
            # save_interval='1iter',
        )

        # trainer.fit(duration='1iter')

        # This is the KL assert that must be true if we are truly loading from the same model.
        # This is only true on the first iteration
        # assert torch.allclose(
        #     trainer.state.loss['kl/ift_kl'], # pyright: ignore
        #     torch.tensor(0.0),
        #     atol=5e-5,
        # )

    def train_1_iter(self):
        self.ppo_trainer.fit(duration='1iter')

    def sync_weight_and_gen(self, vllm_engines: list[Any]):
        self.ppo_callback.round_trip_to_inference_engines(
            device=self.ppo_trainer.state.device,
            vllm_engines=vllm_engines,
            model_update_group=self.model_update_group,
        )


    def get_node_ip(self):
        return ray.util.get_node_ip_address().strip('[]')
    
    def get_free_port(self):
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]
    
    def _allocate_master_address(self):
        """Allocate master address and port for rank 0."""
        if self.master_addr is None:
            # Get the local IP address
            self.master_addr = self.get_node_ip()

        if self.master_port is None:
            # Allocate a free port
            self.master_port = self.get_free_port()
    
    def get_master_address(self) -> Tuple[Optional[str], Optional[int]]:
        """Return the master address and port as a tuple."""
        return (self.master_addr, self.master_port)
    
    def init_default_process_group(self):
        """Initialize the distributed process group."""         
        # Initialize process group
        dist.init_process_group(timeout=timedelta(seconds=30))
        print(f'is distributed initialized: {dist.is_initialized()}')
        # Print debug information
        num_visible_devices = torch.cuda.device_count()
        print(f'num_visible_devices: {num_visible_devices}')
        print('Ray actor init envs:')
        print(f'rank: {dist.get_rank()}')
        print(f'node_rank: {dist.get_rank() // 8}')
        print(f'world_size: {dist.get_world_size()}')
        print(f'local_rank: {dist.get_rank() % 8}')
        print(f'master_addr: {self.master_addr}')
        print(f'master_port: {self.master_port}')
    
    def init_model(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto')
        self.model.to('cuda')

    def sync_weights(self, vllm_engines: list[Any]):
        for name, p in self.model.named_parameters():
            refs = [engine.update_weight.remote(name, p.dtype, p.shape, empty_cache=False) for engine in vllm_engines]
            dist.broadcast(p, src=0, group=self.model_update_group)
            ray.get(refs)

    def tensor_all_reduce(self) -> float:
        """Perform a simple tensor all_reduce operation."""
        # Create a tensor on the GPU and perform all_reduce
        device = torch.device("cuda")
        x = torch.ones(1, device=device)
        dist.all_reduce(x)
        
        return x.item()

    def init_vllm_process_group(self, backend: str, master_addr: str, master_port: int, world_size: int, rank: int, group_name: str):
        """Initialize the vLLM process group."""
        self.model_update_group = init_process_group(backend=backend, init_method=f'tcp://{master_addr}:{master_port}', world_size=world_size, rank=rank, group_name=group_name)
        return dist.get_world_size(self.model_update_group)

@contextmanager
def start_ray_server():
    dist.init_process_group(backend='gloo')
    address = init_ray()
    try:
        yield address
        dist.barrier()
    finally:
        if dist.get_rank() == 0:
            ray.shutdown()
            subprocess.run(['ray', 'stop'], check=True)
        dist.barrier()
        dist.destroy_process_group()


def run(tp_size: int = 8):
    prompts = [
        "what is RAY?",
        "what is vLLM?",
    ]
    pretrain_model_name = 'gpt2'
    with start_ray_server() as address:
        if dist.get_rank() == 0:
            master_addr, _ = address.split(':')
            
            print(f"\n=== STARTING DISTRIBUTED TRAINING WITH RAY ACTORS ===")
            num_train_actors = dist.get_world_size() // 2
            # Create actors - rank 0 will allocate master address/port
            train_actors = []

            # master actor will allocate master_addr and master_port
            master_actor = DistributedGPUActor.remote(0, num_train_actors)
            train_actors.append(master_actor)
            
            # Get master address from rank 0 actor
            master_info = ray.get(master_actor.get_master_address.remote())
            master_addr, master_port = master_info
            print(f"Master address allocated: {master_addr}:{master_port}")
            
            # Create remaining actors with the master address/port
            for i in range(1, num_train_actors):
                actor = DistributedGPUActor.remote(i, num_train_actors, master_addr, master_port)
                train_actors.append(actor)
            
            # # Initialize process groups for all actors
            # init_tasks = [actor.init_default_process_group.remote() for actor in train_actors]
            # ray.get(init_tasks)
            
            # # Perform tensor all_reduce on all actors
            # reduce_tasks = [actor.tensor_all_reduce.remote() for actor in train_actors]
            # results = ray.get(reduce_tasks)
            # print(f"All-reduce results: {results}")

            # build_ref_model_tasks = [actor.build_ref_model.remote() for actor in train_actors]
            # ray.get(build_ref_model_tasks)
            # print('build ref model done')

            build_ppo_trainer_tasks = [actor.build_ppo_trainer.remote(pretrain_model_name) for actor in train_actors]
            ray.get(build_ppo_trainer_tasks)
            print('build ppo trainer done')

            vllm_tensor_parallel_size = min(tp_size, dist.get_world_size() - num_train_actors)
            num_vllm_engines = dist.get_world_size() // 2 // vllm_tensor_parallel_size
            print(f'num_vllm_engines: {num_vllm_engines}')
            vllm_engines = create_vllm_engines(
                num_engines=num_vllm_engines,
                tensor_parallel_size=vllm_tensor_parallel_size,
                enforce_eager=True,
                pretrain=pretrain_model_name,
                revision=None,
                seed=1,
                enable_prefix_caching=False,
                max_model_len=512,
            )

            new_port = ray.get(master_actor.get_free_port.remote())
            print(f'new_port: {new_port}')
            refs = [
                engine.init_process_group.remote(
                    master_addr,
                    new_port,
                    i * vllm_tensor_parallel_size + 1,
                    dist.get_world_size() // 2 + 1,
                    'weight-update',
                    backend='nccl',
                ) for i, engine in enumerate(vllm_engines)
            ]
            refs.append(master_actor.init_vllm_process_group.remote(
                backend='nccl',
                master_addr=master_addr,
                master_port=new_port,
                world_size=dist.get_world_size() // 2 + 1,
                rank=0,
                group_name='weight-update',
            ))
            # should only get refs of both master and vllm_engines together, otherwise it will hang
            print(ray.get(refs))

            refs = [actor.sync_weight_and_gen.remote(vllm_engines) for actor in train_actors]
            ray.get(refs)
            print('sync weight and gen done')

            refs = [actor.train_1_iter.remote() for actor in train_actors]
            ray.get(refs)
            print('train 1 iter done')
            # refs = [actor.init_model.remote(pretrain_model_name) for actor in train_actors]
            # ray.get(refs)
            # print('init model done')

            # ray.get(master_actor.sync_weights.remote(vllm_engines))
            # print('sync weights done')

            # ref = vllm_engines[0].generate.remote(prompts)
            # gen_results = ray.get(ref)
            # for output in gen_results:
            #     prompt = output.prompt
            #     generated_text = output.outputs[0].text
            #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp_size', type=int, default=8)
    args = parser.parse_args()
    run(tp_size=args.tp_size)
