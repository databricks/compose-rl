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
    OnPolicyCallback,
)
from compose_rl.algorithms.online.model_methods import OnPolicyEnum
from compose_rl.algorithms.online.modeling_hf import ComposerHFPolicy
from compose_rl.data import prompt_dataset_collate_fn
from tests.common import PromptDataset, VerifiablePromptDataset, world_size

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

    def build_ref_model(self):
        max_seq_len = 32
        prompt_len = 10

        model_name = 'gpt2'
        tiny_gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')

        dataset = PromptDataset(prompt_len=prompt_len)
        dataloader = DataLoader(
            dataset,
            collate_fn=partial(
                prompt_dataset_collate_fn,
                tiny_gpt2_tokenizer,
                max_seq_len,
            ),
            sampler=composer_dist.get_sampler(dataset),
            batch_size=4,
        )

        # We need to mock this method, since our dataset isn't a StreamingDataset
        dataloader.state_dict = lambda: {}
        dataloader.load_state_dict = lambda x: None

        model_config = {
            'tokenizer': tiny_gpt2_tokenizer,
            'pretrained_model_name_or_path': model_name,
            'pretrained': True,
            'use_flash_attention_2': True,
            'allow_embedding_resizing': True,
        }
        tmp_model = ComposerHFCausalLM(**model_config)

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
            parallelism_config={'fsdp': {}},
            save_folder=tmp_ref_path,
            save_weights_only=True,
            device_train_microbatch_size=2,
        )

        temp_trainer.fit()

        # After making the reference model, we can proceed with the PPO training
        tmp_ref_path = os.path.join(tmp_ref_path, 'latest-rank0.pt')

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
        dist.init_process_group(timeout=timedelta(seconds=30), backend='nccl')
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
    pretrain_model_name = 'meta-llama/Llama-3.2-1B-Instruct'
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
            
            # Initialize process groups for all actors
            init_tasks = [actor.init_default_process_group.remote() for actor in train_actors]
            ray.get(init_tasks)
            
            # Perform tensor all_reduce on all actors
            reduce_tasks = [actor.tensor_all_reduce.remote() for actor in train_actors]
            results = ray.get(reduce_tasks)
            print(f"All-reduce results: {results}")

            build_ref_model_tasks = [actor.build_ref_model.remote() for actor in train_actors]
            ray.get(build_ref_model_tasks)
            print('build ref model done')

            # vllm_tensor_parallel_size = tp_size
            # num_vllm_engines = dist.get_world_size() // 2 // vllm_tensor_parallel_size
            # print(f'num_vllm_engines: {num_vllm_engines}')
            # vllm_engines = create_vllm_engines(
            #     num_engines=num_vllm_engines,
            #     tensor_parallel_size=vllm_tensor_parallel_size,
            #     enforce_eager=True,
            #     pretrain=pretrain_model_name,
            #     revision=None,
            #     seed=1,
            #     enable_prefix_caching=False,
            #     max_model_len=2048,
            # )

            # new_port = ray.get(master_actor.get_free_port.remote())
            # print(f'new_port: {new_port}')
            # refs = [
            #     engine.init_process_group.remote(
            #         master_addr,
            #         new_port,
            #         i * vllm_tensor_parallel_size + 1,
            #         dist.get_world_size() // 2 + 1,
            #         'weight-update',
            #         backend='nccl',
            #     ) for i, engine in enumerate(vllm_engines)
            # ]
            # refs.append(master_actor.init_vllm_process_group.remote(
            #     backend='nccl',
            #     master_addr=master_addr,
            #     master_port=new_port,
            #     world_size=dist.get_world_size() // 2 + 1,
            #     rank=0,
            #     group_name='weight-update',
            # ))
            # print(ray.get(refs))

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
