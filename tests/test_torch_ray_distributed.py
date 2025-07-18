import ray
import torch
import torch.distributed as dist
import os
import pathlib
import pytest

from typing import Optional, Tuple
import argparse
from datetime import timedelta

from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from tests.common import world_size


from compose_rl.algorithms.online.generation_utils import init_process_group, create_vllm_engines

from compose_rl.utils.ray_utils import is_cuda_visible_devices_set, get_node_ip, get_free_port, start_ray_server


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
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        
        # Set LOCAL_RANK based on Ray GPU allocation
        os.environ['LOCAL_RANK'] = '0' if is_cuda_visible_devices_set() else str(ray.get_gpu_ids()[0])
        
        # If this is rank 0 and no master_addr/master_port provided, allocate them
        if rank == 0 and (master_addr is None or master_port is None):
            self._allocate_master_address()

        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = str(self.master_port)

        self.model = None
        self.model_update_group = None
    
    def _allocate_master_address(self):
        """Allocate master address and port for rank 0."""
        if self.master_addr is None:
            # Get the local IP address
            self.master_addr = get_node_ip()

        if self.master_port is None:
            # Allocate a free port
            self.master_port = get_free_port()
    
    def get_master_address(self) -> Tuple[Optional[str], Optional[int]]:
        """Return the master address and port as a tuple."""
        return (self.master_addr, self.master_port)
    
    def get_free_port(self):
        return get_free_port()

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

    def sync_weights(self, vllm_engines: list):
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


@pytest.mark.gpu
@world_size(4)
def test_distributed_ray_actors(world_size: int, tiny_gpt2_model: PreTrainedModel, tiny_gpt2_tokenizer: PreTrainedTokenizerBase, tmp_path: pathlib.Path, tp_size: int = 8):
    """Test distributed training with Ray actors using the tiny_gpt2_model fixture."""
    prompts = [
        "what is RAY?",
        "what is vLLM?",
    ]
    
    # Save the model and tokenizer to a temporary directory
    local_save_path = str(tmp_path / 'tiny_gpt2_model')
    tiny_gpt2_model.save_pretrained(local_save_path)
    tiny_gpt2_tokenizer.save_pretrained(local_save_path)
    
    with start_ray_server() as address:
        if dist.get_rank() == 0:
            master_addr, _ = address.split(':')
            
            print(f"\n=== STARTING DISTRIBUTED TRAINING WITH RAY ACTORS ===")
            num_train_actors = world_size // 2
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


            vllm_tensor_parallel_size = min(tp_size, world_size - num_train_actors)
            num_vllm_engines = world_size // 2 // vllm_tensor_parallel_size
            print(f'num_vllm_engines: {num_vllm_engines}')
            vllm_engines = create_vllm_engines(
                num_engines=num_vllm_engines,
                tensor_parallel_size=vllm_tensor_parallel_size,
                enforce_eager=True,
                pretrain=local_save_path,
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
                    world_size // 2 + 1,
                    'weight-update',
                    backend='nccl',
                ) for i, engine in enumerate(vllm_engines)
            ]
            refs.append(master_actor.init_vllm_process_group.remote(
                backend='nccl',
                master_addr=master_addr,
                master_port=new_port,
                world_size=world_size // 2 + 1,
                rank=0,
                group_name='weight-update',
            ))

            refs = [actor.init_model.remote(local_save_path) for actor in train_actors]
            ray.get(refs)
            print('init model done')

            ray.get(master_actor.sync_weights.remote(vllm_engines))
            print('sync weights done')

            ref = vllm_engines[0].generate.remote(prompts)
            gen_results = ray.get(ref)
            for output in gen_results:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def run(tp_size: int = 8):
    prompts = [
        "what is RAY?",
        "what is vLLM?",
    ]
    pretrain_model_name = os.path.expanduser('~/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e')
    # pretrain_model_name = 'meta-llama/Llama-3.2-1B-Instruct'
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

            refs = [actor.init_model.remote(pretrain_model_name) for actor in train_actors]
            ray.get(refs)
            print('init model done')

            ray.get(master_actor.sync_weights.remote(vllm_engines))
            print('sync weights done')

            ref = vllm_engines[0].generate.remote(prompts)
            gen_results = ray.get(ref)
            for output in gen_results:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp_size', type=int, default=8)
    args = parser.parse_args()
    run(tp_size=args.tp_size)
