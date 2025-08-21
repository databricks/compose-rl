# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import subprocess

import ray
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.utils import wait_for_server, terminate_process

from inference_server_test.sglang_remote import (
    RemoteSGLangEngine,
    InferenceEngineConfig,
    ModelRequest,
    GenerationHyperparameters,
    WeightUpdateMeta,
    ParamSpec,
)
from compose_rl.utils.ray_utils import start_ray_server
from tests.common import BaseDistributedGPUActor


@ray.remote(num_gpus=1)
class DistributedGPUActor(BaseDistributedGPUActor):
    """Distributed GPU actor for testing."""

    def init_model(self, model_name: str):
        """Initialize the model."""
        print(f'rank {self.rank} initializing model')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
        )
        print(f'rank {self.rank} HF model loaded')
        self.model.to('cuda')
        print(f'rank {self.rank} model moved to cuda')

    def get_param_specs(self):
        """Get the parameter specifications for the model."""
        param_specs = []
        for name, p in self.model.named_parameters():
            param_spec = ParamSpec(
                name=name,
                shape=tuple(p.shape),
                dtype=str(p.dtype).split('torch.')[-1]  # Convert torch.float32 to 'float32'
            )
            param_specs.append(param_spec)
        return param_specs

    def sync_weights(self):
        """Sync the weights of the model to the SGLang engines."""
        for _, p in self.model.named_parameters():
            # Broadcast the parameter tensor to all GPUs in the distributed group
            dist.broadcast(p, src=0, group=self.model_update_group)

    def test_tensor_all_reduce(self) -> float:
        """Perform a simple tensor all_reduce operation."""
        # Create a tensor on the GPU and perform all_reduce
        device = torch.device('cuda')
        x = torch.ones(1, device=device, dtype=torch.int32)
        dist.all_reduce(x)

        return x.item()


async def test_distributed_ray_actors(
    model_name: str,
    gen_tp_size: int = 1,
    num_sglang_servers: int = 1,
):
    """Test basic single contrller with Ray."""

    prompts = [
        'what is the capital of France?',
        'what is the population of the capital of France?',
    ]


    with start_ray_server() as address:
        if dist.get_rank() == 0:
            sglang_server_process = subprocess.Popen(
                "CUDA_VISIBLE_DEVICES=2 python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct",
                shell=True
            )
            sglang_addresses = [f"localhost:{30000}"]
            wait_for_server(f"http://{sglang_addresses[0]}")

            # rank 0 is the ray client
            master_addr, _ = address.split(':')

            print(
                f'\n=== STARTING DISTRIBUTED TRAINING WITH RAY ACTORS ===',
            )
            num_train_actors = dist.get_world_size()
            # Create actors - rank 0 will allocate master address/port
            train_actors = []
            # master actor will allocate master_addr and master_port
            master_actor = DistributedGPUActor.remote(0, num_train_actors)
            train_actors.append(master_actor)

            # Get master address from rank 0 actor
            master_info = ray.get(
                master_actor.get_master_address.remote(),  # type: ignore
            )
            master_addr, master_port = master_info
            print(
                f'Master address allocated: {master_addr}:{master_port}',
            )

            # Create remaining actors with the master address/port
            for i in range(1, num_train_actors):
                actor = DistributedGPUActor.remote(
                    i,
                    num_train_actors,
                    master_addr,  # type: ignore
                    master_port,
                )
                train_actors.append(actor)

            # Initialize process groups for all actors
            init_tasks = [
                actor.init_train_process_group.remote()  # type: ignore
                for actor in train_actors
            ]
            ray.get(init_tasks)
            print('init train process group done')

            # Perform tensor all_reduce on all actors
            reduce_tasks = [
                actor.test_tensor_all_reduce.remote()  # type: ignore
                for actor in train_actors
            ]
            results = ray.get(reduce_tasks)
            assert results == [num_train_actors] * num_train_actors
            print('tensor all_reduce done')
            
            print(f'SGLang server addresses: {sglang_addresses}')
            
            # Create SGLang engine configuration
            inference_config = InferenceEngineConfig(
                setup_timeout=60.0,
                request_timeout=300.0,
                request_retries=3
            )
            
            # Create RemoteSGLangEngine
            sglang_engine = RemoteSGLangEngine(
                config=inference_config,
                addresses=sglang_addresses
            )
            
            # Initialize SGLang engine (wait for servers to be ready)
            sglang_engine.initialize()

            new_port = ray.get(
                master_actor.get_free_port.remote(),  # type: ignore
            )
            print(f'new_port to init SGLang distributed group: {new_port}')
            
            # Setup distributed group for weight updates with SGLang
            # Create weight update metadata for distributed group initialization
            weight_update_meta = WeightUpdateMeta(
                nccl_master_address=master_addr,
                nccl_master_port=new_port,
                nccl_group_name="sglang_weight_update",
                gen_tp_size=gen_tp_size,  # Tensor parallel size per server
                gen_world_size=num_sglang_servers * gen_tp_size
            )
            
            # Initialize distributed group on SGLang servers
            await asyncio.gather(
                sglang_engine.ainit_weights_update_group(weight_update_meta),
                # Initialize process group on the training side
                master_actor.add_process_group.remote(  # type: ignore
                    backend='nccl',
                    master_addr=master_addr,
                    master_port=new_port,
                    world_size=num_sglang_servers + 1,  # SGLang servers + trainer
                    rank=0,
                    group_name='sglang_weight_update',
                ),
            )
            print('SGLang distributed group initialization done')

            refs = [
                actor.init_model.remote(model_name)  # type: ignore
                for actor in train_actors
            ]
            ray.get(refs)
            print('Trainer init model done')

            param_specs = ray.get(master_actor.get_param_specs.remote())

            await asyncio.gather(
                sglang_engine.aupdate_weights(param_specs, 'sglang_weight_update'),
                master_actor.sync_weights.remote(),
            )
            print('sync weights done')

            # Initialize tokenizer for proper tokenization/detokenization
            print(f'Loading tokenizer for model: {model_name}')
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f'Tokenizer loaded. Vocab size: {tokenizer.vocab_size}')

            # Test generation with SGLang
            results = []
            for prompt in prompts:
                # Convert prompt string to token IDs using the actual tokenizer
                input_ids = tokenizer.encode(prompt, add_special_tokens=True)
                
                # Create generation request
                gen_config = GenerationHyperparameters(
                    max_new_tokens=50,
                    temperature=1.0,
                    top_p=1.0,
                    n_samples=1
                )
                
                request = ModelRequest(
                    input_ids=input_ids,
                    gconfig=gen_config
                )
                
                # Generate response
                response = await sglang_engine.agenerate(request)
                results.append((prompt, response.output_tokens))            
            
            # Display results with both tokens and decoded text
            for prompt, output_tokens in results:
                # Detokenize the output tokens to get the generated text
                generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
                print(f'Prompt: {prompt!r}')
                print(f'Generated token IDs ({len(output_tokens)}): {output_tokens}')
                print(f'Generated text: {generated_text!r}')
                print('-' * 60)
            terminate_process(sglang_server_process)


if __name__ == "__main__":
    asyncio.run(test_distributed_ray_actors("Qwen/Qwen2.5-0.5B-Instruct"))