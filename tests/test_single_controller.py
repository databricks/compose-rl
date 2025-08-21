# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import logging
import asyncio

import ray
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

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

# Set up logging
logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class DistributedGPUActor(BaseDistributedGPUActor):
    """Distributed GPU actor for testing."""

    def init_model(self, model_name: str):
        """Initialize the model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
        )
        self.model.to('cuda')

    def sync_weights(self, sglang_engine: RemoteSGLangEngine):
        """Sync the weights of the model to the SGLang engines."""
        # Create parameter specifications for all model parameters
        param_specs = []
        for name, p in self.model.named_parameters():
            param_spec = ParamSpec(
                name=name,
                shape=p.shape,
                dtype=str(p.dtype).split('.')[-1]  # Convert torch.float32 to 'float32'
            )
            param_specs.append(param_spec)
            
            # Broadcast the parameter tensor to all GPUs in the distributed group
            dist.broadcast(p, src=0, group=self.model_update_group)
        
        # Create weight update metadata
        weight_update_meta = WeightUpdateMeta(
            nccl_master_address=self.master_addr,
            nccl_master_port=29500,  # Default NCCL port
            nccl_param_specs=[param_specs],  # List of param specs for each rank
            nccl_group_name="sglang_weight_update",
            gen_tp_size=1,  # Tensor parallel size for generation (assuming 1 for now)
            gen_world_size=len(sglang_engine.addresses)  # Number of SGLang servers
        )
        
        # Update weights on SGLang servers
        sglang_engine.update_weights(weight_update_meta)

        for name, p in self.model.named_parameters():
            # Broadcast the parameter tensor to all GPUs in the distributed group
            dist.broadcast(p, src=0, group=self.model_update_group)

    def test_tensor_all_reduce(self) -> float:
        """Perform a simple tensor all_reduce operation."""
        # Create a tensor on the GPU and perform all_reduce
        device = torch.device('cuda')
        x = torch.ones(1, device=device, dtype=torch.int32)
        dist.all_reduce(x)

        return x.item()


def test_distributed_ray_actors(
    model_name: str,
):
    """Test basic single contrller with Ray."""

    prompts = [
        'what is RAY?',
        'what is vLLM?',
    ]

    world_size = dist.get_world_size()

    with start_ray_server() as address:
        if dist.get_rank() == 0:
            # rank 0 is the ray client
            master_addr, _ = address.split(':')

            logger.info(
                f'\n=== STARTING DISTRIBUTED TRAINING WITH RAY ACTORS ===',
            )
            num_train_actors = world_size
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
            logger.info(
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

            # Perform tensor all_reduce on all actors
            reduce_tasks = [
                actor.test_tensor_all_reduce.remote()  # type: ignore
                for actor in train_actors
            ]
            results = ray.get(reduce_tasks)
            assert results == [num_train_actors] * num_train_actors

            # Setup SGLang engine configuration
            # Assume SGLang servers are already running and accessible
            # For this test, we'll use localhost addresses with different ports
            num_sglang_servers = 1
            sglang_addresses = []
            for i in range(num_sglang_servers):
                # Assuming SGLang servers are running on consecutive ports starting from 30000
                sglang_addresses.append(f"localhost:{30000 + i}")
            
            logger.info(f'SGLang server addresses: {sglang_addresses}')
            
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
            logger.info(f'new_port to init SGLang distributed group: {new_port}')
            
            # Setup distributed group for weight updates with SGLang
            # Create weight update metadata for distributed group initialization
            weight_update_meta = WeightUpdateMeta(
                nccl_master_address=master_addr,
                nccl_master_port=new_port,
                nccl_group_name="sglang_weight_update",
                gen_tp_size=1,  # Tensor parallel size per server
                gen_world_size=num_sglang_servers
            )
            
            # Initialize distributed group on SGLang servers
            sglang_engine.init_weights_update_group(weight_update_meta)
            
            # Initialize process group on the training side
            ray.get(
                master_actor.add_process_group.remote(  # type: ignore
                    backend='nccl',
                    master_addr=master_addr,
                    master_port=new_port,
                    world_size=num_sglang_servers + 1,  # SGLang servers + trainer
                    rank=0,
                    group_name='sglang_weight_update',
                ),
            )
            logger.info('SGLang distributed group initialization done')

            refs = [
                actor.init_model.remote(model_name)  # type: ignore
                for actor in train_actors
            ]
            ray.get(refs)
            logger.info('Trainer init model done')

            ray.get(
                master_actor.sync_weights.remote(sglang_engine),  # type: ignore
            )
            logger.info('sync weights done')

            # Test generation with SGLang
            async def test_generation():
                results = []
                for prompt in prompts:
                    # Convert prompt string to token IDs (simplified for testing)
                    # In practice, you'd use a tokenizer here
                    input_ids = [1, 2, 3, 4, 5]  # Dummy token IDs for testing
                    
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
                
                return results
            
            # Run async generation
            gen_results = asyncio.run(test_generation())
            for prompt, output_tokens in gen_results:
                logger.info(
                    f'Prompt: {prompt!r}, Generated tokens: {output_tokens}',
                )
