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
    WeightUpdateMeta,
    ParamSpec,
)
from inference_server_test.client import ArealOpenAI
from compose_rl.utils.ray_utils import start_ray_server
from tests.common import BaseDistributedGPUActor


@ray.remote(num_gpus=1)
class DistributedGPUActor(BaseDistributedGPUActor):
    """Distributed GPU actor for testing."""

    def init_model(self, model_name: str):
        """Initialize the model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
        )
        self.model.to('cuda')

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
        'Where is the capital of France?',
        'what is the population of it?',
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

            # Create ArealOpenAI client 
            print(f'Creating ArealOpenAI client...')
            client = ArealOpenAI(
                engine=sglang_engine,
                tokenizer=tokenizer,
                api_key="none",  # Not used but required
                base_url="none"  # Not used but required
            )
            print(f'ArealOpenAI client created successfully')

            # Test generation with ArealOpenAI chat interface - multi-turn conversation
            results = []
            conversation_messages = []  # Accumulate conversation history
            
            for i, prompt in enumerate(prompts):
                print(f'\n💬 Turn {i+1}: Processing prompt')
                print(f'📝 User: {prompt}')
                
                # Add user message to conversation history
                conversation_messages.append({"role": "user", "content": prompt})
                
                print(f'📋 Current conversation context ({len(conversation_messages)} messages):')
                for j, msg in enumerate(conversation_messages):
                    print(f'   [{j+1}] {msg["role"]}: {msg["content"]}')
                
                try:
                    # Generate response using ArealOpenAI chat interface with full conversation history
                    response = await client.chat.completions.create(
                        messages=conversation_messages,
                        max_tokens=1024,
                        temperature=1.0,
                        top_p=1.0
                    )
                    
                    assistant_reply = response.choices[0].message.content
                    print(f'🤖 Assistant: {assistant_reply}')
                    
                    # Add assistant response to conversation history for next turn
                    conversation_messages.append({"role": "assistant", "content": assistant_reply})
                    
                    # Extract completion with token info
                    completion = client.get_completions(response.id)
                    if completion:
                        results.append((prompt, completion, len(conversation_messages)))
                    else:
                        print(f"⚠️  Warning: Could not retrieve completion for response {response.id}")
                        
                except Exception as e:
                    print(f"❌ Generation failed for prompt '{prompt}': {e}")
                    continue
            
            # Display detailed results with tokens, logprobs and decoded text
            print(f"\n📊 Detailed Generation Results:")
            print("=" * 80)
            for i, (prompt, completion, context_length) in enumerate(results):
                print(f'\n🔄 Turn {i+1}:')
                print(f'   👤 User: {prompt!r}')
                print(f'   📋 Context length at time of generation: {context_length-1} messages (before assistant response)')
                
                # Get assistant response text
                assistant_text = tokenizer.decode(completion.response.output_tokens, skip_special_tokens=True)
                print(f'   🤖 Assistant: {assistant_text!r}')
                
                # Token analysis
                print(f'\n   📊 Token Analysis:')
                print(f'      Completion ID: {completion.completion.id}')
                print(f'      Input tokens: {completion.response.input_len}')
                print(f'      Output tokens: {completion.response.output_len}')
                print(f'      Input token IDs: {completion.response.input_tokens[:10]}...' if len(completion.response.input_tokens) > 10 else f'      Input token IDs: {completion.response.input_tokens}')
                print(f'      Output token IDs: {completion.response.output_tokens}')
                print(f'      Output logprobs: {[f"{lp:.3f}" for lp in completion.response.output_logprobs[:5]]}...' if len(completion.response.output_logprobs) > 5 else f'      Output logprobs: {[f"{lp:.3f}" for lp in completion.response.output_logprobs]}')
                
                # Decode individual output tokens
                print(f'      Output tokens decoded:')
                for j, token_id in enumerate(completion.response.output_tokens[:10]):  # Show first 10 tokens
                    token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                    logprob = completion.response.output_logprobs[j] if j < len(completion.response.output_logprobs) else 0.0
                    print(f'        [{j:2d}] ID:{token_id:5d} → {token_text!r} (logprob: {logprob:.3f})')
                if len(completion.response.output_tokens) > 10:
                    print(f'        ... and {len(completion.response.output_tokens) - 10} more tokens')
                
                print('-' * 60)
            
            # Final conversation summary
            print(f"\n🗨️  Final Conversation Summary:")
            print("=" * 50)
            for i, msg in enumerate(conversation_messages):
                role_emoji = "👤" if msg["role"] == "user" else "🤖"
                print(f'[{i+1:2d}] {role_emoji} {msg["role"].capitalize()}: {msg["content"]}')
            print(f"\n✅ Multi-turn conversation completed with {len(conversation_messages)} total messages!")
            terminate_process(sglang_server_process)


if __name__ == "__main__":
    asyncio.run(test_distributed_ray_actors("Qwen/Qwen2.5-0.5B-Instruct"))