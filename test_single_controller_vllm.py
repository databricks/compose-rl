# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import argparse
import os
import signal
import subprocess

import ray
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from orl_servers.vllm_remote import RemoteVLLMEngine
from orl_servers.structs import InferenceEngineConfig, WeightUpdateMeta, ParamSpec
from orl_servers import ArealOpenAI
from compose_rl.utils.ray_utils import start_ray_server
from tests.common import BaseDistributedGPUActor

from test_async_llm_server import _wait_for_server_ready

WORKER_WRAP = 'orl_servers.vllm_worker_wrap.WorkerWrap'


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
                dtype='bfloat16'  # hardcoded
            )
            param_specs.append(param_spec)
        return param_specs

    # def sync_weights(self):
    #     """Sync the weights of the model to the vLLM engines."""
    #     for _, p in self.model.named_parameters():
    #         # Broadcast the parameter tensor to all GPUs in the distributed group
    #         dist.broadcast(p, src=0, group=self.model_update_group)
    
    def sync_weight(self, param_name: str): 
        """Sync the weights of the model to the vLLM engines."""
        p = self.model.get_parameter(param_name)
        param_data = p.data.bfloat16()
        self._model_update_group.broadcast(param_data, src=0, stream=torch.cuda.current_stream())

    def test_tensor_all_reduce(self) -> float:
        """Perform a simple tensor all_reduce operation."""
        # Create a tensor on the GPU and perform all_reduce
        device = torch.device('cuda')
        x = torch.ones(1, device=device, dtype=torch.int32)
        dist.all_reduce(x)

        return x.item()


async def test_distributed_ray_actors(
    model_name: str,
    gen_tp_size: int,
    num_vllm_servers: int,
):
    """Test basic single contrller with Ray."""

    prompts = [
        'Where is the capital of France?',
        'what is the population of it?',
        'is Louvre Museum located in it?'
    ]

    with start_ray_server() as address:
        if dist.get_rank() == 0:
            # Set environment variable for CUDA device visibility
            env = os.environ.copy()
            training_world_size = dist.get_world_size()
            inference_gpus = range(training_world_size, training_world_size + gen_tp_size)
            print(f'training_world_size: {training_world_size}, gen_tp_size: {gen_tp_size}')
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, inference_gpus))
            
            vllm_server_process = subprocess.Popen([
                'orl-vllm-server',
                '--model', model_name,
                '--worker-extension-cls', WORKER_WRAP,
                '--tensor-parallel-size', str(gen_tp_size),
                '--disable-custom-all-reduce'
            ], env=env)
            vllm_addresses = [f"localhost:{8000}"]
            _wait_for_server_ready()

            try:
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
                
                print(f'vLLM server addresses: {vllm_addresses}')
                
                # Create SGLang engine configuration
                inference_config = InferenceEngineConfig(
                    setup_timeout=60.0,
                    request_timeout=300.0,
                    request_retries=3
                )
                
                # Create RemoteSGLangEngine
                vllm_engine = RemoteVLLMEngine(
                    config=inference_config,
                    addresses=vllm_addresses
                )
                
                # Initialize SGLang engine (wait for servers to be ready)
                vllm_engine.initialize()

                new_port = ray.get(
                    master_actor.get_free_port.remote(),  # type: ignore
                )
                print(f'new_port to init SGLang distributed group: {new_port}')
                
                # Setup distributed group for weight updates with SGLang
                # Create weight update metadata for distributed group initialization
                weight_update_meta = WeightUpdateMeta(
                    nccl_master_address=master_addr,
                    nccl_master_port=new_port,
                    gen_tp_size=gen_tp_size,  # Tensor parallel size per server
                    gen_world_size=num_vllm_servers * gen_tp_size
                )
                
                # Initialize distributed group on SGLang servers
                await asyncio.gather(
                    vllm_engine.ainit_weight_update_group(weight_update_meta),
                    # Initialize process group on the training side
                    master_actor.add_process_group.remote(  # type: ignore
                        master_addr=master_addr,
                        master_port=new_port,
                        world_size=num_vllm_servers * gen_tp_size + 1,  # vLLM servers + trainer
                        rank=0,
                        # group_name='vllm_weight_update',
                    ),
                )
                print('vLLM distributed group initialization done')

                # Initialize tokenizer for proper tokenization/detokenization
                print(f'Loading tokenizer for model: {model_name}')
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                print(f'Tokenizer loaded. Vocab size: {tokenizer.vocab_size}')

                # Create ArealOpenAI client 
                print(f'Creating ArealOpenAI client...')
                client = ArealOpenAI(
                    engine=vllm_engine,
                    tokenizer=tokenizer,
                    api_key="none",  # Not used but required
                    base_url="none"  # Not used but required
                )
                print(f'ArealOpenAI client created successfully')

                async def run_conversation_phase(phase_label: str):
                    print(f"\n===== {phase_label}: Starting multi-turn conversation =====")
                    results_local = []
                    conversation_messages_local = []

                    for i, prompt in enumerate(prompts):
                        print(f'\n💬 Turn {i+1}: Processing prompt')
                        print(f'📝 User: {prompt}')

                        conversation_messages_local.append({"role": "user", "content": prompt})

                        print(f'📋 Current conversation context ({len(conversation_messages_local)} messages):')
                        for j, msg in enumerate(conversation_messages_local):
                            print(f'   [{j+1}] {msg["role"]}: {msg["content"]}')

                        try:
                            response = await client.chat.completions.create(
                                messages=conversation_messages_local,
                                max_tokens=1024,
                                temperature=1.0,
                                top_p=1.0
                            )

                            assistant_reply = response.choices[0].message.content
                            print(f'🤖 Assistant: {assistant_reply}')

                            conversation_messages_local.append({"role": "assistant", "content": assistant_reply})

                            completion = client.get_completions(response.id)
                            if completion:
                                results_local.append((prompt, completion, len(conversation_messages_local)))
                            else:
                                print(f"⚠️  Warning: Could not retrieve completion for response {response.id}")

                        except Exception as e:
                            print(f"❌ Generation failed for prompt '{prompt}': {e}")
                            raise e

                    print(f"\n📊 {phase_label} - Detailed Generation Results:")
                    print("=" * 80)
                    for i, (prompt, completion, context_length) in enumerate(results_local):
                        print(f'\n🔄 Turn {i+1}:')
                        print(f'   👤 User: {prompt!r}')
                        print(f'   📋 Context length at time of generation: {context_length-1} messages (before assistant response)')

                        assistant_text = tokenizer.decode(completion.response.output_tokens, skip_special_tokens=True)
                        print(f'   🤖 Assistant: {assistant_text!r}')

                        print(f'\n   📊 Token Analysis:')
                        print(f'      Completion ID: {completion.completion.id}')
                        print(f'      Input tokens: {completion.response.input_len}')
                        print(f'      Output tokens: {completion.response.output_len}')
                        print(f'      Input token IDs: {completion.response.input_tokens[:10]}...' if len(completion.response.input_tokens) > 10 else f'      Input token IDs: {completion.response.input_tokens}')
                        print(f'      Output token IDs: {completion.response.output_tokens}')
                        print(f'      Output logprobs: {[f"{lp:.3f}" for lp in completion.response.output_logprobs[:5]]}...' if len(completion.response.output_logprobs) > 5 else f'      Output logprobs: {[f"{lp:.3f}" for lp in completion.response.output_logprobs]}')

                        print(f'      Output tokens decoded:')
                        for j, token_id in enumerate(completion.response.output_tokens[:10]):
                            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                            logprob = completion.response.output_logprobs[j] if j < len(completion.response.output_logprobs) else 0.0
                            print(f'        [{j:2d}] ID:{token_id:5d} → {token_text!r} (logprob: {logprob:.3f})')
                        if len(completion.response.output_tokens) > 10:
                            print(f'        ... and {len(completion.response.output_tokens) - 10} more tokens')

                        print('-' * 60)

                    print(f"\n🗨️  {phase_label} - Final Conversation Summary:")
                    print("=" * 50)
                    for i, msg in enumerate(conversation_messages_local):
                        role_emoji = "👤" if msg["role"] == "user" else "🤖"
                        print(f'[{i+1:2d}] {role_emoji} {msg["role"].capitalize()}: {msg["content"]}')
                    print(f"\n✅ {phase_label} completed with {len(conversation_messages_local)} total messages!")

                    return results_local, conversation_messages_local

                # Phase 1: Pre-weight-update generation
                pre_results, _ = await run_conversation_phase("PRE-UPDATE")

                # Initialize trainer model and perform weight update broadcast
                refs = [
                    actor.init_model.remote(model_name)  # type: ignore
                    for actor in train_actors
                ]
                ray.get(refs)
                print('Trainer init model done')

                param_specs = ray.get(master_actor.get_param_specs.remote())

                for param_spec in param_specs:
                    await asyncio.gather(
                        vllm_engine.aupdate_weight(param_spec),
                        master_actor.sync_weight.remote(param_spec.name),
                    )
                print('sync weights done')
                await vllm_engine.areset_prefix_cache()

                # Phase 2: Post-weight-update generation (same prompts)
                post_results, _ = await run_conversation_phase("POST-UPDATE")

                # Optional: simple comparison summary per prompt
                print("\n===== Comparison: PRE-UPDATE vs POST-UPDATE =====")
                for i, prompt in enumerate(prompts):
                    print(f"\nPrompt {i+1}: {prompt!r}")
                    if i < len(pre_results):
                        pre_text = tokenizer.decode(pre_results[i][1].response.output_tokens, skip_special_tokens=True)
                        print(f"  PRE:  {pre_text!r}")
                    if i < len(post_results):
                        post_text = tokenizer.decode(post_results[i][1].response.output_tokens, skip_special_tokens=True)
                        print(f"  POST: {post_text!r}")
            finally:
                # Try graceful shutdown first with SIGINT
                print("🔄 Attempting graceful shutdown with SIGINT (like CTRL+C)...")
                vllm_server_process.send_signal(signal.SIGINT)
                vllm_server_process.wait(timeout=10)  # Wait up to 10 seconds for graceful shutdown
                print("✅ vLLM server shut down gracefully with SIGINT")



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--gen-tp-size', '-tp', type=int, default=1)
    args.add_argument('--num_vllm_servers', type=int, default=1)
    args = args.parse_args()
    asyncio.run(test_distributed_ray_actors("Qwen/Qwen2.5-0.5B-Instruct", args.gen_tp_size, args.num_vllm_servers))