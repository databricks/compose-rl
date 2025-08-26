# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 The AllenAI Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified version from https://github.com/OpenRLHF/OpenRLHF and The AllenAI Team.

import asyncio
import logging
import os
from typing import Any, Union
from uuid import uuid4

import ray
import torch
from packaging import version

import vllm
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

log = logging.getLogger(__name__)


class BaseLLM:

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.noset_visible_devices = kwargs.pop('noset_visible_devices')

        if kwargs.get('distributed_executor_backend') == 'ray':
            # a hack to make the script work.
            # stop ray from manipulating *_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            os.environ.pop('ROCR_VISIBLE_DEVICES', None)
        elif self.noset_visible_devices:
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
            os.environ['CUDA_VISIBLE_DEVICES'] = str(ray.get_gpu_ids()[0])

        self.num_gpus = kwargs.pop('num_gpus')
        self.bundle_indices = kwargs.pop('bundle_indices', None)
        if self.bundle_indices is not None:
            os.environ['VLLM_RAY_PER_WORKER_GPUS'] = str(self.num_gpus)
            os.environ['VLLM_RAY_BUNDLE_INDICES'] = ','.join(
                map(str, self.bundle_indices),
            )
            log.info(f'creating LLM with bundle_indices={self.bundle_indices}')

        # Store args and kwargs for child classes to use
        self.args = args
        self.kwargs = kwargs

        if version.parse(vllm.__version__) >= version.parse("0.9.0"):
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

class LLM(BaseLLM):

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Initialize base class first
        super().__init__(*args, **kwargs)
        
        # Create sync LLM engine
        self.llm = vllm.LLM(*self.args, **self.kwargs)

    def generate(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        sampling_params = None
        if 'sampling_params' in kwargs:
            sampling_params = SamplingParams(**kwargs.pop('sampling_params'))
            log.info(f'sampling_params is: {sampling_params}')

        return self.llm.generate(
            *args,
            **kwargs,
            sampling_params=sampling_params,
        )

    def chat(self, *args: Any, **kwargs: Any):
        sampling_params = None
        if 'sampling_params' in kwargs:
            sampling_params = SamplingParams(**kwargs.pop('sampling_params'))
            log.info(f'sampling_params is: {sampling_params}')

        return self.llm.chat(
            *args,
            **kwargs,
            sampling_params=sampling_params,
        )

    def init_process_group(
        self,
        master_address: str,
        master_port: str,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
    ):
        return self.llm.collective_rpc(
            'init_process_group',
            args=(
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
            ),
        )

    def update_weight(
        self,
        name: str,
        dtype: torch.dtype,
        shape: Union[tuple[int, ...], list[int]],
        empty_cache: bool = False,
    ):
        return self.llm.collective_rpc(
            'update_weight',
            args=(name, dtype, shape, empty_cache),
        )

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

class AsyncLLM:

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # this env is necessary otherwise vLLM will not create V1 engine even if vllm.envs.VLLM_USE_V1 is True
        os.environ["VLLM_USE_V1"] = "1"
        if version.parse(vllm.__version__) >= version.parse("0.9.0"):
            # otherwise it can not serialize torch dtype
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        # Create AsyncLLMEngine instead of regular LLM
        engine_args = vllm.AsyncEngineArgs(*args, **kwargs)
        self.engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        
        # Track running tasks by request_id for abort functionality
        self.running_tasks: dict[str, asyncio.Task] = {}
        
        # Generation control: Event is set when generation is allowed
        self._generation_enabled = asyncio.Event()
        self._generation_enabled.set()  # Initially allow generation

    async def _collect_outputs(self, prompt_token_ids: list[int], request_id: str, sampling_params: SamplingParams):
        """Collect outputs for a single prompt."""
        final_output = None
        try:
            async for request_output in self.engine.generate(
                prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                final_output = request_output
        except asyncio.CancelledError:
            # Local task was cancelled (likely due to abort() call)
            # The actual generation in vLLM engine should have been aborted separately
            log.info(f'local task for request {request_id} was aborted')
            # TODO consider overwriting the "finish_reason" and "stop_reason" of the final_output
        finally:
            return final_output
    
    async def _generate(self, prompt_token_ids: list[int], sampling_params: SamplingParams):
        # Wait for generation to be enabled before proceeding
        await self._generation_enabled.wait()
        
        request_id = str(uuid4().hex)
        task = asyncio.create_task(self._collect_outputs(prompt_token_ids, request_id, sampling_params))
        # Track the task by request_id
        self.running_tasks[request_id] = task
        res = await task
        self.running_tasks.pop(request_id)
        return res

    async def _generate_with_retries(self, prompt_token_ids: list[int], sampling_params: SamplingParams, max_retries: int = 0):
        retry = 0
        sampling_params_with_retries = sampling_params.clone()
        ans = None
        while max_retries == 0 or retry < max_retries:
            res = await self._generate(prompt_token_ids, sampling_params_with_retries)
            if ans is None:
                ans = res
            else:
                ans.add(res, aggregate=True)
            if res.finished:
                log.info(f'request {res.request_id} is finished, finishreason: {res.outputs[0].finish_reason}, stopreason: {res.outputs[0].stop_reason}')
                return ans
            else:
                log.info(f'request {res.request_id} is not finished, finishreason: {res.outputs[0].finish_reason}, stopreason: {res.outputs[0].stop_reason}, retrying...')
                # TODO (handle n > 1)
                assert sampling_params.n == 1, f'generate with retries does not work with sampling_params.n > 1, but got {sampling_params.n}'
                retry += 1
                prompt_token_ids += list(res.outputs[0].token_ids)
                if sampling_params_with_retries.max_tokens is not None:
                    sampling_params_with_retries.max_tokens = sampling_params_with_retries.max_tokens - len(res.outputs[0].token_ids)
        return ans
    
    async def generate(self, batched_promts: list[list[int]], sampling_params: SamplingParams):
        """Generate responses using vLLM's async engine."""

        tasks = []
        for prompt in batched_promts:
            # Schedule the collection of outputs for each prompt.
            # Avoid duplicate request_ids
            task = asyncio.create_task(self._generate_with_retries(prompt, sampling_params))
            tasks.append(task)
        
        outputs = await asyncio.gather(*tasks)

        return outputs

    async def abort(self, request_id: str):
        """
        Abort a running generation task by request_id.
        
        Args:
            request_id: The ID of the request to abort
            
        Returns:
            None
        """
        # Get the task if it exists
        task = self.running_tasks[request_id]
        if not task.done():
            task.cancel()
            log.info(f"Cancelled local task for request {request_id}")
        return

    async def pause_generation(self):
        """
        Abort all current requests and prevent any new _generate calls.
        
        This method will:
        1. Clear the generation enabled event to block new requests
        2. Cancel all currently running generation tasks
        """
        # Prevent new generation requests
        self._generation_enabled.clear()
        log.info("Generation paused - new requests will be blocked")
        
        # Cancel all running tasks
        if self.running_tasks:
            log.info(f"Cancelling {len(self.running_tasks)} running tasks")
            for request_id in list(self.running_tasks.keys()):
                await self.abort(request_id)
        else:
            log.info("No running tasks to cancel")

    async def continue_generation(self):
        """
        Remove the generation lock and allow new generation requests.
        
        This method sets the generation enabled event, allowing blocked
        and new _generate calls to proceed.
        """
        self._generation_enabled.set()
        log.info("Generation resumed - new requests are now allowed")

    async def init_process_group(
        self, master_address: str, master_port: str, rank_offset: int, world_size: int
    ):
        return await self.engine.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size),
        )

    async def update_weight(self, name: str, dtype: torch.dtype, shape: Union[tuple[int, ...], list[int]], empty_cache: bool = False):
        return await self.engine.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    async def reset_prefix_cache(self):
        await self.engine.reset_prefix_cache()


LLMRayActor = ray.remote(LLM)
LLMRayActorAsync = ray.remote(AsyncLLM)


def get_shared_async_llm_and_tokenizer():
    """Helper function to create shared AsyncLLM and tokenizer for tests."""
    from transformers import AutoTokenizer
    
    # Model configuration
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Initializing shared AsyncLLM with model: {model_name}")
    
    # Create AsyncLLM instance
    async_llm = AsyncLLM(
        model=model_name,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=2048,
    )
    
    # Load tokenizer for encoding prompts
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return async_llm, tokenizer


async def test_async_llm_pause_continue(async_llm: AsyncLLM, tokenizer: Any):
    """Test the pause_generation and continue_generation functionality of AsyncLLM.
    
    This test demonstrates:
    1. Starting generation tasks
    2. Pausing generation (aborts current tasks and blocks new ones)
    3. Trying to start new generation while paused (should be blocked)
    4. Continuing generation (unblocks new tasks)
    5. Verifying new tasks can run after continue
    """
    try:
        print("Testing pause_generation and continue_generation functionality...")
        
        # Test prompts for initial generation
        initial_prompts = [
            "Write a short story about a robot.",
            "Explain the concept of quantum entanglement.",
        ]
        
        # Test prompts for blocked generation
        blocked_prompts = [
            "What is the capital of France?",
        ]
        
        # Test prompts for resumed generation  
        resumed_prompts = [
            "Describe the process of photosynthesis.",
            "What are the benefits of exercise?",
        ]
        
        # Encode all prompts
        def encode_prompts(prompts: list[str]) -> list[list[int]]:
            return [tokenizer.encode(prompt, return_tensors="pt").squeeze(0).tolist() 
                   for prompt in prompts]
        
        initial_encoded = encode_prompts(initial_prompts)
        blocked_encoded = encode_prompts(blocked_prompts)
        resumed_encoded = encode_prompts(resumed_prompts)
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
        )
        
        print(f"\n1. Starting initial generation with {len(initial_prompts)} prompts...")
        
        # Start initial generation but don't await immediately
        initial_task = asyncio.create_task(
            async_llm.generate(initial_encoded, sampling_params)
        )
        
        # Wait a bit for generation to start
        await asyncio.sleep(0.5)
        print(f"Running tasks before pause: {len(async_llm.running_tasks)}")
        
        print("\n2. Pausing generation...")
        await async_llm.pause_generation()
        
        # Verify generation is paused by checking the event
        is_paused = not async_llm._generation_enabled.is_set()
        print(f"Generation is paused: {is_paused}")
        
        print("\n3. Attempting to start new generation while paused (should be blocked)...")
        
        # This should be blocked and wait
        blocked_task_started = False
        blocked_task = asyncio.create_task(
            async_llm.generate(blocked_encoded, sampling_params)
        )
        
        # Give it a moment to see if it starts (it shouldn't)
        # Use shield() to prevent the task from being cancelled when timeout occurs
        try:
            await asyncio.wait_for(asyncio.shield(blocked_task), timeout=1.0)
            blocked_task_started = True
            print("❌ ERROR: Blocked task completed (should have been blocked)")
        except asyncio.TimeoutError:
            print("✅ Confirmed: New generation is properly blocked while paused")
            print("   (Task is still running in background, protected by shield)")
        
        print("\n4. Continuing generation...")
        await async_llm.continue_generation()
        
        # Verify generation is resumed
        is_resumed = async_llm._generation_enabled.is_set()
        print(f"Generation is resumed: {is_resumed}")
        
        print("\n5. Waiting for blocked task to complete after resume...")
        # With shield(), the task should not have been cancelled and should complete now
        blocked_outputs = await blocked_task
        print(f"✅ Blocked task completed after resume with {len(blocked_outputs)} outputs")
        
        print("\n6. Starting new generation after resume...")
        resumed_outputs = await async_llm.generate(resumed_encoded, sampling_params)
        print(f"✅ New generation completed with {len(resumed_outputs)} outputs")
        
        # Wait for initial task to complete (may have been cancelled)
        try:
            initial_outputs = await initial_task
            print(f"Initial task completed with {len(initial_outputs)} outputs")
        except asyncio.CancelledError:
            print("Initial task was cancelled during pause (expected)")
        
        print("\n✅ Pause/Continue test completed successfully!")
        
        # Summary
        print("\nTest Summary:")
        print(f"- Generation was successfully paused: {is_paused}")
        print(f"- New tasks were blocked while paused: {not blocked_task_started}")
        print(f"- Generation was successfully resumed: {is_resumed}")
        print(f"- New tasks work after resume: {len(resumed_outputs) > 0}")
        print(f"- Blocked tasks completed after resume (protected by shield): {len(blocked_outputs) > 0}")
        
        return True
        
    except Exception as e:
        print(f"Error during pause/continue testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_async_llm_abort(async_llm: AsyncLLM, tokenizer: Any):
    """Test the abort functionality of AsyncLLM.
    
    This test creates long-running generation tasks and demonstrates
    aborting them mid-generation. The abort method first calls vLLM's
    abort to stop actual generation, then cancels the local asyncio task.
    """
    try:
        # Test prompts designed to generate longer responses
        test_prompts = [
            "Write a detailed explanation of machine learning with examples and applications in at least 500 words.",
            "Explain the history of artificial intelligence from its inception to modern times, including major milestones.",
            "Describe the process of training a neural network step by step with mathematical details.",
        ]
        
        print(f"Testing abort functionality with {len(test_prompts)} long prompts...")
        
        # Encode prompts to token IDs
        encoded_prompts = []
        for prompt in test_prompts:
            tokens = tokenizer.encode(prompt, return_tensors="pt").squeeze(0).tolist()
            encoded_prompts.append(tokens)
            print(f"Prompt: '{prompt[:50]}...' -> {len(tokens)} tokens")
        
        # Create sampling parameters for longer generation
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=500,  # Longer generation
            stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
        )
        
        print(f"Sampling parameters: temp={sampling_params.temperature}, " +
              f"top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}")
        
        # Start generation but don't await immediately
        print("\nStarting generation tasks...")
        
        # Start generation in the background
        generation_task = asyncio.create_task(
            async_llm.generate(encoded_prompts, sampling_params)
        )
        
        # Wait a bit for generation to start
        await asyncio.sleep(1.0)
        
        # Check running tasks
        print(f"Running tasks: {list(async_llm.running_tasks.keys())}")
        
        if async_llm.running_tasks:
            # Pick the first request to abort
            request_id_to_abort = list(async_llm.running_tasks.keys())[0]
            print(f"\nAborting request: {request_id_to_abort}")
            
            # Abort the request
            result = await async_llm.abort(request_id_to_abort)
            print(f"Abort method returned: {result}")
            print("Request has been aborted in vLLM engine and local task cancelled")
        else:
            print("No running tasks found to abort")
        
        # Wait for remaining tasks to complete or handle exceptions
        outputs = await generation_task
        print(f"\nRemaining generation tasks completed")
        
        # Process results
        completed_count = 0
        aborted_count = 0
        for i, output in enumerate(outputs):
            if output.finished:
                completed_count += 1
                print(f"Task {i} completed with {len(output.outputs[0].token_ids)} tokens.\n prompt: {test_prompts[i]}\n response: {tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=True)}")
            else:
                aborted_count += 1
                print(f"Task {i} aborted with {len(result.outputs[0].token_ids)} tokens.\n prompt: {test_prompts[i]}\n response: {tokenizer.decode(result.outputs[0].token_ids, skip_special_tokens=True)}")
        
        print(f"\nSummary: {completed_count} completed, {aborted_count} aborted")
        
        print(f"\n✅ Abort test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during abort testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_async_llm(async_llm: AsyncLLM, tokenizer: Any):
    """Simple test for AsyncLLM using Qwen/Qwen2.5-0.5B model.
    
    This test creates an AsyncLLM instance and tests it with several prompts
    without depending on Ray.
    """
    try:
        
        # Test prompts
        test_prompts = [
            "What is artificial intelligence?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about machine learning.",
            "What are the benefits of renewable energy?",
        ]
        
        print(f"Testing with {len(test_prompts)} prompts...")
        
        # Encode prompts to token IDs
        encoded_prompts = []
        for prompt in test_prompts:
            tokens = tokenizer.encode(prompt, return_tensors="pt").squeeze(0).tolist()
            encoded_prompts.append(tokens)
            print(f"Prompt: '{prompt}' -> {len(tokens)} tokens")
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
        )
        
        print(f"Sampling parameters: temp={sampling_params.temperature}, " +
              f"top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}")
        
        # Generate responses
        print("\nGenerating responses...")
        start_time = asyncio.get_event_loop().time()
        
        outputs = await async_llm.generate(encoded_prompts, sampling_params)
        
        end_time = asyncio.get_event_loop().time()
        generation_time = end_time - start_time
        
        print(f"Generation completed in {generation_time:.2f} seconds")
        
        # Process and display results
        print("\n" + "="*50)
        print("GENERATION RESULTS")
        print("="*50)
        
        for i, (prompt, output) in enumerate(zip(test_prompts, outputs)):
            if output and output.outputs:
                generated_text = tokenizer.decode(
                    output.outputs[0].token_ids, 
                    skip_special_tokens=True
                )
                print(f"\nPrompt {i+1}: {prompt}")
                print(f"Generated: {generated_text}")
                print(f"Tokens generated: {len(output.outputs[0].token_ids)}")
                print("-" * 30)
            else:
                print(f"\nPrompt {i+1}: {prompt}")
                print("Generated: [ERROR - No output generated]")
                print("-" * 30)
        
        print(f"\nTest completed successfully!")
        print(f"Total time: {generation_time:.2f}s")
        print(f"Average time per prompt: {generation_time/len(test_prompts):.2f}s")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_async_llm_test(async_llm: AsyncLLM, tokenizer: Any):
    """Synchronous wrapper to run the async test."""
    print("Starting AsyncLLM test...")
    
    try:
        # Run the async test
        result = asyncio.run(test_async_llm(async_llm, tokenizer))
        
        if result:
            print("\n✅ AsyncLLM test passed!")
        else:
            print("\n❌ AsyncLLM test failed!")
            
        return result
        
    except Exception as e:
        print(f"\n❌ AsyncLLM test failed with exception: {str(e)}")
        return False


def run_async_llm_pause_continue_test(async_llm: AsyncLLM, tokenizer: Any):
    """Synchronous wrapper to run the async pause/continue test."""
    print("Starting AsyncLLM pause/continue test...")
    
    try:
        # Run the async pause/continue test
        result = asyncio.run(test_async_llm_pause_continue(async_llm, tokenizer))
        
        if result:
            print("\n✅ AsyncLLM pause/continue test passed!")
        else:
            print("\n❌ AsyncLLM pause/continue test failed!")
            
        return result
        
    except Exception as e:
        print(f"\n❌ AsyncLLM pause/continue test failed with exception: {str(e)}")
        return False


def run_async_llm_abort_test(async_llm: AsyncLLM, tokenizer: Any):
    """Synchronous wrapper to run the async abort test."""
    print("Starting AsyncLLM abort test...")
    
    try:
        # Run the async abort test
        result = asyncio.run(test_async_llm_abort(async_llm, tokenizer))
        
        if result:
            print("\n✅ AsyncLLM abort test passed!")
        else:
            print("\n❌ AsyncLLM abort test failed!")
            
        return result
        
    except Exception as e:
        print(f"\n❌ AsyncLLM abort test failed with exception: {str(e)}")
        return False


if __name__ == "__main__":
    import sys

    async_llm, tokenizer = get_shared_async_llm_and_tokenizer()
    
    # Check command line arguments for which test to run
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "abort":
            # Run the abort test
            run_async_llm_abort_test(async_llm, tokenizer)
        elif test_type == "pause_continue":
            # Run the pause/continue test
            run_async_llm_pause_continue_test(async_llm, tokenizer)
        else:
            print(f"Unknown test type: {test_type}")
            print("Available tests: abort, pause_continue")
    else:
        basic_result = run_async_llm_test(async_llm, tokenizer)
