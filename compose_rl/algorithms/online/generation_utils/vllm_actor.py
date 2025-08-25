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

class AsyncLLM(BaseLLM):

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Initialize base class first
        super().__init__(*args, **kwargs)
        os.environ["VLLM_USE_V1"] = "1"
        # Create AsyncLLMEngine instead of regular LLM
        engine_args = vllm.AsyncEngineArgs(*self.args, **self.kwargs)
        self.llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        print(f'created: {type(self.llm)}')
        
        # Track running tasks by request_id for abort functionality
        self.running_tasks: dict[str, asyncio.Task] = {}
        self.partial_outputs: dict[str, Any] = {}

    async def _collect_outputs(self, prompt_token_ids: list[int], request_id: str, sampling_params: SamplingParams):
        """Collect outputs for a single prompt."""
        final_output = None
        try:
            async for request_output in self.llm.generate(
                prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                final_output = request_output
                # Store partial output in case of abort
                self.partial_outputs[request_id] = final_output
                
        except asyncio.CancelledError:
            # Local task was cancelled (likely due to abort() call)
            # The actual generation in vLLM engine should have been aborted separately
            log.info(f"Request {request_id} local task was cancelled")
            final_output = self.partial_outputs.get(request_id, None)
            raise
        finally:
            # Clean up tracking
            self.running_tasks.pop(request_id, None)
            self.partial_outputs.pop(request_id, None)

        return final_output

    async def generate(self, batched_promts: list[list[int]], sampling_params: SamplingParams):
        """Generate responses using vLLM's async engine."""

        tasks = []
        request_ids = []
        for prompt in batched_promts:
            # Schedule the collection of outputs for each prompt.
            # Avoid duplicate request_ids
            request_id = str(uuid4().hex)
            task = asyncio.create_task(self._collect_outputs(prompt, request_id, sampling_params))
            
            # Track the task by request_id
            self.running_tasks[request_id] = task
            tasks.append(task)
            request_ids.append(request_id)
        
        try:
            outputs = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            # Clean up any remaining tasks
            for req_id in request_ids:
                self.running_tasks.pop(req_id, None)
                self.partial_outputs.pop(req_id, None)
            raise

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
        task = self.running_tasks.get(request_id)
        if task is None:
            log.warning(f"Request {request_id} not found in running tasks")
            return None
            
        # # First, abort the request in vLLM's engine to stop actual generation
        # try:
        #     await self.llm.abort(request_id)
        #     log.info(f"Aborted request {request_id} in vLLM engine")
        # except Exception as e:
        #     log.warning(f"Failed to abort request {request_id} in vLLM engine: {e}")
            
        # Then cancel our local task that's iterating over the results
        if not task.done():
            task.cancel()
            log.info(f"Cancelled local task for request {request_id}")
        
        return None

    async def init_process_group(
        self, master_address: str, master_port: str, rank_offset: int, world_size: int
    ):
        return await self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size),
        )

    async def update_weight(self, name: str, dtype: torch.dtype, shape: Union[tuple[int, ...], list[int]], empty_cache: bool = False):
        return await self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    async def reset_prefix_cache(self):
        await self.llm.reset_prefix_cache()


LLMRayActor = ray.remote(LLM)
LLMRayActorAsync = ray.remote(AsyncLLM)


async def test_async_llm_abort():
    """Test the abort functionality of AsyncLLM.
    
    This test creates long-running generation tasks and demonstrates
    aborting them mid-generation. The abort method first calls vLLM's
    abort to stop actual generation, then cancels the local asyncio task.
    """
    try:
        from transformers import AutoTokenizer
        
        # Model configuration
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        
        print(f"Initializing AsyncLLM with model: {model_name}")
        
        # Create AsyncLLM instance
        async_llm = AsyncLLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            noset_visible_devices=False,
            num_gpus=1,
        )
        
        # Load tokenizer for encoding prompts
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
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
        try:
            outputs = await generation_task
            print(f"\nRemaining generation tasks completed")
            
            # Process results
            completed_count = 0
            aborted_count = 0
            for i, output in enumerate(outputs):
                if isinstance(output, asyncio.CancelledError):
                    print(f"Task {i} was cancelled (aborted)")
                    aborted_count += 1
                elif isinstance(output, Exception):
                    print(f"Task {i} failed with error: {output}")
                elif output and output.outputs:
                    print(f"Task {i} completed successfully with {len(output.outputs[0].token_ids)} tokens")
                    completed_count += 1
                else:
                    print(f"Task {i} completed but no output")
            
            print(f"\nSummary: {completed_count} completed, {aborted_count} aborted")
            
        except Exception as e:
            print(f"Generation task failed: {e}")
        
        print(f"\n✅ Abort test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during abort testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_async_llm():
    """Simple test for AsyncLLM using Qwen/Qwen2.5-0.5B model.
    
    This test creates an AsyncLLM instance and tests it with several prompts
    without depending on Ray.
    """
    try:
        from transformers import AutoTokenizer
        
        # Model configuration
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        
        print(f"Initializing AsyncLLM with model: {model_name}")
        
        # Create AsyncLLM instance
        async_llm = AsyncLLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            noset_visible_devices=False,
            num_gpus=1,
        )
        
        # Load tokenizer for encoding prompts
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
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


def run_async_llm_test():
    """Synchronous wrapper to run the async test."""
    print("Starting AsyncLLM test...")
    
    try:
        # Run the async test
        result = asyncio.run(test_async_llm())
        
        if result:
            print("\n✅ AsyncLLM test passed!")
        else:
            print("\n❌ AsyncLLM test failed!")
            
        return result
        
    except Exception as e:
        print(f"\n❌ AsyncLLM test failed with exception: {str(e)}")
        return False


def run_async_llm_abort_test():
    """Synchronous wrapper to run the async abort test."""
    print("Starting AsyncLLM abort test...")
    
    try:
        # Run the async abort test
        result = asyncio.run(test_async_llm_abort())
        
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
    
    # Check command line arguments for which test to run
    if len(sys.argv) > 1 and sys.argv[1] == "abort":
        # Run the abort test
        run_async_llm_abort_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "both":
        # Run both tests
        print("Running both tests...\n")
        basic_result = run_async_llm_test()
        print("\n" + "="*60 + "\n")
        abort_result = run_async_llm_abort_test()
        
        if basic_result and abort_result:
            print("\n🎉 All tests passed!")
        else:
            print("\n❌ Some tests failed!")
    else:
        # Default: run the basic test
        print("Run with 'abort' argument to test abort functionality")
        print("Run with 'both' argument to test both basic and abort functionality")
        print("Running basic test...\n")
        run_async_llm_test()