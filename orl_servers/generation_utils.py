import time
import asyncio
import logging
from typing import List, Any

import torch

from .structs import GenerationHyperparameters, ModelRequest, WeightUpdateMeta, ParamSpec
from .vllm_remote import RemoteVLLMEngine
from .async_utils import run_async_sync
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from composer.utils import dist as composer_dist
from compose_rl.algorithms.online.model_methods import OnPolicyEnum, ALGORITHM_TYPE
from compose_rl.algorithms.online.generation_utils.vllm_utils import (
    build_param_fullnames,
    simplify_param_path,
    should_update_torch_module,
)


log = logging.getLogger(__name__)


async def vllm_generate_async(
    remote_engine: RemoteVLLMEngine,
    max_gen_len: int,
    generation_kwargs: dict,
    pad_token_id: int,
    all_prompts: list,
    batch_sizes: list,
) -> tuple[list, list]:
    """Asynchronously generate completions using a RemoteVLLMEngine.

    Mirrors the return contract of the legacy `_vllm_generate` by returning a
    list of lists, where each inner list contains token-id sequences for the
    corresponding original minibatch shard (based on ``batch_sizes``).
    """
    # Normalize sampling params into GenerationHyperparameters
    top_k = generation_kwargs.get("top_k", -1)
    if top_k == -1:
        # Remote engine uses a very large number to effectively disable top-k
        top_k = int(1e8)

    gconfig_base = GenerationHyperparameters(
        n_samples=1,
        max_new_tokens=max_gen_len,
        min_new_tokens=generation_kwargs.get("min_new_tokens", 0),
        greedy=generation_kwargs.get("greedy", False),
        top_p=generation_kwargs.get("top_p", 1.0),
        top_k=top_k,
        temperature=generation_kwargs.get("temperature", 1.0),
        stop_token_ids=generation_kwargs.get("stop_token_ids", []),
        stop=generation_kwargs.get("stop"),
        frequency_penalty=generation_kwargs.get("frequency_penalty", 0.0),
        logprobs=generation_kwargs.get("logprobs", 1),
        prompt_logprobs=generation_kwargs.get("prompt_logprobs"),
    )

    # Remove pad tokens from all prompts
    cleaned_prompts: List[List[int]] = []
    for prompt in all_prompts:
        if isinstance(prompt, torch.Tensor):
            # TODO could speed this up with GPU by masking out pad tokens
            tokens = [
                token
                for token in prompt.detach().cpu().tolist()
                if token != pad_token_id
            ]
        else:
            tokens = [token for token in list(prompt) if token != pad_token_id]
        cleaned_prompts.append(tokens)

    # Dispatch one async generation per prompt
    tasks = []
    for prompt_ids in cleaned_prompts:
        req = ModelRequest(
            input_ids=prompt_ids,
            gconfig=gconfig_base,
        )
        tasks.append(asyncio.create_task(remote_engine.agenerate(req)))

    start_time = time.time()
    results = await asyncio.gather(*tasks)
    log.info(f'took: {time.time() - start_time} to gather async generations')

    # Flatten responses in submission order (one output per input)
    all_responses: List[List[int]] = [resp.output_tokens for resp in results]
    all_logprobs: List[List[float]] = [resp.output_logprobs for resp in results]

    # Distribute responses back to original device shards
    split_responses: List[list] = []
    split_logprobs: List[list] = []
    start = 0
    for size in batch_sizes:
        split_responses.append(all_responses[start:start + size])
        split_logprobs.append(all_logprobs[start:start + size])
        start += size
    return split_responses, split_logprobs


def vllm_generate_sync(
    remote_engine: RemoteVLLMEngine,
    max_gen_len: int,
    generation_kwargs: dict,
    pad_token_id: int,
    all_prompts: list,
    batch_sizes: list,
) -> tuple[list, list]:
    """Synchronous wrapper around `_remote_vllm_generate_async`.

    Keeps a similar call pattern as the original `_vllm_generate` while using
    the async ``agenerate`` under the hood for concurrency.
    """
    return run_async_sync(
        vllm_generate_async(
            remote_engine,
            max_gen_len,
            generation_kwargs,
            pad_token_id,
            all_prompts,
            batch_sizes,
        )
    )


async def setup_process_groups(
    master_actor: Any,
    vllm_engine: RemoteVLLMEngine,
    gen_tp_size: int,
    num_vllm_servers: int,
) -> None:
    """Initialize trainer and vLLM servers' weight-update process group.

    This mirrors the logic used in test_single_controller_vllm.py by:
      - Getting a free TCP port from the master actor
      - Initializing the vLLM servers' NCCL communicators via HTTP
      - Adding a matching process group on the trainer side (rank 0)
    """
    master_addr, _ = await master_actor.get_master_address.remote()  # type: ignore
    new_port = await master_actor.get_free_port.remote()  # type: ignore

    meta = WeightUpdateMeta(
        nccl_master_address=master_addr,
        nccl_master_port=new_port,
        gen_tp_size=gen_tp_size,
        gen_world_size=num_vllm_servers * gen_tp_size,
    )

    # Initialize both sides concurrently
    await asyncio.gather(
        vllm_engine.ainit_weight_update_group(meta),
        master_actor.add_process_group.remote(  # type: ignore
            backend='nccl',
            master_addr=master_addr,
            master_port=new_port,
            world_size=num_vllm_servers * gen_tp_size + 1,
            rank=0,
            group_name='vllm_weight_update',
        ),
    )

    return


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return 'float32'
    if dtype == torch.float16:
        return 'float16'
    if dtype == torch.bfloat16:
        return 'bfloat16'
    if dtype == torch.int8:
        return 'int8'
    if dtype == torch.int16:
        return 'int16'
    if dtype == torch.int32:
        return 'int32'
    if dtype == torch.int64:
        return 'int64'
    raise ValueError(f'Unsupported dtype for RemoteVLLMEngine update: {dtype}')


async def broadcast_to_vllm(
    model: torch.nn.Module,
    vllm_engine: RemoteVLLMEngine,
    device: torch.device,
    loss_type: OnPolicyEnum = OnPolicyEnum.PPO,
    enable_prefix_caching: bool = False,
) -> None:
    """Broadcast model weights (FSDP-aware) to RemoteVLLMEngine servers.

    Mirrors compose_rl.algorithms.online.generation_utils.vllm_utils.broadcast_to_vllm,
    but uses RemoteVLLMEngine HTTP endpoints paired with concurrent NCCL broadcasts
    driven by the trainer master actor.
    """
    # Avoid OOM
    torch.cuda.empty_cache()

    if loss_type == OnPolicyEnum.PPO:
        num_params = len(list(model.model.lm_backbone.named_parameters()))  # type: ignore
    elif loss_type in ALGORITHM_TYPE.CRITIC_FREE:
        num_params = len(list(model.model.named_parameters()))  # type: ignore
    else:
        raise ValueError(f'Unsupported loss type: {loss_type}')

    cache_reset_task = None
    if enable_prefix_caching and composer_dist.get_global_rank() == 0:
        cache_reset_task = asyncio.create_task(vllm_engine.areset_prefix_cache())

    valid_non_leaf_module_names = [
        'model.embed_tokens.weight',
        'lm_head.weight',
        'model.norm.weight',
    ]
    seen_fsdp_modules = set()
    seen_updated_parsed_names = set()
    count = 0
    param_2_full_name = build_param_fullnames(model)

    # Dummy forward to satisfy FSDP state
    with torch.no_grad():
        dummy_batch = {
            'obs': torch.tensor([[0]], dtype=torch.long, device=device),
            'right_padded_attn_mask': torch.tensor([[1]], dtype=torch.bool, device=device),
            'actions': torch.tensor([[0]], dtype=torch.long, device=device),
            'prompt_len': torch.tensor([1], device=device),
            'max_gen_len': torch.tensor([1], device=device),
            'action_mask': torch.tensor([[0]], dtype=torch.long, device=device),
        }
        model(dummy_batch)

    start_time = time.time()
    update_time = 0.0

    for module_name, module in model.named_modules():
        if isinstance(module, FSDP):
            if module_name == 'model' and loss_type == OnPolicyEnum.PPO:
                continue
            if module in seen_fsdp_modules:
                continue
            seen_fsdp_modules.add(module)

            # Materialize and iterate params
            with FSDP.summon_full_params(module, writeback=False, rank0_only=True, recurse=False):
                for _, param in module.named_parameters(recurse=True):
                    full_name = param_2_full_name[param]
                    parsed_name = simplify_param_path(full_name)

                    if 'critic_head' in parsed_name:
                        continue

                    update = should_update_torch_module(
                        parsed_name,
                        full_name,
                        module,
                        loss_type,
                        valid_non_leaf_module_names,
                    )
                    if not update or parsed_name in seen_updated_parsed_names:
                        continue

                    seen_updated_parsed_names.add(parsed_name)
                    count += 1
                    shape = tuple(param.shape)
                    dtype_str = _torch_dtype_to_str(param.dtype)

                    spec = ParamSpec(name=parsed_name, shape=shape, dtype=dtype_str)

                    # Only root trainer rank coordinates HTTP + broadcast
                    # TODO assert model_update_group exists and is not None
                    if hasattr(model, 'model_update_group') and getattr(model, 'model_update_group') is not None and composer_dist.get_global_rank() == 0:
                        # Launch HTTP and NCCL broadcast concurrently.
                        http_task = asyncio.create_task(vllm_engine.aupdate_weight(spec))

                        # TODO try just def this as async w/o using thread as http_task is always run earlier in gather
                        def _do_broadcast():
                            # Use the vLLM communicator exposed on the model (set by the actor)
                            communicator = getattr(model, 'model_update_group')
                            communicator.broadcast(param.data, src=0, stream=torch.cuda.current_stream())

                        bcast_task = asyncio.to_thread(_do_broadcast)
                        await asyncio.gather(http_task, bcast_task)
                        update_time += time.time() - start_time

    log.info(f'for loop took: {time.time() - start_time}')
    log.info(f'update time is: {update_time}')
    log.info(f'number of parameters updated is: {count}')

    if enable_prefix_caching and cache_reset_task is not None:
        # ensure cache reset completed
        await cache_reset_task

    # Global-rank-0 sanity check mirroring original implementation
    if composer_dist.get_global_rank() == 0:
        assert num_params == count, (
            f'Number of parameters updated {count} does not match the number of parameters {num_params}'
            + f'This means that some parameters were not updated.'
        )

    return

