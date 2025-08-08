from datetime import timedelta
from composer.utils import dist
import torch

from compose_rl.algorithms.online.generation_utils.vllm_utils import init_process_group

import logging

logging.basicConfig(
    # Example of format string
    # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: composer.trainer.trainer: Using precision Precision.FP32
    # Including the PID and thread name to help with debugging dataloader workers and callbacks that spawn background
    # threads / processes
    format=
    f'[TRAIN]%(asctime)s: rank{dist.get_global_rank()}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s',
)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

MODEL_UPDATE_PORT=29600
EXPERIENCE_BUFFER_PORT=29601
NUM_INFERENCE_ENGINES=1
MAX_ITERATIONS=10

if __name__ == "__main__":
    # note: the smaller timeout seems to hold, doesn't matter which process gorup you set the timeout to
    torch.distributed.init_process_group(backend="nccl")
    log.info(f"Hello from rank {dist.get_global_rank()}")

    model_update_group = None
    experience_buffer_group = None
    if dist.get_global_rank() == 0:
        log.info("Initializing model update process group")
        model_update_group = init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{MODEL_UPDATE_PORT}",
            world_size=1 + NUM_INFERENCE_ENGINES,
            rank=0,
            group_name="model_update_group",
        )
        experience_buffer_group = init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{EXPERIENCE_BUFFER_PORT}",
            world_size=1 + NUM_INFERENCE_ENGINES,
            rank=0,
            group_name="experience_buffer_group",
        )

    # TODO: broadcast the model weights to the inference engines
    if model_update_group is not None:
        t = torch.tensor([5]).to('cuda')
        torch.distributed.broadcast(group=model_update_group, src=0,tensor=t, async_op=True) # broadcast all the model weights
        log.info(f"Rank {dist.get_global_rank()} Broadcasted model weights{t}")

    # TODO: get the experience buffer results from the rollout process
    if experience_buffer_group is not None:
        t = torch.tensor([0]).to('cuda')
        torch.distributed.broadcast(group=experience_buffer_group, src=1,tensor=t) # block until the broadcast is complete
        log.info(f"Rank {dist.get_global_rank()} Broadcasted experience{t}")

    # all training ranks should wait until we have the experience buffer results
    dist.barrier()

    # distributed the experiences results to each of the training ranks

    # TODO: train the model

