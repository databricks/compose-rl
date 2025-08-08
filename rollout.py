from compose_rl.algorithms.online.generation_utils.vllm_utils import init_process_group
from composer.utils import dist
import torch

import logging

logging.basicConfig(
    # Example of format string
    # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: composer.trainer.trainer: Using precision Precision.FP32
    # Including the PID and thread name to help with debugging dataloader workers and callbacks that spawn background
    # threads / processes
    format=
    f'[ROLLOUT]%(asctime)s: rank{dist.get_global_rank()}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s',
)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


MODEL_UPDATE_PORT=29600
EXPERIENCE_BUFFER_PORT=29601
NUM_INFERENCE_ENGINES=1
MAX_ITERATIONS=10

if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl") # 0, 1
    log.info(f"Hello from rank {dist.get_global_rank()}")
    rank = dist.get_global_rank() + 1
    log.info(f"Rank {rank}")
    model_update_group = None
    experience_buffer_group = None
    if dist.get_global_rank() == 0:
        log.info("Initializing model update process group") # 1
        model_update_group = init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{MODEL_UPDATE_PORT}",
            world_size=1 + NUM_INFERENCE_ENGINES,
            rank=rank,
            group_name="model_update_group",
        )
        experience_buffer_group = init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{EXPERIENCE_BUFFER_PORT}",
            world_size=1 + NUM_INFERENCE_ENGINES,
            rank=rank,
            group_name="experience_buffer_group",
        )

    # TODO: check to see if there's an update to the model weights, if there is update the weights
    # to make it sync, we will wait until there is a weight update
    if model_update_group is not None:
        t = torch.tensor([0]).to('cuda')
        dist.broadcast(group=model_update_group, src=0,tensor=t)
        log.info(f"Rank {dist.get_global_rank()} all gathered {t}")

    # TODO: start generating rollouts and put it in the experience buffer
    dist.barrier() # wait until the model update is complete



    if experience_buffer_group is not None:
        t = torch.tensor([6]).to('cuda')
        torch.distributed.broadcast(group=experience_buffer_group, src=1,tensor=t, async_op=True) # don't block, send it off and continue generating rollouts
        log.info(f"Rank {dist.get_global_rank()} Broadcasted experience{t}")

    

