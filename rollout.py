from compose_rl.algorithms.online.generation_utils.vllm_utils import init_process_group
from composer.utils import dist
import torch

import logging

MODEL_UPDATE_PORT=29600
EXPERIENCE_BUFFER_PORT=29601
NUM_INFERENCE_ENGINES=1
MAX_ITERATIONS=2

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

if __name__ == "__main__":

    rank = 1 # TODO: UPDATE TO SUPPORT MULTIPLE INFERENCE ENGINES
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
    is_ready_to_update = torch.tensor([0]).to('cuda')
    is_ready_to_update_work = None

    for i in range(MAX_ITERATIONS):
        log.info(f"[ITERATION {i + 1}/{MAX_ITERATIONS}] Starting iteration")
        
        if is_ready_to_update_work is None:
            # if we haven't checked if there's an update to the model weights, run the check in the background
            is_ready_to_update_work = torch.distributed.broadcast(group=model_update_group, src=0,tensor=is_ready_to_update, async_op=True)
            if i == 0:
                is_ready_to_update_work.wait() # We need to update the weights for the first iteration before we start generating rollouts

        if is_ready_to_update.item() == 1:
            assert is_ready_to_update_work.is_completed()
            log.info(f"Weights are ready to update")

            log.info("Updating the model weights") # this is a blocking operation, we need to wait until the weights are updated before we can start generating rollouts
            weights = torch.tensor([i]).to('cuda')
            torch.distributed.broadcast(group=model_update_group, src=0,tensor=weights)
            log.info(f"Updating the weights to {weights}")
            # rest the update check
            is_ready_to_update = torch.tensor([0]).to('cuda') 
            is_ready_to_update_work = None


        # TODO: start generating rollouts and put it in the experience buffer

        experience_buffer = torch.tensor([6]).to('cuda')
        experience_buffer_work = torch.distributed.broadcast(group=experience_buffer_group, src=1,tensor=experience_buffer, async_op=True) # don't block, send it off and continue generating rollouts
        log.info(f"Sent experience buffer {experience_buffer}")

        log.info(f"Completed iteration {i + 1}/{MAX_ITERATIONS}")

        if i == MAX_ITERATIONS - 1:
            assert experience_buffer_work is not None
            log.info(f"Waiting for the last experience buffer to be received")
            experience_buffer_work.wait()


    

