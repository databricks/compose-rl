from composer.utils import dist
import torch

from compose_rl.algorithms.online.generation_utils.vllm_utils import init_process_group

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
    f'[TRAIN]%(asctime)s: rank{dist.get_global_rank()}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s',
)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl")

    # Initialize the process groups for communication between train and rollout agents
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
            backend="gloo",
            init_method=f"tcp://localhost:{EXPERIENCE_BUFFER_PORT}",
            world_size=1 + NUM_INFERENCE_ENGINES,
            rank=0,
            group_name="experience_buffer_group",
        )

    for i in range(MAX_ITERATIONS):
        log.info(f"Starting iteration {i + 1}/{MAX_ITERATIONS}")
        
        if model_update_group is not None:
            # Let the rollout agent know that we're ready to update the model weights
            is_ready_to_update = torch.tensor([1]).to('cuda')
            torch.distributed.broadcast(group=model_update_group, src=0,tensor=is_ready_to_update) 
            log.info(f"Rank {dist.get_global_rank()} Broadcasted is_ready_to_update {is_ready_to_update}")

            # Broadcast the model weights
            weights = torch.tensor([10+i]).to('cuda')
            torch.distributed.broadcast(group=model_update_group, src=0,tensor=weights) 
            log.info(f"Rank {dist.get_global_rank()} Broadcasted model weights {weights}") 

        # Get the experience buffer results from the rollout process
        experience_buffer = torch.tensor([0])
        if experience_buffer_group is not None:
            torch.distributed.broadcast(group=experience_buffer_group, src=1,tensor=experience_buffer) 
            log.info(f"Rank {dist.get_global_rank()} Got experience buffer {experience_buffer}")

        # all training ranks should wait until we have the experience buffer results
        dist.barrier()

        # TODO: distributed the experiences results to each of the training ranks
        # TODO: train the model 

        # simulate "long training!""
        import time
        time.sleep(20) 


        log.info(f"Completed iteration {i + 1}/{MAX_ITERATIONS}")

