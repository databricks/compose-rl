# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import re
import subprocess
import sys
import time

import ray
from omegaconf import OmegaConf as om

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


@ray.remote
class SyncActor:

    def __init__(self):
        log.info('SyncActor initialized')
        self.training_done = False

    def mark_done(self):
        log.info('mark_done called')
        self.training_done = True
        log.info('mark_done completed')
        return 'Done'

    def is_training_done(self):
        return self.training_done


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text.

    Handles both color codes and formatting like bold, underline, etc.

    Args:
        text (str): The input string potentially containing ANSI codes.
    """
    ansi_pattern = r'\x1b\[[0-9;]*[a-zA-Z]'
    return re.sub(ansi_pattern, '', text)


def parse_ray_local_ip(log_output: str) -> str:
    """Parse the local node IP from Ray runtime startup logs.

    Args:
        log_output (str): The complete log output from Ray startup

    Returns:
        str: The extracted local node IP address
    """
    # First clean the ANSI escape sequences
    cleaned_output = strip_ansi(log_output)

    # Then look for the IP
    ip_pattern = r'Local node IP: (\d+\.\d+\.\d+\.\d+)'
    match = re.search(ip_pattern, cleaned_output)

    if match:
        return match.group(1)
    else:
        raise ValueError('No local node IP found in the logs')


def broadcast_string(message: str, src_rank: int):
    """Broadcast a string message from the source rank to all other ranks.

    Args:
        message (str): The message to broadcast
        src_rank (int): The rank of the source process
    """
    encoded = message.encode('utf-8')
    length_tensor = torch.LongTensor([len(encoded)])
    dist.broadcast(length_tensor, src=src_rank)

    data_tensor = torch.ByteTensor(list(encoded)) if dist.get_rank() == src_rank else \
                  torch.ByteTensor([0] * length_tensor.item()) # type: ignore
    dist.broadcast(data_tensor, src=src_rank)

    return data_tensor.cpu().numpy().tobytes().decode('utf-8')


def recv_string(src: int) -> str:
    """Receive the length of a string, then receive the actual bytes and decode.

    Args:
        src (int): The rank of the source process
    """
    length_tensor = torch.LongTensor([0])
    dist.recv(length_tensor, src=src)

    data_tensor = torch.ByteTensor(length_tensor.item())
    dist.recv(data_tensor, src=src)

    return data_tensor.numpy().tobytes().decode('utf-8')

def start_ray_nodes(num_inference_gpus):
    local_rank = os.getenv('LOCAL_RANK', None)
    assert local_rank is not None, 'LOCAL_RANK is usually set via composer'
    local_rank = int(local_rank)

    is_training_proc = os.getenv('TRAINING', None)
    
    if is_training_proc is not None and local_rank != 0:
        log.info('Not starting ray on non-master local rank, exiting.')
        return
    
    if local_rank == 0:
        log.info('starting train ray server')
        result = subprocess.run(
            ['ray', 'start', '--head', '--port=6379'],
            check=True,
            capture_output=True,
            text=True,
        )

    else:
        log.info('starting inference ray server')
        resources = json.dumps({"worker_node": num_inference_gpus})
        result = subprocess.run(
            ['ray', 'start', '--address=127.0.0.1:6379', f"--resources={resources}"]
        )
    log.info('done starting ray server')


def reassign_train_and_inference_gpus(num_train_gpus, num_inference_gpus):
    local_rank = os.getenv('LOCAL_RANK', None)
    assert local_rank is not None, 'LOCAL_RANK is usually set via composer'
    local_rank = int(local_rank)

    if local_rank < num_train_gpus:
        os.environ['LOCAL_WORLD_SIZE'] = str(num_train_gpus)
        os.environ['WORLD_SIZE'] = str(num_train_gpus)
        os.environ['TRAINING'] = '1'
        os.environ['NUM_INFERENCE_GPUS'] = str(num_inference_gpus)
    
    else:
        log.info('Reassigning env vars for inference')
        # TODO: figure out if we need to mess around with this more...
        # We should set the visible divices on inference nodes so they only see the inference GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{os.getenv('LOCAL_RANK')}"
        log.info(f"Set inference visible devices to: {os.environ['CUDA_VISIBLE_DEVICES']}")
        os.environ['LOCAL_WORLD_SIZE'] = str(num_inference_gpus)
        os.environ['MASTER_PORT'] = str(40977)

if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]

    with open(yaml_path) as f:
        yaml_cfg = om.load(f)

    num_gpus = os.getenv('LOCAL_WORLD_SIZE', None)
    assert num_gpus is not None
    num_gpus = int(num_gpus)

    num_nodes = os.getenv('NUM_NODES', None)
    assert num_nodes is not None, 'NUM_NODES must be set'
    num_nodes = int(num_nodes)
    assert num_nodes == 1

    # Set the environment variables for the total number of nodes
    # since NUM_NODES is overridden by train_num_node
    os.environ['TOTAL_NUM_NODES'] = str(num_nodes)
    os.environ['TRAIN_NUM_NODES'] = str(num_nodes)

    num_train_gpus = yaml_cfg['variables']['num_train_gpus']  # type: ignore
    num_inference_gpus = num_gpus - num_train_gpus

    reassign_train_and_inference_gpus(num_train_gpus, num_inference_gpus)
    start_ray_nodes(num_inference_gpus)

    sync_actor = None
    if os.getenv('LOCAL_RANK', None) == '0':
         # Adding a ray sync actor on global rank 0 to make it work
        sync_actor = SyncActor.options(name='sync_actor',
                                       namespace='default').remote()

    log.info('After start ray nodes and sync actor')

    if os.getenv('TRAINING', None) is not None:
        from llmfoundry.command_utils import train_from_yaml

        train_from_yaml(yaml_path, args_list)
        log.info('After calling `train_from_yaml`.')
        if os.getenv('NODE_RANK', None) == '0' and os.getenv('LOCAL_RANK', None) == '0':
            status = ray.get(sync_actor.mark_done.remote())  # type: ignore
    else:
        # Wait until the actor is available
        while True:
            try:
                log.info('Trying to get sync actor on inference node.')
                sync_actor = ray.get_actor(
                    'sync_actor',
                    namespace='default',
                )
                log.info('Got sync actor on inference node.')
                break
            except ValueError:  # Actor not found
                time.sleep(1)  # Retry after a short delay
        while True:
            is_training_done = ray.get(sync_actor.is_training_done.remote())
            if is_training_done:
                break
            time.sleep(10)
        log.info('After waiting for training.')
