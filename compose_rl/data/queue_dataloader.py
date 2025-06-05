# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Ray queue-based dataloader for prompt data."""

import logging
import threading
import time
from typing import Any, Optional

import ray

log = logging.getLogger(__name__)


# Ray queue supports actor_options for naming - no custom wrapper needed


@ray.remote
class PromptDataProducer:
    """Ray actor that produces prompt data and populates a distributed queue."""
    
    def __init__(
        self,
        tokenizer_config: dict[str, Any],  # Pass config instead of object
        device_batch_size: int,
        dataset_config: dict[str, Any],
        max_seq_len: int,
        queue_name: str = "prompt_queue",
        namespace: str = "default",
        preload_batches: int = 100,
    ):
        """Initialize the data producer.
        
        Args:
            tokenizer_config: Tokenizer configuration (to recreate tokenizer)
            device_batch_size: Batch size per device
            dataset_config: Dataset configuration 
            max_seq_len: Maximum sequence length
            queue_name: Name of the Ray queue to populate
            namespace: Ray namespace for the queue
            preload_batches: Number of batches to preload in queue
        """
        # Store lightweight config instead of heavy objects
        self.tokenizer_config = tokenizer_config
        self.device_batch_size = device_batch_size
        self.dataset_config = dataset_config
        self.max_seq_len = max_seq_len
        self.queue_name = queue_name
        self.namespace = namespace
        self.preload_batches = preload_batches
        self.running = False
        self.producer_thread = None
        self.dataloader = None
        
        # Setup queue - this is lightweight
        self._setup_queue()
    
    def _setup_queue(self):
        """Setup the Ray queue."""
        from ray.util.queue import Queue
        
        try:
            self.queue = ray.get_actor(self.queue_name, namespace=self.namespace)
            log.info(f"Connected to existing queue: {self.queue_name}")
        except ValueError:
            # Queue doesn't exist, create it with proper actor options
            max_queue_size = self.preload_batches * 2  # Allow some buffer
            
            # Use Ray's built-in queue with actor_options for naming
            actor_options = {
                "name": self.queue_name,
                "namespace": self.namespace
            }
            
            self.queue = Queue(
                maxsize=max_queue_size,
                actor_options=actor_options
            )
            
            log.info(f"Created new queue: {self.queue_name} with max size {max_queue_size}")
    
    def _setup_dataloader(self):
        """Setup the underlying streaming dataloader - done lazily when needed."""
        # Import heavy dependencies only when needed
        from functools import partial
        import torch
        from streaming import Stream, StreamingDataLoader
        from transformers import AutoTokenizer
        from compose_rl.data.prompt_data import (
            PromptStreamingDataset,
            prompt_dataset_collate_fn,
        )
        
        # Recreate tokenizer from config
        tokenizer = AutoTokenizer.from_pretrained(**self.tokenizer_config)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        dataset_cfg = self.dataset_config.copy()
        streams_dict = dataset_cfg.pop('streams', None)
        
        # Build streams
        streams = None
        if streams_dict is not None:
            streams = [Stream(**stream) for stream in streams_dict.values()]
        
        streaming_dataset = PromptStreamingDataset(
            streams=streams,
            batch_size=self.device_batch_size,
            **dataset_cfg,
        )
        
        self.dataloader = StreamingDataLoader(
            streaming_dataset,
            collate_fn=partial(prompt_dataset_collate_fn, tokenizer, self.max_seq_len),
            batch_size=self.device_batch_size,
            drop_last=True,
            num_workers=4,  # Use multiple workers for the producer
            pin_memory=False,  # Don't pin memory on producer side
            prefetch_factor=2,
            persistent_workers=True,
            timeout=0,
        )
        log.info("Setup dataloader for prompt data producer")
    
    def start_producing(self):
        """Start producing data in a background thread."""
        if self.running:
            log.warning("Producer already running")
            return
        
        # Setup dataloader now (lazy initialization)
        if self.dataloader is None:
            self._setup_dataloader()
        
        self.running = True
        self.producer_thread = threading.Thread(target=self._produce_data)
        self.producer_thread.daemon = True
        self.producer_thread.start()
        log.info("Started prompt data producer thread")
    
    def stop_producing(self):
        """Stop the data producer."""
        self.running = False
        if self.producer_thread:
            self.producer_thread.join(timeout=10)
        log.info("Stopped prompt data producer")
    
    def _produce_data(self):
        """Main producer loop - runs in background thread."""
        log.info("Starting data production loop")
        batch_count = 0
        
        while self.running:
            try:
                for batch in self.dataloader:
                    if not self.running:
                        break
                    
                    # Put batch in queue (this will block if queue is full)
                    ray.get(self.queue.put.remote(batch))
                    batch_count += 1
                    
                    if batch_count % 10 == 0:
                        queue_size = ray.get(self.queue.qsize.remote())
                        log.debug(f"Produced {batch_count} batches, queue size: {queue_size}")
                
                # If we reach here, dataloader is exhausted
                log.info(f"Dataloader exhausted after {batch_count} batches")
                break
                
            except Exception as e:
                log.error(f"Error in data producer: {e}")
                time.sleep(1)  # Brief pause before retry
        
        log.info(f"Data producer finished. Total batches produced: {batch_count}")
    
    def get_queue_size(self):
        """Get current queue size."""
        return ray.get(self.queue.qsize.remote())
    
    def is_running(self):
        """Check if producer is running."""
        return self.running


class QueuePromptDataLoader:
    """Dataloader that pulls prompt batches from a Ray queue."""
    
    def __init__(
        self,
        queue_name: str = "prompt_queue",
        namespace: str = "default", 
        timeout: Optional[float] = None,
    ):
        """Initialize queue-based dataloader.
        
        Args:
            queue_name: Name of the Ray queue to pull from
            namespace: Ray namespace for the queue
            timeout: Timeout for queue operations (None = block indefinitely)
        """
        self.queue_name = queue_name
        self.namespace = namespace
        self.timeout = timeout
        self.queue = None
        
    def _get_queue(self):
        """Get reference to the Ray queue."""
        if self.queue is None:
            # Wait for queue to be available
            max_retries = 60  # Wait up to 60 seconds
            for i in range(max_retries):
                try:
                    self.queue = ray.get_actor(self.queue_name, namespace=self.namespace)
                    log.info(f"Connected to queue: {self.queue_name}")
                    break
                except ValueError:
                    if i == max_retries - 1:
                        raise RuntimeError(f"Queue {self.queue_name} not found after {max_retries} seconds")
                    time.sleep(1)
        return self.queue
    
    def get_batch(self):
        """Get a single batch from the queue.
        
        Returns:
            Batch dictionary or None if queue is empty and timeout reached
        """
        queue = self._get_queue()
        
        try:
            if self.timeout is not None:
                batch = ray.get(queue.get.remote(block=True, timeout=self.timeout))
            else:
                batch = ray.get(queue.get.remote(block=True))
            return batch
        except Exception as e:
            log.warning(f"Failed to get batch from queue: {e}")
            return None
    
    def __iter__(self):
        """Iterator interface for the dataloader."""
        return self
    
    def __next__(self):
        """Get next batch from queue."""
        batch = self.get_batch()
        if batch is None:
            raise StopIteration
        return batch
    
    def qsize(self) -> int:
        """Get current queue size."""
        queue = self._get_queue()
        return ray.get(queue.qsize.remote())


def build_queue_prompt_dataloader(
    tokenizer,  # Can be tokenizer object or config
    device_batch_size: int,
    dataset: dict[str, Any],
    drop_last: bool = True,
    num_workers: int = 0,  # Ignored for queue dataloader
    pin_memory: bool = True,  # Ignored for queue dataloader  
    prefetch_factor: int = 2,  # Ignored for queue dataloader
    persistent_workers: bool = True,  # Ignored for queue dataloader
    timeout: int = 0,
    queue_name: str = "prompt_queue",
    namespace: str = "default",
    preload_batches: int = 100,
    **kwargs,
) -> QueuePromptDataLoader:
    """Build a queue-based prompt dataloader.
    
    Args:
        tokenizer: The model's tokenizer
        device_batch_size: Batch size per device
        dataset: Dataset configuration
        drop_last: Whether to drop last batch (compatibility, ignored)
        num_workers: Number of workers (compatibility, ignored)
        pin_memory: Whether to pin memory (compatibility, ignored)
        prefetch_factor: Prefetch factor (compatibility, ignored)
        persistent_workers: Whether to use persistent workers (compatibility, ignored)
        timeout: Timeout for queue operations
        queue_name: Name of the Ray queue
        namespace: Ray namespace 
        preload_batches: Number of batches to preload
        **kwargs: Additional arguments (ignored for compatibility)
    
    Returns:
        QueuePromptDataLoader instance
    """
    # Import here to avoid circular imports and ensure Ray is available
    from composer.utils import dist
    
    max_seq_len = dataset.get('max_seq_len')
    if max_seq_len is None:
        raise ValueError('max_seq_len must be provided in the dataset configuration')
    
    # Only global rank 0 starts the producer
    is_producer_rank = dist.get_global_rank() == 0
    
    # Start producer if this is the producer rank
    if is_producer_rank:
        log.info("Starting prompt data producer on global rank 0")
        
        # Extract tokenizer config to pass to actor (avoid heavy object serialization)
        if hasattr(tokenizer, 'name_or_path'):
            tokenizer_config = {
                'pretrained_model_name_or_path': tokenizer.name_or_path,
                'model_max_length': getattr(tokenizer, 'model_max_length', None),
                'padding_side': getattr(tokenizer, 'padding_side', 'right'),
                'trust_remote_code': True,
            }
        else:
            # Fallback for cases where tokenizer doesn't have name_or_path
            tokenizer_config = {
                'pretrained_model_name_or_path': 'gpt2',  # Default fallback
                'model_max_length': dataset.get('max_seq_len', 2048),
                'padding_side': 'left',
                'trust_remote_code': True,
            }
        
        producer = PromptDataProducer.options(
            name="prompt_producer",
            namespace=namespace,
            num_cpus=2,  # Give producer some CPU resources
        ).remote(
            tokenizer_config=tokenizer_config,
            device_batch_size=device_batch_size,
            dataset_config=dataset,
            max_seq_len=max_seq_len,
            queue_name=queue_name,
            namespace=namespace,
            preload_batches=preload_batches,
        )
        
        # Start the producer
        ray.get(producer.start_producing.remote())
        log.info("Producer started successfully")
    else:
        log.info(f"Rank {dist.get_global_rank()} will consume from queue: {queue_name}")
    
    # Return the consumer dataloader
    return QueuePromptDataLoader(
        queue_name=queue_name,
        namespace=namespace,
        timeout=timeout if timeout > 0 else None,
    )