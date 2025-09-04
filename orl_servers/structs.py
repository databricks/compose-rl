
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal
import uuid

import torch
import numpy as np
from transformers import PreTrainedTokenizerFast



@dataclass
class InferenceEngineConfig:
    """Configuration for inference engine settings."""
    
    setup_timeout: float = 60.0  # Timeout in seconds for server setup/initialization
    request_timeout: float = 300.0  # Timeout in seconds for HTTP requests
    request_retries: int = 3  # Maximum number of retry attempts for requests


@dataclass
class GenerationHyperparameters:
    """Controls text generation behavior for RL training."""

    n_samples: int = field(
        default=1, metadata={"help": "Number of sequences to generate per prompt."}
    )
    max_new_tokens: int = field(
        default=16384, metadata={"help": "Maximum number of tokens to generate."}
    )
    min_new_tokens: int = field(
        default=0, metadata={"help": "Minimum number of tokens to generate."}
    )
    greedy: bool = field(
        default=False,
        metadata={"help": "Whether to use greedy decoding (max probability)."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Nucleus sampling probability threshold (0.0, 1.0]."},
    )
    top_k: int = field(
        default=int(1e8),
        metadata={"help": "Number of highest probability tokens to consider."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature. Higher values increase diversity."},
    )
    stop_token_ids: list[int] = field(
        default_factory=list,
        metadata={"help": "Stop generation when encoutering these token ids."},
    )
    stop: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "One or multiple stop words. Generation will stop if one of these words is sampled."
        },
    )
    frequency_penalty: float = field(
        default=0.0,
        metadata={
            "help": (
                "Penalizes tokens based on their frequency in generation so far. "
                "Must be between -2 and 2 where negative numbers encourage repeatment."
            )
        },
    )
    # vLLM needs following two fields to be set to non-None to return logprobs
    logprobs: int | None = field(
        default=1,
        metadata={"help": "Number of log probabilities to return per output token."},
    )
    prompt_logprobs: int | None = field(
        default=None,
        metadata={"help": "Number of log probabilities to return per prompt token."},
    )

@dataclass
class ModelRequest:
    rid: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_ids: list[int] = field(default_factory=list)
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    metadata: Dict[str, Any] = field(default_factory=dict)
    # tokenizer is used for encode-decode in the inference engine
    tokenizer: Optional[PreTrainedTokenizerFast] = None


@dataclass
class ModelResponse:
    # outputs
    input_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    output_logprobs: list[float] = field(default_factory=list)
    # what's this used for? tracking num of updates of the model?
    # output_versions: list[int] = field(default_factory=list)
    stop_reason: Literal["length", "stop", "interrupt"] = "stop"
    # tokenizer is used for encode-decode in the inference engine
    tokenizer: Optional[PreTrainedTokenizerFast] = None

    # statistics
    latency: float = float("inf")
    ttft: float = float("inf")  # Time to first token
    itl: list[float] = field(default_factory=list)  # List of inter-token latencies

    @property
    def input_len(self) -> int:
        return len(self.input_tokens)

    @property
    def output_len(self) -> int:
        return len(self.output_tokens)


@dataclass
class ParamSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def size(self) -> int:
        """Param bytes"""
        return getattr(torch, self.dtype).itemsize * np.prod(self.shape)


@dataclass
class WeightUpdateMeta:
    nccl_master_address: str = "127.0.0.1"
    nccl_master_port: int = 29500
    nccl_group_name: str = "update_weight_group"
    gen_tp_size: int = 1
    gen_world_size: int = 1
