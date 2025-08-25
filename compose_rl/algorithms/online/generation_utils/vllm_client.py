import os
import time
import uuid
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Set, Union
from dataclasses import dataclass, field

import torch
from openai import AsyncOpenAI
from openai._types import NOT_GIVEN, Body, NotGiven
from openai.resources.chat.completions.completions import (
    AsyncCompletions as BaseAsyncCompletions,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.shared_params.metadata import Metadata

from transformers import PreTrainedTokenizerFast
from vllm import SamplingParams
from vllm.outputs import RequestOutput

from .vllm_actor import AsyncLLM
from .tool_call_parser import process_tool_calls


# reset OpenAI keys when using the wrapped client.
os.environ["OPENAI_API_KEY"] = "none"
os.environ["OPENAI_BASE_URL"] = "none"


@dataclass
class CompletionWithTokenLogp:
    """Internal structure to store completion with its token logprobs."""

    completion: ChatCompletion
    request_output: RequestOutput
    messages: list[dict] = field(default_factory=list)

    def to_tensor_dict(self) -> dict[str, torch.Tensor]:
        """Convert to tensor format for training."""
        output = self.request_output.outputs[0] if self.request_output.outputs else None
        if output is None:
            raise ValueError("No output available in request_output")
        
        # Get token IDs and logprobs from vLLM output
        output_tokens = output.token_ids
        output_logprobs = [logprob_dict.logprob if logprob_dict else 0.0 
                          for logprob_dict in (output.logprobs or [])]
        
        # Create input tokens from prompt (we'll need to reconstruct this)
        # For now, create a placeholder - this would need to be stored separately
        input_tokens = getattr(self.request_output, 'prompt_token_ids', [])
        
        seq = input_tokens + output_tokens
        logprobs = [0.0] * len(input_tokens) + output_logprobs
        loss_mask = [0] * len(input_tokens) + [1] * len(output_tokens)
        versions = [-1] * len(input_tokens) + [-1] * len(output_tokens)  # vLLM doesn't have versions
        
        res = dict(
            # unsqueeze to add an additional batch dimension
            input_ids=torch.tensor(seq).unsqueeze(0),
            loss_mask=torch.tensor(loss_mask).unsqueeze(0),
            logprobs=torch.tensor(logprobs).unsqueeze(0),
            versions=torch.tensor(versions).unsqueeze(0),
            attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
        )
        return res


class AsyncCompletionsWithReward(BaseAsyncCompletions):
    """Extended AsyncCompletions that adds caching and reward functionality using vLLM."""

    # Class-level set to track which parameters have been warned about (shared across all instances)
    _warned_parameters: Set[str] = set()

    def __init__(
        self,
        client: AsyncOpenAI,
        async_engine: AsyncLLM,
        tokenizer: PreTrainedTokenizerFast,
        cache: Dict[str, CompletionWithTokenLogp],
        tool_call_parser: Optional[str] = None,
    ):
        super().__init__(client)
        self.async_engine = async_engine
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser
        self._cache = cache

    async def create(
        self,
        *,
        messages: Iterable[ChatCompletionMessageParam],
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        max_completion_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        metadata: Optional[Metadata] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] | NotGiven = NOT_GIVEN,
        store: Optional[bool] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        extra_body: Body | None = None,
    ) -> ChatCompletion:
        """Override create method to use vLLM AsyncLLM and cache responses."""
        # Extract and validate supported parameters
        messages_list = list(messages)
        if not messages_list:
            raise ValueError("messages cannot be empty")
        if extra_body is None:
            extra_body = {}
            
        # Convert messages to prompt format
        tools = tools if tools is not NOT_GIVEN else None
        prompt_token_ids = self.tokenizer.apply_chat_template(
            messages_list,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            **extra_body.get("chat_template_kwargs", {}),
        )

        # Process parameters for vLLM SamplingParams
        temp = 1.0 if temperature is NOT_GIVEN else (temperature or 0.0)
        max_new_tokens = 512
        
        if max_tokens is not NOT_GIVEN and max_tokens is not None:
            max_new_tokens = max_tokens - len(prompt_token_ids)
            if max_new_tokens <= 0:
                raise RuntimeError(
                    "max_tokens must be greater than the number of prompt tokens"
                )
                
        if max_completion_tokens is not NOT_GIVEN and max_completion_tokens is not None:
            max_new_tokens = min(max_new_tokens, max_completion_tokens)

        top_p_val = 1.0 if top_p is NOT_GIVEN else (top_p or 1.0)
        
        # Process stop tokens
        stop_tokens = None
        if stop is not NOT_GIVEN and stop is not None:
            if isinstance(stop, str):
                stop_tokens = [stop]
            else:
                stop_tokens = stop

        # Create vLLM SamplingParams
        sampling_params = SamplingParams(
            n=1,  # number of samples
            temperature=temp,
            max_tokens=max_new_tokens,
            top_p=top_p_val,
            stop=stop_tokens,
            frequency_penalty=frequency_penalty if frequency_penalty is not NOT_GIVEN else 0.0,
            stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.pad_token_id],
        )

        # Call vLLM AsyncLLM generate method (expects batch of prompts)
        request_outputs = await self.async_engine.generate([prompt_token_ids], sampling_params)
        
        if not request_outputs or not request_outputs[0].outputs:
            raise RuntimeError("No output generated from vLLM")
            
        request_output = request_outputs[0]
        completion_output = request_output.outputs[0]
        
        # Convert response to OpenAI format
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        current_time = int(time.time())

        # Decode the generated tokens
        output_text = self.tokenizer.decode(
            completion_output.token_ids, 
            skip_special_tokens=True
        )

        # Parse tool calls if needed
        tool_calls = None
        finish_reason = completion_output.finish_reason
        if tool_choice != "none" and tools:
            tool_calls, output_text, finish_reason = process_tool_calls(
                output_text,
                tools,
                self.tool_call_parser,
                finish_reason,
            )

        # Create proper ChatCompletion object with all required fields
        chat_completion = ChatCompletion(
            id=completion_id,
            choices=[
                Choice(
                    finish_reason=finish_reason,
                    index=0,
                    logprobs=None,  # For simplicity
                    message=ChatCompletionMessage(
                        content=output_text,
                        role="assistant",
                        tool_calls=tool_calls,
                    ),
                )
            ],
            created=current_time,
            model="None",
            object="chat.completion",
            service_tier=None,
            system_fingerprint=None,
            usage=CompletionUsage(
                completion_tokens=len(completion_output.token_ids),
                prompt_tokens=len(prompt_token_ids),
                total_tokens=len(prompt_token_ids) + len(completion_output.token_ids),
            ),
        )

        if store is NOT_GIVEN or store:
            # Cache the completion with its input messages
            # Store prompt_token_ids in request_output for later use
            request_output.prompt_token_ids = prompt_token_ids
            self._cache[completion_id] = CompletionWithTokenLogp(
                completion=deepcopy(chat_completion),
                request_output=request_output,  # Store the vLLM output
                messages=deepcopy(messages_list),  # Store a copy of the input messages
            )
            
        return chat_completion


class VllmOpenAI(AsyncOpenAI):
    """Extended AsyncOpenAI client that uses vLLM's AsyncLLM engine and supports reward setting."""

    def __init__(
        self,
        async_engine: AsyncLLM,
        tokenizer: PreTrainedTokenizerFast,
        tool_call_parser: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.async_engine = async_engine
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser
        self._completion_cache: Dict[str, CompletionWithTokenLogp] = {}

        # Override chat.completions with our extended implementation
        self.chat.completions = AsyncCompletionsWithReward(
            self,
            async_engine,
            tokenizer,
            self._completion_cache,
            tool_call_parser=self.tool_call_parser,
        )

    def get_completions(
        self, completion_id: str
    ) -> Optional[CompletionWithTokenLogp]:
        """Get completion from cache."""
        return self._completion_cache.get(completion_id)
