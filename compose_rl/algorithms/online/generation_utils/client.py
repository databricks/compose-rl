import os
import time
import uuid
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Set, Union

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

from .sglang_remote import GenerationHyperparameters, ModelRequest, ModelResponse, RemoteSGLangEngine
from .tool_call_parser import process_tool_calls

from transformers import PreTrainedTokenizerFast


# reset OpenAI keys when using the wrapped client.
os.environ["OPENAI_API_KEY"] = "none"
os.environ["OPENAI_BASE_URL"] = "none"


from dataclasses import dataclass, field

import torch


@dataclass
class CompletionWithTokenLogp:
    """Internal structure to store completion with its logprobs."""

    completion: ChatCompletion
    response: ModelResponse
    messages: list[dict] = field(default_factory=list)

    def to_tensor_dict(self) -> dict[str, torch.Tensor]:
        resp = self.response
        seq = resp.input_tokens + resp.output_tokens
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
        versions = [-1] * resp.input_len + resp.output_versions
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
    """Extended AsyncCompletions that adds caching and reward functionality."""

    # Class-level set to track which parameters have been warned about (shared across all instances)
    _warned_parameters: Set[str] = set()

    def __init__(
        self,
        client: AsyncOpenAI,
        engine: RemoteSGLangEngine,
        tokenizer: PreTrainedTokenizerFast,
        cache: Dict[str, CompletionWithTokenLogp],
        tool_call_parser: Optional[str] = None,
    ):
        super().__init__(client)
        self.engine = engine
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
        """Override create method to use AReaL engine and cache responses."""
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
        stop_tokens = None if stop is NOT_GIVEN else stop

        if frequency_penalty is NOT_GIVEN or frequency_penalty is None:
            frequency_penalty = 0.0

        # Create generation config
        gconfig = GenerationHyperparameters(
            n_samples=1,
            temperature=temp,
            max_new_tokens=max_new_tokens,
            top_p=top_p_val,
            stop=(
                stop_tokens
                if isinstance(stop_tokens, list)
                else [stop_tokens] if stop_tokens else None
            ),
            greedy=temp == 0,
            frequency_penalty=frequency_penalty,
            stop_token_ids=list(
                set([self.tokenizer.eos_token_id, self.tokenizer.pad_token_id])
            ),
        )

        model_request = ModelRequest(
            input_ids=prompt_token_ids,
            gconfig=gconfig,
            rid=str(uuid.uuid4()),
            metadata=metadata,
            tokenizer=self.tokenizer,
        )

        # Call inference engine
        response = await self.engine.agenerate(model_request)

        # Convert response to OpenAI format
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        current_time = int(time.time())

        output_text = self.tokenizer.decode(response.output_tokens)

        # Parse tool calls.
        tool_calls = None
        if tool_choice != "none" and tools:
            tool_calls, output_text, response.stop_reason = process_tool_calls(
                output_text,
                tools,
                self.tool_call_parser,
                response.stop_reason,
            )

        # Create proper ChatCompletion object with all required fields
        chat_completion = ChatCompletion(
            id=completion_id,
            choices=[
                Choice(
                    finish_reason=response.stop_reason,
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
                completion_tokens=len(response.output_tokens),
                prompt_tokens=len(response.input_tokens),
                total_tokens=len(response.input_tokens) + len(response.output_tokens),
            ),
        )

        if store is NOT_GIVEN or store:
            # Cache the completion with its input messages
            self._cache[completion_id] = CompletionWithTokenLogp(
                completion=deepcopy(chat_completion),
                response=response,  # Should not deepcopy response because of tokenizer
                messages=deepcopy(messages_list),  # Store a copy of the input messages
            )
        return chat_completion


class ArealOpenAI(AsyncOpenAI):
    """Extended AsyncOpenAI client that uses AReaL's inference engine and supports reward setting."""

    def __init__(
        self,
        engine: RemoteSGLangEngine,
        tokenizer: PreTrainedTokenizerFast,
        tool_call_parser: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.engine = engine
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser
        self._completion_cache: Dict[str, CompletionWithTokenLogp] = {}

        # Override chat.completions with our extended implementation
        self.chat.completions = AsyncCompletionsWithReward(
            self,
            engine,
            tokenizer,
            self._completion_cache,
            tool_call_parser=self.tool_call_parser,
        )

    def get_completions(
        self, completion_id: str
    ) -> Optional[CompletionWithTokenLogp]:
        """Get completion from cache."""
        return self._completion_cache.get(completion_id)


