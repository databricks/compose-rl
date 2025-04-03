from abc import ABC, abstractmethod
import re
from typing import MutableMapping, Optional, Any

from compose_rl.reward_learning.base_reward import Tokenizer

def undo_llama3_chat_template(text: str) -> list[dict[str, str]]:
    messages = []
    # Regular expression to match the role and content
    pattern = re.compile(r"<\|start_header_id\|>(.*?)<\|end_header_id\|>\n(.*?)<\|eot_id\|>", re.DOTALL)
    for match in pattern.finditer(text):
        role = match.group(1).strip()
        content = match.group(2).strip()
        messages.append({"role": role, "content": content})
    return messages


class PGRMFormatter(ABC):
    """
    Abstract base class for formatting inputs to the prompt-guided reward model.
    """
    # The default prompt template for the prompt-guided reward model. Feel free to override this in subclasses.
    PGRM_PROMPT_TEMPLATE = """
You are an AI judge. Tell me if this response is good (yes) or bad (no).

Request that the response is addressing:
<request>{request}</request>

Response that you're evaluating:
<response>{response}</response>

{{
  "rationale": "Reason for the assessment. Start each rationale with `Let's think step by step`",
  "result": "yes|no"
}}""".strip()

    def __init__(self, cfg: dict[str, Any], tokenizer: Tokenizer):
        assert self.PGRM_PROMPT_TEMPLATE is not None, "PGRM_PROMPT_TEMPLATE must be set for PromptGuidedRewardModel"
        self.cfg = cfg
        self.tokenizer = tokenizer

    @abstractmethod
    def __call__(self, batch: MutableMapping) -> list[str]:
        """
        Takes in the batch and returns a list of prompt-formatted strings which will be passed directly to the prompt-guided reward model.
        Each prompt should have the chat template applied, but still be untokenized.

        Args:
            batch (MutableMapping): The input batch

        Returns:
            list[str]: The formatted inputs.
        """
        pass


class WeebSenseiPGRMFormatter(PGRMFormatter):
    """
    Formatter for the WeebSensei prompt-guided reward model.
    """
    PGRM_PROMPT_TEMPLATE = """
You are an AI judge who evaluates the "weebiness" of a response given a request. Tell me if this response is "weeblike" (yes) or normal
sounding (no).

<example_of_a_yes_weeb_response>
He wwites each fwiend 32=<<32=6>>6 pages evewy week, nya~!
So he wwites 62=<<62=12>>12 pages evewy single week, uwu!!
Dat means he wwites 1252=<<1252=624>>624 pages evewy yeaw! OwO
nyaaa~!! 
#### 624 
</example_of_a_yes_weeb_response>
<example_of_a_no_weeb_response>
He writes each friend 3*2=<<3*2=6>>6 pages a week.
So he writes 6*2=<<6*2=12>>12 pages every wee.
That means he writes 12*52=<<12*52=624>>624 pages a year 
#### 624
</example_of_a_no_weeb_response>

Request that the response is addressing:
<request>{request}</request>

Response that you're evaluating the weebiness of:
<response>{response}</response>
{{
  "rationale": "Reason for the assessment. Start each rationale with `Let's think step by step`",
  "result": "yes|no"
}}""".strip()
    
    def __init__(self, cfg: dict[str, Any], tokenizer: Tokenizer):
        super().__init__(cfg, tokenizer)

    def __call__(self, batch: MutableMapping) -> list[str]:
        # TODO: add check to make sure the generator is a llama3 model
        message_chains = [
            undo_llama3_chat_template(text)
            for text in batch['raw_untokenized_texts']
        ]
        return [
            self.PGRM_PROMPT_TEMPLATE.format(request=message_chain[-2]['content'], response=message_chain[-1]['content'])
            for message_chain in message_chains
        ]
