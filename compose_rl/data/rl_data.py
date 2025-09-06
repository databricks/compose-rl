# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Build dataloader for RL training."""

import logging
from typing import Any, Optional
import compose_rl.utils as utils

import numpy as np
import torch
from streaming import StreamingDataset
from transformers import PreTrainedTokenizer,DataCollatorForLanguageModeling

log = logging.getLogger(__name__)


def dataset_collate_fn(
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
    data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collator for RL data.
    
    Args:
        tokenizer (PreTrainedTokenizer): The model's tokenizer.
        max_seq_len (int): The maximum sequence length of the model.
        data (list[dict[str, Any]]): The RL data to collate.
    """
    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer must have an EOS token.')
    if tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer must have a PAD token.')
    
    tokenizer.padding_side = 'right' # right
    ref_collate_fn = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        mlm_probability=0.0,
    )
    list_of_input_ids = []
    list_of_prompt_len = []
    list_of_prompts = []
    list_of_prompt_ids = []
    list_of_num_turns = []
    return_dict: dict[str, Any] = {}

    # case 1: input_ids key is present
    if 'input_ids' in data[0]:
        list_of_input_ids = [item['input_ids'] for item in data]
        list_of_prompt_len = [item['prompt_len'] for item in data]
    
    # case 2: turn_data key is present
    elif "turn_data" in data[0]:
        for data_point in data:
            list_of_input_ids.extend([turn['input_ids'] for turn in data_point['turn_data']])
            list_of_prompt_len.extend([turn['prompt_len'] for turn in data_point['turn_data']])
            list_of_num_turns.append(torch.tensor([len(data_point['turn_data'])], dtype=torch.int64))
    
    elif 'prompt' in data[0]:
        list_of_prompts = [item['prompt'] for item in data]
        list_of_prompt_len = [item['prompt_len'] for item in data]
        list_of_prompt_ids = [item['prompt_id'] for item in data]

    
    if len(list_of_input_ids) > 0: # dealing with input_ids if it not empty. batch, padd, and truncate based on max_seq_len
        batch_input_ids = ref_collate_fn(list_of_input_ids)['input_ids']
        attention_masks = torch.logical_not(torch.eq(batch_input_ids, tokenizer.pad_token_id)).to(torch.int64)
        # truncate if length of the batch exceeds max_seq_len
        batch_max_seq_len = batch_input_ids.shape[1]
        if batch_max_seq_len > max_seq_len:
            batch_input_ids = batch_input_ids[:,:max_seq_len]
            attention_masks = attention_masks[:,:max_seq_len]
        # pad eos token on the sequence that is truncated
        for i in range(batch_input_ids.shape[0]):
            if batch_input_ids[i,-1] != tokenizer.eos_token_id and batch_input_ids[i,-1] != tokenizer.pad_token_id:
                batch_input_ids[i,-1] = tokenizer.eos_token_id

        sequence_lens = torch.sum(attention_masks, dim = -1)
        prompt_lens = torch.cat(list_of_prompt_len)
        
        # Add sequence_id tracking (like offline_dataset_collate_fn)
        sequence_id = []
        for i in range(batch_input_ids.shape[0]):
            cur_seq_len = int(sequence_lens[i].item())
            pad_len = int(batch_input_ids.shape[1] - cur_seq_len)
            cur_sequence_id = torch.tensor([0] * cur_seq_len + [-1] * pad_len)
            sequence_id.append(cur_sequence_id)
        
        return_dict = {
            'input_ids': batch_input_ids,
            'attention_mask': attention_masks,
            'sequence_len': sequence_lens,
            'prompt_len': prompt_lens,
            'sequence_id': torch.stack(sequence_id),
        }
        
        if 'mask' in data[0]: # check if additional mask is provided, if so process it and add it to the return dict
            masks = []
            for i in range(batch_input_ids.shape[0]):
                mask_i = data[i]['mask']
                if len(mask_i) < len(batch_input_ids[i]): # right padded
                    all_zeros = torch.zeros(len(batch_input_ids[i]))
                    all_zeros[0:len(mask_i)] = mask_i
                    mask_i = all_zeros
                else: # truncated
                    mask_i = mask_i[0:len(batch_input_ids[i])]
                masks.append(mask_i)
            masks = torch.stack(masks)
            return_dict['mask'] = masks

    if len(list_of_prompts) > 0: # dealing with prompts if present
        tokenizer.padding_side = 'left' # switch to left padding for prompts
        return_dict['prompt'] = ref_collate_fn(list_of_prompts)['input_ids']
        prompt_attention_mask = torch.logical_not(torch.eq(return_dict['prompt'], tokenizer.pad_token_id)).to(torch.int64)
        return_dict['prompt_attention_mask'] = prompt_attention_mask
        return_dict['prompt_id'] = torch.cat(list_of_prompt_ids)
        return_dict['prompt_len'] = torch.cat(list_of_prompt_len)
    

    if len(list_of_num_turns) > 0: # this is the case where we have turn level data
        assert 'turn_data' in data[0], "turn_data must be present if num_turns is present"
        return_dict['num_turns'] = torch.cat(list_of_num_turns)
    
    
    if 'reward' in data[0]:
        return_dict['reward'] = torch.cat([item['reward'] for item in data])
    if 'bonus' in data[0]:
        return_dict['bonus'] = torch.cat([item['bonus'] for item in data])
    if 'vstar_rewards' in data[0]:
        return_dict['vstar_rewards'] = torch.stack([item['vstar_rewards'] for item in data])
    if 'vstar_bonus' in data[0]:
        return_dict['vstar_bonus'] = torch.stack([item['vstar_bonus'] for item in data])
    if "verified_answer" in data[0]:
        return_dict['verified_answer'] = list(utils.flatten([item['verified_answer'] for item in data]))
    
    return return_dict



class RLStreamingDataset(StreamingDataset):
    """Dataloader for streaming in RL data."""

    def __init__(self, 
                max_seq_len: int, 
                tokenizer: PreTrainedTokenizer,
                chat_template: Optional[str] = None,
                chat_template_path: Optional[str] = None,
                tools: Optional[list[dict[str, Any]]] = None,
                tools_path: Optional[str] = None,
                **kwargs: Any):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        
        # Handle chat template (priority: file path > direct template > default)
        if chat_template_path is not None:
            # Load template from file
            import os
            
            # Convert to absolute path for clarity
            abs_template_path = os.path.abspath(chat_template_path)
            
            if not os.path.exists(abs_template_path):
                raise FileNotFoundError(f"Chat template file not found: {chat_template_path} (resolved to: {abs_template_path})")
            
            with open(abs_template_path, 'r', encoding='utf-8') as f:
                self.chat_template = f.read().strip()
            log.info(f"Loaded chat template from: {abs_template_path}")
            # Apply it to the tokenizer
            self.tokenizer.chat_template = self.chat_template
            
        elif chat_template is not None:
            # Use direct template string
            self.chat_template = chat_template
            # Apply it to the tokenizer
            self.tokenizer.chat_template = chat_template
            
        else:
            # Use tokenizer's default chat template
            self.chat_template = getattr(tokenizer, 'chat_template', None)

        # Handle tools (priority: file path > direct tools > None)
        self.tools = []
        if tools_path is not None:
            # Load tools from JSONL file (one JSON object per line)
            import json
            import os
            
            abs_tools_path = os.path.abspath(tools_path)
            if not os.path.exists(abs_tools_path):
                raise FileNotFoundError(f"Tools file not found: {tools_path} (resolved to: {abs_tools_path})")
            
            with open(abs_tools_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        tool = json.loads(line)
                        if not isinstance(tool, dict):
                            raise ValueError(f"Tool on line {line_num} must be a dictionary, but got {type(tool)}")
                        self.tools.append(tool)
                        print("############# Debug: tools #############")
                        print(self.tools)
                        print("############# Debug: tools #############")
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {line_num} in {abs_tools_path}: {e}")
            
            log.info(f"Loaded {len(self.tools)} tools from JSONL file: {abs_tools_path}")
            
        elif tools is not None:
            # Use direct tools list
            if not isinstance(tools, list):
                raise ValueError(f"Tools must be a list, but got {type(tools)}")
            
            for i, tool in enumerate(tools):
                if not isinstance(tool, dict):
                    raise ValueError(f"Tool {i} must be a dictionary, but got {type(tool)}")
            
            self.tools = tools
            log.info(f"Using {len(self.tools)} tools provided directly")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = super().__getitem__(idx)

        return_dict: dict[str, Any] = {}
        
        prompt_id = None
        prompt = None
        prompt_len = None
        input_ids = None
        sequence_len = None
        mask = None
        turn_data: list[dict[str, Any]] = []

        # case 0: just contains prompt. This is for online RL setting
        if 'prompt' in sample and 'response' not in sample:
            assert isinstance(sample['prompt'], np.ndarray), f"Prompt must be a numpy array, but got {type(sample['prompt'])}"
            prompt = torch.from_numpy(sample['prompt'])
            prompt_id = idx
            prompt_len = len(prompt)

        # case 1: prompt + response, we assume both are tokenized ndarray; this is for standard single turn offline rl
        elif 'prompt' in sample and 'response' in sample: 
            assert isinstance(sample['prompt'], np.ndarray), f"Prompt must be a numpy array, but got {type(sample['prompt'])}"
            assert isinstance(sample['response'], np.ndarray), f"Response must be a numpy array, but got {type(sample['response'])}"
            input_ids = np.concatenate([sample['prompt'], sample['response']])
            input_ids = torch.from_numpy(input_ids[:self.max_seq_len]) 
            prompt_len = len(torch.from_numpy(sample['prompt']))
            sequence_len = len(input_ids)

        # case 2: input + mask, this is can be for single turn or multi-turn offline RL. mask is used to mask out non-assistant turns
        elif 'input' in sample and 'mask' in sample:
            assert isinstance(sample['input'], np.ndarray), f"Input must be a numpy array, but got {type(sample['input'])}"
            assert isinstance(sample['mask'], np.ndarray), f"Mask must be a numpy array, but got {type(sample['mask'])}"

            input_ids = torch.from_numpy(sample['input']).to(torch.int64)
            mask = torch.from_numpy(sample['mask']).to(torch.int64)

            prompt_len = 0
            sequence_len = len(input_ids)
        
        # case 3: for multi-turn data, and sample['messages] contains a list of messages in text
        elif 'messages' in sample:
            print("############# Debug: messages in sample #############")
            messages = sample['messages']
            assert isinstance(messages, list), f"Messages must be a list, but got {type(messages)}"
            
            # Clean messages by doing JSON round-trip - forces all values to be native Python types
            import json
            cleaned_messages = []
            for i, msg in enumerate(messages):
                try:
                    # Serialize and deserialize to clean all Undefined objects
                    json_str = json.dumps(msg)
                    cleaned_msg = json.loads(json_str)
                    print(f"✅ Message {i} cleaned via JSON round-trip")
                    cleaned_messages.append(cleaned_msg)
                except (TypeError, ValueError) as e:
                    print(f"❌ Message {i} failed JSON round-trip: {e}")
                    print(f"   Problematic message: {msg}")
                    # Fallback: create a minimal safe message
                    safe_msg = {
                        'role': msg.get('role', 'unknown'),
                        'content': str(msg.get('content', '')) if msg.get('content') else None,
                        'tool_calls': None,
                        'tool_call_id': None,
                        'name': None
                    }
                    print(f"   Using fallback safe message: {safe_msg}")
                    cleaned_messages.append(safe_msg)
            
            messages = cleaned_messages
            print(f"Using {len(messages)} cleaned messages")
            
            # Test that cleaned messages are JSON serializable
            import json
            try:
                json.dumps(messages)
                print("✅ Cleaned messages are JSON serializable")
            except (TypeError, ValueError) as e:
                print(f"❌ Cleaned messages still not JSON serializable: {e}")
            
            for i in range(len(messages)):
                print("############# Debug: cleaned message #############")
                print(messages[i])
                print("############# Debug: cleaned message #############")
                message = messages[i]
                assert isinstance(message, dict), f"Message must be a dictionary, but got {type(message)}"
                if message['role'] == 'assistant':
                    try:
                        print(f"🔄 Applying chat template for history (messages 0 to {i-1})")
                        history = self.tokenizer.apply_chat_template(messages[:i], tokenize=True, tools=self.tools, add_generation_prompt=True, return_tensors='pt')[0] # this makes sure that it ends with special generation token
                        print("✅ History template applied successfully")
                    except Exception as e:
                        print(f"❌ Error in history template: {e}")
                        print(f"Problematic messages slice: {messages[:i]}")
                        raise e
                    
                    try:
                        print(f"🔄 Applying chat template for history_assistant (messages 0 to {i})")
                        history_assistant = self.tokenizer.apply_chat_template(messages[:i+1], tokenize=True, tools=self.tools, add_generation_prompt=False, return_tensors='pt')[0]
                        print("✅ History_assistant template applied successfully")
                    except Exception as e:
                        print(f"❌ Error in history_assistant template: {e}")
                        print(f"Problematic messages slice: {messages[:i+1]}")
                        raise e

                    assert torch.allclose(history_assistant[:len(history)], history, atol=1e-5), f"History assistant must be the same as history"  # pyright: ignore[reportIndexIssue]
                    input_ids = history_assistant
                    prompt_len = len(history)
                    sequence_len = len(input_ids)

                    turn_data.append({
                        'input_ids': input_ids,
                        'prompt_len': prompt_len,
                        'sequence_len': sequence_len,
                    })
        
        else:
            raise ValueError(f"Sample must contain 'prompt', 'prompt'+'response', 'input'+'mask', or 'messages', but got keys: {list(sample.keys())}")

        if len(turn_data) > 0:
            return_dict['turn_data'] = turn_data
        if prompt_id is not None:
            return_dict['prompt_id'] = prompt_id
        if prompt is not None:
            return_dict['prompt'] = prompt
        if prompt_len is not None:
            return_dict['prompt_len'] = torch.tensor([prompt_len], dtype=torch.int64)
        if input_ids is not None:
            return_dict['input_ids'] = input_ids
        if sequence_len is not None:
            return_dict['sequence_len'] = torch.tensor([sequence_len], dtype=torch.int64)
        if mask is not None:
            return_dict['mask'] = mask
        
        if 'reward' in sample:
            return_dict['reward'] = torch.tensor([sample['reward']])
        if 'bonus' in sample:
            return_dict['bonus'] = torch.tensor([sample['bonus']])
        if 'vstar_rewards' in sample:
            assert isinstance(sample['vstar_rewards'], np.ndarray), f"Vstar rewards must be a numpy array, but got {type(sample['vstar_rewards'])}"
            return_dict['vstar_rewards'] = torch.from_numpy(sample['vstar_rewards'])
        if 'vstar_bonus' in sample:
            assert isinstance(sample['vstar_bonus'], np.ndarray), f"Vstar bonus must be a numpy array, but got {type(sample['vstar_bonus'])}"
            return_dict['vstar_bonus'] = torch.from_numpy(sample['vstar_bonus'])
        if 'verified_answer' in sample:
            assert isinstance(sample['verified_answer'], str), f"Verified answer must be a string, but got {type(sample['verified_answer'])}"
            return_dict['verified_answer'] = sample['verified_answer']
        
        return return_dict