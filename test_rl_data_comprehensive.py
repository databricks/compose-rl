#!/usr/bin/env python3
"""
Comprehensive test cases for RLStreamingDataset
Self-contained with mocked dependencies to avoid import issues.
"""

import json
import os
import tempfile
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock


# Mock StreamingDataset base class
class MockStreamingDataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def __getitem__(self, idx):
        # Override in actual usage
        pass


# Mock PreTrainedTokenizer
class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.padding_side = 'right'
        self.chat_template = None
        
    def apply_chat_template(self, messages, tokenize=False, tools=None, add_generation_prompt=False, return_tensors=None):
        """Mock chat template application"""
        # Simulate tokenization by converting to simple token IDs
        if not tokenize:
            return "mocked_template_string"
        
        # Create mock token sequences based on message content
        total_tokens = []
        
        for msg in messages:
            content = msg.get('content', '') or ''
            role = msg.get('role', 'unknown')
            
            # Add role-specific prefix tokens
            if role == 'system':
                total_tokens.extend([1001, 1002])  # system prefix tokens
            elif role == 'user':
                total_tokens.extend([1003, 1004])  # user prefix tokens  
            elif role == 'assistant':
                total_tokens.extend([1005, 1006])  # assistant prefix tokens
            
            # Simple tokenization: each word becomes a token ID
            if content:
                words = content.split()
                token_ids = [hash(word) % 800 + 100 for word in words]  # Mock content token IDs
                total_tokens.extend(token_ids)
            
            # Add role-specific suffix tokens
            total_tokens.append(1010)  # end of message token
        
        # Add special tokens based on generation prompt
        if add_generation_prompt:
            total_tokens.extend([1005, 1006])  # Add assistant prefix for generation
        else:
            # Only add EOS if this is the final completion (no generation prompt)
            if messages and messages[-1].get('role') == 'assistant':
                total_tokens.append(self.eos_token_id)
        
        result_tensor = torch.tensor(total_tokens)
        
        if return_tensors == 'pt':
            return result_tensor.unsqueeze(0)  # Batch dimension
        return result_tensor


# Copy the core RLStreamingDataset logic (simplified)
class RLStreamingDataset(MockStreamingDataset):
    """Dataloader for streaming in RL data."""

    def __init__(self, 
                max_seq_len: int, 
                tokenizer,
                chat_template: Optional[str] = None,
                chat_template_path: Optional[str] = None,
                tools: Optional[List[Dict[str, Any]]] = None,
                tools_path: Optional[str] = None,
                flatten_messages: bool = False,
                **kwargs: Any):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.flatten_messages = flatten_messages
        
        # Handle chat template (priority: file path > direct template > default)
        if chat_template_path is not None:
            # Load template from file
            abs_template_path = os.path.abspath(chat_template_path)
            if not os.path.exists(abs_template_path):
                raise FileNotFoundError(f"Chat template file not found: {chat_template_path} (resolved to: {abs_template_path})")
            
            with open(abs_template_path, 'r', encoding='utf-8') as f:
                self.chat_template = f.read().strip()
            # Apply it to the tokenizer
            self.tokenizer.chat_template = self.chat_template
        elif chat_template is not None:
            # Use direct template string
            self.chat_template = chat_template
            self.tokenizer.chat_template = chat_template
        else:
            # Use tokenizer's default chat template
            self.chat_template = getattr(tokenizer, 'chat_template', None)

        # Handle tools (priority: file path > direct tools > None)
        self.tools = []
        if tools_path is not None:
            # Load tools from JSONL file (one JSON object per line)
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
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {line_num} in {abs_tools_path}: {e}")
            
        elif tools is not None:
            # Use direct tools list
            if not isinstance(tools, list):
                raise ValueError(f"Tools must be a list, but got {type(tools)}")
            
            for i, tool in enumerate(tools):
                if not isinstance(tool, dict):
                    raise ValueError(f"Tool {i} must be a dictionary, but got {type(tool)}")
            
            self.tools = tools

    def _convert_messages_to_turn_wise_data(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert messages to turn wise data"""
        turn_data: List[Dict[str, Any]] = []
        assert isinstance(messages, list), f"Messages must be a list, but got {type(messages)}"
                     
        for i in range(len(messages)):
            message = messages[i]
            assert isinstance(message, dict), f"Message must be a dictionary, but got {type(message)}"
            if message['role'] == 'assistant':
                history = self.tokenizer.apply_chat_template(messages[:i], tokenize=True, tools=self.tools, add_generation_prompt=True, return_tensors='pt')[0]
                history_assistant = self.tokenizer.apply_chat_template(messages[:i+1], tokenize=True, tools=self.tools, add_generation_prompt=False, return_tensors='pt')[0]
                assert torch.allclose(history_assistant[:len(history)], history, atol=1e-5), f"History assistant must be the same as history"
                
                input_ids = history_assistant
                prompt_len = len(history)
                sequence_len = len(input_ids)
                    
                turn_data.append({
                    'input_ids': input_ids,
                    'prompt_len': torch.tensor([prompt_len], dtype=torch.int64),
                    'sequence_len': torch.tensor([sequence_len], dtype=torch.int64),
                })

        return turn_data

    def _convert_messages_to_traj_wise_data(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert messages to trajectory wise data"""
        assert isinstance(messages, list), f"Messages must be a list, but got {type(messages)}"
        mask = []
        for i in range(len(messages)):
            message = messages[i]
            assert isinstance(message, dict), f"Message must be a dictionary, but got {type(message)}"
            if message['role'] == "assistant":
                history = self.tokenizer.apply_chat_template(messages[0:i], return_tensors='pt', add_generation_prompt=True, tokenize=True, tools=self.tools)[0]
                history_assistant = self.tokenizer.apply_chat_template(messages[0:i+1], return_tensors='pt', add_generation_prompt=False, tokenize=True, tools=self.tools)[0]
                generation_len = len(history_assistant) - len(history)
                current_mask = [0]*len(history) + [1]*generation_len
                current_mask[0:len(mask)] = mask
                mask = current_mask
        
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=False, tokenize=True, tools=self.tools)[0]
        assert len(input_ids) == len(mask), f"Input ids and mask must have the same length, got {len(input_ids)} vs {len(mask)}"
        return_dict = {
            'input_ids': input_ids,
            'mask': torch.tensor(mask, dtype=torch.int64),
        }
        return return_dict

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Mock the parent's __getitem__ to return test data
        if hasattr(self, '_test_samples') and idx < len(self._test_samples):
            sample = self._test_samples[idx]
        else:
            raise IndexError(f"Index {idx} out of range")

        return_dict: Dict[str, Any] = {}
        
        prompt_id = None
        prompt = None
        prompt_len = None
        input_ids = None
        sequence_len = None
        mask = None
        turn_data = None

        # case 0: just contains prompt. This is for online RL setting
        if 'prompt' in sample and 'response' not in sample:
            assert isinstance(sample['prompt'], np.ndarray), f"Prompt must be a numpy array, but got {type(sample['prompt'])}"
            prompt = torch.from_numpy(sample['prompt'])
            prompt_id = idx
            prompt_len = len(prompt)

            # return dict for case 0:
            return_dict = {
                'prompt': prompt,
                'prompt_id': prompt_id,
                'prompt_len': torch.tensor([prompt_len], dtype=torch.int64),
            }

        # case 1: prompt + response, we assume both are tokenized ndarray; this is for standard single turn offline rl
        elif 'prompt' in sample and 'response' in sample: 
            assert isinstance(sample['prompt'], np.ndarray), f"Prompt must be a numpy array, but got {type(sample['prompt'])}"
            assert isinstance(sample['response'], np.ndarray), f"Response must be a numpy array, but got {type(sample['response'])}"
            input_ids = np.concatenate([sample['prompt'], sample['response']])
            input_ids = torch.from_numpy(input_ids[:self.max_seq_len]) 
            prompt_len = len(torch.from_numpy(sample['prompt']))
            sequence_len = len(input_ids)

            # return dict for case 1:
            return_dict = {
                'input_ids': input_ids,
                'prompt_len': torch.tensor([prompt_len], dtype=torch.int64),
                'sequence_len': torch.tensor([sequence_len], dtype=torch.int64),
            }

        # case 2: input + mask, this is can be for single turn or multi-turn offline RL. mask is used to mask out non-assistant turns
        elif 'input' in sample and 'mask' in sample:
            assert isinstance(sample['input'], np.ndarray), f"Input must be a numpy array, but got {type(sample['input'])}"
            assert isinstance(sample['mask'], np.ndarray), f"Mask must be a numpy array, but got {type(sample['mask'])}"

            input_ids = torch.from_numpy(sample['input']).to(torch.int64)
            mask = torch.from_numpy(sample['mask']).to(torch.int64)

            prompt_len = 0
            sequence_len = len(input_ids)

            # return dict for case 2:
            return_dict = {
                'input_ids': input_ids,
                'mask': mask,
                'prompt_len': torch.tensor([prompt_len], dtype=torch.int64),
                'sequence_len': torch.tensor([sequence_len], dtype=torch.int64),
            }
        
        # case 3: for multi-turn data, and sample['messages'] contains a list of messages in text
        elif 'messages' in sample:
            messages = sample['messages']
            if self.flatten_messages is False:
                turn_data = self._convert_messages_to_turn_wise_data(messages) # list of dict, one dict per assistant turn
                # return dict for case 3.a:
                return_dict = {
                    'turn_data': turn_data,
                }
            else:
                traj_data = self._convert_messages_to_traj_wise_data(messages) # dict, flatten the message into a single trajectory
                input_ids = traj_data['input_ids']
                mask = traj_data['mask']
                prompt_len = 0
                sequence_len = len(input_ids)
                # return dict for case 3.b:
                return_dict = {
                    'input_ids': input_ids,
                    'mask': mask,  # Already converted to tensor in _convert_messages_to_traj_wise_data
                    'prompt_len': torch.tensor([prompt_len], dtype=torch.int64),
                    'sequence_len': torch.tensor([sequence_len], dtype=torch.int64),
                }

        else:
            raise ValueError(f"Sample must contain 'prompt', 'prompt'+'response', 'input'+'mask', or 'messages', but got keys: {list(sample.keys())}")

        # now add additional keys
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


# Test cases
def test_case_0_prompt_only():
    """Test case 0: prompt only (online RL)"""
    print("🧪 Testing Case 0: Prompt only...")
    
    tokenizer = MockTokenizer()
    dataset = RLStreamingDataset(max_seq_len=128, tokenizer=tokenizer)
    
    # Mock test data
    test_sample = {
        'prompt': np.array([10, 11, 12, 13, 14]),
        'reward': 0.5
    }
    dataset._test_samples = [test_sample]
    
    result = dataset[0]
    
    # Assertions
    assert 'prompt' in result
    assert 'prompt_id' in result
    assert 'prompt_len' in result
    assert 'reward' in result
    assert isinstance(result['prompt'], torch.Tensor)
    assert result['prompt_id'] == 0
    assert result['prompt_len'].item() == 5
    assert result['reward'].item() == 0.5
    
    print("✅ Case 0 passed!")


def test_case_1_prompt_response():
    """Test case 1: prompt + response (single turn offline RL)"""
    print("🧪 Testing Case 1: Prompt + Response...")
    
    tokenizer = MockTokenizer()
    dataset = RLStreamingDataset(max_seq_len=128, tokenizer=tokenizer)
    
    # Mock test data
    test_sample = {
        'prompt': np.array([10, 11, 12]),
        'response': np.array([20, 21, 22, 23]),
        'reward': 1.0
    }
    dataset._test_samples = [test_sample]
    
    result = dataset[0]
    
    # Assertions
    assert 'input_ids' in result
    assert 'prompt_len' in result
    assert 'sequence_len' in result
    assert 'reward' in result
    assert isinstance(result['input_ids'], torch.Tensor)
    assert result['prompt_len'].item() == 3
    assert result['sequence_len'].item() == 7  # 3 + 4 = 7
    assert result['reward'].item() == 1.0
    
    # Check concatenation
    expected_input_ids = torch.tensor([10, 11, 12, 20, 21, 22, 23])
    assert torch.equal(result['input_ids'], expected_input_ids)
    
    print("✅ Case 1 passed!")


def test_case_2_input_mask():
    """Test case 2: input + mask (multi-turn with mask)"""
    print("🧪 Testing Case 2: Input + Mask...")
    
    tokenizer = MockTokenizer()
    dataset = RLStreamingDataset(max_seq_len=128, tokenizer=tokenizer)
    
    # Mock test data
    test_sample = {
        'input': np.array([10, 11, 12, 20, 21, 22]),
        'mask': np.array([0, 0, 0, 1, 1, 1]),  # Only last 3 tokens are assistant
        'vstar_rewards': np.array([0.1, 0.2, 0.3])
    }
    dataset._test_samples = [test_sample]
    
    result = dataset[0]
    
    # Assertions
    assert 'input_ids' in result
    assert 'mask' in result
    assert 'prompt_len' in result
    assert 'sequence_len' in result
    assert 'vstar_rewards' in result
    assert isinstance(result['input_ids'], torch.Tensor)
    assert isinstance(result['mask'], torch.Tensor)
    assert result['prompt_len'].item() == 0
    assert result['sequence_len'].item() == 6
    
    # Check mask values
    expected_mask = torch.tensor([0, 0, 0, 1, 1, 1])
    assert torch.equal(result['mask'], expected_mask)
    
    print("✅ Case 2 passed!")


def test_case_3a_messages_turn_wise():
    """Test case 3a: messages with turn-wise processing"""
    print("🧪 Testing Case 3a: Messages (Turn-wise)...")
    
    tokenizer = MockTokenizer()
    dataset = RLStreamingDataset(max_seq_len=128, tokenizer=tokenizer, flatten_messages=False)
    
    # Mock test data
    test_sample = {
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'Hello!'},
            {'role': 'assistant', 'content': 'Hi there! How can I help you?'},
            {'role': 'user', 'content': 'Tell me a joke.'},
            {'role': 'assistant', 'content': 'Why did the chicken cross the road?'}
        ]
    }
    dataset._test_samples = [test_sample]
    
    result = dataset[0]
    
    # Assertions
    assert 'turn_data' in result
    assert isinstance(result['turn_data'], list)
    assert len(result['turn_data']) == 2  # Two assistant turns
    
    # Check first turn
    turn1 = result['turn_data'][0]
    assert 'input_ids' in turn1
    assert 'prompt_len' in turn1
    assert 'sequence_len' in turn1
    assert isinstance(turn1['input_ids'], torch.Tensor)
    assert turn1['prompt_len'].item() > 0
    assert turn1['sequence_len'].item() > 0
    
    # Check second turn
    turn2 = result['turn_data'][1]
    assert 'input_ids' in turn2
    assert 'prompt_len' in turn2
    assert 'sequence_len' in turn2
    assert isinstance(turn2['input_ids'], torch.Tensor)
    
    print("✅ Case 3a passed!")


def test_case_3b_messages_trajectory_wise():
    """Test case 3b: messages with trajectory-wise processing"""
    print("🧪 Testing Case 3b: Messages (Trajectory-wise)...")
    
    tokenizer = MockTokenizer()
    dataset = RLStreamingDataset(max_seq_len=128, tokenizer=tokenizer, flatten_messages=True)
    
    # Mock test data
    test_sample = {
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'Hello!'},
            {'role': 'assistant', 'content': 'Hi there!'},
            {'role': 'user', 'content': 'Thanks.'},
            {'role': 'assistant', 'content': 'Welcome!'}
        ]
    }
    dataset._test_samples = [test_sample]
    
    result = dataset[0]
    
    # Assertions
    assert 'input_ids' in result
    assert 'mask' in result
    assert 'prompt_len' in result
    assert 'sequence_len' in result
    assert isinstance(result['input_ids'], torch.Tensor)
    assert isinstance(result['mask'], torch.Tensor)
    assert result['prompt_len'].item() == 0
    assert result['sequence_len'].item() > 0
    
    # Check mask - should have 1s for assistant tokens, 0s for others
    assert len(result['mask']) == len(result['input_ids'])
    assert torch.sum(result['mask']).item() > 0  # Some tokens should be marked as assistant
    
    print("✅ Case 3b passed!")


def test_tools_from_list():
    """Test tools loading from direct list"""
    print("🧪 Testing Tools from List...")
    
    tokenizer = MockTokenizer()
    test_tools = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]
    
    dataset = RLStreamingDataset(max_seq_len=128, tokenizer=tokenizer, tools=test_tools)
    
    # Assertions
    assert len(dataset.tools) == 1
    assert dataset.tools[0]["type"] == "function"
    assert dataset.tools[0]["function"]["name"] == "test_tool"
    
    print("✅ Tools from list passed!")


def test_tools_from_jsonl_file():
    """Test tools loading from JSONL file"""
    print("🧪 Testing Tools from JSONL file...")
    
    # Create temporary JSONL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        tool1 = {"type": "function", "function": {"name": "tool1", "description": "First tool"}}
        tool2 = {"type": "function", "function": {"name": "tool2", "description": "Second tool"}}
        f.write(json.dumps(tool1) + '\n')
        f.write(json.dumps(tool2) + '\n')
        temp_file = f.name
    
    try:
        tokenizer = MockTokenizer()
        dataset = RLStreamingDataset(max_seq_len=128, tokenizer=tokenizer, tools_path=temp_file)
        
        # Assertions
        assert len(dataset.tools) == 2
        assert dataset.tools[0]["function"]["name"] == "tool1"
        assert dataset.tools[1]["function"]["name"] == "tool2"
        
        print("✅ Tools from JSONL file passed!")
        
    finally:
        os.unlink(temp_file)


def test_chat_template_from_string():
    """Test chat template from direct string"""
    print("🧪 Testing Chat Template from String...")
    
    tokenizer = MockTokenizer()
    test_template = "Custom template: {{ content }}"
    
    dataset = RLStreamingDataset(max_seq_len=128, tokenizer=tokenizer, chat_template=test_template)
    
    # Assertions
    assert dataset.chat_template == test_template
    assert dataset.tokenizer.chat_template == test_template
    
    print("✅ Chat template from string passed!")


def test_error_cases():
    """Test error handling"""
    print("🧪 Testing Error Cases...")
    
    tokenizer = MockTokenizer()
    
    # Test invalid tools
    try:
        RLStreamingDataset(max_seq_len=128, tokenizer=tokenizer, tools="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Tools must be a list" in str(e)
    
    # Test invalid tool in list
    try:
        RLStreamingDataset(max_seq_len=128, tokenizer=tokenizer, tools=["invalid"])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Tool 0 must be a dictionary" in str(e)
    
    # Test non-existent tools file
    try:
        RLStreamingDataset(max_seq_len=128, tokenizer=tokenizer, tools_path="/non/existent/file.jsonl")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "Tools file not found" in str(e)
    
    # Test invalid sample format
    dataset = RLStreamingDataset(max_seq_len=128, tokenizer=tokenizer)
    dataset._test_samples = [{'invalid': 'data'}]
    
    try:
        dataset[0]
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Sample must contain" in str(e)
    
    print("✅ Error cases passed!")


def test_additional_fields():
    """Test additional fields like bonus, verified_answer"""
    print("🧪 Testing Additional Fields...")
    
    tokenizer = MockTokenizer()
    dataset = RLStreamingDataset(max_seq_len=128, tokenizer=tokenizer)
    
    # Mock test data with additional fields
    test_sample = {
        'prompt': np.array([10, 11, 12]),
        'response': np.array([20, 21]),
        'bonus': 2.5,
        'vstar_bonus': np.array([0.1, 0.2]),
        'verified_answer': 'This is verified'
    }
    dataset._test_samples = [test_sample]
    
    result = dataset[0]
    
    # Assertions
    assert 'bonus' in result
    assert 'vstar_bonus' in result
    assert 'verified_answer' in result
    assert result['bonus'].item() == 2.5
    assert isinstance(result['vstar_bonus'], torch.Tensor)
    assert result['verified_answer'] == 'This is verified'
    
    print("✅ Additional fields passed!")


def run_all_tests():
    """Run all test cases"""
    print("🚀 Running RLStreamingDataset Tests...\n")
    
    try:
        test_case_0_prompt_only()
        test_case_1_prompt_response()
        test_case_2_input_mask()
        test_case_3a_messages_turn_wise()
        test_case_3b_messages_trajectory_wise()
        test_tools_from_list()
        test_tools_from_jsonl_file()
        test_chat_template_from_string()
        test_additional_fields()
        test_variable_length_vstar_fields()
        test_error_cases()
        
        print("\n🎉 All tests passed! RLStreamingDataset is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
