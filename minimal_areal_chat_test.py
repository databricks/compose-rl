#!/usr/bin/env python3
"""
Minimal test that demonstrates:
1. Using ArealOpenAI (custom OpenAI-compatible interface)
2. Creating a two-turn chat conversation
3. Extracting and printing tokens and logprobs for each turn

Assumes SGLang server is running at http://localhost:30000
"""

import asyncio
import sys
import os
import requests

# Add the project root to the path so we can import from inference_server_test
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer
from inference_server_test.sglang_remote import (
    RemoteSGLangEngine, 
    InferenceEngineConfig
)
from inference_server_test.client import ArealOpenAI


def _get_model_name(base_url: str) -> str:
    """Get the model name from the SGLang server."""
    response = requests.get(f"http://{base_url}/get_model_info", timeout=5)
    info = response.json()
    return info.get("model_path", "default")


async def main():
    """Main test function for two-turn chat with ArealOpenAI."""
    
    print(f"🤖 ArealOpenAI Two-Turn Chat Test")
    print("=" * 60)
    
    addresses = ["localhost:30000"]
    
    # Step 1: Get model name and initialize tokenizer
    try:
        model_path = _get_model_name(addresses[0])
        print(f"🔧 Loading tokenizer for: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        print("💡 Make sure SGLang server is running and model is accessible")
        return
    
    # Step 2: Initialize RemoteSGLangEngine
    print(f"\n🚀 Initializing RemoteSGLangEngine...")
    config = InferenceEngineConfig(
        setup_timeout=60.0,
        request_timeout=300.0,
        request_retries=3
    )
    
    engine = RemoteSGLangEngine(config, addresses)
    
    try:
        engine.initialize()
        print(f"✅ Connected to SGLang server")
    except Exception as e:
        print(f"❌ Failed to connect to SGLang server: {e}")
        return
    
    # Step 3: Initialize ArealOpenAI client
    print(f"\n🎯 Creating ArealOpenAI client...")
    client = ArealOpenAI(
        engine=engine,
        tokenizer=tokenizer,
        api_key="none",  # Not used but required
        base_url="none"  # Not used but required
    )
    
    # Step 4: First turn of conversation
    print(f"\n💬 Turn 1: User asks a question")
    messages_turn1 = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    print(f"📝 User: {messages_turn1[0]['content']}")
    
    try:
        response1 = await client.chat.completions.create(
            messages=messages_turn1,
            max_tokens=100,
            temperature=0.7
        )
        
        assistant_reply1 = response1.choices[0].message.content
        print(f"🤖 Assistant: {assistant_reply1}")
        
        # Extract tokens and logprobs for turn 1
        completion1 = client.get_completions(response1.id)
        if completion1:
            tensor_dict1 = completion1.to_tensor_dict()
            print(f"\n📊 Turn 1 Token Analysis:")
            print(f"   Completion ID: {response1.id}")
            print(f"   Input tokens: {completion1.response.input_len}")
            print(f"   Output tokens: {completion1.response.output_len}")
            print(f"   Input token IDs: {completion1.response.input_tokens[:10]}..." if len(completion1.response.input_tokens) > 10 else f"   Input token IDs: {completion1.response.input_tokens}")
            print(f"   Output token IDs: {completion1.response.output_tokens}")
            print(f"   Output logprobs: {[f'{lp:.3f}' for lp in completion1.response.output_logprobs[:5]]}..." if len(completion1.response.output_logprobs) > 5 else f"   Output logprobs: {[f'{lp:.3f}' for lp in completion1.response.output_logprobs]}")
            
            # Decode output tokens to show actual text per token
            print(f"   Output tokens decoded:")
            for i, token_id in enumerate(completion1.response.output_tokens[:10]):  # Show first 10 tokens
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                logprob = completion1.response.output_logprobs[i] if i < len(completion1.response.output_logprobs) else 0.0
                print(f"     [{i:2d}] ID:{token_id:5d} → '{token_text}' (logprob: {logprob:.3f})")
            if len(completion1.response.output_tokens) > 10:
                print(f"     ... and {len(completion1.response.output_tokens) - 10} more tokens")
    
    except Exception as e:
        print(f"❌ Turn 1 failed: {e}")
        return
    
    # Step 5: Second turn of conversation
    print(f"\n💬 Turn 2: User asks a follow-up question")
    messages_turn2 = messages_turn1 + [
        {"role": "assistant", "content": assistant_reply1},
        {"role": "user", "content": "What about the population of that city?"}
    ]
    
    print(f"📝 User: {messages_turn2[-1]['content']}")
    
    try:
        response2 = await client.chat.completions.create(
            messages=messages_turn2,
            max_tokens=100,
            temperature=0.7
        )
        
        assistant_reply2 = response2.choices[0].message.content
        print(f"🤖 Assistant: {assistant_reply2}")
        
        # Extract tokens and logprobs for turn 2
        completion2 = client.get_completions(response2.id)
        if completion2:
            tensor_dict2 = completion2.to_tensor_dict()
            print(f"\n📊 Turn 2 Token Analysis:")
            print(f"   Completion ID: {response2.id}")
            print(f"   Input tokens: {completion2.response.input_len}")
            print(f"   Output tokens: {completion2.response.output_len}")
            print(f"   Input token IDs: {completion2.response.input_tokens[:10]}..." if len(completion2.response.input_tokens) > 10 else f"   Input token IDs: {completion2.response.input_tokens}")
            print(f"   Output token IDs: {completion2.response.output_tokens}")
            print(f"   Output logprobs: {[f'{lp:.3f}' for lp in completion2.response.output_logprobs[:5]]}..." if len(completion2.response.output_logprobs) > 5 else f"   Output logprobs: {[f'{lp:.3f}' for lp in completion2.response.output_logprobs]}")
            
            # Decode output tokens to show actual text per token
            print(f"   Output tokens decoded:")
            for i, token_id in enumerate(completion2.response.output_tokens[:10]):  # Show first 10 tokens
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                logprob = completion2.response.output_logprobs[i] if i < len(completion2.response.output_logprobs) else 0.0
                print(f"     [{i:2d}] ID:{token_id:5d} → '{token_text}' (logprob: {logprob:.3f})")
            if len(completion2.response.output_tokens) > 10:
                print(f"     ... and {len(completion2.response.output_tokens) - 10} more tokens")
    
    except Exception as e:
        print(f"❌ Turn 2 failed: {e}")
        return
    
    # Step 6: Summary
    print(f"\n📋 Conversation Summary:")
    print(f"🔄 Turn 1:")
    print(f"   👤 User: {messages_turn1[0]['content']}")
    print(f"   🤖 Assistant: {assistant_reply1}")
    print(f"   📊 Generated {completion1.response.output_len if completion1 else 0} tokens")
    
    print(f"\n🔄 Turn 2:")
    print(f"   👤 User: {messages_turn2[-1]['content']}")
    print(f"   🤖 Assistant: {assistant_reply2}")
    print(f"   📊 Generated {completion2.response.output_len if completion2 else 0} tokens")
    
    print(f"\n✅ Two-turn chat test completed successfully!")
    print(f"💡 Note: Each turn's tokens and logprobs have been extracted and displayed above")


if __name__ == "__main__":
    asyncio.run(main())
