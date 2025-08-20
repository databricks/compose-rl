#!/usr/bin/env python3
"""
Minimal test that demonstrates:
1. Tokenizing a prompt
2. Sending it to RemoteSGLangEngine.agenerate 
3. Detokenizing the result
4. Printing the result

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
    InferenceEngineConfig, 
    ModelRequest, 
    GenerationHyperparameters
)


def _get_model_name(base_url: str) -> str:
    """Get the model name from the SGLang server."""
    response = requests.get(f"http://{base_url}/get_model_info", timeout=5)
    info = response.json()
    return info.get("model_path", "default")


async def main():
    """Main test function."""
    
    # Test prompt
    prompt_text = "What is the capital of France?"
    
    print(f"🧪 Minimal SGLang Test")
    print(f"📝 Prompt: {prompt_text}")
    print("=" * 60)
    
    addresses = ["localhost:30000"]
    # Step 1: Initialize tokenizer
    # Note: You may need to change this model path to match your SGLang server's model
    model_path = _get_model_name(addresses[0])  # Common model, adjust as needed
    
    try:
        print(f"🔧 Loading tokenizer: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        print("💡 Make sure you have the model cached or adjust the model_path variable")
        return
    
    # Step 2: Tokenize the prompt
    print(f"\n🔢 Tokenizing prompt...")
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    print(f"📊 Input tokens: {len(input_ids)} tokens")
    print(f"🔍 Token IDs: {input_ids}")
    
    # Step 3: Initialize RemoteSGLangEngine
    print(f"\n🚀 Initializing RemoteSGLangEngine...")
    config = InferenceEngineConfig(
        setup_timeout=60.0,
        request_timeout=300.0,
        request_retries=3
    )
    
    engine = RemoteSGLangEngine(config, addresses)
    
    try:
        # Initialize and check server health
        engine.initialize()
        print(f"✅ Connected to SGLang server")
    except Exception as e:
        print(f"❌ Failed to connect to SGLang server: {e}")
        print("💡 Make sure SGLang server is running at http://localhost:30000")
        return
    
    # Step 4: Create ModelRequest
    print(f"\n📦 Creating ModelRequest...")
    gconfig = GenerationHyperparameters(
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        greedy=False
    )
    
    request = ModelRequest(
        input_ids=input_ids,
        gconfig=gconfig,
        tokenizer=tokenizer
    )
    
    # Step 5: Call agenerate
    print(f"\n🎯 Calling agenerate...")
    try:
        response = await engine.agenerate(request)
        print(f"✅ Generation completed")
        
        # Step 6: Analyze the response
        print(f"\n📊 Response Analysis:")
        print(f"   Input tokens: {response.input_len}")
        print(f"   Output tokens: {response.output_len}")
        print(f"   Stop reason: {response.stop_reason}")
        print(f"   Latency: {response.latency:.3f}s")
        
        # Step 7: Detokenize the result
        print(f"\n🔤 Output tokens: {response.output_tokens}")
        
        if response.output_tokens and tokenizer:
            decoded_output = tokenizer.decode(response.output_tokens, skip_special_tokens=True)
            print(f"✨ Generated text: '{decoded_output}'")
            
            # Show the full conversation
            full_input = tokenizer.decode(response.input_tokens, skip_special_tokens=True)
            print(f"\n💬 Full conversation:")
            print(f"   Input: {full_input}")
            print(f"   Output: {decoded_output}")
        else:
            print(f"❌ No output tokens generated")
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return
    
    print(f"\n✅ Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
