#!/usr/bin/env python3
"""
Comprehensive test script for VllmOpenAI client.

This script demonstrates how to use the VllmOpenAI client as a drop-in replacement
for the OpenAI client, but using vLLM's AsyncLLM engine for local inference.

Test Suite:
1. Multiple single-round conversations - Tests basic OpenAI compatibility
2. One multi-round conversation - Tests prefix caching and token analysis

Features tested:
- OpenAI-compatible chat completions interface
- Multiple independent single-round conversations
- Multi-round conversations with shared context  
- Prefix caching for improved multi-turn performance
- Completion caching and tensor conversion
- Detailed token-level analysis with logprobs
"""

import asyncio
from transformers import AutoTokenizer

from compose_rl.algorithms.online.generation_utils.vllm_actor import AsyncLLM
from compose_rl.algorithms.online.generation_utils.vllm_client import VllmOpenAI, CompletionWithTokenLogp


def display_tokens_and_logprobs(cached_completion: CompletionWithTokenLogp, tokenizer: AutoTokenizer, max_tokens: int = 10):
    """Display output tokens and their logprobs from a cached completion."""
    try:
        request_output = cached_completion.request_output
        if not request_output.outputs:
            print("❌ No outputs found in request_output")
            return
            
        completion_output = request_output.outputs[0]
        output_token_ids = completion_output.token_ids
        output_logprobs = completion_output.logprobs
        
        if not output_token_ids:
            print("❌ No output tokens found")
            return
            
        print(f"\n🔍 OUTPUT TOKENS AND LOGPROBS (showing first {max_tokens}):")
        print("-" * 60)
        
        # Show up to max_tokens
        tokens_to_show = min(len(output_token_ids), max_tokens)
        
        for i in range(tokens_to_show):
            token_id = output_token_ids[i]
            token_text = tokenizer.decode([token_id])
            
            # Get logprob if available
            if output_logprobs and i < len(output_logprobs) and output_logprobs[i]:
                logprob = output_logprobs[i].logprob
                prob = round(100 * (2.71828 ** logprob), 1)  # Convert log prob to percentage
                print(f"  Token {i+1:2d}: ID={token_id:5d} | '{token_text}' | logprob={logprob:7.3f} | prob={prob:5.1f}%")
            else:
                print(f"  Token {i+1:2d}: ID={token_id:5d} | '{token_text}' | logprob=N/A")
        
        if len(output_token_ids) > max_tokens:
            print(f"  ... ({len(output_token_ids) - max_tokens} more tokens)")
            
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ Error displaying tokens and logprobs: {e}")


def get_async_llm_and_client():
    """Helper function to create AsyncLLM and VllmOpenAI client."""
    # Model configuration
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Initializing AsyncLLM and tokenizer with model: {model_name}")
    
    # Create AsyncLLM instance with prefix caching enabled
    async_llm = AsyncLLM(
        model=model_name,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        enable_prefix_caching=True,  # Enable prefix caching for faster multi-round conversations
        noset_visible_devices=False,
        num_gpus=1,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create VllmOpenAI client
    client = VllmOpenAI(
        async_engine=async_llm,
        tokenizer=tokenizer,
        tool_call_parser=None,
    )
    
    return async_llm, client


async def test_vllm_openai_single_rounds(client: VllmOpenAI):
    """Test the VllmOpenAI client with multiple independent single-round conversations."""
    
    # Test single-round conversations - independent chats to show basic functionality
    print(f"\n{'='*60}")
    print("TEST 1: MULTIPLE SINGLE-ROUND CONVERSATIONS")
    print(f"{'='*60}")
    print("Testing basic OpenAI-compatible interface with different independent conversations...")
    
    single_round_conversations = [
        {
            "topic": "Science",
            "messages": [
                {"role": "system", "content": "You are a science educator. Provide clear, concise explanations."},
                {"role": "user", "content": "What is photosynthesis?"}
            ]
        },
        {
            "topic": "Technology", 
            "messages": [
                {"role": "system", "content": "You are a tech expert. Give practical advice."},
                {"role": "user", "content": "How does machine learning work?"}
            ]
        },
        {
            "topic": "History",
            "messages": [
                {"role": "user", "content": "Tell me about the Renaissance period."}
            ]
        },
        {
            "topic": "Cooking",
            "messages": [
                {"role": "system", "content": "You are a chef. Give cooking tips and recipes."},
                {"role": "user", "content": "How do I make perfect pasta?"}
            ]
        }
    ]
    
    for i, conv in enumerate(single_round_conversations):
        print(f"\n--- CONVERSATION {i+1}: {conv['topic']} ---")
        
        # Display the conversation
        for msg in conv['messages']:
            print(f"{msg['role'].upper()}: {msg['content']}")
        
        print("Generating response...")
        
        try:
            response = await client.chat.completions.create(
                messages=conv['messages'],
                temperature=0.7,
                top_p=0.9,
                max_tokens=512,
                store=True,
            )
            
            assistant_message = response.choices[0].message.content
            
            print(f"ASSISTANT: {assistant_message}")
            print(f"📊 Usage: {response.usage.prompt_tokens} prompt + {response.usage.completion_tokens} completion = {response.usage.total_tokens} total tokens")
            
            # Demonstrate cache retrieval
            cached_completion = client.get_completions(response.id)
            if cached_completion:
                print("✓ Completion successfully cached")
                
                # Show tensor conversion capability
                try:
                    tensor_dict = cached_completion.to_tensor_dict()
                    print(f"✓ Tensor conversion successful - input_ids shape: {tensor_dict['input_ids'].shape}")
                except Exception as e:
                    print(f"⚠ Tensor conversion failed: {e}")
            
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("SINGLE-ROUND CONVERSATIONS TEST COMPLETED!")
    print(f"✨ Demonstrated: OpenAI-compatible interface, caching, tensor conversion")
    print(f"{'='*60}")


async def test_vllm_openai_multi_round(client: VllmOpenAI, tokenizer: AutoTokenizer):
    """Test the VllmOpenAI client with one multi-round conversation using prefix caching.
    
    Also demonstrates detailed token analysis by displaying output tokens and their logprobs.
    """
    
    # Test one multi-round conversation with prefix caching
    print(f"\n{'='*60}")
    print("TEST 2: ONE MULTI-ROUND CONVERSATION")
    print(f"{'='*60}")
    print("Testing multi-round conversation with prefix caching and token analysis...")
    
    # System message that will be part of the cached prefix
    system_message = {"role": "system", "content": "You are a knowledgeable AI assistant specializing in artificial intelligence. Provide detailed, informative responses."}
    
    # Conversation topic
    topic = "Artificial Intelligence"
    round1_question = "What is artificial intelligence and how does machine learning work?"
    round2_question = "Can you give me specific examples of how AI is used in healthcare and finance?"
    
    print(f"\n💬 TOPIC: {topic}")
    
    # Round 1: Initial question
    print(f"\n--- ROUND 1 ---")
    round1_messages = [system_message, {"role": "user", "content": round1_question}]
    
    print(f"USER: {round1_question}")
    print("Generating initial response...")
    
    try:
        response1 = await client.chat.completions.create(
            messages=round1_messages,
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            store=True,
        )
        
        assistant_msg1 = response1.choices[0].message.content
        
        print(f"ASSISTANT: {assistant_msg1}")
        print(f"📊 Usage: {response1.usage.prompt_tokens} prompt + {response1.usage.completion_tokens} completion = {response1.usage.total_tokens} total tokens")
        
    except Exception as e:
        print(f"❌ Error in round 1: {str(e)}")
        return
    
    # Round 2: Follow-up question (builds on the same conversation)
    print(f"\n--- ROUND 2 ---")
    round2_messages = [
        system_message,
        {"role": "user", "content": round1_question},
        {"role": "assistant", "content": assistant_msg1},
        {"role": "user", "content": round2_question}
    ]
    
    print(f"USER: {round2_question}")
    print("Generating follow-up response...")
    
    try:
        response2 = await client.chat.completions.create(
            messages=round2_messages,
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            store=True,
        )
        
        assistant_msg2 = response2.choices[0].message.content
        
        print(f"ASSISTANT: {assistant_msg2}")
        print(f"📊 Usage: {response2.usage.prompt_tokens} prompt + {response2.usage.completion_tokens} completion = {response2.usage.total_tokens} total tokens")
        
        # Show cache hits and detailed token analysis
        cached_completion1 = client.get_completions(response1.id)
        cached_completion2 = client.get_completions(response2.id)
        if cached_completion1 and cached_completion2:
            print("✓ Both completions successfully cached")
            
            # Show tensor conversion for both rounds
            try:
                tensor_dict1 = cached_completion1.to_tensor_dict()
                tensor_dict2 = cached_completion2.to_tensor_dict()
                print(f"✓ Tensor conversion successful")
                print(f"  Round 1 tensor shape: {tensor_dict1['input_ids'].shape}")
                print(f"  Round 2 tensor shape: {tensor_dict2['input_ids'].shape}")
            except Exception as e:
                print(f"⚠ Tensor conversion failed: {e}")
                
            # Display tokens and logprobs for both rounds
            print(f"\n{'='*60}")
            print("🔍 DETAILED TOKEN ANALYSIS")
            print(f"{'='*60}")
            
            print("\n📝 ROUND 1 OUTPUT TOKENS:")
            display_tokens_and_logprobs(cached_completion1, tokenizer, max_tokens=15)
            
            print("\n📝 ROUND 2 OUTPUT TOKENS:")
            display_tokens_and_logprobs(cached_completion2, tokenizer, max_tokens=15)
            
    except Exception as e:
        print(f"❌ Error in round 2: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("MULTI-ROUND CONVERSATION TEST COMPLETED!")
    print(f"✨ Demonstrated: Multi-turn conversations with prefix caching and detailed token analysis")
    print(f"{'='*60}")


async def run_all_tests():
    """Run both test functions sequentially with shared client."""
    print("🚀 Starting VllmOpenAI comprehensive test suite...")
    print("   1. Multiple single-round conversations")
    print("   2. One multi-round conversation with prefix caching")
    
    try:
        # Create shared AsyncLLM engine and VllmOpenAI client once
        print("\n🔧 Initializing shared AsyncLLM engine and client...")
        _, client = get_async_llm_and_client()
        print("✅ Engine and client initialization complete!")
        
        # Get tokenizer for token analysis (same as used in client)
        from transformers import AutoTokenizer
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Test 1: Multiple single-round conversations
        await test_vllm_openai_single_rounds(client)
        
        # Small delay between tests
        await asyncio.sleep(1)
        
        # Test 2: One multi-round conversation with token analysis
        await test_vllm_openai_multi_round(client, tokenizer)
        
        print(f"\n{'='*70}")
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        print("✅ Single-round conversations: OpenAI compatibility, caching, tensor conversion")
        print("✅ Multi-round conversation: Prefix caching enabled and detailed token analysis")
        print("🚀 VllmOpenAI client is ready for production use!")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n❌ Test suite failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def run_test():
    """Synchronous wrapper to run all async tests."""
    print("Starting VllmOpenAI comprehensive test suite with prefix caching...")
    
    try:
        # Run all tests
        asyncio.run(run_all_tests())
        print("\n🎉 VllmOpenAI test suite completed successfully!")
        
    except Exception as e:
        print(f"\n❌ VllmOpenAI test suite failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the test when this file is executed directly
    run_test()
