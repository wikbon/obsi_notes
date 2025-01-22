import pytest
import logging
from src.core.llm_handler import LLMHandler

def test_llm_handler_initialization():
    """Test that LLMHandler initializes correctly."""
    handler = LLMHandler(verbose=True)
    assert handler.model_path.endswith("Yi-1.5-34B-Chat-Q6_K.gguf")
    assert handler.llm is not None
    print("\nLLM Handler initialized successfully")

def test_chat_history():
    """Test chat history functionality."""
    handler = LLMHandler(verbose=True)
    
    # Add some messages
    handler.add_message("user", "Hello!")
    handler.add_message("assistant", "Hi there! How can I help you?")
    
    history = handler.get_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    
    # Clear history
    handler.clear_history()
    assert len(handler.get_history()) == 0
    print("\nChat history functionality working correctly")

def test_interactive_chat():
    """Test interactive chat with message history."""
    handler = LLMHandler(verbose=True)
    
    # First interaction
    handler.add_message("system", "You are a helpful AI assistant.")
    handler.add_message("user", "What's your favorite color?")
    
    response = handler.create_chat_completion(temperature=0.7)
    print("\nFirst interaction complete")
    print("Response:", response["choices"][0]["message"]["content"])
    
    # Second interaction using history
    handler.add_message("user", "Why do you like that color?")
    response = handler.create_chat_completion(temperature=0.7)
    print("\nSecond interaction complete")
    print("Response:", response["choices"][0]["message"]["content"])
    
    # Print full history
    print("\nFull chat history:")
    for msg in handler.get_history():
        print(f"{msg['role']}: {msg['content']}")

def test_chat_completion_with_function_calling():
    """Test chat completion with function calling capability."""
    handler = LLMHandler(verbose=True)
    messages = [
        {"role": "user", "content": "Extract the age: John is 30 years old"}
    ]
    
    tools = [{
        "type": "function",
        "function": {
            "name": "extract_age",
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "integer",
                        "description": "The age extracted from the text"
                    }
                },
                "required": ["age"]
            }
        }
    }]
    
    response = handler.create_chat_completion(
        messages=messages,
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": "extract_age"}
        },
        temperature=0.0
    )
    
    print("\nFunction calling test complete")
    print("Response:", response["choices"][0]["message"])

def test_invalid_model_path():
    """Test that invalid model path raises appropriate error."""
    with pytest.raises(Exception) as exc_info:  # Specific exception type from llama_cpp
        LLMHandler(model_path="/nonexistent/path/model.gguf")
    print("\nInvalid model path test complete")
    print("Error caught successfully:", str(exc_info.value))

if __name__ == '__main__':
    # Configure logging for better visibility
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("\n=== Running LLM Handler Tests ===\n")
    
    # Run all tests
    pytest.main([__file__, '-v', '--capture=no'])
    
    print("\n=== Interactive Chat Demo ===\n")
    # Additional interactive demo
    handler = LLMHandler(verbose=True)
    
    # Example conversation
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello! Can you help me with a task?"},
        {"role": "assistant", "content": "Of course! I'd be happy to help. What task do you need assistance with?"},
        {"role": "user", "content": "Tell me a short joke about programming."}
    ]
    
    for msg in messages:
        handler.add_message(msg["role"], msg["content"])
        print(f"{msg['role'].capitalize()}: {msg['content']}")
    
    response = handler.create_chat_completion(temperature=0.7)
    print(f"\nAssistant: {response['choices'][0]['message']['content']}")
