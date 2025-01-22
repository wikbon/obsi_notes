#!/usr/bin/env python3
import sys
from src.core.llm_handler import LLMHandler

def main():
    # Initialize the LLM with verbose output
    print("\n=== Initializing LLM Chat ===\n")
    handler = LLMHandler(verbose=True)
    
    # Set up the initial system message
    handler.add_message("system", "You are a helpful AI assistant. You provide clear, concise, and accurate responses.")
    
    print("\nChat initialized! Type your messages and press Enter to chat.")
    print("Type 'exit', 'quit', or press Ctrl+C to end the conversation.")
    print("Type 'history' to view the conversation history.")
    print("Type 'clear' to clear the conversation history.\n")
    
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye! Chat ended.")
                break
                
            # Check for history command
            elif user_input.lower() == 'history':
                print("\n=== Conversation History ===")
                for msg in handler.get_history():
                    if msg['role'] == 'system':
                        continue  # Skip system messages
                    print(f"{msg['role'].capitalize()}: {msg['content']}")
                continue
                
            # Check for clear command
            elif user_input.lower() == 'clear':
                handler.clear_history()
                # Re-add the system message
                handler.add_message("system", "You are a helpful AI assistant. You provide clear, concise, and accurate responses.")
                print("\nConversation history cleared!")
                continue
                
            # Add user message and get response
            handler.add_message("user", user_input)
            
            # Get LLM response
            response = handler.create_chat_completion(temperature=0.7)
            
            # Extract and print assistant's response
            assistant_response = response['choices'][0]['message']['content']
            print(f"\nAssistant: {assistant_response}")
            
    except KeyboardInterrupt:
        print("\n\nChat ended by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
