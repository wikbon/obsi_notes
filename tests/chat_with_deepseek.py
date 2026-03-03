#!/usr/bin/env python3
import sys

from src.core.deepseek_handler import DeepSeekHandler


def main():
    # Initialize DeepSeek with verbose output
    print("\n=== Initializing DeepSeek Chat ===\n")
    handler = DeepSeekHandler(verbose=True)

    print("\nChat initialized! Type your messages and press Enter to chat.")
    print("Type 'exit', 'quit', or press Ctrl+C to end the conversation.")
    print("Type 'history' to view the conversation history.")
    print("Type 'clear' to clear the conversation history.")
    print(
        "For mathematical problems, add: 'Please reason step by step, and put your final answer within \\boxed{}'"
    )
    print("\n")

    try:
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Check for exit commands
            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye! Chat ended.")
                break

            # Check for history command
            elif user_input.lower() == "history":
                print("\n=== Conversation History ===")
                for msg in handler.get_history():
                    print(f"{msg['role'].capitalize()}: {msg['content']}")
                continue

            # Check for clear command
            elif user_input.lower() == "clear":
                handler.clear_history()
                print("\nConversation history cleared!")
                continue

            # Add user message and get response
            handler.add_message("user", user_input)

            # Get DeepSeek response
            response = handler.create_chat_completion(
                temperature=0.6
            )  # Using recommended temperature

            # Extract and print assistant's response
            assistant_msg = response["choices"][0]["message"]["content"]
            print(f"\nAssistant: {assistant_msg}")

            # If there's boxed content, print it separately
            boxed_content = response["choices"][0].get("boxed_content")
            if boxed_content:
                print(f"\nFinal Answer: {boxed_content}")

    except KeyboardInterrupt:
        print("\n\nChat ended by user (Ctrl+C)")
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
