from unittest.mock import MagicMock, patch

import pytest

from src.core.llm_handler import LLMHandler


def test_model_path_required():
    """Test that LLMHandler raises ValueError when model_path is empty."""
    with pytest.raises(ValueError, match="model_path is required"):
        LLMHandler()


def test_model_path_required_explicit_empty():
    """Test that explicitly passing empty string raises ValueError."""
    with pytest.raises(ValueError, match="model_path is required"):
        LLMHandler(model_path="")


@patch("src.core.llm_handler.Llama")
def test_initialization_with_valid_path(mock_llama):
    """Test that LLMHandler initializes when a model_path is provided."""
    mock_llama.return_value = MagicMock()
    handler = LLMHandler(model_path="/some/model.gguf")
    assert handler.model_path == "/some/model.gguf"
    assert handler.message_history == []
    mock_llama.assert_called_once()


@patch("src.core.llm_handler.Llama")
def test_chat_history(mock_llama):
    """Test chat history functionality."""
    mock_llama.return_value = MagicMock()
    handler = LLMHandler(model_path="/some/model.gguf")

    handler.add_message("user", "Hello!")
    handler.add_message("assistant", "Hi there!")

    history = handler.get_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"

    handler.clear_history()
    assert len(handler.get_history()) == 0


@patch("src.core.llm_handler.Llama")
def test_create_chat_completion(mock_llama):
    """Test chat completion uses message history when no messages passed."""
    mock_instance = MagicMock()
    mock_instance.create_chat_completion.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "test response"}}]
    }
    mock_llama.return_value = mock_instance

    handler = LLMHandler(model_path="/some/model.gguf")
    handler.add_message("user", "Hello")
    response = handler.create_chat_completion()

    assert response["choices"][0]["message"]["content"] == "test response"
    # Assistant response should be appended to history
    assert len(handler.get_history()) == 2


def test_invalid_model_path():
    """Test that a nonexistent model path raises an error from llama_cpp."""
    with pytest.raises(Exception):
        LLMHandler(model_path="/nonexistent/path/model.gguf")
