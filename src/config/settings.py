"""Configuration manager for the obsidian notes processing tool."""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """Loads and provides access to application configuration."""

    DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return {}

    def get_vault_path(self) -> str:
        """Get the vault path with ~ expansion."""
        path = self.config.get('vault', {}).get('path', '')
        return os.path.expanduser(path)

    def get_processing_settings(self) -> Dict[str, Any]:
        """Get processing configuration."""
        return self.config.get('processing', {
            'output_dir': '',
            'json_extension': '.json',
        })

    def get_file_settings(self) -> Dict[str, Any]:
        """Get file handling settings."""
        return self.config.get('files', {
            'ignore_patterns': ['.git', '.obsidian', '.trash', '.DS_Store'],
            'supported_extensions': ['.md', '.markdown', '.txt', '.srt'],
        })

    def get_openai_settings(self) -> Dict[str, Any]:
        """Get OpenAI API settings."""
        return self.config.get('openai', {
            'default_model': 'gpt-5.2-latest',
            'temperature': 0.7,
            'max_output_tokens': 4096,
            'instructions': 'You are a helpful assistant specialized in processing and analyzing notes.',
        })

    def get_local_llm_settings(self) -> Dict[str, Any]:
        """Get local LLM settings for DeepSeek/LMStudio."""
        return self.config.get('local_llm', {
            'temperature': 0.6,
            'max_tokens': 4096,
            'n_ctx': 8192,
        })

    def get_models(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations for llama-cpp-python."""
        return self.config.get('models', {})
