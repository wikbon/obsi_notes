"""Configuration manager for the obsidian notes processing tool."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """Loads and provides access to application configuration.

    Priority: environment variables > config.yaml values > sensible defaults.
    """

    DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return {}

    def get_vault_path(self) -> str:
        """Get vault path. Env var OBSIDIAN_VAULT_PATH takes priority."""
        env_path = os.getenv("OBSIDIAN_VAULT_PATH")
        if env_path:
            return os.path.expanduser(env_path)
        path = self.config.get("vault", {}).get("path", "")
        if not path:
            raise ValueError(
                "Vault path not configured. Set OBSIDIAN_VAULT_PATH environment variable "
                "or vault.path in config.yaml"
            )
        return os.path.expanduser(path)

    def get_output_dir(self) -> str:
        """Get output directory. Env var OBSIDIAN_OUTPUT_DIR takes priority."""
        env_path = os.getenv("OBSIDIAN_OUTPUT_DIR")
        if env_path:
            return os.path.expanduser(env_path)
        path = self.config.get("processing", {}).get("output_dir", "")
        if path:
            return os.path.expanduser(path)
        # Default: vault_path/processed
        return str(Path(self.get_vault_path()) / "processed")

    def get_processing_settings(self) -> Dict[str, Any]:
        """Get processing configuration with env var override for output_dir."""
        settings = self.config.get(
            "processing",
            {
                "output_dir": "",
                "json_extension": ".json",
            },
        )
        settings["output_dir"] = self.get_output_dir()
        return settings

    def get_file_settings(self) -> Dict[str, Any]:
        """Get file handling settings."""
        return self.config.get(
            "files",
            {
                "ignore_patterns": [".git", ".obsidian", ".trash", ".DS_Store"],
                "supported_extensions": [".md", ".markdown", ".txt", ".srt"],
            },
        )

    def get_openai_settings(self) -> Dict[str, Any]:
        """Get OpenAI API settings."""
        return self.config.get(
            "openai",
            {
                "default_model": "gpt-5.2-latest",
                "temperature": 0.7,
                "max_output_tokens": 4096,
                "instructions": "You are a helpful assistant specialized in processing and analyzing notes.",
            },
        )

    def get_local_llm_settings(self) -> Dict[str, Any]:
        """Get local LLM settings for DeepSeek/LMStudio."""
        return self.config.get(
            "local_llm",
            {
                "temperature": 0.6,
                "max_tokens": 4096,
                "n_ctx": 8192,
            },
        )

    def get_lmstudio_settings(self) -> Dict[str, str]:
        """Get LM Studio connection settings. Env vars take priority."""
        settings = self.config.get("lmstudio", {})
        base_url = os.getenv("LMSTUDIO_BASE_URL") or settings.get(
            "base_url", "http://127.0.0.1:1234/v1"
        )
        model_id = os.getenv("LMSTUDIO_MODEL_ID") or settings.get("model_id", "")
        return {
            "base_url": base_url,
            "model_id": model_id,
        }

    def get_models(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations. Env vars override GGUF paths from config."""
        models = dict(self.config.get("models", {}))

        env_mapping = {
            "DeepSeek-Llama-8B": "DEEPSEEK_LLAMA_8B_PATH",
            "DeepSeek-Qwen-32B": "DEEPSEEK_QWEN_32B_PATH",
            "DeepSeek-Llama-70B": "DEEPSEEK_LLAMA_70B_PATH",
        }
        for model_name, env_var in env_mapping.items():
            path = os.getenv(env_var)
            if path:
                if model_name not in models:
                    models[model_name] = {"n_gpu_layers": -1, "n_ctx": 8192}
                models[model_name]["path"] = path

        # Filter out models with no path configured
        return {k: v for k, v in models.items() if v.get("path")}
