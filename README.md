# Obsidian Notes Processor

A CLI tool that processes Obsidian vault notes using local LLMs. Supports multiple backends: **llama-cpp-python** (local GGUF models), **LM Studio**, and **OpenAI**.

## Features

- **Atomic note extraction** — Break daily notes into self-contained ideas using LLM analysis
- **Hub note generation** — Create structured daily summaries with themes, action items, and cross-references
- **Flashcard generation** — Generate study flashcards from note content
- **Interactive chat** — Chat with an LLM about your notes
- **Multiple LLM backends** — DeepSeek (llama-cpp-python), LM Studio, OpenAI
- **Batch processing** — Process entire directories of notes at once

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- One of the supported LLM backends:
  - **Local GGUF models** via llama-cpp-python (optional CUDA for GPU acceleration)
  - **LM Studio** running locally
  - **OpenAI API** key

## Installation

```bash
git clone <repo-url>
cd obsidian_notes
uv sync
```

## Configuration

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your settings:
   ```bash
   # Required: path to your Obsidian vault
   OBSIDIAN_VAULT_PATH=/path/to/your/obsidian/vault

   # For OpenAI backend
   OPENAI_API_KEY=your-api-key

   # For LM Studio backend
   LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
   LMSTUDIO_MODEL_ID=your-model-id

   # For local GGUF models (configure the ones you use)
   DEEPSEEK_LLAMA_8B_PATH=/path/to/model.gguf
   ```

3. Optionally edit `src/config/config.yaml` for non-secret settings (temperature, context window, file patterns).

## Usage

```bash
uv run python -m src.cli_interface
```

The interactive CLI lets you:
- Select a note or directory to process
- Choose a processing mode (atomic notes, hub notes, flashcards, chat)
- Select which LLM model/backend to use

## Project Structure

```
obsidian_notes/
├── src/
│   ├── cli_interface.py          # Interactive CLI entry point
│   ├── config/
│   │   ├── config.yaml           # Non-secret configuration
│   │   └── settings.py           # ConfigManager (env var priority)
│   ├── core/
│   │   ├── deepseek_handler.py   # llama-cpp-python backend
│   │   ├── llm_handler.py        # Generic llama-cpp-python backend
│   │   ├── lmstudio_handler.py   # LM Studio API backend
│   │   └── openai_handler.py     # OpenAI API backend
│   └── utils/
│       ├── atomic_note_extractor.py
│       ├── flashcard_generator.py
│       ├── helpers.py
│       └── note_parser.py        # Obsidian markdown parser
├── tests/
├── .env.example
├── pyproject.toml
└── LICENSE
```

## License

MIT License. See [LICENSE](LICENSE) for details.
