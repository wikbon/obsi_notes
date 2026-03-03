"""OpenAI handler using the modern Responses API."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FORMAT_NOTE_PROMPT = (
    "You are a highly skilled assistant designed to process raw notes into a "
    "well-structured Markdown format. Your task:\n\n"
    "Reorganize and combine information:\n"
    "    Analyze the provided text and organize it into logical sections.\n"
    "    Combine related points, move information as necessary, and ensure the content flows logically.\n"
    "    Format the notes using appropriate headings, subheadings, and bullet points.\n"
    "    Make sure to respect the specific wording and examples used by user in raw notes.\n"
    "    Do not talk about anything else, just format the notes.\n"
    "    Your task is to clean up, fix typos, fix transcription errors and organize. "
    "DON'T rephrase or rewrite using different words.\n"
    '    Do not include "```markdown" or "```" in the output.\n\n'
    "Output format:\n\n"
    "<well-organized notes>"
)


class OpenAIHandler:
    """Handler for OpenAI API interactions using the Responses API."""

    def __init__(self, model: str = None, verbose: bool = False):
        """Initialize the OpenAI handler.

        Args:
            model: Model ID to use (defaults to config or gpt-5.2-latest)
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Create a .env file with OPENAI_API_KEY=your-key"
            )
        self.client = OpenAI(api_key=api_key)

        # Load config for defaults
        from src.config.settings import ConfigManager

        config = ConfigManager()
        openai_settings = config.get_openai_settings()

        self.model = model or openai_settings.get("default_model", "gpt-5.2-latest")
        self.default_temp = openai_settings.get("temperature", 0.7)
        self.default_max_output_tokens = openai_settings.get("max_output_tokens", 4096)
        self.instructions = openai_settings.get(
            "instructions",
            "You are a helpful assistant specialized in processing and analyzing notes.",
        )

        self.message_history: List[Dict[str, str]] = []
        self._previous_response_id: Optional[str] = None

        if verbose:
            logger.info(f"OpenAI handler initialized with model: {self.model}")

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the chat history."""
        self.message_history.append({"role": role, "content": content})
        if self.verbose:
            logger.info(f"Added message - Role: {role}")

    def clear_history(self) -> None:
        """Clear the message history and conversation state."""
        self.message_history = []
        self._previous_response_id = None
        if self.verbose:
            logger.info("Message history cleared")

    def get_history(self) -> List[Dict[str, str]]:
        """Get the current message history."""
        return self.message_history

    def _build_input(self, messages: List[Dict[str, str]]) -> list:
        """Build Responses API input from message dicts.

        Filters out system messages (those go into instructions param).
        """
        input_items = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                continue  # system messages handled via instructions param
            input_items.append(
                {
                    "role": role,
                    "content": msg["content"],
                }
            )
        return input_items

    def _extract_instructions(self, messages: List[Dict[str, str]]) -> str:
        """Extract system/developer instructions from messages."""
        system_parts = [msg["content"] for msg in messages if msg["role"] == "system"]
        if system_parts:
            return "\n\n".join(system_parts)
        return self.instructions

    def _call_responses_api(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        use_previous_response: bool = False,
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call the OpenAI Responses API and return a compatible response dict.

        Args:
            messages: Message dicts with role and content
            temperature: Sampling temperature
            max_output_tokens: Max output tokens
            use_previous_response: Whether to use previous_response_id for conversation
            instructions: Override instructions (system prompt)

        Returns:
            Dict in the handler-compatible format
        """
        temp = temperature if temperature is not None else self.default_temp
        max_tokens = max_output_tokens or self.default_max_output_tokens
        instr = instructions or self._extract_instructions(messages)
        input_items = self._build_input(messages)

        # Build API call kwargs
        kwargs = {
            "model": self.model,
            "instructions": instr,
            "input": input_items,
            "temperature": temp,
            "max_output_tokens": max_tokens,
            "store": False,
        }

        if use_previous_response and self._previous_response_id:
            kwargs["previous_response_id"] = self._previous_response_id

        if self.verbose:
            logger.info(f"Calling OpenAI Responses API (model={self.model}, temp={temp})")

        try:
            response = self.client.responses.create(**kwargs)

            # Extract text from response
            output_text = response.output_text or ""

            # Store response ID for conversation chaining
            self._previous_response_id = response.id

            if self.verbose:
                logger.info(f"Response received (id={response.id}, status={response.status})")
                if hasattr(response, "usage") and response.usage:
                    logger.info(
                        f"Tokens: input={response.usage.input_tokens}, "
                        f"output={response.usage.output_tokens}"
                    )

            # Return in handler-compatible format
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": output_text,
                        },
                        "boxed_content": None,
                    }
                ]
            }

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"Error: {e}",
                        },
                        "boxed_content": None,
                    }
                ]
            }

    def create_chat_completion(
        self, messages: Optional[List[Dict[str, str]]] = None, temperature: float = None, **kwargs
    ) -> Dict[str, Any]:
        """Create a chat completion using the OpenAI Responses API.

        Uses previous_response_id for multi-turn conversation state.

        Args:
            messages: List of message dicts. If None, uses internal history.
            temperature: Sampling temperature
            **kwargs: Additional arguments (max_tokens mapped to max_output_tokens)

        Returns:
            Dict with choices[0].message.content
        """
        if messages is None:
            messages = self.message_history

        max_tokens = kwargs.pop("max_tokens", None)

        response = self._call_responses_api(
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_tokens,
            use_previous_response=True,
        )

        # Add response to history
        content = response["choices"][0]["message"]["content"]
        self.add_message("assistant", content)

        return response

    def create_chat_completion_no_history(
        self, messages: List[Dict[str, str]], temperature: float = None, **kwargs
    ) -> Dict[str, Any]:
        """Create a chat completion without affecting message history.

        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            Dict with choices[0].message.content
        """
        max_tokens = kwargs.pop("max_tokens", None)

        return self._call_responses_api(
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_tokens,
            use_previous_response=False,
        )

    def extract_atomic_notes(
        self,
        note_segments: List[Dict[str, Any]],
        temperature: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Extract atomic notes from note segments.

        Args:
            note_segments: List of segments with 'content' and optional 'timestamp'
            temperature: Sampling temperature

        Returns:
            List of dicts with 'note' key
        """
        all_extracted_notes = []

        for segment in note_segments:
            content = segment["content"]
            timestamp = segment.get("timestamp", "")

            user_prompt = (
                "You are an assistant specialized in note summarization and extraction. "
                "Your task is to extract atomic notes and return them in JSON format. "
                "An atomic note should capture one complete, self-contained idea - "
                "don't split related concepts that form a single coherent thought. "
                "DO NOT include any markdown formatting or explanation text. "
                "Return ONLY the raw JSON array."
                f"Here is a raw note segment{f' (timestamp: {timestamp})' if timestamp else ''}:\n"
                f"{content}\n"
                "Extract atomic notes following these rules:\n"
                "1. Each note should be one complete, self-contained idea.\n"
                "2. Keep related concepts together if they form a single coherent thought.\n"
                "3. Don't split definitions or explanations of a single concept.\n"
                "4. If multiple sentences make up a single idea, keep them together.\n"
                "5. If in doubt, don't split a sentence into multiple atomic notes.\n"
                "\n"
                "Return a JSON array in this exact format (no markdown, no explanation):\n"
                "[\n"
                "  {\n"
                '    "note": "the complete atomic idea here"\n'
                "  }\n"
                "]"
            )

            if self.verbose:
                logger.info(f"Processing segment{f' from {timestamp}' if timestamp else ''}")

            response = self.create_chat_completion_no_history(
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=4096,
            )

            raw_text = ""
            if response.get("choices"):
                raw_text = response["choices"][0]["message"]["content"]

            try:
                raw_text = raw_text.lstrip("\ufeff").strip()
                if raw_text.startswith("```json") and raw_text.endswith("```"):
                    raw_text = raw_text[7:-3].strip()
                elif raw_text.startswith("```") and raw_text.endswith("```"):
                    raw_text = raw_text[3:-3].strip()
                raw_text = re.sub(r",\s*}", "}", raw_text)

                if not raw_text.startswith(("{", "[")):
                    logger.error("Response does not appear to be valid JSON.")
                    continue

                extracted_list = json.loads(raw_text)
                if not isinstance(extracted_list, list):
                    extracted_list = [extracted_list]

                for note in extracted_list:
                    note_text = note.get("note", "")
                    click.echo(f"\nOriginal input to LLM:\n{content}\n")
                    click.echo(f"\nExtracted note: {note_text}")

                all_extracted_notes.extend(extracted_list)

                if self.verbose:
                    logger.info(f"Extracted {len(extracted_list)} atomic notes")

            except json.JSONDecodeError:
                logger.error(
                    f"JSON parse error for segment{f' at {timestamp}' if timestamp else ''}"
                )
                logger.error(f"Raw response:\n{raw_text}\n")

        return all_extracted_notes

    def generate_daily_hub_note(
        self,
        parsed_notes: List[Dict[str, Any]],
        source_file: str,
        date_str: str = None,
        temperature: float = 0.3,
        save_markdown: bool = True,
        output_dir: Optional[Path] = None,
        vault_path: Optional[Path] = None,
    ) -> str:
        """Generate a structured daily hub note from parsed notes.

        Args:
            parsed_notes: List of dicts with 'timestamp' and 'content'
            source_file: Path to the source file
            date_str: Date string (extracted from filename if None)
            temperature: Sampling temperature
            save_markdown: Whether to save to file
            output_dir: Output directory (required if save_markdown)
            vault_path: Vault root path (required if save_markdown)

        Returns:
            Formatted markdown string
        """
        if date_str is None:
            source_path = Path(source_file)
            parts = source_path.stem.split("-")
            if len(parts) >= 3:
                date_str = f"{parts[0]}-{parts[1]}-{parts[2]}"
            else:
                date_str = source_path.stem

        lines_for_prompt = []
        for note in parsed_notes:
            timestamp = note.get("timestamp", "N/A")
            content = note.get("content", "")
            lines_for_prompt.append(f"- Timestamp: {timestamp} | Content: {content}")

        combined_text = "\n".join(lines_for_prompt)

        user_prompt = (
            f"Here is a list of timestamped notes for {date_str}:\n"
            f"{combined_text}\n\n"
            "Please transform this note into a well-structured daily note using the following guidelines:\n"
            "1) Create a clear hierarchical structure with appropriate headings\n"
            "2) Organize related thoughts and ideas under common themes\n"
            "3) Enhance readability with bullet points and consistent formatting\n"
            "4) Include these sections:\n"
            "   - Daily Overview\n"
            "   - Key Insights or Learnings\n"
            "   - Project Updates (if applicable)\n"
            "   - Questions & Ideas for Further Exploration\n"
            "   - Action Items & Next Steps\n\n"
            f"Use '# Daily Note - {date_str}' as the main heading.\n"
            "Remove original timestamps while preserving logical flow.\n"
            'Do not include "```markdown" or "```" wrappers in the output.'
        )

        hub_instructions = (
            "You are an assistant that organizes daily mind dumps into a nicely formatted "
            "Markdown document, complete with headings, subheadings, summaries, and an Action Items list."
        )

        if self.verbose:
            logger.info(f"Generating hub note for {date_str}")

        response = self.create_chat_completion_no_history(
            messages=[
                {"role": "system", "content": hub_instructions},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=4096,
        )

        if response.get("choices") and len(response["choices"]) > 0:
            markdown_result = response["choices"][0]["message"]["content"]
            # Clean markdown wrappers
            if markdown_result.startswith("```markdown\n"):
                markdown_result = markdown_result[11:]
            elif markdown_result.startswith("```\n"):
                markdown_result = markdown_result[4:]
            if markdown_result.endswith("\n```"):
                markdown_result = markdown_result[:-4]
            markdown_result = markdown_result.strip()
        else:
            markdown_result = f"# Daily Note - {date_str}\n\n*(No response from LLM)*"

        if save_markdown:
            if output_dir is None or vault_path is None:
                raise ValueError(
                    "output_dir and vault_path must be provided when save_markdown is True"
                )

            source_path = Path(source_file)
            rel_path = source_path.relative_to(vault_path)
            output_path = output_dir / rel_path.with_suffix(".hub.md")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_result)

            if self.verbose:
                logger.info(f"Saved hub note to {output_path}")

        return markdown_result

    def format_note(self, raw_content: str, temperature: float = 0.3) -> str:
        """Format raw notes into well-structured markdown.

        Args:
            raw_content: Raw note text to format
            temperature: Sampling temperature (low for consistency)

        Returns:
            Formatted markdown string
        """
        response = self.create_chat_completion_no_history(
            messages=[
                {"role": "system", "content": FORMAT_NOTE_PROMPT},
                {"role": "user", "content": raw_content},
            ],
            temperature=temperature,
            max_tokens=4096,
        )

        result = response["choices"][0]["message"]["content"]
        # Clean markdown wrappers
        if result.startswith("```markdown\n"):
            result = result[11:]
        elif result.startswith("```\n"):
            result = result[4:]
        if result.endswith("\n```"):
            result = result[:-4]
        return result.strip()

    def clear_chat_history(self) -> None:
        """Clear the chat history."""
        self.message_history = []
        self._previous_response_id = None
        if self.verbose:
            logger.info("Chat history cleared")
