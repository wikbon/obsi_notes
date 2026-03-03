import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from src.config.settings import ConfigManager


class NoteParser:
    """A class to handle loading and parsing of raw notes for LLM processing."""

    def __init__(self, vault_path: str = None, file_path: str = None, config_path: str = None):
        """Initialize the NoteParser with optional vault and file paths.

        Args:
            vault_path (str, optional): Path to the Obsidian vault
            file_path (str, optional): Path to the specific note file to be parsed
            config_path (str, optional): Path to config file, defaults to standard location
        """
        self._config_manager = ConfigManager(config_path=config_path)
        self.config = self._config_manager.config
        self.vault_path = (
            Path(vault_path) if vault_path else Path(self._config_manager.get_vault_path())
        )
        self.file_path = Path(file_path) if file_path else None
        self.raw_content = None
        self.parsed_notes = []
        self.frontmatter = {}

    def set_vault_path(self, vault_path: str) -> None:
        """Set the Obsidian vault path.

        Args:
            vault_path (str): Path to the Obsidian vault
        """
        self.vault_path = Path(vault_path)

    def _extract_frontmatter(self, content: str) -> tuple[dict, str]:
        """Extract YAML frontmatter from note content.

        Args:
            content (str): Raw note content

        Returns:
            tuple[dict, str]: (frontmatter dict, content without frontmatter)
        """
        if content.startswith("---\n"):
            parts = content[4:].split("\n---\n", 1)
            if len(parts) >= 2:
                try:
                    return yaml.safe_load(parts[0]), parts[1]
                except yaml.YAMLError:
                    return {}, content
        return {}, content

    def load_file(self, file_path: Optional[str] = None) -> bool:
        """Load content from a file.

        Args:
            file_path (str, optional): Path to the file to load. If not provided,
                                     uses the path from initialization.

        Returns:
            bool: True if file was successfully loaded, False otherwise.
        """
        if file_path:
            self.file_path = Path(file_path)

        if not self.file_path:
            raise ValueError("No file path provided")

        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                content = file.read()
                self.frontmatter, self.raw_content = self._extract_frontmatter(content)
            return True
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            return False

    def get_vault_files(self, extension: str = ".md") -> List[Path]:
        """Get all note files in the vault with specified extension.

        Args:
            extension (str): File extension to filter by (default: ".md")

        Returns:
            List[Path]: List of paths to matching files
        """
        if not self.vault_path:
            raise ValueError("No vault path set. Use set_vault_path() first.")

        return list(self.vault_path.rglob(f"*{extension}"))

    def _process_internal_links(self, content: str) -> str:
        """Convert Obsidian internal links to a normalized format.

        Args:
            content (str): Content containing internal links

        Returns:
            str: Content with processed internal links
        """
        # Convert [[Note Name]] to "Note Name"
        content = re.sub(r"\[\[(.*?)\]\]", r'"\1"', content)
        # Convert [[Note Name|Alias]] to "Alias"
        content = re.sub(r"\[\[(.*?)\|(.*?)\]\]", r'"\2"', content)
        return content

    def extract_tags(self, content: str) -> List[str]:
        """Extract hashtags from content.

        Args:
            content (str): Content to extract tags from

        Returns:
            List[str]: List of extracted tags without the '#' symbol
        """
        # Match hashtags that are part of words (e.g., #idea, #task)
        pattern = r"#([\w-]+)"
        return re.findall(pattern, content)

    def extract_tasks(self, content: str) -> List[Dict[str, str]]:
        """Extract tasks from content.

        Args:
            content (str): Content to extract tasks from

        Returns:
            List[Dict[str, str]]: List of tasks with status and content
        """
        tasks = []

        # Match Markdown task items with optional status
        task_pattern = r"- \[([ xX])\] (.*?)(?=\n|$)"
        tasks.extend(
            [
                {
                    "type": "checkbox",
                    "status": bool(status.strip()),
                    "content": task_content.strip(),
                }
                for status, task_content in re.findall(task_pattern, content)
            ]
        )

        # Match #task tagged items
        task_tag_pattern = r"#task\s+(.*?)(?=\n|$)"
        tasks.extend(
            [
                {
                    "type": "tagged",
                    "status": False,  # Default to not completed
                    "content": task.strip(),
                }
                for task in re.findall(task_tag_pattern, content)
            ]
        )

        return tasks

    def extract_structure(self, content: str) -> List[Dict[str, Any]]:
        """Extract structural elements like headings and bullet points.

        Args:
            content (str): Content to analyze

        Returns:
            List[Dict[str, Any]]: List of structural elements with their content
        """
        elements = []
        lines = content.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match headings (## Heading)
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading_match:
                elements.append(
                    {
                        "type": "heading",
                        "level": len(heading_match.group(1)),
                        "content": heading_match.group(2).strip(),
                    }
                )
                continue

            # Match bullet points (- or * or +)
            bullet_match = re.match(r"^[-*+]\s+(.+)$", line)
            if bullet_match:
                elements.append({"type": "bullet", "content": bullet_match.group(1).strip()})
                continue

            # Match numbered lists (1. 2. etc)
            number_match = re.match(r"^\d+\.\s+(.+)$", line)
            if number_match:
                elements.append({"type": "numbered", "content": number_match.group(1).strip()})
                continue

            # Regular text
            elements.append({"type": "text", "content": line})

        return elements

    def split_by_timestamp(self) -> List[Dict[str, Any]]:
        """Split the content by timestamp markers.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing timestamp and content
        """
        if not self.raw_content:
            raise ValueError("No content loaded. Call load_file() first.")

        # Process internal links before splitting
        processed_content = self._process_internal_links(self.raw_content)

        # Regular expression for common timestamp formats
        timestamp_pattern = (
            r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\+\d{2}:\d{2})?|\s\d{2}:\d{2}(?::\d{2})?)"
        )

        # Split content by timestamps
        segments = re.split(f"({timestamp_pattern})", processed_content)

        parsed_notes = []
        current_timestamp = None

        for segment in segments:
            if re.match(timestamp_pattern, segment):
                current_timestamp = segment
            elif segment.strip() and current_timestamp:
                content = segment.strip()
                parsed_notes.append(
                    {
                        "timestamp": current_timestamp,
                        "content": content,
                        "tags": self.extract_tags(content),
                        "tasks": self.extract_tasks(content),
                        "structure": self.extract_structure(content),
                        "metadata": {
                            "frontmatter": self.frontmatter,
                            "source_relative_path": str(self.file_path.relative_to(self.vault_path))
                            if self.file_path
                            else None,
                        },
                    }
                )

        self.parsed_notes = parsed_notes
        return self.parsed_notes

    def get_llm_ready_format(self) -> List[str]:
        """Convert parsed notes into a format ready for LLM processing.

        Returns:
            List[str]: List of formatted strings ready for LLM processing
        """
        if not self.parsed_notes:
            self.split_by_timestamp()

        llm_ready = []
        for note in self.parsed_notes:
            # Format tasks
            task_str = ""
            if note["tasks"]:
                task_str = "\nTasks:\n" + "\n".join(
                    f"- [{' x' if task['status'] else ' '}] {task['content']}"
                    for task in note["tasks"]
                )

            # Format structure
            structure_str = ""
            if note["structure"]:
                structure_str = "\nStructure:\n" + "\n".join(
                    f"{'#' * elem['level'] + ' ' if elem['type'] == 'heading' else '- '}{elem['content']}"
                    for elem in note["structure"]
                    if elem["type"] != "text"
                )

            # Format tags
            tags_str = ""
            if note["tags"]:
                tags_str = "\nTags: " + ", ".join(f"#{tag}" for tag in note["tags"])

            formatted_note = (
                f"[{note['timestamp']}]\n{note['content']}{tags_str}{task_str}{structure_str}"
            )
            llm_ready.append(formatted_note)

        return llm_ready

    def parse_file(
        self, file_path: Optional[str] = None, export_json: bool = False
    ) -> Union[List[str], str]:
        """Convenience method to process a file in one go.

        Args:
            file_path (str, optional): Path to the file to process
            export_json (bool): If True, export results to JSON instead of returning LLM format

        Returns:
            Union[List[str], str]: List of LLM-ready formatted notes or path to JSON file if export_json=True
        """
        self.load_file(file_path)
        self.split_by_timestamp()

        if export_json:
            return self.export_to_json()
        return self.get_llm_ready_format()

    def export_to_json(self, output_path: Optional[str] = None) -> str:
        """Export processed notes to a JSON file.

        Args:
            output_path (str, optional): Path to save the JSON file. If not provided,
                                       will create a file in the configured output directory

        Returns:
            str: Path to the created JSON file
        """
        if not self.parsed_notes:
            self.split_by_timestamp()

        if not output_path:
            if not self.file_path:
                raise ValueError("No output path provided and no source file path available")

            # Get relative path from vault root
            rel_path = self.file_path.relative_to(self.vault_path)

            # Construct output path in configured directory
            processing = self._config_manager.get_processing_settings()
            output_dir = Path(processing["output_dir"])
            output_path = output_dir / rel_path.with_suffix(
                processing.get("json_extension", ".json")
            )

        # Create export data structure
        export_data = {
            "source_file": str(self.file_path),
            "processed_at": datetime.now().isoformat(),
            "note_segments": self.parsed_notes,
        }

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        return str(output_path)
