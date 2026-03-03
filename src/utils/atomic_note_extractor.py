import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click

from src.core.deepseek_handler import DeepSeekHandler
from src.core.llm_handler import LLMHandler
from src.utils.note_parser import NoteParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AtomicNoteExtractor:
    """A class to extract atomic notes from raw note content using LLM."""

    def __init__(
        self,
        vault_path: Optional[str] = None,
        config_path: Optional[str] = None,
        verbose: bool = False,
        llm_handler: Optional[Union[LLMHandler, DeepSeekHandler]] = None,
    ):
        """Initialize the AtomicNoteExtractor.

        Args:
            vault_path (str, optional): Path to the Obsidian vault
            config_path (str, optional): Path to config file
            verbose (bool): Whether to print detailed logs
            llm_handler (Union[LLMHandler, DeepSeekHandler], optional): Existing LLM handler instance to use
        """
        self.parser = NoteParser(vault_path=vault_path, config_path=config_path)
        self.llm_handler = llm_handler if llm_handler else LLMHandler(verbose=verbose)
        self.verbose = verbose

    def process_file(
        self, file_path: str, export_json: bool = False, temperature: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Process a single file and extract atomic notes.

        Args:
            file_path (str): Path to the file to process
            export_json (bool): Whether to export results to JSON
            temperature (float): Temperature setting for LLM output

        Returns:
            List[Dict[str, Any]]: List of extracted atomic notes
        """
        # Process the file and get segments
        self.parser.parse_file(file_path)
        segments = self.parser.parsed_notes

        if self.verbose:
            logger.info(f"Processing {len(segments)} segments from {file_path}")

        # Extract atomic notes using LLM
        atomic_notes = self.llm_handler.extract_atomic_notes(
            note_segments=segments, temperature=temperature
        )

        if export_json:
            self._export_to_json(atomic_notes, file_path)

        return atomic_notes

    def process_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        export_json: bool = False,
        temperature: float = 0.7,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Process all markdown files in a directory.

        Args:
            directory_path (str): Path to directory containing notes
            recursive (bool): Whether to process subdirectories
            export_json (bool): Whether to export results to JSON
            temperature (float): Temperature setting for LLM output

        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary mapping file paths to their atomic notes
        """
        directory = Path(directory_path)
        pattern = "**/*.md" if recursive else "*.md"

        results = {}
        for file_path in directory.glob(pattern):
            try:
                atomic_notes = self.process_file(
                    str(file_path), export_json=export_json, temperature=temperature
                )
                results[str(file_path)] = atomic_notes
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")

        return results

    def _export_to_json(self, atomic_notes: List[Dict[str, Any]], source_file: str) -> str:
        """Export atomic notes to a JSON file.

        Args:
            atomic_notes (List[Dict[str, Any]]): List of atomic notes to export
            source_file (str): Path to the source file

        Returns:
            str: Path to the created JSON file
        """
        source_path = Path(source_file)
        output_dir = Path(self.parser.config["processing"]["output_dir"])

        # Get relative path from vault root and construct output path
        rel_path = source_path.relative_to(self.parser.vault_path)
        output_path = output_dir / rel_path.with_suffix(".atomic.json")

        # Create export data structure
        export_data = {"source_file": str(source_file), "atomic_notes": atomic_notes}

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        # Always log the export, not just in verbose mode
        logger.info(f"Exported atomic notes to: {output_path}")
        click.echo(f"Saved atomic notes to: {output_path}")

        return str(output_path)
