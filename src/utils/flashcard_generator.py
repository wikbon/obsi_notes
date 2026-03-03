#!/usr/bin/env python3

import logging
from pathlib import Path
from typing import List

from src.utils.helpers import generate_frontmatter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FlashcardGenerator:
    """Generates Obsidian-compatible flashcards for spaced repetition."""

    def __init__(self, llm_handler):
        """Initialize the flashcard generator.

        Args:
            llm_handler: LLM handler instance (DeepSeek, LMStudio, or OpenAI)
        """
        self.llm = llm_handler

    def generate_flashcards(self, note_path: Path, deck_tag: str = "#flashcard") -> str:
        """Generate flashcards from a note and save them next to the original file.

        Args:
            note_path: Path to the original note file
            deck_tag: Tag to assign flashcards to a specific deck

        Returns:
            Path to the generated flashcards file
        """
        # Read the original note
        note_content = note_path.read_text(encoding="utf-8")

        # Prepare system prompt for the LLM
        system_prompt = """
        Create spaced repetition flashcards from the given note content. Follow these rules:
        1. Extract key concepts and create question-answer pairs
        2. Use :: for single-line basic cards
        3. Use ::: for single-line bidirectional cards when appropriate
        6. Each card should test one specific concept
        7. Questions should be clear and unambiguous
        8. Answers should be concise but complete
        9. Don't use markdown in questions or answers
        10. Structure each line as a simple text

        For single line cards, the question and answer should be separated by a double colon (::). the question and the answer should be on the same line:
        Question :: Answer

        For multi-line cards, the question and answer should be separated by a question mark (?). the question and the answer should be on separate lines:
        Question
        ?
        Answer

        it's important to have question mark (?) or (??) for multi-line cards on a separate line between the Question and Answer.

        Examples:

        === Single-line Basic Cards (::) ===
        Geography Example:
        What is the capital of France? :: Paris

        Science Example:
        What is the chemical symbol for Gold? :: Au

        History Example:
        Who was the first president of the United States? :: George Washington

        === Single-line Bidirectional Cards (:::) ===
        Biology Example:
        Photosynthesis ::: Process where plants convert sunlight into chemical energy

        Language Example:
        Bonjour ::: Hello in French

        Math Example:
        Pi ::: Mathematical constant approximately equal to 3.14159

        === Multi-line Basic Cards (?) ===
        Computer Components Example:
        What are the main components of a computer?
        ?
        CPU (Central Processing Unit)
        RAM (Random Access Memory)
        Storage (HDD/SSD)
        Motherboard
        Power Supply Unit

        Solar System Example:
        List the planets in our solar system in order from the Sun
        ?
        Mercury
        Venus
        Earth
        Mars
        Jupiter
        Saturn
        Uranus
        Neptune

        === Multi-line Bidirectional Cards (??) ===
        Programming Example:
        Python List Methods
        ??
        append() adds element to end
        pop() removes and returns last element
        extend() adds elements from iterable
        insert() adds element at specific index

        States of Matter Example:
        Three states of matter and their properties
        ??
        Solid has fixed shape and volume
        Liquid has fixed volume but takes container shape
        Gas has no fixed shape or volume
        """

        # Generate flashcards using LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create flashcards from this note:\n\n{note_content}"},
        ]
        response = self.llm.create_chat_completion_no_history(
            messages,
            max_tokens=4050,  # Limit response length for flashcards
        )
        flashcard_content = response["choices"][0]["message"]["content"]

        # Create flashcards file with proper frontmatter
        frontmatter = generate_frontmatter(note_path.stem, "flashcards")
        flashcard_content = frontmatter + flashcard_content
        flashcard_path = note_path.parent / f"{note_path.stem}_flashcards.md"
        flashcard_path.write_text(flashcard_content, encoding="utf-8")

        logger.info(f"Generated flashcards at: {flashcard_path}")
        return str(flashcard_path)

    def generate_flashcards_content(self, note_content: str, deck_tag: str = "#flashcard") -> str:
        """Generate flashcard text content without saving to file.

        Args:
            note_content: Raw note text to generate flashcards from
            deck_tag: Tag for the flashcard deck

        Returns:
            Flashcard content string (without frontmatter)
        """
        system_prompt = self._get_flashcard_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create flashcards from this note:\n\n{note_content}"},
        ]
        response = self.llm.create_chat_completion_no_history(
            messages,
            max_tokens=4050,
        )
        return response["choices"][0]["message"]["content"]

    def _get_flashcard_prompt(self) -> str:
        """Return the flashcard generation system prompt."""
        return """
        Create spaced repetition flashcards from the given note content. Follow these rules:
        1. Extract key concepts and create question-answer pairs
        2. Use :: for single-line basic cards
        3. Use ::: for single-line bidirectional cards when appropriate
        6. Each card should test one specific concept
        7. Questions should be clear and unambiguous
        8. Answers should be concise but complete
        9. Don't use markdown in questions or answers
        10. Structure each line as a simple text

        For single line cards, the question and answer should be separated by a double colon (::). the question and the answer should be on the same line:
        Question :: Answer

        For multi-line cards, the question and answer should be separated by a question mark (?). the question and the answer should be on separate lines:
        Question
        ?
        Answer

        it's important to have question mark (?) or (??) for multi-line cards on a separate line between the Question and Answer.

        Examples:

        === Single-line Basic Cards (::) ===
        What is the capital of France? :: Paris
        What is the chemical symbol for Gold? :: Au

        === Single-line Bidirectional Cards (:::) ===
        Photosynthesis ::: Process where plants convert sunlight into chemical energy
        Bonjour ::: Hello in French

        === Multi-line Basic Cards (?) ===
        What are the main components of a computer?
        ?
        CPU (Central Processing Unit)
        RAM (Random Access Memory)
        Storage (HDD/SSD)
        Motherboard
        Power Supply Unit

        === Multi-line Bidirectional Cards (??) ===
        Python List Methods
        ??
        append() adds element to end
        pop() removes and returns last element
        extend() adds elements from iterable
        insert() adds element at specific index
        """

    def batch_generate_flashcards(
        self, note_paths: List[Path], deck_tag: str = "#flashcard"
    ) -> List[str]:
        """Generate flashcards for multiple notes.

        Args:
            note_paths: List of paths to note files
            deck_tag: Tag to assign flashcards to a specific deck

        Returns:
            List of paths to generated flashcard files
        """
        generated_files = []
        for note_path in note_paths:
            try:
                flashcard_path = self.generate_flashcards(note_path, deck_tag)
                generated_files.append(flashcard_path)
            except Exception as e:
                logger.error(f"Failed to generate flashcards for {note_path}: {str(e)}")
        return generated_files
