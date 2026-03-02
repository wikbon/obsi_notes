"""Utility functions for note processing output."""

from pathlib import Path

SKIP_SUFFIXES = ('_processed', '_formatted', '_flashcards', '_hub', '_atomic')


def generate_frontmatter(original_stem: str, process_type: str) -> str:
    """Generate YAML frontmatter for processed note output files.

    Args:
        original_stem: Filename stem of the original note (without extension)
        process_type: One of 'format', 'flashcards', 'hub', 'atomic', 'both'

    Returns:
        YAML frontmatter string including opening/closing ---
    """
    tags = ['processed']
    if process_type == 'flashcards':
        tags.append('flashcards')
    elif process_type == 'format':
        tags.append('formatted_notes')
    elif process_type == 'hub':
        tags.append('hub_note')
    elif process_type == 'atomic':
        tags.append('atomic_notes')
    elif process_type == 'both':
        tags.extend(['formatted_notes', 'flashcards'])

    tag_lines = "\n".join(f"  - {tag}" for tag in tags)
    return f'---\noriginal_note: "[[{original_stem}]]"\ntags:\n{tag_lines}\n---\n\n'


def should_skip_file(filename: str) -> bool:
    """Check if a file should be skipped during batch processing.

    Skips files that have already been processed (contain processing suffixes).

    Args:
        filename: The filename to check

    Returns:
        True if the file should be skipped
    """
    stem = Path(filename).stem
    return any(stem.endswith(suffix) for suffix in SKIP_SUFFIXES)
