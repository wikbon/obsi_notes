#!/usr/bin/env python3

import click
import inquirer
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import sys
import logging

from src.core.deepseek_handler import DeepSeekHandler
from src.core.lmstudio_handler import LMStudioHandler
from src.core.openai_handler import OpenAIHandler
from src.utils.note_parser import NoteParser
from src.utils.atomic_note_extractor import AtomicNoteExtractor
from src.utils.flashcard_generator import FlashcardGenerator
from src.utils.helpers import should_skip_file, generate_frontmatter
from src.config.settings import ConfigManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load model configurations from config
_config_manager = ConfigManager()
MODELS = _config_manager.get_models()

class NoteCLI:
    """CLI interface for interacting with notes."""
    
    def __init__(self, vault_path: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize the CLI interface.
        
        Args:
            vault_path: Path to the Obsidian vault
            config_path: Path to config file
        """
        # First select implementation
        impl_questions = [
            inquirer.List('implementation',
                        message="Select the LLM implementation to use:",
                        choices=['llama-cpp-python', 'LM Studio', 'OpenAI (gpt-5.2)'],
                        carousel=True)
        ]
        impl_answers = inquirer.prompt(impl_questions)
        selected_impl = impl_answers['implementation']
        
        if selected_impl == 'llama-cpp-python':
            # Ask user to select model
            questions = [
                inquirer.List('model',
                            message="Select the model to use:",
                            choices=list(MODELS.keys()))
            ]
            answers = inquirer.prompt(questions)
            selected_model = answers['model']
            model_config = MODELS[selected_model]

            logger.info(f"Initializing with llama-cpp-python model: {selected_model}")
            self.llm = DeepSeekHandler(
                model_path=model_config['path'],
                n_gpu_layers=model_config['n_gpu_layers'],
                n_ctx=model_config['n_ctx'],
                verbose=True
            )
        elif selected_impl == 'LM Studio':
            logger.info("Initializing with LM Studio")
            self.llm = LMStudioHandler(verbose=True)
        else:
            logger.info("Initializing with OpenAI (gpt-5.2)")
            self.llm = OpenAIHandler(verbose=True)
        
        self.parser = NoteParser(vault_path=vault_path, config_path=config_path)
        self.extractor = AtomicNoteExtractor(
            vault_path=vault_path,
            config_path=config_path,
            verbose=True,
            llm_handler=self.llm
        )
        self.flashcard_generator = FlashcardGenerator(llm_handler=self.llm)
        
    def get_daily_notes(self) -> List[Path]:
        """Get list of daily notes sorted by date.
        
        Returns:
            List of paths to daily notes
        """
        # Get all markdown files
        all_files = self.parser.get_vault_files()
        
        # Filter for daily notes (assuming format YYYY-MM-DD.md)
        daily_notes = [
            f for f in all_files 
            if f.stem.replace('.md', '').replace('-', '').isdigit() 
            and len(f.stem) == 10
        ]
        
        # Sort by date newest first)
        return sorted(daily_notes, reverse=True)
        
    def select_note(self) -> Optional[Path]:
        """Prompt user to select a daily note.
        
        Returns:
            Selected note path or None if cancelled
        """
        daily_notes = self.get_daily_notes()
        if not daily_notes:
            click.echo("No daily notes found!")
            return None
            
        # Create choices list with dates and relative paths
        choices = [
            f"{note.stem} - {note.relative_to(self.parser.vault_path)}"
            for note in daily_notes
        ]
        
        questions = [
            inquirer.List(
                'note',
                message="Select a daily note",
                choices=choices,
                carousel=True
            )
        ]
        
        try:
            answers = inquirer.prompt(questions)
            if not answers:  # User cancelled
                return None
                
            selected = answers['note'].split(' - ')[1]  # Get the relative path part
            return self.parser.vault_path / selected
            
        except KeyboardInterrupt:
            click.echo("\nOperation cancelled by user")
            return None
            
    def chat_about_note(self, note_path: Path) -> None:
        """Start a chat session about the entire note.
        
        Args:
            note_path: Path to the note file
        """
        # Load the note content
        self.parser.load_file(str(note_path))
        content = self.parser.raw_content
        
        click.echo("\nStarting chat session about the note. Type 'exit' to end.\n")
        
        # Initialize chat with the note content
        self.llm.clear_history()
        
        # Add the note content as user context
        context_message = f"Here is the note I want to discuss:\n\n{content}"
        self.llm.add_message("user", context_message)
        
        # Get initial response from assistant
        initial_response = self.llm.create_chat_completion(
            messages=self.llm.get_history(),
            temperature=0.7
        )
        
        if initial_response.get("choices"):
            assistant_msg = initial_response["choices"][0]["message"]["content"]
            click.echo(f"\nAssistant: {assistant_msg}")
        
        while True:
            try:
                # Get user input
                user_input = click.prompt("\nYou", type=str)
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                    
                # Add user message to history
                self.llm.add_message("user", user_input)
                
                # Get AI response using full conversation history
                response = self.llm.create_chat_completion(
                    messages=self.llm.get_history(),
                    temperature=0.7
                )
                
                # Extract and display response
                if response.get("choices"):
                    ai_response = response["choices"][0]["message"]["content"]
                    click.echo(f"\nAssistant: {ai_response}")
                    # Add assistant's response to history
                    self.llm.add_message("assistant", ai_response)
                else:
                    click.echo("\nError: Failed to get response from AI")
                    
            except KeyboardInterrupt:
                break
                
        click.echo("\nChat session ended.")
        
    def process_atomic_notes(self, note_path: Path) -> None:
        """Process note into atomic notes.
        
        Args:
            note_path: Path to the note file
        """
        click.echo(f"\nProcessing atomic notes from {note_path.name}...")
        
        try:
            # Ask about JSON export
            questions = [
                inquirer.List(
                    'export_json',
                    message="Export results to JSON?",
                    choices=[
                        ('Yes', True),
                        ('No', False)
                    ],
                    carousel=True
                )
            ]
            
            answers = inquirer.prompt(questions)
            if not answers:
                return
                
            export_json = answers['export_json']
            
            atomic_notes = self.extractor.process_file(
                file_path=str(note_path),
                export_json=export_json,
                temperature=0.7
            )
            
            # Display results
            click.echo("\nExtracted atomic notes:")
            for note in atomic_notes:
                click.echo(f"- {note.get('note', '')}")
                
        except Exception as e:
            click.echo(f"Error processing note: {str(e)}")

    def process_directory_atomic_notes(self, directory_path: Path) -> None:
        """Process all markdown files in a directory and extract atomic notes.
        
        Args:
            directory_path: Path to the directory containing notes
        """
        click.echo(f"\nProcessing atomic notes in directory: {directory_path}")
        
        try:
            # Ask for recursive processing and JSON export
            questions = [
                inquirer.List(
                    'recursive',
                    message="Process subdirectories recursively?",
                    choices=[
                        ('Yes', True),
                        ('No', False)
                    ],
                    carousel=True
                ),
                inquirer.List(
                    'export_json',
                    message="Export results to JSON?",
                    choices=[
                        ('Yes', True),
                        ('No', False)
                    ],
                    carousel=True
                )
            ]
            
            answers = inquirer.prompt(questions)
            if not answers:
                return
                
            recursive = answers['recursive']
            export_json = answers['export_json']
            
            # Process the directory
            results = self.extractor.process_directory(
                directory_path=str(directory_path),
                recursive=recursive,
                export_json=export_json,
                temperature=0.7
            )
            
            # Display results
            click.echo("\nProcessing complete!")
            for file_path, atomic_notes in results.items():
                click.echo(f"\nFile: {file_path}")
                click.echo("Atomic notes:")
                for note in atomic_notes:
                    click.echo(f"- {note.get('note', '')}")
                
        except Exception as e:
            click.echo(f"Error processing directory: {str(e)}")

    def generate_hub_note(self, note_path: Path) -> None:
        """Generate a structured hub note from the daily note.
        
        Args:
            note_path: Path to the note file
        """
        click.echo(f"\nGenerating hub note from {note_path.name}...")
        
        try:
            # First process the file to get parsed notes
            self.parser.parse_file(str(note_path))
            parsed_notes = self.parser.parsed_notes
            
            # Generate hub note
            hub_note = self.llm.generate_daily_hub_note(
                parsed_notes=parsed_notes,
                source_file=str(note_path),
                output_dir=Path(_config_manager.get_output_dir()),
                vault_path=self.parser.vault_path,
                save_markdown=True,
                temperature=0.7
            )
            
            # Display preview
            click.echo("\nGenerated hub note preview:")
            click.echo("=" * 40)
            click.echo(hub_note[:500] + "..." if len(hub_note) > 500 else hub_note)
            click.echo("=" * 40)
            
        except Exception as e:
            click.echo(f"Error generating hub note: {str(e)}")

    def process_directory_hub_notes(self, directory_path: Path) -> None:
        """Process all markdown files in a directory and generate hub notes.
        
        Args:
            directory_path: Path to the directory containing notes
        """
        click.echo(f"\nProcessing directory and generating hub notes: {directory_path}")
        
        try:
            # Ask for recursive processing
            questions = [
                inquirer.List(
                    'recursive',
                    message="Process subdirectories recursively?",
                    choices=[
                        ('Yes', True),
                        ('No', False)
                    ],
                    carousel=True
                )
            ]
            
            answers = inquirer.prompt(questions)
            if not answers:
                return
                
            recursive = answers['recursive']
            pattern = "**/*.md" if recursive else "*.md"
            
            # Process each markdown file in the directory
            for note_path in directory_path.glob(pattern):
                if should_skip_file(note_path.name):
                    click.echo(f"  Skipping already-processed: {note_path.name}")
                    continue
                try:
                    click.echo(f"\nProcessing {note_path.relative_to(directory_path)}...")
                    self.parser.parse_file(str(note_path))
                    parsed_notes = self.parser.parsed_notes

                    # Generate hub note
                    hub_note = self.llm.generate_daily_hub_note(
                        parsed_notes=parsed_notes,
                        source_file=str(note_path),
                        output_dir=Path(_config_manager.get_output_dir()),
                        vault_path=self.parser.vault_path,
                        save_markdown=True,
                        temperature=0.7
                    )
                    
                    # Display preview
                    click.echo("\nGenerated hub note preview:")
                    click.echo("=" * 40)
                    click.echo(hub_note[:500] + "..." if len(hub_note) > 500 else hub_note)
                    click.echo("=" * 40)
                    
                except Exception as e:
                    click.echo(f"Error processing {note_path}: {str(e)}")
                    continue
                    
        except Exception as e:
            click.echo(f"Error processing directory: {str(e)}")

    def process_format_note(self, note_path: Path) -> None:
        """Format a raw note into well-structured markdown.

        Args:
            note_path: Path to the note file
        """
        click.echo(f"\nFormatting note: {note_path.name}...")

        try:
            raw_content = note_path.read_text(encoding='utf-8')
            formatted = self.llm.format_note(raw_content, temperature=0.3)

            frontmatter = generate_frontmatter(note_path.stem, 'format')
            output_path = note_path.parent / f"{note_path.stem}_formatted.md"
            output_path.write_text(frontmatter + formatted, encoding='utf-8')

            click.echo(f"Formatted note saved to: {output_path}")
            click.echo("\nPreview:")
            click.echo("=" * 40)
            click.echo(formatted[:500] + "..." if len(formatted) > 500 else formatted)
            click.echo("=" * 40)
        except Exception as e:
            click.echo(f"Error formatting note: {str(e)}")

    def process_format_and_flashcards(self, note_path: Path) -> None:
        """Format a note and generate flashcards in a single output file.

        Args:
            note_path: Path to the note file
        """
        click.echo(f"\nProcessing format + flashcards for: {note_path.name}...")

        try:
            raw_content = note_path.read_text(encoding='utf-8')

            # Format the note
            click.echo("  Formatting note...")
            formatted = self.llm.format_note(raw_content, temperature=0.3)

            # Generate flashcards from the raw content
            click.echo("  Generating flashcards...")
            flashcards = self.flashcard_generator.generate_flashcards_content(raw_content)

            # Combine into a single output file
            frontmatter = generate_frontmatter(note_path.stem, 'both')
            combined = f"{frontmatter}{formatted}\n\n---\n\n# Flashcards\n\n{flashcards}"

            output_path = note_path.parent / f"{note_path.stem}_processed.md"
            output_path.write_text(combined, encoding='utf-8')

            click.echo(f"Processed note saved to: {output_path}")
        except Exception as e:
            click.echo(f"Error processing note: {str(e)}")

    def process_note(self, note_path: Path) -> None:
        """Process a single note file."""
        questions = [
            inquirer.List('action',
                         message="What would you like to do with this note?",
                         choices=[
                             'Extract atomic notes',
                             'Generate flashcards',
                             'Format notes',
                             'Format + Flashcards (both)',
                             'Generate hub note',
                             'Chat about the note',
                             'Skip'
                         ])
        ]

        answers = inquirer.prompt(questions)
        if answers['action'] == 'Extract atomic notes':
            self.process_atomic_notes(note_path)
        elif answers['action'] == 'Generate flashcards':
            self.flashcard_generator.generate_flashcards(note_path)
        elif answers['action'] == 'Format notes':
            self.process_format_note(note_path)
        elif answers['action'] == 'Format + Flashcards (both)':
            self.process_format_and_flashcards(note_path)
        elif answers['action'] == 'Generate hub note':
            self.generate_hub_note(note_path)
        elif answers['action'] == 'Chat about the note':
            self.chat_about_note(note_path)

    def process_directory_flashcards(self, directory_path: Path) -> None:
        """Process all markdown files in a directory and generate flashcards.
        
        Args:
            directory_path: Path to the directory containing notes
        """
        click.echo(f"\nProcessing directory and generating flashcards: {directory_path}")
        
        try:
            # Ask for recursive processing
            questions = [
                inquirer.List(
                    'recursive',
                    message="Process subdirectories recursively?",
                    choices=[
                        ('Yes', True),
                        ('No', False)
                    ],
                    carousel=True
                )
            ]
            
            answers = inquirer.prompt(questions)
            if not answers:
                return
                
            recursive = answers['recursive']
            pattern = "**/*.md" if recursive else "*.md"
            
            # Process each markdown file in the directory
            note_paths = []
            for note_path in directory_path.glob(pattern):
                if not should_skip_file(note_path.name):
                    note_paths.append(note_path)
            
            if not note_paths:
                click.echo("No markdown files found to process.")
                return
                
            # Generate flashcards for all notes
            generated_files = self.flashcard_generator.batch_generate_flashcards(note_paths)
            
            # Display summary
            click.echo(f"\nGenerated {len(generated_files)} flashcard files:")
            for file_path in generated_files:
                click.echo(f"- {file_path}")
                
        except Exception as e:
            click.echo(f"Error processing directory: {str(e)}")

    def process_directory_format_notes(self, directory_path: Path) -> None:
        """Format all markdown files in a directory.

        Args:
            directory_path: Path to the directory containing notes
        """
        click.echo(f"\nFormatting notes in directory: {directory_path}")

        try:
            questions = [
                inquirer.List(
                    'recursive',
                    message="Process subdirectories recursively?",
                    choices=[('Yes', True), ('No', False)],
                    carousel=True
                )
            ]
            answers = inquirer.prompt(questions)
            if not answers:
                return

            recursive = answers['recursive']
            pattern = "**/*.md" if recursive else "*.md"

            processed = 0
            for note_path in directory_path.glob(pattern):
                if should_skip_file(note_path.name):
                    click.echo(f"  Skipping already-processed: {note_path.name}")
                    continue
                try:
                    click.echo(f"\nFormatting {note_path.name}...")
                    raw_content = note_path.read_text(encoding='utf-8')
                    formatted = self.llm.format_note(raw_content, temperature=0.3)

                    frontmatter = generate_frontmatter(note_path.stem, 'format')
                    output_path = note_path.parent / f"{note_path.stem}_formatted.md"
                    output_path.write_text(frontmatter + formatted, encoding='utf-8')

                    click.echo(f"  Saved: {output_path.name}")
                    processed += 1
                except Exception as e:
                    click.echo(f"  Error formatting {note_path.name}: {str(e)}")
                    continue

            click.echo(f"\nFormatted {processed} notes.")
        except Exception as e:
            click.echo(f"Error processing directory: {str(e)}")

    def process_directory_both(self, directory_path: Path) -> None:
        """Format + generate flashcards for all markdown files in a directory.

        Args:
            directory_path: Path to the directory containing notes
        """
        click.echo(f"\nProcessing format + flashcards in directory: {directory_path}")

        try:
            questions = [
                inquirer.List(
                    'recursive',
                    message="Process subdirectories recursively?",
                    choices=[('Yes', True), ('No', False)],
                    carousel=True
                )
            ]
            answers = inquirer.prompt(questions)
            if not answers:
                return

            recursive = answers['recursive']
            pattern = "**/*.md" if recursive else "*.md"

            processed = 0
            for note_path in directory_path.glob(pattern):
                if should_skip_file(note_path.name):
                    click.echo(f"  Skipping already-processed: {note_path.name}")
                    continue
                try:
                    click.echo(f"\nProcessing {note_path.name}...")
                    raw_content = note_path.read_text(encoding='utf-8')

                    click.echo("  Formatting...")
                    formatted = self.llm.format_note(raw_content, temperature=0.3)

                    click.echo("  Generating flashcards...")
                    flashcards = self.flashcard_generator.generate_flashcards_content(raw_content)

                    frontmatter = generate_frontmatter(note_path.stem, 'both')
                    combined = f"{frontmatter}{formatted}\n\n---\n\n# Flashcards\n\n{flashcards}"

                    output_path = note_path.parent / f"{note_path.stem}_processed.md"
                    output_path.write_text(combined, encoding='utf-8')

                    click.echo(f"  Saved: {output_path.name}")
                    processed += 1
                except Exception as e:
                    click.echo(f"  Error processing {note_path.name}: {str(e)}")
                    continue

            click.echo(f"\nProcessed {processed} notes (format + flashcards).")
        except Exception as e:
            click.echo(f"Error processing directory: {str(e)}")


@click.command()
@click.option('--vault-path', '-v', type=str, help='Path to Obsidian vault')
@click.option('--config-path', '-c', type=str, help='Path to config file')
def main(vault_path: Optional[str], config_path: Optional[str]):
    """Interactive CLI for working with notes."""
    try:
        cli = NoteCLI(vault_path=vault_path, config_path=config_path)
        
        while True:
            # First select process type
            process_questions = [
                inquirer.List(
                    'process_type',
                    message="What would you like to process?",
                    choices=[
                        ('Process directory', 'dir'),
                        ('Process single file', 'file'),
                        ('Exit', 'exit')
                    ],
                    carousel=True
                )
            ]
            
            process_answer = inquirer.prompt(process_questions)
            if not process_answer or process_answer['process_type'] == 'exit':
                break
                
            process_type = process_answer['process_type']
            target_path = None
            
            # Initialize with vault path
            current_dir = Path(_config_manager.get_vault_path())
            
            while True:  # Navigation loop
                # List contents of current directory
                directories = [d for d in current_dir.iterdir() if d.is_dir()]
                dir_choices = [f"📁 {str(d.relative_to(current_dir))}" for d in directories]
                
                if process_type == 'file':
                    # For file processing, show both files and directories
                    files = [f for f in current_dir.iterdir() if f.is_file() and f.suffix == '.md']
                    file_choices = [f"📄 {str(f.relative_to(current_dir))}" for f in files]
                    all_choices = dir_choices + file_choices
                    
                    if current_dir != Path(_config_manager.get_vault_path()):
                        all_choices = ["📁 .."] + all_choices
                    
                    all_choices.append("Cancel")
                    
                    nav_question = [
                        inquirer.List(
                            'selection',
                            message=f"Current directory: {current_dir.name}\nSelect file or directory:",
                            choices=all_choices,
                            carousel=True
                        )
                    ]
                    
                    nav_answer = inquirer.prompt(nav_question)
                    if not nav_answer or nav_answer['selection'] == 'Cancel':
                        break
                        
                    selection = nav_answer['selection']
                    
                    if selection == "📁 ..":
                        current_dir = current_dir.parent
                        continue
                        
                    item_name = selection[2:]  # Remove icon prefix
                    selected_path = current_dir / item_name
                    
                    if selected_path.is_dir():
                        current_dir = selected_path
                        continue
                    else:
                        target_path = selected_path
                        break
                        
                else:  # Directory processing
                    all_choices = ["Select current directory"]
                    
                    if current_dir != Path(_config_manager.get_vault_path()):
                        all_choices = ["📁 .."] + all_choices
                        
                    all_choices.extend(dir_choices)
                    all_choices.append("Cancel")
                    
                    nav_question = [
                        inquirer.List(
                            'selection',
                            message=f"Current directory: {current_dir.name}\nSelect directory:",
                            choices=all_choices,
                            carousel=True
                        )
                    ]
                    
                    nav_answer = inquirer.prompt(nav_question)
                    if not nav_answer or nav_answer['selection'] == 'Cancel':
                        break
                        
                    selection = nav_answer['selection']
                    
                    if selection == "Select current directory":
                        target_path = current_dir
                        break
                    elif selection == "📁 ..":
                        current_dir = current_dir.parent
                        continue
                    else:
                        dir_name = selection[2:]  # Remove icon prefix
                        current_dir = current_dir / dir_name
                        continue
            
            if target_path:
                # Choose action for the selected path
                if target_path.is_file():
                    cli.process_note(target_path)
                else:
                    choices = [
                        ('Process directory and generate hub notes', 'dir_hub'),
                        ('Process directory and extract atomic notes', 'dir_atomic'),
                        ('Process directory and generate flashcards', 'dir_flashcards'),
                        ('Process directory and format notes', 'dir_format'),
                        ('Process directory - Format + Flashcards', 'dir_both'),
                        ('Go back', 'back'),
                        ('Exit', 'exit')
                    ]
                    
                    questions = [
                        inquirer.List(
                            'action',
                            message=f"What would you like to do with {target_path.name}?",
                            choices=choices,
                            carousel=True
                        )
                    ]
                    
                    answers = inquirer.prompt(questions)
                    if not answers:
                        break
                        
                    action = answers['action']
                    
                    if action == 'dir_hub':
                        cli.process_directory_hub_notes(target_path)
                    elif action == 'dir_atomic':
                        cli.process_directory_atomic_notes(target_path)
                    elif action == 'dir_flashcards':
                        cli.process_directory_flashcards(target_path)
                    elif action == 'dir_format':
                        cli.process_directory_format_notes(target_path)
                    elif action == 'dir_both':
                        cli.process_directory_both(target_path)
                    elif action == 'exit':
                        break
                    
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user")
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        sys.exit(1)
        
    click.echo("\nGoodbye!")

if __name__ == '__main__':
    main()
