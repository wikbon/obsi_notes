#!/usr/bin/env python3

import click
import inquirer
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import sys
import logging

from src.core.llm_handler import LLMHandler
from src.utils.note_parser import NoteParser
from src.utils.atomic_note_extractor import AtomicNoteExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NoteCLI:
    """CLI interface for interacting with notes."""
    
    def __init__(self, vault_path: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize the CLI interface.
        
        Args:
            vault_path: Path to the Obsidian vault
            config_path: Path to config file
        """
        self.parser = NoteParser(vault_path=vault_path, config_path=config_path)
        self.llm = LLMHandler(verbose=True)
        self.extractor = AtomicNoteExtractor(
            vault_path=vault_path,
            config_path=config_path,
            verbose=True,
            llm_handler=self.llm
        )
        
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
        
        system_prompt = (
            "You are an AI assistant helping to analyze and discuss notes. "
            "The user will ask questions about a note, and you should provide "
            "helpful, concise responses. When asked to analyze or summarize, "
            "please provide a structured response with main points and key ideas."
        )
        
        # Initialize chat with the note content
        self.llm.clear_history()
        self.llm.add_message("system", system_prompt)
        
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
                output_dir=Path(self.parser.config['processing']['output_dir']),
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
                try:
                    click.echo(f"\nProcessing {note_path.relative_to(directory_path)}...")
                    self.parser.parse_file(str(note_path))
                    parsed_notes = self.parser.parsed_notes
                    
                    # Generate hub note
                    hub_note = self.llm.generate_daily_hub_note(
                        parsed_notes=parsed_notes,
                        source_file=str(note_path),
                        output_dir=Path(self.parser.config['processing']['output_dir']),
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

@click.command()
@click.option(
    '--vault-path',
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help='Path to Obsidian vault'
)
@click.option(
    '--config-path',
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help='Path to config file'
)
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
            current_dir = Path(cli.parser.config['vault']['path'])
            
            while True:  # Navigation loop
                # List contents of current directory
                directories = [d for d in current_dir.iterdir() if d.is_dir()]
                dir_choices = [f"📁 {str(d.relative_to(current_dir))}" for d in directories]
                
                if process_type == 'file':
                    # For file processing, show both files and directories
                    files = [f for f in current_dir.iterdir() if f.is_file() and f.suffix == '.md']
                    file_choices = [f"📄 {str(f.relative_to(current_dir))}" for f in files]
                    all_choices = dir_choices + file_choices
                    
                    if current_dir != Path(cli.parser.config['vault']['path']):
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
                    
                    if current_dir != Path(cli.parser.config['vault']['path']):
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
                choices = []
                if target_path.is_file():
                    choices = [
                        ('Chat about the note', 'chat'),
                        ('Extract atomic notes', 'atomic'),
                        ('Generate hub note', 'hub'),
                        ('Go back', 'back'),
                        ('Exit', 'exit')
                    ]
                else:  # is directory
                    choices = [
                        ('Process directory and generate hub notes', 'dir_hub'),
                        ('Process directory and extract atomic notes', 'dir_atomic'),
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
                
                if action == 'chat':
                    cli.chat_about_note(target_path)
                elif action == 'atomic':
                    cli.process_atomic_notes(target_path)
                elif action == 'hub':
                    cli.generate_hub_note(target_path)
                elif action == 'dir_hub':
                    cli.process_directory_hub_notes(target_path)
                elif action == 'dir_atomic':
                    cli.process_directory_atomic_notes(target_path)
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
