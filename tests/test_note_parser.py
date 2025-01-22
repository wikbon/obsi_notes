#!/usr/bin/env python3

import sys
import logging
import argparse
import yaml
from pathlib import Path
from typing import Optional, List
import os
from datetime import datetime

# Add parent directory to Python path to import note_parser
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.note_parser import NoteParser

CONFIG_PATH = Path(__file__).parent.parent / "src" / "config" / "config.yaml"

def load_config() -> dict:
    """Load configuration from YAML file."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return {"vault": {"path": None}}

def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration.
    
    Args:
        debug (bool): If True, set log level to DEBUG
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def select_note_interactive(files: List[Path]) -> Optional[Path]:
    """Display an interactive menu to select a note.
    
    Args:
        files: List of note files
        
    Returns:
        Optional[Path]: Selected note path or None if cancelled
    """
    if not files:
        print("No files found in vault!")
        return None
        
    # Sort files by modification time (most recent first)
    files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    while True:
        print("\nAvailable notes (most recent first):")
        print("0: Exit")
        
        # Show last 20 files with option to show more
        display_count = min(20, len(files))
        for i, file in enumerate(files[:display_count], 1):
            try:
                mod_time = datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                rel_path = str(file.relative_to(Path(load_config()['vault']['path'])))
                print(f"{i}: [{mod_time}] {rel_path}")
            except Exception as e:
                print(f"{i}: [Error reading file info] {file.name}")
            
        if len(files) > display_count:
            print(f"... and {len(files) - display_count} more files")
            print("m: Show more files")
            
        try:
            choice = input("\nSelect a note number (or 'm' for more, 0 to exit): ").strip().lower()
            
            if choice == '0':
                return None
            elif choice == 'm':
                display_count = min(display_count + 20, len(files))
                continue
                
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None

def process_vault(vault_path: str, debug: bool = False, interactive: bool = False, export_json: bool = False) -> None:
    """Process all notes in the vault and display debug information.
    
    Args:
        vault_path (str): Path to Obsidian vault
        debug (bool): Enable debug output
        interactive (bool): Enable interactive note selection
        export_json (bool): Export results to JSON file
    """
    setup_logging(debug)
    logger = logging.getLogger(__name__)
    
    parser = NoteParser(vault_path=vault_path)
    logger.info(f"Processing vault: {vault_path}")
    
    try:
        files = parser.get_vault_files()
        logger.info(f"Found {len(files)} markdown files in vault")
        
        if interactive:
            selected_file = select_note_interactive(files)
            if selected_file:
                process_single_file(str(selected_file), debug, export_json)
            return
            
        for file_path in files:
            logger.debug(f"\nProcessing file: {file_path}")
            
            if parser.load_file(file_path):
                logger.debug(f"Frontmatter: {parser.frontmatter}")
                
                if export_json:
                    json_path = parser.process_file(export_json=True)
                    logger.info(f"Exported to JSON: {json_path}")
                    continue
                
                notes = parser.split_by_timestamp()
                logger.debug(f"Found {len(notes)} timestamped segments")
                
                for i, note in enumerate(notes, 1):
                    logger.debug(f"\nSegment {i}:")
                    logger.debug(f"Timestamp: {note['timestamp']}")
                    logger.debug(f"Content length: {len(note['content'])} chars")
                    if debug:
                        logger.debug("Content preview (first 100 chars):")
                        logger.debug(note['content'][:100] + "..." if len(note['content']) > 100 else note['content'])
            else:
                logger.error(f"Failed to load file: {file_path}")
                
    except Exception as e:
        logger.error(f"Error processing vault: {str(e)}")
        sys.exit(1)

def process_single_file(file_path: str, debug: bool = False, export_json: bool = False) -> None:
    """Process a single note file and display debug information.
    
    Args:
        file_path (str): Path to note file
        debug (bool): Enable debug output
        export_json (bool): Export results to JSON file
    """
    setup_logging(debug)
    logger = logging.getLogger(__name__)
    
    parser = NoteParser(file_path=file_path)
    logger.info(f"Processing file: {file_path}")
    
    try:
        if parser.load_file():
            logger.debug(f"Frontmatter: {parser.frontmatter}")
            
            if export_json:
                json_path = parser.process_file(export_json=True)
                logger.info(f"Exported to JSON: {json_path}")
                return
                
            notes = parser.split_by_timestamp()
            logger.debug(f"Found {len(notes)} timestamped segments")
            
            for i, note in enumerate(notes, 1):
                logger.debug(f"\nSegment {i}:")
                logger.debug(f"Timestamp: {note['timestamp']}")
                logger.debug(f"Content length: {len(note['content'])} chars")
                if debug:
                    logger.debug("Content preview (first 100 chars):")
                    logger.debug(note['content'][:100] + "..." if len(note['content']) > 100 else note['content'])
        else:
            logger.error(f"Failed to load file: {file_path}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        sys.exit(1)

def main():
    config = load_config()
    default_vault = config['vault']['path']

    parser = argparse.ArgumentParser(description="Test Obsidian Note Parser")
    parser.add_argument("path", nargs='?', default=default_vault,
                      help="Path to Obsidian vault or single note file (optional if configured)")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output")
    parser.add_argument("-f", "--file", action="store_true", help="Process single file instead of vault")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactively select a note to process")
    parser.add_argument("-j", "--json", action="store_true", help="Export processed notes to JSON")
    
    args = parser.parse_args()
    
    if not args.path:
        parser.error("No vault path provided and none configured in config.yaml")
        
    path = str(Path(args.path).resolve())
    
    if args.file:
        process_single_file(path, args.debug, args.json)
    else:
        process_vault(path, args.debug, args.interactive, args.json)

if __name__ == "__main__":
    main()
