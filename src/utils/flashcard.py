"""Flashcard generation utilities for spaced repetition learning."""

import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import yaml
import json

class FlashcardGenerator:
    """Handles generation and management of flashcards from notes."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the FlashcardGenerator.
        
        Args:
            config_path (str, optional): Path to config file
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config['vault']['path']) / self.config['processing']['spaced_repetition_dir']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: Optional[str] = None) -> dict:
        """Load configuration from YAML file."""
        if not config_path:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def generate_qa_pairs(self, content: str, topic: str = None) -> List[Dict[str, str]]:
        """Generate Q&A pairs from note content using LLM.
        
        Args:
            content (str): Note content to generate flashcards from
            topic (str, optional): Topic or category for the flashcards
            
        Returns:
            List[Dict[str, str]]: List of Q&A pairs
        """
        from ..core.llm_handler import LLMHandler
        
        llm = LLMHandler(verbose=False)
        system_prompt = """You are a flashcard generator. Create concise, atomic Q&A pairs from the given content.
        Focus on key concepts, facts, and relationships. Each Q&A pair should test a single piece of knowledge."""
        
        user_prompt = f"""Generate Q&A pairs from this content. Return a JSON array of objects with 'question' and 'answer' fields.
        Content: {content}
        Topic: {topic if topic else 'General'}
        
        Return format:
        [
            {{"question": "What is X?", "answer": "X is Y"}},
            ...
        ]"""
        
        llm.add_message("system", system_prompt)
        llm.add_message("user", user_prompt)
        
        response = llm.create_chat_completion(temperature=0.7)
        try:
            qa_pairs = json.loads(response['choices'][0]['message']['content'])
            return qa_pairs
        except (json.JSONDecodeError, KeyError, IndexError):
            return []
            
    def save_flashcards(self, qa_pairs: List[Dict[str, str]], topic: str = None) -> Path:
        """Save Q&A pairs as a markdown file.
        
        Args:
            qa_pairs (List[Dict[str, str]]): List of Q&A pairs
            topic (str, optional): Topic or category for the flashcards
            
        Returns:
            Path: Path to the created flashcard file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flashcards_{topic}_{timestamp}.md" if topic else f"flashcards_{timestamp}.md"
        output_path = self.output_dir / filename
        
        frontmatter = {
            'date': datetime.now().isoformat(),
            'topic': topic if topic else 'General',
            'type': 'flashcards',
            'count': len(qa_pairs)
        }
        
        with open(output_path, 'w') as f:
            # Write frontmatter
            f.write('---\n')
            yaml.dump(frontmatter, f)
            f.write('---\n\n')
            
            # Write Q&A pairs
            for i, qa in enumerate(qa_pairs, 1):
                f.write(f"## Q{i}: {qa['question']}\n\n")
                f.write(f"A: {qa['answer']}\n\n")
                f.write("---\n\n")
                
        return output_path
        
    def generate_from_note(self, note_content: str, topic: str = None, tags: List[str] = None) -> Optional[Path]:
        """Generate and save flashcards from a note.
        
        Args:
            note_content (str): Note content to generate flashcards from
            topic (str, optional): Topic or category for the flashcards
            tags (List[str], optional): Tags associated with the note
            
        Returns:
            Optional[Path]: Path to the created flashcard file, if successful
        """
        # Only process if #flashcard tag is present
        if tags and '#flashcard' not in tags:
            return None
            
        qa_pairs = self.generate_qa_pairs(note_content, topic)
        if qa_pairs:
            return self.save_flashcards(qa_pairs, topic)
        return None
