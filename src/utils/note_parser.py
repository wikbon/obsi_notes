import os
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
import re
import yaml
import json
from pathlib import Path

class NoteParser:
    """A class to handle loading and parsing of raw notes for LLM processing."""
    
    def __init__(self, vault_path: str = None, file_path: str = None, config_path: str = None):
        """Initialize the NoteParser with optional vault and file paths.
        
        Args:
            vault_path (str, optional): Path to the Obsidian vault
            file_path (str, optional): Path to the specific note file to be parsed
            config_path (str, optional): Path to config file, defaults to standard location
        """
        self.config = self._load_config(config_path)
        self.vault_path = Path(vault_path) if vault_path else Path(self.config['vault']['path'])
        self.file_path = Path(file_path) if file_path else None
        self.raw_content = None
        self.parsed_notes = []
        self.frontmatter = {}
        
    def _load_config(self, config_path: Optional[str] = None) -> dict:
        """Load configuration from YAML file.
        
        Args:
            config_path (str, optional): Path to config file
            
        Returns:
            dict: Configuration dictionary
        """
        if not config_path:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
            
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Error loading config: {e}")
        
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
        if content.startswith('---\n'):
            parts = content[4:].split('\n---\n', 1)
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
            with open(self.file_path, 'r', encoding='utf-8') as file:
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
        content = re.sub(r'\[\[(.*?)\]\]', r'"\1"', content)
        # Convert [[Note Name|Alias]] to "Alias"
        content = re.sub(r'\[\[(.*?)\|(.*?)\]\]', r'"\2"', content)
        return content

    def extract_tags(self, content: str) -> List[str]:
        """Extract hashtags from content.
        
        Args:
            content (str): Content to extract tags from
            
        Returns:
            List[str]: List of extracted tags without the '#' symbol
        """
        # Match hashtags that are part of words (e.g., #idea, #task)
        pattern = r'#([\w-]+)'
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
        task_pattern = r'- \[([ xX])\] (.*?)(?=\n|$)'
        tasks.extend([{
            'type': 'checkbox',
            'status': bool(status.strip()),
            'content': task_content.strip()
        } for status, task_content in re.findall(task_pattern, content)])
        
        # Match #task tagged items
        task_tag_pattern = r'#task\s+(.*?)(?=\n|$)'
        tasks.extend([{
            'type': 'tagged',
            'status': False,  # Default to not completed
            'content': task.strip()
        } for task in re.findall(task_tag_pattern, content)])
        
        return tasks
        
    def extract_structure(self, content: str) -> List[Dict[str, Any]]:
        """Extract structural elements like headings and bullet points.
        
        Args:
            content (str): Content to analyze
            
        Returns:
            List[Dict[str, Any]]: List of structural elements with their content
        """
        elements = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Match headings (## Heading)
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                elements.append({
                    'type': 'heading',
                    'level': len(heading_match.group(1)),
                    'content': heading_match.group(2).strip()
                })
                continue
                
            # Match bullet points (- or * or +)
            bullet_match = re.match(r'^[-*+]\s+(.+)$', line)
            if bullet_match:
                elements.append({
                    'type': 'bullet',
                    'content': bullet_match.group(1).strip()
                })
                continue
                
            # Match numbered lists (1. 2. etc)
            number_match = re.match(r'^\d+\.\s+(.+)$', line)
            if number_match:
                elements.append({
                    'type': 'numbered',
                    'content': number_match.group(1).strip()
                })
                continue
                
            # Regular text
            elements.append({
                'type': 'text',
                'content': line
            })
            
        return elements

    def parse_timestamped_segments(self, content: Optional[str] = None) -> List[Dict[str, Any]]:
        """Parse content into timestamped segments with enhanced metadata.
        
        Args:
            content (str, optional): Content to parse. If None, uses self.raw_content
            
        Returns:
            List[Dict[str, Any]]: List of parsed segments with metadata
        """
        if content is None:
            content = self.raw_content
            
        if not content:
            return []
            
        # Split content by timestamp pattern
        timestamp_pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?)\]'
        segments = re.split(f'({timestamp_pattern})', content)
        
        parsed_segments = []
        current_segment = None
        
        for i, segment in enumerate(segments[1:], 1):  # Skip first empty segment
            if i % 2 == 1:  # Timestamp
                if current_segment:
                    parsed_segments.append(current_segment)
                timestamp_match = re.match(timestamp_pattern, segment)
                current_segment = {
                    'timestamp': timestamp_match.group(1),
                    'content': '',
                    'tags': [],
                    'tasks': [],
                    'structure': [],
                    'links': []
                }
            else:  # Content
                if current_segment:
                    # Clean and store content
                    cleaned_content = segment.strip()
                    current_segment['content'] = cleaned_content
                    
                    # Extract metadata
                    current_segment['tags'] = self.extract_tags(cleaned_content)
                    current_segment['tasks'] = self.extract_tasks(cleaned_content)
                    current_segment['structure'] = self.extract_structure(cleaned_content)
                    current_segment['links'] = self.extract_links(cleaned_content)
                    
                    # Extract atomic notes based on structure
                    current_segment['atomic_notes'] = self.extract_atomic_notes(cleaned_content)
        
        # Add last segment
        if current_segment:
            parsed_segments.append(current_segment)
            
        return parsed_segments
        
    def extract_atomic_notes(self, content: str) -> List[str]:
        """Extract atomic notes based on structure and content.
        
        Args:
            content (str): Content to extract atomic notes from
            
        Returns:
            List[str]: List of atomic notes
        """
        atomic_notes = []
        current_heading = None
        current_note = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                if current_note:
                    note_text = ' '.join(current_note)
                    if current_heading:
                        note_text = f"{current_heading}: {note_text}"
                    atomic_notes.append(note_text)
                    current_note = []
                continue
                
            # Check for headings
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                if current_note:
                    note_text = ' '.join(current_note)
                    if current_heading:
                        note_text = f"{current_heading}: {note_text}"
                    atomic_notes.append(note_text)
                    current_note = []
                current_heading = heading_match.group(2)
                continue
                
            # Check for bullet points
            bullet_match = re.match(r'^[-*+]\s+(.+)$', line)
            if bullet_match:
                if current_note:
                    note_text = ' '.join(current_note)
                    if current_heading:
                        note_text = f"{current_heading}: {note_text}"
                    atomic_notes.append(note_text)
                current_note = [bullet_match.group(1)]
                continue
                
            # Append to current note
            current_note.append(line)
            
        # Add final note if exists
        if current_note:
            note_text = ' '.join(current_note)
            if current_heading:
                note_text = f"{current_heading}: {note_text}"
            atomic_notes.append(note_text)
            
        return atomic_notes
        
    def extract_links(self, content: str) -> List[str]:
        """Extract internal links from content.
        
        Args:
            content (str): Content to extract links from
            
        Returns:
            List[str]: List of extracted link titles
        """
        # Match [[Link]] and [[Link|Alias]] patterns
        link_pattern = r'\[\[(.*?)(?:\|.*?)?\]\]'
        return [link.split('|')[0] for link in re.findall(link_pattern, content)]
        
    def process_note_with_llm(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """Process a note segment with LLM for enhanced extraction.
        
        Args:
            segment (Dict[str, Any]): Note segment to process
            
        Returns:
            Dict[str, Any]: Enhanced segment with LLM-extracted data
        """
        from ..core.llm_handler import LLMHandler
        
        llm = LLMHandler(verbose=False)
        
        # Get existing vault titles for linking
        vault_titles = [f.stem for f in self.get_vault_files()]
        
        system_prompt = """You are a note processing assistant. Extract structured information from the given note segment."""
        
        user_prompt = f"""Process this note segment and extract:
        1. Atomic ideas (one per item)
        2. Tasks (marked with #task or '- [ ]')
        3. Tags
        4. Potential links to these existing vault titles: {', '.join(vault_titles)}
        
        Note segment:
        [Timestamp: {segment['timestamp']}]
        {segment['content']}
        
        Return a JSON object with these fields:
        {{
            "atomic_notes": ["idea 1", "idea 2", ...],
            "tasks": ["task 1", "task 2", ...],
            "tags": ["#tag1", "#tag2", ...],
            "suggested_links": ["Title1", "Title2", ...]
        }}"""
        
        llm.add_message("system", system_prompt)
        llm.add_message("user", user_prompt)
        
        response = llm.create_chat_completion(temperature=0.7)
        try:
            llm_extracted = json.loads(response['choices'][0]['message']['content'])
            
            # Merge LLM-extracted data with existing data
            segment['atomic_notes'] = list(set(segment.get('atomic_notes', []) + llm_extracted.get('atomic_notes', [])))
            segment['tasks'] = list(set(segment.get('tasks', []) + llm_extracted.get('tasks', [])))
            segment['tags'] = list(set(segment.get('tags', []) + llm_extracted.get('tags', [])))
            segment['suggested_links'] = llm_extracted.get('suggested_links', [])
            
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
            
        return segment
        
    def process_note(self, file_path: Optional[str] = None, use_llm: bool = True) -> Dict[str, Any]:
        """Process a note file completely, including LLM-based extraction.
        
        Args:
            file_path (str, optional): Path to note file
            use_llm (bool): Whether to use LLM for enhanced extraction
            
        Returns:
            Dict[str, Any]: Processed note data
        """
        if file_path:
            self.load_file(file_path)
            
        segments = self.parse_timestamped_segments()
        
        if use_llm:
            segments = [self.process_note_with_llm(segment) for segment in segments]
            
        # Generate flashcards if appropriate
        from .flashcard import FlashcardGenerator
        flashcard_gen = FlashcardGenerator()
        
        for segment in segments:
            if '#flashcard' in segment['tags']:
                flashcard_path = flashcard_gen.generate_from_note(
                    segment['content'],
                    topic=next((tag[1:] for tag in segment['tags'] if tag.startswith('#topic_')), None),
                    tags=segment['tags']
                )
                if flashcard_path:
                    segment['flashcard_file'] = str(flashcard_path)
        
        return {
            'file_path': str(self.file_path),
            'frontmatter': self.frontmatter,
            'segments': segments
        }

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
        timestamp_pattern = r'\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\+\d{2}:\d{2})?|\s\d{2}:\d{2}(?::\d{2})?)'
        
        # Split content by timestamps
        segments = re.split(f'({timestamp_pattern})', processed_content)
        
        parsed_notes = []
        current_timestamp = None
        
        for segment in segments:
            if re.match(timestamp_pattern, segment):
                current_timestamp = segment
            elif segment.strip() and current_timestamp:
                content = segment.strip()
                parsed_notes.append({
                    'timestamp': current_timestamp,
                    'content': content,
                    'tags': self.extract_tags(content),
                    'tasks': self.extract_tasks(content),
                    'structure': self.extract_structure(content),
                    'metadata': {
                        'frontmatter': self.frontmatter,
                        'source_relative_path': str(self.file_path.relative_to(self.vault_path)) if self.file_path else None
                    }
                })
                
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
            if note['tasks']:
                task_str = "\nTasks:\n" + "\n".join(
                    f"- [{' x' if task['status'] else ' '}] {task['content']}" 
                    for task in note['tasks']
                )
            
            # Format structure
            structure_str = ""
            if note['structure']:
                structure_str = "\nStructure:\n" + "\n".join(
                    f"{'#' * elem['level'] + ' ' if elem['type'] == 'heading' else '- '}{elem['content']}"
                    for elem in note['structure'] if elem['type'] != 'text'
                )
            
            # Format tags
            tags_str = ""
            if note['tags']:
                tags_str = "\nTags: " + ", ".join(f"#{tag}" for tag in note['tags'])
            
            formatted_note = f"[{note['timestamp']}]\n{note['content']}{tags_str}{task_str}{structure_str}"
            llm_ready.append(formatted_note)
            
        return llm_ready

    def process_file(self, file_path: Optional[str] = None, export_json: bool = False) -> Union[List[str], str]:
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
            output_dir = Path(self.config['processing']['output_dir'])
            output_path = output_dir / rel_path.with_suffix(self.config['processing']['json_extension'])
            
        # Create export data structure
        export_data = {
            "source_file": str(self.file_path),
            "processed_at": datetime.now().isoformat(),
            "note_segments": self.parsed_notes
        }
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
            
        return str(output_path)