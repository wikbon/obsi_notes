from llama_cpp import Llama
from typing import Optional, Dict, Any, List
import logging
import json
import re
from pathlib import Path
import click
import sys

# Add the project root to Python path for imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.note_parser import NoteParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepSeekHandler:
    """Handler for DeepSeek-R1 model interactions."""

    CHAT_TEMPLATE = "<｜begin▁of▁sentence｜>{system_prompt}<｜User｜>{prompt}<｜Assistant｜>"
    BOXED_PATTERN = r"\\boxed\{([^}]+)\}"

    def __init__(
        self,
        model_path: str = "/path/to/DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf",
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        verbose: bool = False
    ):
        """Initialize the DeepSeek handler.
        
        Args:
            model_path: Path to the GGUF model file
            n_gpu_layers: Number of layers to offload to GPU
            n_ctx: Context window size
            verbose: Whether to print detailed logs
        """
        self.model_path = model_path
        self.verbose = verbose
        if verbose:
            logger.info(f"Initializing DeepSeek with model: {model_path}")
            
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx
        )
        self.message_history: List[Dict[str, str]] = []
        
        if verbose:
            logger.info("DeepSeek model initialized successfully")
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the chat history.
        
        Args:
            role: Role of the message sender (e.g., 'user', 'assistant')
            content: Content of the message
        """
        message = {"role": role, "content": content}
        self.message_history.append(message)
        if self.verbose:
            logger.info(f"Added message - Role: {role}")
            logger.info(f"Content: {content}")
    
    def clear_history(self) -> None:
        """Clear the message history."""
        self.message_history = []
        if self.verbose:
            logger.info("Message history cleared")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the current message history.
        
        Returns:
            List of message dictionaries
        """
        return self.message_history

    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages according to DeepSeek chat template.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string
        """
        # DeepSeek doesn't use system prompts, so we'll combine any system messages
        # into the user prompt
        system_parts = []
        user_parts = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            elif msg["role"] == "user":
                user_parts.append(msg["content"])
            # We don't include assistant messages as we're formatting for a new response

        system_prompt = " ".join(system_parts)
        user_prompt = " ".join(user_parts)
        
        # Format using DeepSeek template
        return self.CHAT_TEMPLATE.format(
            system_prompt=system_prompt,
            prompt=user_prompt
        )

    def _extract_boxed_content(self, text: str) -> Optional[str]:
        """Extract content from \\boxed{} if present.
        
        Args:
            text: Text to extract from
            
        Returns:
            Content within boxed or None if not found
        """
        match = re.search(self.BOXED_PATTERN, text)
        return match.group(1) if match else None

    def _clean_think_blocks(self, text: str) -> str:
        """Remove <think> blocks from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text with think blocks removed
        """
        # Remove <think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove any leftover empty lines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def _clean_markdown_blocks(self, text: str) -> str:
        """Remove markdown code block wrapping from text.
        
        Args:
            text: Text that may be wrapped in markdown code blocks
            
        Returns:
            Text with markdown code block wrapping removed
        """
        text = text.strip()
        # remove leading ``` or ```json plus any whitespace, line breaks, etc.
        text = re.sub(r'^```(?:json|[a-zA-Z0-9_-]+)?\s*', '', text)
        # remove trailing ```
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()
        return text

    def create_chat_completion(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.6,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a chat completion using the DeepSeek model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
                     If None, uses internal message history.
            temperature: Sampling temperature (recommended 0.6 for DeepSeek)
            **kwargs: Additional arguments to pass to create_completion
            
        Returns:
            Dict containing the chat completion response with extracted boxed content
        """
        if messages is None:
            messages = self.message_history
            
        prompt = self._format_prompt(messages)
        
        if self.verbose:
            logger.info(f"Sending prompt to DeepSeek: {prompt}")
            
        completion = self.llm.create_completion(
            prompt=prompt,
            temperature=temperature,
            **kwargs
        )
        
        response_text = completion["choices"][0]["text"]
        # Clean think blocks from response
        response_text = self._clean_think_blocks(response_text)
        boxed_content = self._extract_boxed_content(response_text)
        if boxed_content:
            boxed_content = self._clean_think_blocks(boxed_content)
        
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "boxed_content": boxed_content
            }]
        }
        
        # Add response to history
        self.add_message("assistant", response_text)
        
        return response

    def create_stateless_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a chat completion without affecting message history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (recommended 0.6 for DeepSeek)
            **kwargs: Additional arguments to pass to create_completion
            
        Returns:
            Dict containing the chat completion response with extracted boxed content
        """
        prompt = self._format_prompt(messages)
        
        if self.verbose:
            logger.info(f"Sending prompt to DeepSeek: {prompt}")
            
        completion = self.llm.create_completion(
            prompt=prompt,
            temperature=temperature,
            **kwargs
        )
        
        response_text = completion["choices"][0]["text"]
        # Clean think blocks from response
        response_text = self._clean_think_blocks(response_text)
        boxed_content = self._extract_boxed_content(response_text)
        if boxed_content:
            boxed_content = self._clean_think_blocks(boxed_content)
            boxed_content = self._clean_markdown_blocks(boxed_content)
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "boxed_content": boxed_content
            }]
        }

    def extract_atomic_notes(
        self, 
        note_segments: List[Dict[str, Any]], 
        temperature: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Extract atomic notes from each segment using the DeepSeek model.
        
        Args:
            note_segments (List[Dict[str, Any]]):
                A list of note segments, each typically has keys like 
                'timestamp' and 'content'.
            temperature (float):
                The temperature setting for the LLM output (creativity).
                
        Returns:
            List[Dict[str, Any]]:
                Returns a combined list of atomic note objects. Each object 
                contains fields like 'note'.
        """
        all_extracted_notes = []
        
        for segment in note_segments:
            content = segment["content"]
            timestamp = segment.get("timestamp", "")
            
            # Build the system and user messages
            system_prompt = (
                "You are an assistant specialized in note summarization and extraction. "
                "Your task is to extract atomic notes and return them in JSON format. "
                "An atomic note should capture one complete, self-contained idea - "
                "don't split related concepts that form a single coherent thought. "
                "DO NOT include any markdown formatting or explanation text. "
                "Return ONLY the raw JSON array."
            )
            
            user_prompt = f"""Here is a raw note segment{f' (timestamp: {timestamp})' if timestamp else ''}:

{content}

Extract atomic notes following these rules:
1. Each note should be one complete, self-contained idea.
2. Keep related concepts together if they form a single coherent thought.
3. Don't split definitions or explanations of a single concept.
4. If multiple sentences make up a single idea, keep them together as a single atomic note.
5. If in doubt, don't split a sentence into multiple atomic notes.

Return a JSON array in this exact format (no markdown, no explanation):
[
  {{
    "note": "the complete atomic idea here",
  }}
]"""
            
            # Use the LLM
            if self.verbose:
                logger.info(f"Processing segment{f' from {timestamp}' if timestamp else ''}")
                
            response = self.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=2048,  # adjust as needed
            )
            
            # Get the raw response
            raw_assistant_message = ""
            if response.get("choices"):
                raw_assistant_message = response["choices"][0]["message"]["content"]
                # Clean think blocks
                raw_assistant_message = self._clean_think_blocks(raw_assistant_message)
 
            # Attempt to parse JSON
            try:
                # Preprocess input to remove BOM and strip whitespace
                raw_assistant_message = raw_assistant_message.lstrip("\ufeff").strip()
                
                # Remove triple backticks with `json` marker if present
                if raw_assistant_message.startswith("```json") and raw_assistant_message.endswith("```"):
                    raw_assistant_message = raw_assistant_message[7:-3].strip()
                elif raw_assistant_message.startswith("```") and raw_assistant_message.endswith("```"):
                    raw_assistant_message = raw_assistant_message[3:-3].strip()
                # Remove trailing commas inside objects
                raw_assistant_message = re.sub(r",\s*}", "}", raw_assistant_message)

                # Validate JSON structure before parsing
                if not raw_assistant_message.startswith(("{", "[")):
                    logger.error("Input does not appear to be valid JSON.")
                    return []  # Return empty list instead of None
                
                # Parse JSON   
                extracted_list = json.loads(raw_assistant_message)
                
                # If it's not a list, wrap it in a list for uniformity
                if not isinstance(extracted_list, list):
                    extracted_list = [extracted_list]
                    
                # Print each extracted note to CLI
                for note in extracted_list:
                    note_text = note.get("note", "")
                    click.echo(f"\nOriginal input to LLM:\n{content}\n")
                    click.echo(f"\nExtracted note: {note_text}")
                    
                all_extracted_notes.extend(extracted_list)
                
                if self.verbose:
                    logger.info(f"Successfully extracted {len(extracted_list)} atomic notes")
                    
            except json.JSONDecodeError:
                logger.error(f"JSON parse error for segment{f' at {timestamp}' if timestamp else ''}")
                logger.error(f"Raw response was:\n{raw_assistant_message}\n")
                
        return all_extracted_notes

    def generate_daily_hub_note(
        self,
        parsed_notes: List[Dict[str, Any]],
        source_file: str,
        date_str: str = None,
        temperature: float = 0.6,
        save_markdown: bool = True,
        output_dir: Optional[Path] = None,
        vault_path: Optional[Path] = None
    ) -> str:
        """
        Generate a structured daily hub note in Markdown format from parsed notes.
        
        Args:
            parsed_notes (List[Dict[str, Any]]): List of parsed notes, each containing at least
                                               'timestamp' and 'content' keys.
            source_file (str): Path to the source file being processed
            date_str (str, optional): The date string for the note. If None, will be extracted from filename
            temperature (float): LLM creativity level (default: 0.6).
            save_markdown (bool): Whether to save the generated markdown to a file
            output_dir (Path, optional): Directory to save output files. Required if save_markdown is True
            vault_path (Path, optional): Path to the vault root. Required if save_markdown is True
            
        Returns:
            str: A formatted Markdown string containing the organized daily hub note.
        """
        if save_markdown and (output_dir is None or vault_path is None):
            raise ValueError("output_dir and vault_path must be provided when save_markdown is True")

        # Extract date from filename if not provided
        if date_str is None:
            source_path = Path(source_file)
            parts = source_path.stem.split('-')
            if len(parts) >= 3:
                date_str = f"{parts[0]}-{parts[1]}-{parts[2]}"  # Combines YYYY-MM-DD
            else:
                date_str = source_path.stem  # Fallback to full stem if not in expected format
            
        # Format notes for the prompt
        lines_for_prompt = []
        for note in parsed_notes:
            timestamp = note.get('timestamp', 'N/A')
            content = note.get('content', '')
            lines_for_prompt.append(f"- Timestamp: {timestamp} | Content: {content}")
            
        combined_text = "\n".join(lines_for_prompt)
        
        # Create system and user prompts
        system_prompt = (
            "You are an assistant that organizes daily mind dumps into a nicely formatted "
            "Markdown document, complete with headings, subheadings, summaries, and an Action Items list."
        )
        
        user_prompt = f"""Here is a list of timestamped notes for {date_str}:

{combined_text}

Please transform this note into a well-structured daily note using the following guidelines:

1) Create a clear hierarchical structure with appropriate headings (# for main, ## for sections, ### for subsections)
2) Organize related thoughts and ideas under common themes or categories
3) Enhance readability by:
   - Breaking long paragraphs into concise bullet points
   - Using consistent formatting for similar types of information
   - Adding clear transitions between different topics
4) Include the following sections:
   - Daily Overview (brief summary of main activities/thoughts)
   - Key Insights or Learnings
   - Project Updates (if applicable)
   - Questions & Ideas for Further Exploration
   - Action Items & Next Steps

The final note should follow this structure:

# Daily Note - {date_str}

## Daily Overview
(A concise summary of the day's main activities and focus areas)

## Key Topics
(Major themes or areas of focus from the notes, organized into clear sections)

### Topic A
- Main points and insights
- Supporting details or examples
- Related thoughts and connections

### Topic B
(Similar structure as Topic A)

## Questions & Ideas
- Open questions that arose
- Ideas for future exploration
- Potential connections to investigate

## Action Items
- [ ] Specific tasks or next steps
- [ ] Follow-up items
- [ ] Reminders for future reference

Important Formatting Guidelines:
1) Use consistent bullet points for lists
2) Keep paragraphs concise and focused
3) Use bold or italic text sparingly and purposefully
4) Include cross-references to related notes where relevant
5) Remove any redundant information or unnecessary details

Note: Original timestamps should be removed from the final note while preserving the logical flow of ideas.

Please reason step by step, and put your final markdown within \\boxed{{ }}"""
        
        # Call the LLM
        if self.verbose:
            logger.info(f"Generating hub note for {date_str}")
            
        response = self.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=4096,
        )
        
        # Extract the Markdown content
        if response.get("choices") and len(response["choices"]) > 0:
            markdown_result = response["choices"][0]["message"]["content"]
            boxed_content = response["choices"][0].get("boxed_content")
            if boxed_content:
                markdown_result = boxed_content
            # Clean think blocks
            markdown_result = self._clean_think_blocks(markdown_result)
            # Remove wrapping markdown code block if present
            if markdown_result.startswith("```markdown\n"):
                markdown_result = markdown_result[11:]  # Remove ```markdown\n
            elif markdown_result.startswith("```\n"):
                markdown_result = markdown_result[4:]  # Remove ```\n
            if markdown_result.endswith("\n```"):
                markdown_result = markdown_result[:-4]  # Remove \n```
            markdown_result = markdown_result.strip()
        else:
            markdown_result = f"# Daily Note - {date_str}\n\n*(No response from LLM)*"
            if self.verbose:
                logger.error("Failed to get response from LLM")
        
        # Save markdown and process PARA links if requested
        if save_markdown:
            source_path = Path(source_file)
            # Get relative path from vault root and construct output path
            rel_path = source_path.relative_to(vault_path)
            output_path = output_dir / rel_path.with_suffix('.hub.md')
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write initial markdown file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_result)
                
            if self.verbose:
                logger.info(f"Saved initial hub note to {output_path}")
            
            try:
                # Create NoteParser instance for vault info
                note_parser = NoteParser(vault_path=str(vault_path))
                vault_info = note_parser.extract_vault_info()
                
                # Process PARA links
                self.process_daily_hub_note(
                    daily_hub_path=str(output_path),
                    vault_info=vault_info,
                    temperature=temperature
                )
                
                if self.verbose:
                    logger.info(f"Processed PARA links for {output_path}")
                    
                # Read the updated content with PARA links
                with open(output_path, 'r', encoding='utf-8') as f:
                    markdown_result = f.read()
                    
            except Exception as e:
                logger.error(f"Error processing PARA links: {str(e)}")
        
        return markdown_result

    def clear_chat_history(self) -> None:
        """
        Clear the chat history for the current LLM session.
        This can be useful to reset context between different tasks.
        """
        if hasattr(self, '_chat_history'):
            self._chat_history = []
        if self.verbose:
            logger.info("Chat history cleared")

    def classify_note_themes(
        self,
        content: str,
        vault_info: Dict[str, Any],
        hub_note_path: str,
        temperature: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Classify note themes according to PARA structure and match with existing vault structure.
        
        Args:
            content: The note content to classify
            vault_info: Dictionary containing vault information from NoteParser.extract_vault_info()
            hub_note_path: Path to the hub note being processed
            temperature: LLM temperature setting
            
        Returns:
            List of dictionaries containing theme classifications and links
        """
        # Extract existing PARA categories from vault structure
        para_categories = {
            'projects': [],
            'areas': [],
            'resources': [],
            'archive': []
        }
        
        # Extract existing project and area names from vault info
        for note in vault_info['hub_notes']:
            folder = note['folder']
            if '1_projects' in folder:
                para_categories['projects'].append(note['title'])
            elif '2_areas' in folder:
                para_categories['areas'].append(note['title'])
        
        # Build the system and user prompts
        system_prompt = """You are an AI note organizer specialized in the PARA method (Projects, Areas, Resources, Archive).
Your task is to analyze note content and classify its themes according to PARA, matching them with existing categories when possible.
Return your analysis in a structured JSON format."""

        user_prompt = f"""Analyze the following note content and classify its themes according to PARA:

{content}

Existing Categories in the Vault:
Projects: {', '.join(para_categories['projects'])}
Areas: {', '.join(para_categories['areas'])}

For each major theme or heading in the note:
1. Classify it as: Project (time-bound goal), Area (ongoing responsibility), Resource (reference), or Archive (completed/inactive)
2. Match it with an existing category if applicable
3. Suggest a new category name if no match exists

Return a JSON array in this format (just the JSON):
[
  {{
    "theme": "theme or heading text",
    "classification": "Project|Area|Resource|Archive",
    "matched_category": "existing category name or null",
    "suggested_category": "suggested new category name or null",
    "reasoning": "brief explanation of classification"
  }}
]"""

        response = self.create_stateless_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=4096  # Ensure enough tokens for response
        )

        try:
            raw_response = response["choices"][0]["message"]["content"]
            boxed_content = response["choices"][0].get("boxed_content")
            
            if boxed_content:
                content_to_parse = self._clean_markdown_blocks(boxed_content)
            else:
                # also remove fences from raw response
                content_to_parse = self._clean_markdown_blocks(raw_response)
            
            # Save raw response next to the hub note
            hub_path = Path(hub_note_path)
            raw_response_path = hub_path.parent / f"{hub_path.stem}.llm_raw"
            raw_response_path.write_text(raw_response, encoding='utf-8')
            
            logger.info(f"Saved raw LLM response to: {raw_response_path}")
            
            try:
                classifications = json.loads(content_to_parse)
                return classifications
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing theme classifications: {str(e)}")
                logger.error(f"Failed to parse content: {content_to_parse}")
                return []
            
        except (KeyError, AttributeError) as e:
            logger.error(f"Error accessing LLM response: {str(e)}")
            return []

    def link_daily_hub_to_para(
        self,
        daily_hub_path: str,
        daily_content: str,
        vault_info: Dict[str, Any],
        temperature: float = 0.6
    ) -> Dict[str, Any]:
        """Link daily hub note content to appropriate PARA structure hub notes.
        
        Args:
            daily_hub_path: Path to the daily hub note
            daily_content: Content of the daily hub note
            vault_info: Dictionary containing vault information
            temperature: LLM temperature setting
            
        Returns:
            Dictionary containing update information and modified files
        """
        # Classify themes in the daily note
        classifications = self.classify_note_themes(daily_content, vault_info, daily_hub_path, temperature)
        
        updates = {
            'daily_hub': {'path': daily_hub_path, 'links_added': []},
            'para_hubs': []
        }
        
        # Process each classification
        for theme in classifications:
            theme_category = theme['classification']
            matched_category = theme['matched_category']
            
            # Find the corresponding hub note
            target_hub = None
            if matched_category:
                for hub in vault_info['hub_notes']:
                    if hub['title'] == matched_category:
                        target_hub = hub
                        break
            
            if target_hub:
                # Create bidirectional links
                # 1. Add link in daily hub to PARA hub
                daily_section = f"\n### {theme['theme']}\n"
                daily_section += f"[[{target_hub['title']}]]\n"
                updates['daily_hub']['links_added'].append({
                    'theme': theme['theme'],
                    'link': target_hub['title']
                })
                
                # 2. Add transclusion in PARA hub from daily hub
                para_section = f"\n### Daily Notes References\n"
                para_section += f"![[{Path(daily_hub_path).stem}#{theme['theme']}]]\n"
                
                updates['para_hubs'].append({
                    'path': target_hub['path'],
                    'title': target_hub['title'],
                    'section_to_add': para_section,
                    'linked_theme': theme['theme']
                })
        
        return updates

    def process_daily_hub_note(
        self,
        daily_hub_path: str,
        vault_info: Dict[str, Any],
        temperature: float = 0.6
    ) -> None:
        """Process a daily hub note, analyze its themes, and create bidirectional links with PARA structure.
        
        Args:
            daily_hub_path: Path to the daily hub note
            vault_info: Dictionary containing vault information
            temperature: LLM temperature setting
        """
        try:
            # Read the daily hub note content
            with open(daily_hub_path, 'r', encoding='utf-8') as f:
                daily_content = f.read()
            
            # Get linking updates
            updates = self.link_daily_hub_to_para(daily_hub_path, daily_content, vault_info, temperature)
            
            if not updates['para_hubs']:
                logger.info(f"No PARA links found for {daily_hub_path}")
                return
                
            # Apply updates to daily hub
            with open(daily_hub_path, 'a', encoding='utf-8') as f:
                f.write("\n\n## PARA Links\n")
                for link in updates['daily_hub']['links_added']:
                    f.write(f"\n### {link['theme']}\n")
                    f.write(f"[[{link['link']}]]\n")
            
            # Apply updates to PARA hubs
            for hub_update in updates['para_hubs']:
                try:
                    hub_path = hub_update['path']
                    # Check if section already exists to avoid duplicates
                    with open(hub_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if "## Daily Notes References" not in content:
                            with open(hub_path, 'a', encoding='utf-8') as f:
                                f.write("\n\n## Daily Notes References\n")
                                
                    # Add new reference
                    with open(hub_path, 'a', encoding='utf-8') as f:
                        f.write(f"\n### {Path(daily_hub_path).stem}\n")
                        f.write(f"![[{Path(daily_hub_path).stem}#{hub_update['linked_theme']}]]\n")
                        
                except Exception as e:
                    logger.error(f"Error updating PARA hub {hub_update['path']}: {str(e)}")
                    continue
            
            logger.info(f"Successfully processed daily hub note and created PARA links: {daily_hub_path}")
            
        except Exception as e:
            logger.error(f"Error processing daily hub note: {str(e)}")
