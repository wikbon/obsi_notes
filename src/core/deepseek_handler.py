from llama_cpp import Llama
from typing import Optional, Dict, Any, List
import logging
import json
import re
from pathlib import Path
import click

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
            
            # Build the user prompt (no system prompt for DeepSeek)
            user_prompt = f"""Here is a raw note segment{f' (timestamp: {timestamp})' if timestamp else ''}:

{content}

Extract atomic notes following these rules:
1. Each note should be one complete, self-contained idea.
2. Keep related concepts together if they form a single coherent thought.
3. Don't split definitions or explanations of a single concept.
4. If multiple sentences make up a single idea, keep them together as a single atomic note.
5. If in doubt, don't split a sentence into multiple atomic notes.

Please reason step by step, and put your final answer within \\boxed{{
[
  {{
    "note": "the complete atomic idea here",
  }}
]
}}"""
            
            # Use the LLM
            if self.verbose:
                logger.info(f"Processing segment{f' from {timestamp}' if timestamp else ''}")
                
            response = self.create_chat_completion(
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=2048,  # adjust as needed
            )
            
            # The LLM should respond with boxed JSON
            raw_assistant_message = ""
            if response.get("choices"):
                raw_assistant_message = response["choices"][0]["message"]["content"]
                boxed_content = response["choices"][0].get("boxed_content")
                if boxed_content:
                    raw_assistant_message = boxed_content
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
                    return
                
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

    def clear_chat_history(self) -> None:
        """
        Clear the chat history for the current LLM session.
        This can be useful to reset context between different tasks.
        """
        if hasattr(self, '_chat_history'):
            self._chat_history = []
        if self.verbose:
            logger.info("Chat history cleared")

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
        
        # Create user prompt (no system prompt for DeepSeek)
        user_prompt = f"""Here is a list of timestamped notes for {date_str}:

{combined_text}

Please do the following:
1) Identify main topics or themes across these notes
2) Group them under subheadings in Markdown (## or ###, etc.)
3) Summarize each group in a concise way
4) Provide an overall daily summary at the top
5) Extract any actionable items (to-dos, next steps) and place them in a separate 'Action Items' section
6) Return the final output in valid Markdown
7) The final note's top heading should be '# Daily Note - {date_str}'

Make sure the final format is well-structured, with bullet points or short paragraphs where relevant.

Please reason step by step, and put your final markdown within \\boxed{{ }}"""
        
        # Call the LLM
        if self.verbose:
            logger.info(f"Generating hub note for {date_str}")
            
        response = self.create_chat_completion(
            messages=[
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
        
        # Save markdown if requested
        if save_markdown:
            if output_dir is None or vault_path is None:
                raise ValueError("output_dir and vault_path must be provided when save_markdown is True")
                
            source_path = Path(source_file)
            # Get relative path from vault root and construct output path
            rel_path = source_path.relative_to(vault_path)
            output_path = output_dir / rel_path.with_suffix('.hub.md')
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write markdown file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_result)
                
            if self.verbose:
                logger.info(f"Saved hub note to {output_path}")
        
        return markdown_result
