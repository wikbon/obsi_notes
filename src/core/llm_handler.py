from llama_cpp import Llama
from typing import Optional, Dict, Any, List
import logging
import json
import re
import click
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMHandler:
    """Handler for LLaMA model interactions."""


    def __init__(
        self,
        # model_path: str = "/path/to/Yi-1.5-34B-Chat-Q6_K.gguf",
        model_path: str = "/path/to/granite-3.1-8b-instruct-Q6_K_L.gguf",
        # model_path: str = "/path/to/DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf",

        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        chat_format: str = "chatml-function-calling",
        verbose: bool = False
    ):
        """Initialize the LLM handler.
        
        Args:
            model_path: Path to the GGUF model file
            n_gpu_layers: Number of layers to offload to GPU
            n_ctx: Context window size
            chat_format: Format for chat interactions
            verbose: Whether to print detailed logs
        """
        self.model_path = model_path
        self.verbose = verbose
        if verbose:
            logger.info(f"Initializing LLM with model: {model_path}")
            
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            chat_format=chat_format,
            n_ctx=n_ctx
        )
        self.message_history: List[Dict[str, str]] = []
        
        if verbose:
            logger.info("LLM initialized successfully")
    
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
    
    def create_chat_completion(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a chat completion using the LLaMA model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
                     If None, uses internal message history.
            tools: Optional list of tool specifications for function calling
            tool_choice: Optional specification for tool selection
            temperature: Sampling temperature (0.0 to 1.0)
            **kwargs: Additional arguments to pass to create_chat_completion
            
        Returns:
            Dict containing the chat completion response
        """
        if messages is None:
            messages = self.message_history
            
        if self.verbose:
            logger.info("Creating chat completion")
            logger.info(f"Number of messages: {len(messages)}")
            logger.info(f"Temperature: {temperature}")
            if tools:
                logger.info(f"Number of tools available: {len(tools)}")
        
        response = self.llm.create_chat_completion(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            **kwargs
        )
        
        if self.verbose:
            logger.info("Received response from LLM")
            
        # Add assistant's response to history
        if response.get("choices") and len(response["choices"]) > 0:
            assistant_message = response["choices"][0].get("message", {})
            if assistant_message:
                self.message_history.append(assistant_message)
                if self.verbose:
                    logger.info("Added assistant's response to history")
        
        return response

    def extract_atomic_notes(
        self, 
        note_segments: List[Dict[str, Any]], 
        temperature: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Extract atomic notes from each segment using the LLM.
        
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
            
            # The LLM should respond with JSON. Let's parse it safely:
            raw_assistant_message = ""
            if response.get("choices"):
                raw_assistant_message = response["choices"][0]["message"]["content"]
 
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
        temperature: float = 0.7,
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
            temperature (float): LLM creativity level (default: 0.7).
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
        
        # Create system and user prompts
        system_prompt = (
            "You are an assistant that organizes daily mind dumps into a nicely formatted "
            "Markdown document, complete with headings, subheadings, summaries, and an Action Items list."
        )
        
        user_prompt = f"""
Here is a list of timestamped notes for {date_str}:

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
"""
        
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
