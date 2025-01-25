import requests
import json
import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LMStudioHandler:
    """Handler for LM Studio API interactions."""

    CHAT_TEMPLATE = "<｜begin▁of▁sentence｜>{system_prompt}<｜User｜>{prompt}<｜Assistant｜>"
    BOXED_PATTERN = r"\\boxed\{([^}]+)\}"

    def __init__(self, verbose: bool = False):
        """Initialize the LM Studio handler.
        
        Args:
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose
        self.base_url = "http://127.0.0.1:1234/v1"
        self.message_history: List[Dict[str, str]] = []
        
        if verbose:
            logger.info("Initializing LM Studio handler")
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the chat history.
        
        Args:
            role: Role of the message sender (system/user/assistant)
            content: Content of the message
        """
        self.message_history.append({"role": role, "content": content})
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
        """Format messages according to LM Studio chat template.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string
        """
        system_parts = []
        user_parts = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            elif msg["role"] == "user":
                user_parts.append(msg["content"])

        system_prompt = " ".join(system_parts)
        user_prompt = " ".join(user_parts)
        
        return self.CHAT_TEMPLATE.format(
            system_prompt=system_prompt,
            prompt=user_prompt
        )

    def _extract_boxed_content(self, text: str) -> Optional[str]:
        """Extract content from \boxed{} if present.
        
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
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def create_chat_completion(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        stream: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a chat completion using the LM Studio API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
                     If None, uses internal message history.
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional arguments
            
        Returns:
            Dict containing the chat completion response
        """
        if messages is None:
            messages = self.message_history
        
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "your-lmstudio-model-id",  # Model ID
            "messages": messages,
            "temperature": temperature,
            "max_tokens": -1,
            "stream": stream
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        if line.startswith(b"data: "):
                            try:
                                json_str = line[6:].decode('utf-8')
                                if json_str.strip() == "[DONE]":
                                    break
                                chunk = json.loads(json_str)
                                if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                                    content = chunk["choices"][0]["delta"]["content"]
                                    full_response += content
                                    if self.verbose:
                                        print(content, end="", flush=True)
                            except json.JSONDecodeError:
                                continue
                return {"choices": [{"message": {"role": "assistant", "content": full_response}}]}
            else:
                response_json = response.json()
                content = response_json["choices"][0]["message"]["content"]
                self.add_message("assistant", content)
                return {"choices": [{"message": {"role": "assistant", "content": content}}]}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with LM Studio: {str(e)}")
            return {"error": str(e)}

    def create_chat_completion_no_history(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a chat completion using the LM Studio API without affecting message history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Sampling temperature
            **kwargs: Additional arguments
            
        Returns:
            Dict containing the chat completion response
        """
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "your-lmstudio-model-id",  # Model ID
            "messages": messages,
            "temperature": temperature,
            "max_tokens": -1,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"]
            content = self._clean_think_blocks(content)
            boxed_content = self._extract_boxed_content(content)
            if boxed_content:
                boxed_content = self._clean_think_blocks(boxed_content)
            
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "boxed_content": boxed_content
                }]
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with LM Studio: {str(e)}")
            return {"error": str(e)}

    def extract_atomic_notes(
        self, 
        note_segments: List[Dict[str, Any]], 
        temperature: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Extract atomic notes from each segment using the LM Studio API.
        
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
            
            response = self.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                stream=False
            )
            
            raw_assistant_message = ""
            if response.get("choices"):
                raw_assistant_message = response["choices"][0]["message"]["content"]
                raw_assistant_message = self._clean_think_blocks(raw_assistant_message)
 
            try:
                raw_assistant_message = raw_assistant_message.lstrip("\ufeff").strip()
                
                if raw_assistant_message.startswith("```json") and raw_assistant_message.endswith("```"):
                    raw_assistant_message = raw_assistant_message[7:-3].strip()
                elif raw_assistant_message.startswith("```") and raw_assistant_message.endswith("```"):
                    raw_assistant_message = raw_assistant_message[3:-3].strip()
                raw_assistant_message = re.sub(r",\s*}", "}", raw_assistant_message)

                if not raw_assistant_message.startswith(("{", "[")):
                    logger.error("Input does not appear to be valid JSON.")
                    return []
                
                extracted_list = json.loads(raw_assistant_message)
                
                if not isinstance(extracted_list, list):
                    extracted_list = [extracted_list]
                    
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
            temperature (float): LLM creativity level.
            save_markdown (bool): Whether to save the generated markdown to a file
            output_dir (Path, optional): Directory to save output files. Required if save_markdown is True
            vault_path (Path, optional): Path to the vault root. Required if save_markdown is True
            
        Returns:
            str: A formatted Markdown string containing the organized daily hub note.
        """
        if date_str is None:
            source_path = Path(source_file)
            parts = source_path.stem.split('-')
            if len(parts) >= 3:
                date_str = f"{parts[0]}-{parts[1]}-{parts[2]}"
            else:
                date_str = source_path.stem
            
        lines_for_prompt = []
        for note in parsed_notes:
            timestamp = note.get('timestamp', 'N/A')
            content = note.get('content', '')
            lines_for_prompt.append(f"- Timestamp: {timestamp} | Content: {content}")
            
        combined_text = "\n".join(lines_for_prompt)
        
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

Please reason step by step, and put your final markdown within \boxed{{ }}"""
        
        if self.verbose:
            logger.info(f"Generating hub note for {date_str}")
            
        response = self.create_chat_completion_no_history(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            stream=False
        )
        
        if response.get("choices") and len(response["choices"]) > 0:
            markdown_result = response["choices"][0]["message"]["content"]
            boxed_content = response["choices"][0].get("boxed_content")
            if boxed_content:
                markdown_result = boxed_content
            markdown_result = self._clean_think_blocks(markdown_result)
            if markdown_result.startswith("```markdown\n"):
                markdown_result = markdown_result[11:]
            elif markdown_result.startswith("```\n"):
                markdown_result = markdown_result[4:]
            if markdown_result.endswith("\n```"):
                markdown_result = markdown_result[:-4]
            markdown_result = markdown_result.strip()
        else:
            markdown_result = f"# Daily Note - {date_str}\n\n*(No response from LLM)*"
            if self.verbose:
                logger.error("Failed to get response from LLM")
        
        if save_markdown:
            if output_dir is None or vault_path is None:
                raise ValueError("output_dir and vault_path must be provided when save_markdown is True")
                
            source_path = Path(source_file)
            rel_path = source_path.relative_to(vault_path)
            output_path = output_dir / rel_path.with_suffix('.hub.md')
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_result)
                
            if self.verbose:
                logger.info(f"Saved hub note to {output_path}")
        
        return markdown_result

    def clear_chat_history(self) -> None:
        """
        Clear the chat history for the current LLM session.
        This can be useful to reset context between different tasks.
        """
        self.message_history = []
        if self.verbose:
            logger.info("Chat history cleared")
