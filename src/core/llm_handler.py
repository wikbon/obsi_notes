from llama_cpp import Llama
from typing import Optional, Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMHandler:
    """Handler for LLaMA model interactions."""
    
    def __init__(
        self,
        model_path: str = "/path/to/Yi-1.5-34B-Chat-Q6_K.gguf",
        n_gpu_layers: int = 20,
        n_ctx: int = 4096,
        chat_format: str = "chatml-function-calling",
        verbose: bool = False,
        config: Dict[str, Any] = {}
    ):
        """Initialize the LLM handler.
        
        Args:
            model_path: Path to the GGUF model file
            n_gpu_layers: Number of layers to offload to GPU
            n_ctx: Context window size
            chat_format: Format for chat interactions
            verbose: Whether to print detailed logs
            config: Configuration for external tools
        """
        self.model_path = model_path
        self.verbose = verbose
        self.config = config
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
            
        Raises:
            ValueError: If messages are invalid or empty
            RuntimeError: If LLM completion fails
        """
        if messages is None:
            messages = self.message_history
            
        if not messages:
            raise ValueError("No messages provided for chat completion")
            
        if self.verbose:
            logger.info("Creating chat completion")
            logger.info(f"Number of messages: {len(messages)}")
            logger.info(f"Temperature: {temperature}")
            if tools:
                logger.info(f"Number of tools available: {len(tools)}")
        
        try:
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
            
        except Exception as e:
            error_msg = f"Error during chat completion: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
            
    def google_search(self, query: str) -> List[Dict[str, str]]:
        """Perform a Google search using the configured API.
        
        Args:
            query (str): Search query
            
        Returns:
            List[Dict[str, str]]: List of search results with title and snippet
        """
        if not self.config.get('external_tools', {}).get('google_search', {}).get('enabled'):
            logger.warning("Google Search is not enabled in config")
            return []
            
        api_key = self.config.get('external_tools', {}).get('google_search', {}).get('api_key')
        search_engine_id = self.config.get('external_tools', {}).get('google_search', {}).get('search_engine_id')
        
        if not api_key or not search_engine_id:
            logger.warning("Google Search API credentials not configured")
            return []
            
        try:
            from googleapiclient.discovery import build
            service = build("customsearch", "v1", developerKey=api_key)
            
            result = service.cse().list(
                q=query,
                cx=search_engine_id,
                num=5  # Limit to 5 results
            ).execute()
            
            search_results = []
            if 'items' in result:
                for item in result['items']:
                    search_results.append({
                        'title': item['title'],
                        'snippet': item.get('snippet', ''),
                        'link': item['link']
                    })
                    
            return search_results
            
        except Exception as e:
            logger.error(f"Error during Google search: {str(e)}")
            return []
            
    def summarize_search_results(self, results: List[Dict[str, str]]) -> str:
        """Summarize search results using the LLM.
        
        Args:
            results (List[Dict[str, str]]): List of search results
            
        Returns:
            str: Summarized information
        """
        if not results:
            return "No search results available."
            
        system_prompt = """You are a research assistant. Summarize the key information from these search results."""
        
        user_prompt = "Here are the search results to summarize:\n\n"
        for i, result in enumerate(results, 1):
            user_prompt += f"{i}. {result['title']}\n{result['snippet']}\n\n"
            
        self.add_message("system", system_prompt)
        self.add_message("user", user_prompt)
        
        try:
            response = self.create_chat_completion(temperature=0.3)
            return response['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Error summarizing search results: {str(e)}")
            return "Error summarizing search results."
            
    def research_topic(self, topic: str) -> Dict[str, Any]:
        """Research a topic using Google Search and summarize results.
        
        Args:
            topic (str): Topic to research
            
        Returns:
            Dict[str, Any]: Research results including summary and sources
        """
        search_results = self.google_search(topic)
        summary = self.summarize_search_results(search_results)
        
        return {
            'topic': topic,
            'summary': summary,
            'sources': [result['link'] for result in search_results]
        }
