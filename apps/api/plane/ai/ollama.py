"""
Ollama LLM Integration Module
Provides integration with Ollama local language models
"""
import requests
import logging
from typing import Optional, Dict, Any
from django.conf import settings

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or getattr(settings, 'OLLAMA_API_URL', 'http://localhost:11434')
        self.model = model or getattr(settings, 'OLLAMA_MODEL', 'llama2')
        self.generate_endpoint = f"{self.base_url}/api/generate"
        self.chat_endpoint = f"{self.base_url}/api/chat"
        self.tags_endpoint = f"{self.base_url}/api/tags"
    
    def is_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama service not available: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        stream: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
    ) -> Dict[str, Any]:
        """
        Generate text using Ollama
        
        Args:
            prompt: The input prompt
            stream: Whether to stream the response
            temperature: Controls randomness (0-1)
            top_p: Nucleus sampling parameter
            top_k: Top k sampling parameter
        
        Returns:
            Dictionary with response data
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                }
            }
            
            response = requests.post(
                self.generate_endpoint,
                json=payload,
                timeout=300,  # 5 minutes timeout for long generations
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise
    
    def chat(
        self,
        messages: list,
        stream: bool = False,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Chat with Ollama model
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response
            temperature: Controls randomness (0-1)
        
        Returns:
            Dictionary with response data
        """
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                }
            }
            
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                timeout=300,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama chat API: {e}")
            raise
    
    def _handle_streaming_response(self, response):
        """Handle streaming response from Ollama"""
        chunks = []
        try:
            for line in response.iter_lines():
                if line:
                    chunks.append(line)
            return {"chunks": chunks, "stream": True}
        except Exception as e:
            logger.error(f"Error processing streaming response: {e}")
            raise
    
    def list_models(self) -> list:
        """Get list of available models"""
        try:
            response = requests.get(self.tags_endpoint, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            logger.error(f"Error listing Ollama models: {e}")
            return []
