"""
AI API Views - Ollama Integration Endpoints
"""
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.conf import settings
import logging

from .ollama import OllamaClient

logger = logging.getLogger(__name__)

class OllamaHealthView(APIView):
    """Check Ollama service health"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Check if Ollama is available"""
        try:
            client = OllamaClient()
            is_available = client.is_available()
            
            if is_available:
                return Response({
                    "status": "healthy",
                    "service": "ollama",
                    "available": True
                })
            else:
                return Response({
                    "status": "unavailable",
                    "service": "ollama",
                    "available": False
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
        except Exception as e:
            logger.error(f"Error checking Ollama health: {e}")
            return Response({
                "status": "error",
                "message": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class OllamaGenerateView(APIView):
    """Generate text using Ollama"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        """
        Generate text using Ollama
        
        Request body:
        {
            "prompt": "Your prompt here",
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "stream": false
        }
        """
        try:
            prompt = request.data.get("prompt")
            if not prompt:
                return Response({
                    "error": "Prompt is required"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            temperature = request.data.get("temperature", 0.7)
            top_p = request.data.get("top_p", 0.9)
            top_k = request.data.get("top_k", 40)
            stream = request.data.get("stream", False)
            
            client = OllamaClient()
            
            if not client.is_available():
                return Response({
                    "error": "Ollama service is not available"
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            
            result = client.generate(
                prompt=prompt,
                stream=stream,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
            
            return Response(result)
        
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return Response({
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class OllamaChatView(APIView):
    """Chat with Ollama"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        """
        Chat with Ollama model
        
        Request body:
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ],
            "temperature": 0.7,
            "stream": false
        }
        """
        try:
            messages = request.data.get("messages")
            if not messages or not isinstance(messages, list):
                return Response({
                    "error": "Messages array is required"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            temperature = request.data.get("temperature", 0.7)
            stream = request.data.get("stream", False)
            
            client = OllamaClient()
            
            if not client.is_available():
                return Response({
                    "error": "Ollama service is not available"
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            
            result = client.chat(
                messages=messages,
                stream=stream,
                temperature=temperature
            )
            
            return Response(result)
        
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return Response({
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class OllamaModelsView(APIView):
    """List available Ollama models"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get list of available models"""
        try:
            client = OllamaClient()
            models = client.list_models()
            
            return Response({
                "models": models,
                "count": len(models)
            })
        
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return Response({
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
