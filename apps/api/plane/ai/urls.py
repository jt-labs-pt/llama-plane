"""
AI API URLs
"""
from django.urls import path
from .views import (
    OllamaHealthView,
    OllamaGenerateView,
    OllamaChatView,
    OllamaModelsView,
)

urlpatterns = [
    # Ollama endpoints
    path("ollama/health/", OllamaHealthView.as_view(), name="ollama-health"),
    path("ollama/generate/", OllamaGenerateView.as_view(), name="ollama-generate"),
    path("ollama/chat/", OllamaChatView.as_view(), name="ollama-chat"),
    path("ollama/models/", OllamaModelsView.as_view(), name="ollama-models"),
]
