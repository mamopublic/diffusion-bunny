"""
Stage 4.5: LLM Analysis
Remote character identification and scene description using OpenRouter LLMs.
"""

from .llm_analyzer import LLMAnalyzer, FrameLLMAnalysis
from .composite_creator import CompositeImageCreator, CompositeConfig
from .openrouter_client import OpenRouterClient, OpenRouterConfig, LLMResponse

__all__ = [
    'LLMAnalyzer',
    'FrameLLMAnalysis',
    'CompositeImageCreator',
    'CompositeConfig',
    'OpenRouterClient',
    'OpenRouterConfig',
    'LLMResponse'
]

__version__ = "0.1.0"
__author__ = "Diffusion Bunny Team"
__description__ = "LLM-based character identification and scene analysis for movie frames"
