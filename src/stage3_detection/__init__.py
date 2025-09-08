"""
Stage 3: Character Detection
Detects anime faces and recognizes characters using YOLOv8 and feature matching.
"""

from .detector import CharacterDetector

__all__ = ['CharacterDetector']
