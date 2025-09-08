"""
Stage 2: Quality Filtering Module

This module provides frame filtering capabilities based on quality metrics
including blur detection, brightness analysis, and contrast measurement.
"""

from .filter import FrameFilter, QualityMetrics, FilteredFrame

__all__ = ['FrameFilter', 'QualityMetrics', 'FilteredFrame']
