"""
Pipeline Parallelism Schedules.

This module will contain different pipeline schedules:
- 1F1B (One Forward One Backward)
- GPipe
- Interleaved 1F1B
- Virtual pipeline
"""

from abc import ABC, abstractmethod


class PipelineSchedule(ABC):
    """Base class for pipeline schedules."""
    
    @abstractmethod
    def execute(self, *args, **kwargs):
        """Execute the pipeline schedule."""
        pass


class Schedule1F1B(PipelineSchedule):
    """
    TODO: Extract from pipeline_trainer.py
    
    1F1B (One Forward One Backward) schedule.
    Reduces memory usage compared to GPipe.
    """
    
    def execute(self, *args, **kwargs):
        pass


class ScheduleGPipe(PipelineSchedule):
    """
    TODO: Implement GPipe schedule
    
    GPipe schedule with micro-batching.
    """
    
    def execute(self, *args, **kwargs):
        pass
