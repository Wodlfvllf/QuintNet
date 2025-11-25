"""
Pipeline Parallelism Module
"""


from ...core.communication import pipeline_communicate, bidirectional_pipeline_communicate
from .wrapper import PipelineParallelWrapper
from .trainer import PipelineTrainer

__all__ = [
    'pipeline_communicate',
    'bidirectional_pipeline_communicate',
    'PipelineParallelWrapper',
    'PipelineTrainer',
]
