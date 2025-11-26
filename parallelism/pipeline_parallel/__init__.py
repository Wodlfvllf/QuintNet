"""
Pipeline Parallelism Module
"""


from .wrapper import PipelineParallelWrapper
from .trainer import PipelineTrainer

__all__ = [
    'PipelineParallelWrapper',
    'PipelineTrainer',
]
