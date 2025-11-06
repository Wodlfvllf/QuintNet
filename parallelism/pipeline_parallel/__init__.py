"""
Pipeline Parallelism Implementation.

This module contains:
- Pipeline parallel wrapper
- Pipeline trainer
- Schedule implementations (1F1B, GPipe)
- Stage communication

Migration Source: QuintNet/PipelineParallelism/
"""

from QuintNet.parallelism.pipeline_parallel.wrapper import PipelineParallelWrapper
from QuintNet.parallelism.pipeline_parallel.trainer import PipelineTrainer, PipelineParallel
from QuintNet.parallelism.pipeline_parallel.schedule import Schedule1F1B, ScheduleGPipe
from QuintNet.parallelism.pipeline_parallel.operations import send_forward, recv_forward, send_backward, recv_backward

__all__ = [
    'PipelineParallelWrapper',
    'PipelineTrainer',
    'PipelineParallel',
    'Schedule1F1B',
    'ScheduleGPipe',
    'send_forward',
    'recv_forward',
    'send_backward',
    'recv_backward',
]
