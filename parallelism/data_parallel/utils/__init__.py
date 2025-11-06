"""Utility functions and factory methods."""

from QuintNet.parallelism.data_parallel.utils.factory import create_local_ddp, create_distributed_ddp

__all__ = ["create_local_ddp", "create_distributed_ddp"]