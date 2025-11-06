"""Distributed backends for CustomDDP."""

from QuintNet.parallelism.data_parallel.backends.base import DistributedBackend
from QuintNet.parallelism.data_parallel.backends.torch_backend import TorchDistributedBackend
from QuintNet.parallelism.data_parallel.backends.local_backend import LocalBackend

__all__ = ["DistributedBackend", "TorchDistributedBackend", "LocalBackend"]