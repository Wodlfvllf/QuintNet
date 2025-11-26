"""Distributed backends for DataParallel."""

from .base import DistributedBackend
from .torch_backend import TorchDistributedBackend
from .local_backend import LocalBackend

__all__ = ["DistributedBackend", "TorchDistributedBackend", "LocalBackend"]