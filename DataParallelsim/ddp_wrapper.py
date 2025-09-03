# corrected_custom_ddp.py
import torch
import torch.distributed as dist
from typing import Optional
import torch.nn as nn

class CustomDDP(nn.Module):
    """
    Minimal custom DDP-like wrapper demonstrating:
    - bucketed gradient hooks (tensor.register_hook)
    - flattened all-reduce per bucket
    This version is robust to running without an initialized torch.distributed (will just
    do local averaging when dist isn't initialized) so it's runnable for testing.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        rank: int = 0,
        world_size: int = 1,
        broadcast_buffers: bool = True,
        process_group: Optional[dist.ProcessGroup] = None,
        bucket_cap_mb: int = 25,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = False,
    ):
        super().__init__()
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.broadcast_buffers = broadcast_buffers
        self.process_group = process_group
        self.bucket_cap_mb = bucket_cap_mb
        self.find_unused_parameters = find_unused_parameters
        self.gradient_as_bucket_view = gradient_as_bucket_view

        # buckets: dict idx -> list[param, param, ...]
        self.gradient_buckets = {}
        # bucket readiness flags
        self.bucket_is_ready = {}
        # store hook handles so we can remove later
        self._hook_handles = []

        # create buckets (if asked). we can also always create — here we follow the flag.
        if self.gradient_as_bucket_view:
            self._create_buckets()

        # broadcast parameters once at construction if the process group is initialized.
        # If dist isn't initialized, we just skip (so local testing works).
        if dist.is_available() and dist.is_initialized():
            self._broadcast_parameters()

        # register hooks on parameters that require_grad
        if self.gradient_as_bucket_view and self.gradient_buckets:
            self._register_hooks()

    # ----------------- bucket creation -----------------
    def _create_buckets(self):
        capacity = self.bucket_cap_mb * 1024 * 1024  # bytes
        buckets = {}
        current_bucket = []
        current_size = 0
        idx = 0

        # iterate parameters in reversed order (like your original intent)
        for param in reversed(list(self.model.parameters())):
            if not param.requires_grad:
                # skip params that won't get gradients
                continue

            num_elements = param.numel()
            bytes_per_element = param.element_size()
            param_bytes = num_elements * bytes_per_element

            # if adding this param exceeds capacity and current_bucket is non-empty -> start new bucket
            if current_bucket and (current_size + param_bytes > capacity):
                buckets[idx] = current_bucket
                self.bucket_is_ready[idx] = False
                idx += 1
                current_bucket = []
                current_size = 0

            current_bucket.append(param)
            current_size += param_bytes

        # add final bucket
        if current_bucket:
            buckets[idx] = current_bucket
            self.bucket_is_ready[idx] = False

        self.gradient_buckets = buckets

    # ----------------- forward wrapper -----------------
    def forward(self, *args, **kwargs):
        # simple pass-through to the wrapped model
        return self.model(*args, **kwargs)

    # ----------------- broadcast params -----------------
    def _broadcast_parameters(self):
        """Broadcast model parameters from rank 0 to all ranks (if dist initialized)."""
        if not dist.is_available() or not dist.is_initialized():
            # nothing to do for local test
            return

        print(f"CustomDDP Rank {self.rank}: Broadcasting parameters from rank 0")
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0, group=self.process_group)
        print(f"CustomDDP Rank {self.rank}: Parameter broadcast complete")

    # ----------------- register hooks -----------------
    def _register_hooks(self):
        def make_hook(bucket_idx):
            def hook(grad):
                # The hook is invoked when gradient for that tensor is computed.
                # We mark the bucket and, if complete, trigger the bucket all-reduce.
                self._mark_bucket_ready(bucket_idx)
                # must return grad (or modified grad) — we don't modify here
                return grad

            return hook

        for idx, param_list in self.gradient_buckets.items():
            for p in param_list:
                if p.requires_grad:
                    handle = p.register_hook(make_hook(idx))
                    self._hook_handles.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks (call when cleaning up)."""
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles = []

    # ----------------- bucket marking + all-reduce -----------------
    def _mark_bucket_ready(self, bucket_idx: int):
        # mark this bucket as ready
        self.bucket_is_ready[bucket_idx] = True

        # if all params in bucket have gradients, reduce the bucket
        bucket = self.gradient_buckets[bucket_idx]
        if all(param.grad is not None for param in bucket):
            self._all_reduce(bucket_idx)

    def allreduce(self, bucket_idx: int):
        """
        Flatten all grads in the bucket, all-reduce them (or skip if dist not initialized),
        divide by world_size, and copy back to param.grad.
        """
        bucket = self.gradient_buckets[bucket_idx]
        
        # Collect flattened grad tensors
        flat_grads = []
        devices = []
        for p in bucket:
            if p.grad is not None:
                flat = p.grad.detach().view(-1)
                flat_grads.append(flat)
                devices.append(p.grad.device)
        
        if len(flat_grads) == 0:
            self.bucket_is_ready[bucket_idx] = False
            return
        
        # Use consistent device (first gradient's device)
        device = devices[0]
        flat_grads = [g.to(device) for g in flat_grads]
        tensor_grads = torch.cat(flat_grads)
        
        # Perform distributed all-reduce if available
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor_grads, op=dist.ReduceOp.SUM, group=self.process_group)
            world_size = dist.get_world_size(self.process_group) if self.process_group else dist.get_world_size()
        
        # Average by actual world size
        if world_size > 1:
            tensor_grads.div_(world_size)
        
        # Copy averaged grads back to param.grad (keep on original devices)
        offset = 0
        for p in bucket:
            if p.grad is None:
                continue
            numel = p.grad.numel()
            chunk = tensor_grads[offset:offset + numel]
            p.grad.data.copy_(chunk.view_as(p.grad).to(p.grad.device))
            offset += numel
        
        self.bucket_is_ready[bucket_idx] = False

    # ----------------- utility: get param memory sizes -----------------
    def param_memory_table(self):
        """Return list of tuples (name, size_bytes, size_mb) for all parameters."""
        rows = []
        for name, p in self.model.named_parameters():
            numel = p.numel()
            bpe = p.element_size()
            total_bytes = numel * bpe
            rows.append((name, total_bytes, total_bytes / (1024 ** 2)))
        return rows
    
    # def train(self):
    #     for child in self.model.children():
    #         child.train()
