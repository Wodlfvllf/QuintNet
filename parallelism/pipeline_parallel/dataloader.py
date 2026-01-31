
class PipelineDataLoader:
    """
    Wrapper for DataLoader to support gradient accumulation with pipeline parallelism.
    
    This is a task-agnostic wrapper that passes batches through unchanged.
    The batch format depends on the DataLoader/Collator used:
    
    - For images (MNIST): {'image': tensor, 'label': tensor}
    - For text (GPT-2): {'input_ids': tensor, 'labels': tensor, 'attention_mask': tensor}
    
    Args:
        dataloader: The underlying DataLoader to wrap
        grad_acc_steps: Number of gradient accumulation steps (micro-batches per optimizer step)
        task_type: 'classification' for images, 'clm' for causal LM (default: auto-detect)
    """
    def __init__(self, dataloader, grad_acc_steps, task_type='auto'):
        self.dataloader = dataloader
        self.grad_acc_steps = grad_acc_steps
        self.task_type = task_type
        self.iterator = iter(dataloader)
    
    def __iter__(self):
        self.iterator = iter(self.dataloader)
        return self
    
    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        
        # Auto-detect task type from batch keys
        if self.task_type == 'auto':
            if 'input_ids' in batch:
                self.task_type = 'clm'
            elif 'image' in batch:
                self.task_type = 'classification'
        
        # For CLM tasks (text), pass through as-is
        if self.task_type == 'clm':
            return batch
        
        # For classification (images), convert to expected format for backward compatibility
        if 'image' in batch:
            return {
                "images": batch['image'],
                "labels": batch['label']
            }
        
        # Default: pass through unchanged
        return batch
    
    def __len__(self):
        return len(self.dataloader)