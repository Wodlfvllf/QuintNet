

class PipelineDataLoader:
    """
    Wrapper for DataLoader to support gradient accumulation.
    """
    def __init__(self, dataloader, grad_acc_steps):
        self.dataloader = dataloader
        self.grad_acc_steps = grad_acc_steps
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
        
        # Convert to expected format
        return {
            "images": batch['image'],
            "labels": batch['label']
        }
    
    def __len__(self):
        return len(self.dataloader)