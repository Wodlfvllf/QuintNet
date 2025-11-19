import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
import os

from QuintNet.parallelism.pipeline_parallel import PipelineTrainer, PipelineDataLoader
from QuintNet.core.process_groups import ProcessGroupManager

class Trainer:
    """
    A high-level trainer class to orchestrate the training and validation process.
    It abstracts away the training loop logic, making the main script cleaner.
    """
    def __init__(self, model: nn.Module, train_loader, val_loader, config: dict, pg_manager: ProcessGroupManager):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.pg_manager = pg_manager
        
        self.device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
        self.global_rank = dist.get_rank()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()

        self.pp_size = self.config['mesh_dim'][self.config['mesh_name'].index('pp')]
        self.is_pipeline = self.pp_size > 1
        
        if self.is_pipeline:
            self._setup_pipeline_training()

    def _setup_pipeline_training(self):
        """Prepares the PipelineTrainer and PipelineDataLoader if pipeline parallelism is active."""
        coords = self.pg_manager.get_coordinates_tensor_search(self.global_rank)
        pp_rank = coords[self.config['mesh_name'].index('pp')]
        pp_group = self.pg_manager.get_group('pp')

        self.pipeline_trainer = PipelineTrainer(
            self.model,
            self.pg_manager.device_mesh,
            pp_rank,
            pp_group,
            self.criterion,
            self.device,
            optimizer=self.optimizer,
            max_grad_norm=self.config['max_grad_norm'],
            schedule_type=self.config.get('schedule', '1f1b')
        )
        self.train_loader = PipelineDataLoader(self.train_loader, self.config['grad_acc_steps'])
        
        # Access the underlying pipeline wrapper to get tensor shapes for communication
        # This assumes DDP -> PP wrapping order.
        if hasattr(self.model, 'module') and hasattr(self.model.module, 'local_module'):
             pipeline_wrapper = self.model.module
             self.tensor_shapes = pipeline_wrapper.get_tensor_shapes(self.config['batch_size'])
        else:
            self.tensor_shapes = None

    def fit(self):
        """The main entry point to start the training and validation process."""
        best_val_acc = 0.0
        epochs_without_improvement = 0

        for epoch in range(self.config['num_epochs']):
            if self.global_rank == 0:
                print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}\n" + "-" * 50)

            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc = self._validate_epoch(epoch)

            if self.global_rank == 0:
                print(f"Epoch {epoch+1} Results:")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        print("\nTraining complete.")

    def _train_epoch(self, epoch: int):
        """Handles the logic for a single training epoch."""
        self.model.train()
        
        if self.is_pipeline:
            return self.pipeline_trainer.train_step(
                self.train_loader,
                self.tensor_shapes,
                self.device,
                torch.float32,
            )
        else:
            # Standard non-pipeline training loop (e.g., for DP or TP only)
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}", disable=(self.global_rank != 0))
            
            for batch in pbar:
                images = batch['image'].to(self.device, non_blocking=True)        
                labels = batch['label'].to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if self.global_rank == 0:
                    accuracy = 100 * correct / total
                    pbar.set_postfix({
                        'Loss': f'{running_loss / (pbar.n + 1):.4f}',
                        'Acc': f'{accuracy:.2f}%'
                    })
            
            avg_loss = running_loss / len(self.train_loader)
            accuracy = 100 * correct / total
            return avg_loss, accuracy

    def _validate_epoch(self, epoch: int):
        """Handles the logic for a single validation epoch."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        if self.is_pipeline:
            # Use the pipeline trainer's evaluation method
            return self.pipeline_trainer.evaluate(self.val_loader, self.tensor_shapes, self.device, torch.float32)
        else:
            with torch.no_grad():
                pbar = tqdm(self.val_loader, desc=f"Validating Epoch {epoch+1}", disable=(self.global_rank != 0))
                for batch in pbar:
                    images = batch['image'].to(self.device, non_blocking=True)
                    labels = batch['label'].to(self.device, non_blocking=True)
                    
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    total_loss += loss.item() * labels.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

            avg_loss = total_loss / total_samples
            accuracy = 100 * total_correct / total_samples
            return avg_loss, accuracy
