"""
High-Level Trainer for QuintNet

This module provides a high-level `Trainer` class that abstracts away the
entire training and validation loop. It is designed to be agnostic to the
specific parallelism strategy being used, providing a single, clean interface
to run the training process.

===============================================================================
CONCEPTUAL EXAMPLE:
===============================================================================

A user's training script (e.g., `examples/full_3d.py`) would use the Trainer
as follows:

.. code-block:: python

    # 1. Load Config, Create Model and DataLoaders
    config = load_config('config.yaml')
    model = MyModel(config)
    train_loader, val_loader = create_dataloaders(config)

    # 2. Initialize Process Groups and Apply Parallelism Strategy
    pg_manager = init_process_groups(config)
    strategy = get_strategy(config['strategy_name'], pg_manager, config)
    parallel_model = strategy.apply(model)

    # 3. Create and Run the Trainer
    trainer = Trainer(
        model=parallel_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        pg_manager=pg_manager
    )
    trainer.fit()

The `Trainer` handles:
- Iterating over epochs.
- Calling the correct training step function (either for pipeline or non-pipeline).
- Running the validation loop.
- Printing and logging metrics.
- (Future) Checkpointing and early stopping.

===============================================================================
"""
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

    This class encapsulates the entire training loop, including epoch iteration,
    training steps, validation, and metric logging. It is designed to work
    seamlessly with any parallelized model produced by the QuintNet strategy
    and coordinator system.
    """
    def __init__(self, model: nn.Module, train_loader, val_loader, config: dict, pg_manager: ProcessGroupManager):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The model to be trained. This should be the model
                that has already been parallelized by a QuintNet strategy.
            train_loader: The DataLoader for the training set.
            val_loader: The DataLoader for the validation set.
            config (dict): The configuration dictionary, loaded from a YAML file.
            pg_manager (ProcessGroupManager): The process group manager, which
                provides information about the distributed setup.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.pg_manager = pg_manager
        
        self.device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
        self.global_rank = dist.get_rank()

        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()

        # Check if pipeline parallelism is active to determine the training step strategy
        self.pp_size = self.config['mesh_dim'][self.config['mesh_name'].index('pp')]
        self.is_pipeline = self.pp_size > 1
        
        if self.is_pipeline:
            self._setup_pipeline_training()

    def _setup_pipeline_training(self):
        """
        Prepares the specialized `PipelineTrainer` and `PipelineDataLoader`
        if pipeline parallelism is being used.
        """
        coords = self.pg_manager.get_coordinates_tensor_search(self.global_rank)
        pp_rank = coords[self.config['mesh_name'].index('pp')]
        pp_group = self.pg_manager.get_group('pp')

        # The PipelineTrainer handles the complex 1F1B or AFAB schedules
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
        # The PipelineDataLoader wraps the train loader to handle gradient accumulation
        self.train_loader = PipelineDataLoader(self.train_loader, self.config['grad_acc_steps'])
        
        # To communicate between pipeline stages, we need to know the shape of the tensors being passed.
        # This logic accesses the underlying model to get the shapes.
        # Note: This assumes a DDP -> PP wrapping order for the model.
        if hasattr(self.model, 'module') and hasattr(self.model.module, 'local_module'):
             pipeline_wrapper = self.model.module
             self.tensor_shapes = pipeline_wrapper.get_tensor_shapes(self.config['batch_size'])
        else:
            # This case handles pure PP without a DDP wrapper
            self.tensor_shapes = self.model.get_tensor_shapes(self.config['batch_size'])


    def fit(self):
        """
        The main entry point to start the training and validation process.
        This method iterates over the configured number of epochs and handles
        the training and validation for each epoch.
        """
        # TODO: Implement state for best metrics and early stopping
        best_val_acc = 0.0
        epochs_without_improvement = 0

        for epoch in range(self.config['num_epochs']):
            if self.global_rank == 0:
                print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}\n" + "-" * 50)

            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc = self._validate_epoch(epoch)

            # Logging should only happen on the main rank to avoid clutter
            if self.global_rank == 0:
                print(f"Epoch {epoch+1} Results:")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        print("\nTraining complete.")

    def _train_epoch(self, epoch: int):
        """
        Handles the logic for a single training epoch.

        It checks if pipeline parallelism is active and delegates to the
        appropriate training step function.

        Args:
            epoch (int): The current epoch number.

        Returns:
            A tuple containing the average training loss and accuracy for the epoch.
        """
        self.model.train()
        
        if self.is_pipeline:
            # For pipeline parallelism, we use the specialized PipelineTrainer's train_step
            return self.pipeline_trainer.train_step(
                self.train_loader,
                self.tensor_shapes,
                self.device,
                torch.float32,
            )
        else:
            # For other strategies (DP, TP), a standard training loop is sufficient
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Progress bar is only shown on the main rank
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
            
            # For DP, gradients are already averaged. For TP, each rank has the full loss.
            # We can just return the metrics from rank 0.
            avg_loss = running_loss / len(self.train_loader)
            accuracy = 100 * correct / total
            return avg_loss, accuracy

    def _validate_epoch(self, epoch: int):
        """
        Handles the logic for a single validation epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            A tuple containing the average validation loss and accuracy for the epoch.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        if self.is_pipeline:
            # Use the pipeline trainer's specialized evaluation method
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

            # For DP, we need to aggregate results from all ranks
            if dist.is_initialized() and dist.get_world_size() > 1:
                total_loss_tensor = torch.tensor(total_loss, device=self.device)
                total_correct_tensor = torch.tensor(total_correct, device=self.device)
                total_samples_tensor = torch.tensor(total_samples, device=self.device)

                dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

                total_loss = total_loss_tensor.item()
                total_correct = total_correct_tensor.item()
                total_samples = total_samples_tensor.item()

            avg_loss = total_loss / total_samples if total_samples > 0 else 0
            accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0
            return avg_loss, accuracy