"""
High-Level GPT-2 Trainer for QuintNet

This module provides a specialized `GPT2Trainer` class for training GPT-2 models
with 3D parallelism (DP + PP + TP) on causal language modeling tasks like summarization.

===============================================================================
KEY DIFFERENCES FROM CLASSIFICATION TRAINER:
===============================================================================

1. **Loss Function**: CrossEntropyLoss with ignore_index=-100 for padding
2. **Metrics**: Perplexity (exp(loss)) instead of accuracy  
3. **Batch Format**: input_ids, labels, attention_mask instead of images
4. **Weight Tying**: Calls sync_tied_weights_grad() after backward pass
5. **Position IDs**: Generates position IDs for GPT-2 forward pass

===============================================================================
USAGE EXAMPLE:
===============================================================================

.. code-block:: python

    from QuintNet.GPT2_Trainer import GPT2Trainer
    from QuintNet.utils.Dataloader import SummarizationDataset, SummarizationCollator
    
    # Setup data
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    train_dataset = SummarizationDataset('./data', split='train')
    collator = SummarizationCollator(tokenizer)
    train_loader = DataLoader(train_dataset, collate_fn=collator, batch_size=8)
    
    # Setup model with 3D parallelism
    strategy = get_strategy('3d', pg_manager, config, checkpoint_path=..., is_staged=True)
    model = strategy.apply(None)  # Model built from checkpoint
    
    # Train
    trainer = GPT2Trainer(model, train_loader, val_loader, config, pg_manager)
    trainer.fit()

===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
import os
import math

from .parallelism import PipelineTrainer, PipelineDataLoader
from .core.process_groups import ProcessGroupManager
from .utils import SummarizationDataLoader, SummarizationCollator, SummarizationDataset

class GPT2Trainer:
    """
    Trainer for GPT-2 causal language modeling with 3D parallelism.
    
    Handles:
    - CLM loss computation with padding ignored
    - Perplexity metrics
    - Weight tying gradient synchronization
    - Pipeline parallel scheduling
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        train_loader, 
        val_loader, 
        config: dict, 
        pg_manager: ProcessGroupManager,
        model_config = None,  # GPT2Config object for model params
    ):
        """
        Initialize the GPT2Trainer.
        
        Args:
            model: Parallelized GPT-2 model from strategy.apply()
            train_loader: DataLoader with SummarizationCollator
            val_loader: Validation DataLoader
            config: Training config dict
            pg_manager: Process group manager
            model_config: Optional GPT2Config for model parameters
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.pg_manager = pg_manager
        self.model_config = model_config
        
        self.device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
        self.global_rank = dist.get_rank()
        
        # ─────────────────────────────────────────────────────────────────
        # Optimizer: AdamW is standard for language models
        # ─────────────────────────────────────────────────────────────────
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=float(self.config['learning_rate']),
            weight_decay=0.01,
        )
        
        # ─────────────────────────────────────────────────────────────────
        # Loss: CrossEntropy with ignore_index for padding tokens (-100)
        # ─────────────────────────────────────────────────────────────────
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Get vocab size from model config or default
        if model_config is not None:
            self.vocab_size = model_config.vocab_size
        else:
            self.vocab_size = self.config.get('vocab_size', 50257)
        
        # ─────────────────────────────────────────────────────────────────
        # Pipeline parallelism setup
        # ─────────────────────────────────────────────────────────────────
        self.pp_size = self.config['mesh_dim'][self.config['mesh_name'].index('pp')]
        self.is_pipeline = self.pp_size > 1
        
        # Get pipeline rank info for weight tying sync
        coords = self.pg_manager.get_coordinates_tensor_search(self.global_rank)
        self.pp_rank = coords[self.config['mesh_name'].index('pp')]
        self.is_first_stage = (self.pp_rank == 0)
        self.is_last_stage = (self.pp_rank == self.pp_size - 1)
        
        if self.is_pipeline:
            self._setup_pipeline_training()
        
        # ─────────────────────────────────────────────────────────────────
        # Get reference to the underlying GPT2Stage for weight sync
        # ─────────────────────────────────────────────────────────────────
        self.gpt2_stage = self._get_gpt2_stage()
    
    def _get_gpt2_stage(self):
        """
        Extract the GPT2Stage module from wrapped model.
        
        The model is wrapped as: DataParallel -> PipelineParallelWrapper -> GPT2Stage
        """
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'local_module'):
            return self.model.model.local_module
        elif hasattr(self.model, 'local_module'):
            return self.model.local_module
        else:
            return self.model
    
    def _setup_pipeline_training(self):
        """Setup pipeline trainer and data loader for PP."""
        pp_group = self.pg_manager.get_group('pp')
        
        # Create pipeline trainer with task_type='clm' for GPT-2
        self.pipeline_trainer = PipelineTrainer(
            model=self.model,
            device_mesh=self.pg_manager.device_mesh,
            pp_rank=self.pp_rank,
            pp_group=pp_group,
            criterion=self.criterion,
            device=self.device,
            optimizer=self.optimizer,
            max_grad_norm=float(self.config['max_grad_norm']),
            schedule_type=self.config.get('schedule', '1f1b'),
            task_type='clm',  # GPT-2 uses causal language modeling
            vocab_size=self.vocab_size,
        )
        
        # Wrap DataLoader with task_type='clm' for proper batch handling
        self.train_loader = PipelineDataLoader(
            self.train_loader, 
            self.config['grad_acc_steps'],
            task_type='clm'
        )
        
        # Calculate tensor shapes for pipeline communication
        dp_size = self.config['mesh_dim'][self.config['mesh_name'].index('dp')]
        micro_batch_size = self.config['batch_size'] // (self.config['grad_acc_steps'] * dp_size)
        if micro_batch_size < 1:
            micro_batch_size = 1
        
        max_seq_length = self.config.get('max_seq_length', 512)
        hidden_dim = self.model_config.n_embd if self.model_config else 768
        
        # Tensor shape: [batch, seq_len, hidden_dim]
        self.tensor_shapes = (micro_batch_size, max_seq_length, hidden_dim)
    
    def fit(self):
        """Main training loop."""
        best_val_ppl = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            if self.global_rank == 0:
                print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}\n" + "-" * 50)
            
            train_loss, train_ppl = self._train_epoch(epoch)
            val_loss, val_ppl = self._validate_epoch(epoch)
            
            # Aggregate metrics for pipeline parallelism
            if self.is_pipeline:
                metrics_tensor = torch.zeros(4, device=self.device)
                
                if self.is_last_stage:
                    metrics_tensor[0] = train_loss if train_loss is not None else 0.0
                    metrics_tensor[1] = train_ppl if train_ppl is not None else 0.0
                    metrics_tensor[2] = val_loss if val_loss is not None else 0.0
                    metrics_tensor[3] = val_ppl if val_ppl is not None else 0.0
                
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.MAX)
                
                train_loss = metrics_tensor[0].item()
                train_ppl = metrics_tensor[1].item()
                val_loss = metrics_tensor[2].item()
                val_ppl = metrics_tensor[3].item()
            
            if self.global_rank == 0:
                print(f"Epoch {epoch+1} Results:")
                print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
                print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
            
            # Save checkpoint from ALL ranks (each saves its shard)
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                if dist.is_initialized():
                    dist.barrier()  # Sync before saving
                self._save_checkpoint('best_model.pt')
                if dist.is_initialized():
                    dist.barrier()  # Sync after saving
                if self.global_rank == 0:
                    print(f"  ✅ New best model saved (PPL: {best_val_ppl:.2f})")
        
        # Final save from ALL ranks
        if dist.is_initialized():
            dist.barrier()
        self._save_checkpoint('final_model.pt')
        if dist.is_initialized():
            dist.barrier()
        
        if self.global_rank == 0:
            print("\nTraining complete.")
            
            # Run generation-based evaluation (ROUGE/BLEU)
            if self.config.get('compute_generation_metrics', True):
                self._evaluate_generation_metrics()
    
    def _train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        
        if self.is_pipeline:
            return self._train_epoch_pipeline(epoch)
        else:
            return self._train_epoch_standard(epoch)
    
    def _train_epoch_standard(self, epoch: int):
        """Standard training loop (no pipeline parallelism)."""
        total_loss = 0.0
        total_tokens = 0
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}", disable=(self.global_rank != 0))
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            
            # Create position IDs
            batch_size, seq_len = input_ids.shape
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_ids, position_ids=position_ids)
            
            # Compute loss (reshape for CrossEntropyLoss)
            # logits: [B, T, V] -> [B*T, V]
            # labels: [B, T] -> [B*T]
            loss = self.criterion(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # ─────────────────────────────────────────────────────────────
            # CRITICAL: Sync tied weight gradients between first and last stages
            # ─────────────────────────────────────────────────────────────
            if hasattr(self.gpt2_stage, 'sync_tied_weights_grad'):
                self.gpt2_stage.sync_tied_weights_grad()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
            
            self.optimizer.step()
            
            # Accumulate metrics (only count non-padding tokens)
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            if self.global_rank == 0:
                current_ppl = math.exp(loss.item()) if loss.item() < 20 else float('inf')
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'PPL': f'{current_ppl:.2f}'})
        
        # Aggregate across DP ranks
        if dist.is_initialized() and dist.get_world_size() > 1:
            loss_tensor = torch.tensor(total_loss, device=self.device)
            tokens_tensor = torch.tensor(total_tokens, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor.item()
            total_tokens = tokens_tensor.item()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        
        return avg_loss, perplexity
    
    def _train_epoch_pipeline(self, epoch: int):
        """Training loop with pipeline parallelism."""
        total_loss = 0.0
        num_steps = 0
        
        num_optimizer_steps = len(self.train_loader.dataloader) // self.config['grad_acc_steps']
        if num_optimizer_steps < 1:
            num_optimizer_steps = 1
        
        # Progress bar for pipeline training
        pbar = tqdm(
            range(num_optimizer_steps),
            desc=f"Epoch {epoch+1} [Pipeline]",
            disable=(self.global_rank != 0),
            ncols=100,
        )
        
        for step in pbar:
            step_loss, step_ppl = self.pipeline_trainer.train_step(
                self.train_loader,
                self.tensor_shapes,
                self.device,
                torch.float32,
            )
            
            # Sync tied weights after each optimization step
            if hasattr(self.gpt2_stage, 'sync_tied_weights_grad'):
                self.gpt2_stage.sync_tied_weights_grad()
            
            if step_loss is not None:
                total_loss += step_loss
                num_steps += 1
                
                # Update progress bar with metrics
                if self.global_rank == 0:
                    avg_loss = total_loss / num_steps
                    current_ppl = math.exp(step_loss) if step_loss < 20 else float('inf')
                    pbar.set_postfix({
                        'loss': f'{step_loss:.3f}',
                        'ppl': f'{current_ppl:.1f}',
                        'avg_loss': f'{avg_loss:.3f}'
                    })
        
        if num_steps > 0:
            avg_loss = total_loss / num_steps
            perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
            return avg_loss, perplexity
        else:
            return None, None
    
    def _validate_epoch(self, epoch: int):
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        if self.is_pipeline:
            # Use pipeline evaluator
            from .parallelism import PipelineDataLoader
            
            # Wrap validation loader if not already wrapped
            val_loader = self.val_loader
            if not isinstance(val_loader, PipelineDataLoader):
                val_loader = PipelineDataLoader(val_loader, grad_acc_steps=1, task_type='clm')
            
            # Compute validation-specific tensor shapes (uses different batch size)
            val_batch_size = 128  # Must match val_loader batch_size
            max_seq_length = self.config.get('max_seq_length', 512)
            hidden_dim = self.config['model_config']['n_embd']
            val_tensor_shapes = (val_batch_size, max_seq_length, hidden_dim)
            
            # Run pipeline evaluation
            val_loss, val_ppl = self.pipeline_trainer.evaluate(
                val_loader,
                val_tensor_shapes,  # Use validation-specific shapes
                self.device,
                torch.float32,
            )
            
            # Broadcast result from last stage to all ranks for consistent reporting
            if val_loss is None:
                val_loss = 0.0
                val_ppl = 0.0
            
            # Broadcast from last stage
            loss_tensor = torch.tensor([val_loss, val_ppl], device=self.device)
            if dist.is_initialized():
                # Get last stage rank in pipeline group
                pp_group = self.pg_manager.get_group('pp')
                pp_size = dist.get_world_size(pp_group)
                last_stage_rank = pp_size - 1
                dist.broadcast(loss_tensor, src=last_stage_rank, group=pp_group)
            
            return loss_tensor[0].item(), loss_tensor[1].item()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validating Epoch {epoch+1}", disable=(self.global_rank != 0), ncols=100)
            
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                batch_size, seq_len = input_ids.shape
                position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
                
                logits = self.model(input_ids, position_ids=position_ids)
                
                loss = self.criterion(
                    logits.view(-1, self.vocab_size),
                    labels.view(-1)
                )
                
                num_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                
                # Show running metrics
                if self.global_rank == 0 and total_tokens > 0:
                    curr_loss = total_loss / total_tokens
                    curr_ppl = math.exp(curr_loss) if curr_loss < 20 else float('inf')
                    pbar.set_postfix({'loss': f'{curr_loss:.3f}', 'ppl': f'{curr_ppl:.1f}'})
        
        # Aggregate across ranks
        if dist.is_initialized() and dist.get_world_size() > 1:
            loss_tensor = torch.tensor(total_loss, device=self.device)
            tokens_tensor = torch.tensor(total_tokens, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor.item()
            total_tokens = tokens_tensor.item()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        
        return avg_loss, perplexity
    
    def _save_checkpoint(self, path: str):
        """
        Save model checkpoint.
        
        In 3D parallelism mode, each rank saves its shard with naming:
            {path}_pp{pp_rank}_tp{tp_rank}.pt
        
        The shards can be merged later using the merge_checkpoints utility.
        """
        import os
        
        # Get parallelism info
        pp_rank = getattr(self, 'pp_rank', 0)
        tp_rank = 0
        if hasattr(self, 'pg_manager'):
            try:
                tp_group = self.pg_manager.get_group('tp')
                tp_rank = dist.get_rank(tp_group)
            except:
                pass
        
        # Extract state dict
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'local_module'):
            state_dict = self.model.model.local_module.state_dict()
        elif hasattr(self.model, 'module'):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        
        # Create filename with rank info
        base_path = path.replace('.pt', '')
        shard_path = f"{base_path}_pp{pp_rank}_tp{tp_rank}.pt"
        
        # Save with metadata
        checkpoint = {
            'model_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': self.config,
            'parallelism_info': {
                'pp_rank': pp_rank,
                'tp_rank': tp_rank,
                'pp_size': getattr(self, 'pp_size', 1),
                'tp_size': self.config.get('tp_size', 1),
                'dp_size': self.config.get('dp_size', 1),
            }
        }
        
        torch.save(checkpoint, shard_path)
        
        if self.global_rank == 0:
            print(f"  💾 Saved shard: {shard_path}")
    
    def _evaluate_generation_metrics(self):
        """
        Evaluate model using generation-based metrics (ROUGE, BLEU).
        Only runs on rank 0 and requires non-pipeline mode for generation.
        """
        print("\n" + "=" * 60)
        print("📊 COMPUTING GENERATION METRICS (ROUGE/BLEU)")
        print("=" * 60)
        
        if self.is_pipeline:
            # For pipeline mode, generation is complex - skip for now
            print("⚠️  Generation metrics skipped in pipeline mode.")
            print("    (Generation requires autoregressive decoding across stages)")
            return
        
        try:
            from .utils.metrics import evaluate_generation
            
            # Get underlying model for generation
            if hasattr(self.model, 'module'):
                gen_model = self.model.module
            else:
                gen_model = self.model
            
            # Run generation evaluation on a subset
            num_eval_samples = self.config.get('num_eval_samples', 50)
            
            metrics = evaluate_generation(
                model=gen_model,
                tokenizer=self.tokenizer,
                val_dataset=self.val_loader.dataset,
                device=self.device,
                num_samples=num_eval_samples,
                max_new_tokens=self.config.get('max_target_length', 128),
            )
            
            print("\n📈 Generation Metrics:")
            print(f"   ROUGE-1: {metrics['rouge1']:.2f}")
            print(f"   ROUGE-2: {metrics['rouge2']:.2f}")
            print(f"   ROUGE-L: {metrics['rougeL']:.2f}")
            print(f"   BLEU:    {metrics['bleu']:.2f}")
            print("=" * 60)
            
        except Exception as e:
            print(f"⚠️  Error computing generation metrics: {e}")
            import traceback
            traceback.print_exc()